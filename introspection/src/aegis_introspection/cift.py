from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import (
    BinaryFoldMetrics,
    BinaryMethodReport,
    BinaryTaskConfig,
    BinaryTaskDataset,
    BinaryTaskDefinition,
    BinaryTaskError,
    EvaluationStrategy,
    build_activation_classifier,
    build_binary_task_dataset,
    default_binary_task_definitions,
    evaluate_grouped_activation_method,
    stratified_group_splits,
)
from aegis_introspection.features import PoolingMethod, build_feature_key
from aegis_introspection.probe import IntVector, JsonValue, encode_labels, tensor_to_float_matrix


@dataclass(frozen=True)
class CiftProbeConfig:
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    ridge: float
    output_feature_key: str


@dataclass(frozen=True)
class CiftDiagonalLayerCalibration:
    source_feature_key: str
    mean: torch.Tensor
    variance: torch.Tensor


@dataclass(frozen=True)
class CiftDiagonalCalibration:
    source_feature_keys: tuple[str, ...]
    ridge: float
    layers: tuple[CiftDiagonalLayerCalibration, ...]


@dataclass(frozen=True)
class CiftComparisonDataset:
    dataset_id: str
    artifact: ActivationArtifact


@dataclass(frozen=True)
class DatasetCiftProbeComparison:
    dataset_id: str
    source_model_id: str
    source_revision: str
    source_selected_device: str
    baseline: BinaryMethodReport
    cift: BinaryMethodReport
    macro_f1_delta: float
    accuracy_delta: float
    winning_feature_key: str


@dataclass(frozen=True)
class CiftProbeComparisonReport:
    evaluation_strategy: EvaluationStrategy
    task_name: str
    task_description: str
    baseline_feature_key: str
    cift_feature_key: str
    cift_source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    ridge: float
    fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    dataset_count: int
    cift_win_count: int
    baseline_win_count: int
    tie_count: int
    datasets: tuple[DatasetCiftProbeComparison, ...]


def last_quarter_readout_feature_keys(
    artifact: ActivationArtifact,
    pooling_method: PoolingMethod,
) -> tuple[str, ...]:
    layer_indices = tuple(sorted(artifact["metadata"]["layer_indices"]))
    if len(layer_indices) == 0:
        raise BinaryTaskError("Artifact metadata contains no layer indices.")

    layer_count = len(layer_indices)
    selected_count = max(1, int(np.floor(0.25 * layer_count)))
    selected_layers = layer_indices[-selected_count:]
    feature_keys = tuple(build_feature_key(pooling_method, layer_index) for layer_index in selected_layers)
    missing_feature_keys = tuple(key for key in feature_keys if key not in artifact["features"])
    if len(missing_feature_keys) > 0:
        missing = ", ".join(missing_feature_keys)
        raise BinaryTaskError(f"Artifact is missing CIFT source features: {missing}.")
    return feature_keys


def _validate_cift_config(config: CiftProbeConfig) -> None:
    if len(config.source_feature_keys) == 0:
        raise BinaryTaskError("CIFT probe requires at least one source feature.")
    if len(set(config.source_feature_keys)) != len(config.source_feature_keys):
        raise BinaryTaskError("CIFT source feature keys must be unique.")
    if len(config.calibration_source_labels) == 0:
        raise BinaryTaskError("CIFT probe requires at least one calibration source label.")
    if config.ridge <= 0:
        raise BinaryTaskError("CIFT ridge must be greater than 0.")
    if config.output_feature_key == "":
        raise BinaryTaskError("CIFT output feature key must not be empty.")


def _artifact_indices(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    row_indices: tuple[int, ...],
) -> tuple[int, ...]:
    example_index_by_id = {example_id: index for index, example_id in enumerate(artifact["example_ids"])}
    artifact_indices: list[int] = []
    for row_index in row_indices:
        example_id = dataset.example_ids[row_index]
        artifact_index = example_index_by_id.get(example_id)
        if artifact_index is None:
            raise BinaryTaskError(f"Artifact does not contain binary task example '{example_id}'.")
        artifact_indices.append(artifact_index)
    return tuple(artifact_indices)


def _feature_tensor(
    artifact: ActivationArtifact,
    feature_key: str,
) -> torch.Tensor:
    feature_tensor = artifact["features"].get(feature_key)
    if feature_tensor is None:
        raise BinaryTaskError(f"CIFT source feature '{feature_key}' is not present in the artifact.")
    return feature_tensor.float()


def _feature_rows(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    feature_key: str,
    row_indices: tuple[int, ...],
) -> torch.Tensor:
    artifact_indices = _artifact_indices(artifact, dataset, row_indices)
    return _feature_tensor(artifact, feature_key)[list(artifact_indices)]


def fit_cift_diagonal_calibration(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    calibration_row_indices: tuple[int, ...],
    config: CiftProbeConfig,
) -> CiftDiagonalCalibration:
    _validate_cift_config(config)
    if len(calibration_row_indices) == 0:
        raise BinaryTaskError("CIFT calibration requires at least one calibration row.")

    layers: list[CiftDiagonalLayerCalibration] = []
    for source_feature_key in config.source_feature_keys:
        rows = _feature_rows(
            artifact=artifact,
            dataset=dataset,
            feature_key=source_feature_key,
            row_indices=calibration_row_indices,
        )
        layers.append(
            CiftDiagonalLayerCalibration(
                source_feature_key=source_feature_key,
                mean=rows.mean(dim=0),
                variance=rows.var(dim=0, unbiased=False),
            )
        )

    return CiftDiagonalCalibration(
        source_feature_keys=config.source_feature_keys,
        ridge=config.ridge,
        layers=tuple(layers),
    )


def transform_cift_diagonal(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    row_indices: tuple[int, ...],
    calibration: CiftDiagonalCalibration,
) -> torch.Tensor:
    if len(row_indices) == 0:
        raise BinaryTaskError("CIFT transform requires at least one row.")

    layer_scores: list[torch.Tensor] = []
    for layer in calibration.layers:
        rows = _feature_rows(
            artifact=artifact,
            dataset=dataset,
            feature_key=layer.source_feature_key,
            row_indices=row_indices,
        )
        denominator = layer.variance + calibration.ridge
        squared_distance = ((rows - layer.mean) ** 2) / denominator
        layer_scores.append(torch.sqrt(squared_distance.sum(dim=1)))

    return torch.stack(layer_scores, dim=1)


def _calibration_row_indices(
    dataset: BinaryTaskDataset,
    train_indices: IntVector,
    config: CiftProbeConfig,
) -> tuple[int, ...]:
    calibration_labels = set(config.calibration_source_labels)
    return tuple(
        row_index
        for row_index in train_indices.tolist()
        if dataset.source_labels[row_index] in calibration_labels
    )


def _matrix_to_tuple(matrix: NDArray[np.int64]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _mean(values: tuple[float, ...]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: tuple[float, ...]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64)))


def _method_report(
    feature_name: str,
    label_names: tuple[str, ...],
    true_labels: IntVector,
    fold_predictions: tuple[tuple[int, IntVector, IntVector], ...],
) -> BinaryMethodReport:
    label_indices = np.arange(len(label_names), dtype=np.int64)
    confusion_total = np.zeros((len(label_names), len(label_names)), dtype=np.int64)
    folds: list[BinaryFoldMetrics] = []

    for fold_index, y_true, predictions in fold_predictions:
        fold_confusion = confusion_matrix(y_true, predictions, labels=label_indices).astype(np.int64, copy=False)
        confusion_total += fold_confusion
        folds.append(
            BinaryFoldMetrics(
                fold_index=fold_index,
                accuracy=float(accuracy_score(y_true, predictions)),
                macro_f1=float(
                    f1_score(
                        y_true,
                        predictions,
                        average="macro",
                        labels=label_indices,
                        zero_division=0,
                    )
                ),
                confusion_matrix=_matrix_to_tuple(fold_confusion),
            )
        )

    accuracies = tuple(fold.accuracy for fold in folds)
    macro_f1_scores = tuple(fold.macro_f1 for fold in folds)
    return BinaryMethodReport(
        method_name="activation_probe",
        feature_name=feature_name,
        label_names=label_names,
        example_count=int(true_labels.shape[0]),
        accuracy_mean=_mean(accuracies),
        accuracy_std=_std(accuracies),
        macro_f1_mean=_mean(macro_f1_scores),
        macro_f1_std=_std(macro_f1_scores),
        confusion_matrix=_matrix_to_tuple(confusion_total),
        folds=tuple(folds),
    )


def evaluate_grouped_cift_method(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    binary_config: BinaryTaskConfig,
    cift_config: CiftProbeConfig,
) -> BinaryMethodReport:
    _validate_cift_config(cift_config)
    label_encoding = encode_labels(dataset.target_labels)
    encoded_labels = label_encoding.encoded_labels
    splits = stratified_group_splits(encoded_labels, dataset.families, binary_config)
    fold_predictions: list[tuple[int, IntVector, IntVector]] = []

    for split in splits:
        calibration_row_indices = _calibration_row_indices(
            dataset=dataset,
            train_indices=split.train_indices,
            config=cift_config,
        )
        calibration = fit_cift_diagonal_calibration(
            artifact=artifact,
            dataset=dataset,
            calibration_row_indices=calibration_row_indices,
            config=cift_config,
        )
        train_matrix = tensor_to_float_matrix(
            transform_cift_diagonal(
                artifact=artifact,
                dataset=dataset,
                row_indices=tuple(int(index) for index in split.train_indices.tolist()),
                calibration=calibration,
            )
        )
        test_matrix = tensor_to_float_matrix(
            transform_cift_diagonal(
                artifact=artifact,
                dataset=dataset,
                row_indices=tuple(int(index) for index in split.test_indices.tolist()),
                calibration=calibration,
            )
        )
        classifier = build_activation_classifier(binary_config)
        classifier.fit(train_matrix, encoded_labels[split.train_indices])
        predictions = classifier.predict(test_matrix).astype(np.int64, copy=False)
        fold_predictions.append((split.fold_index, encoded_labels[split.test_indices], predictions))

    return _method_report(
        feature_name=cift_config.output_feature_key,
        label_names=label_encoding.label_names,
        true_labels=encoded_labels,
        fold_predictions=tuple(fold_predictions),
    )


def _task_definition(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise BinaryTaskError(f"Expected exactly one binary task named '{task_name}', found {len(matches)}.")
    return matches[0]


def _winning_feature_key(
    baseline: BinaryMethodReport,
    cift: BinaryMethodReport,
) -> str:
    baseline_score = (baseline.macro_f1_mean, baseline.accuracy_mean)
    cift_score = (cift.macro_f1_mean, cift.accuracy_mean)
    if cift_score > baseline_score:
        return cift.feature_name
    if baseline_score > cift_score:
        return baseline.feature_name
    return "tie"


def _compare_dataset(
    dataset: CiftComparisonDataset,
    definition: BinaryTaskDefinition,
    baseline_feature_key: str,
    cift_config: CiftProbeConfig,
    binary_config: BinaryTaskConfig,
) -> DatasetCiftProbeComparison:
    task_dataset = build_binary_task_dataset(dataset.artifact, definition)
    baseline_report = evaluate_grouped_activation_method(
        artifact=dataset.artifact,
        dataset=task_dataset,
        config=replace(binary_config, activation_feature_key=baseline_feature_key),
    )
    cift_report = evaluate_grouped_cift_method(
        artifact=dataset.artifact,
        dataset=task_dataset,
        binary_config=binary_config,
        cift_config=cift_config,
    )
    metadata = dataset.artifact["metadata"]
    return DatasetCiftProbeComparison(
        dataset_id=dataset.dataset_id,
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        baseline=baseline_report,
        cift=cift_report,
        macro_f1_delta=cift_report.macro_f1_mean - baseline_report.macro_f1_mean,
        accuracy_delta=cift_report.accuracy_mean - baseline_report.accuracy_mean,
        winning_feature_key=_winning_feature_key(baseline_report, cift_report),
    )


def compare_grouped_cift_probe(
    datasets: tuple[CiftComparisonDataset, ...],
    task_name: str,
    baseline_feature_key: str,
    cift_config: CiftProbeConfig,
    binary_config: BinaryTaskConfig,
) -> CiftProbeComparisonReport:
    if len(datasets) == 0:
        raise BinaryTaskError("At least one dataset is required for CIFT comparison.")
    if baseline_feature_key == "":
        raise BinaryTaskError("CIFT comparison baseline feature key must not be empty.")
    _validate_cift_config(cift_config)

    definition = _task_definition(task_name)
    dataset_reports = tuple(
        _compare_dataset(
            dataset=dataset,
            definition=definition,
            baseline_feature_key=baseline_feature_key,
            cift_config=cift_config,
            binary_config=binary_config,
        )
        for dataset in datasets
    )
    cift_win_count = sum(1 for dataset in dataset_reports if dataset.winning_feature_key == cift_config.output_feature_key)
    baseline_win_count = sum(1 for dataset in dataset_reports if dataset.winning_feature_key == baseline_feature_key)
    tie_count = sum(1 for dataset in dataset_reports if dataset.winning_feature_key == "tie")

    return CiftProbeComparisonReport(
        evaluation_strategy="stratified_group_kfold",
        task_name=definition.name,
        task_description=definition.description,
        baseline_feature_key=baseline_feature_key,
        cift_feature_key=cift_config.output_feature_key,
        cift_source_feature_keys=cift_config.source_feature_keys,
        calibration_source_labels=cift_config.calibration_source_labels,
        ridge=cift_config.ridge,
        fold_count=binary_config.fold_count,
        random_seed=binary_config.random_seed,
        regularization_c=binary_config.regularization_c,
        max_iter=binary_config.max_iter,
        dataset_count=len(dataset_reports),
        cift_win_count=cift_win_count,
        baseline_win_count=baseline_win_count,
        tie_count=tie_count,
        datasets=dataset_reports,
    )


def _fold_to_json(fold: BinaryFoldMetrics) -> dict[str, JsonValue]:
    return {
        "fold_index": fold.fold_index,
        "accuracy": fold.accuracy,
        "macro_f1": fold.macro_f1,
        "confusion_matrix": [list(row) for row in fold.confusion_matrix],
    }


def _method_to_json(method: BinaryMethodReport) -> dict[str, JsonValue]:
    return {
        "method_name": method.method_name,
        "feature_name": method.feature_name,
        "label_names": list(method.label_names),
        "example_count": method.example_count,
        "accuracy_mean": method.accuracy_mean,
        "accuracy_std": method.accuracy_std,
        "macro_f1_mean": method.macro_f1_mean,
        "macro_f1_std": method.macro_f1_std,
        "confusion_matrix": [list(row) for row in method.confusion_matrix],
        "folds": [_fold_to_json(fold) for fold in method.folds],
    }


def _dataset_comparison_to_json(dataset: DatasetCiftProbeComparison) -> dict[str, JsonValue]:
    return {
        "dataset_id": dataset.dataset_id,
        "source_model_id": dataset.source_model_id,
        "source_revision": dataset.source_revision,
        "source_selected_device": dataset.source_selected_device,
        "baseline": _method_to_json(dataset.baseline),
        "cift": _method_to_json(dataset.cift),
        "macro_f1_delta": dataset.macro_f1_delta,
        "accuracy_delta": dataset.accuracy_delta,
        "winning_feature_key": dataset.winning_feature_key,
    }


def cift_probe_comparison_report_to_json(report: CiftProbeComparisonReport) -> dict[str, JsonValue]:
    return {
        "evaluation_strategy": report.evaluation_strategy,
        "task_name": report.task_name,
        "task_description": report.task_description,
        "baseline_feature_key": report.baseline_feature_key,
        "cift_feature_key": report.cift_feature_key,
        "cift_source_feature_keys": list(report.cift_source_feature_keys),
        "calibration_source_labels": list(report.calibration_source_labels),
        "ridge": report.ridge,
        "fold_count": report.fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "dataset_count": report.dataset_count,
        "cift_win_count": report.cift_win_count,
        "baseline_win_count": report.baseline_win_count,
        "tie_count": report.tie_count,
        "datasets": [_dataset_comparison_to_json(dataset) for dataset in report.datasets],
    }


def write_cift_probe_comparison_json(path: Path, report: CiftProbeComparisonReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cift_probe_comparison_report_to_json(report), file, indent=2)
        file.write("\n")


def render_cift_probe_comparison_markdown(report: CiftProbeComparisonReport) -> str:
    source_features = "`, `".join(report.cift_source_feature_keys)
    calibration_labels = "`, `".join(report.calibration_source_labels)
    lines = [
        "# CIFT-Like Probe Comparison",
        "",
        "## Source",
        "",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Baseline feature: `{report.baseline_feature_key}`",
        f"- CIFT-like feature: `{report.cift_feature_key}`",
        f"- CIFT-like source features: `{source_features}`",
        f"- Calibration source labels: `{calibration_labels}`",
        f"- Ridge: `{report.ridge:.6g}`",
        f"- Dataset count: `{report.dataset_count}`",
        f"- CIFT-like wins: `{report.cift_win_count}`",
        f"- Baseline wins: `{report.baseline_win_count}`",
        f"- Ties: `{report.tie_count}`",
        "",
        "## Dataset Comparison",
        "",
        "| Dataset | Baseline Macro F1 | CIFT-like Macro F1 | Delta Macro F1 | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in report.datasets:
        lines.append(
            f"| `{dataset.dataset_id}` | "
            f"{dataset.baseline.macro_f1_mean:.4f} | "
            f"{dataset.cift.macro_f1_mean:.4f} | "
            f"{dataset.macro_f1_delta:+.4f} | "
            f"`{dataset.winning_feature_key}` |"
        )

    lines.extend(
        [
            "",
            "| Dataset | Baseline Accuracy | CIFT-like Accuracy | Delta Accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset in report.datasets:
        lines.append(
            f"| `{dataset.dataset_id}` | "
            f"{dataset.baseline.accuracy_mean:.4f} | "
            f"{dataset.cift.accuracy_mean:.4f} | "
            f"{dataset.accuracy_delta:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Confusion Matrices",
            "",
        ]
    )
    for dataset in report.datasets:
        lines.append(f"### {dataset.dataset_id} / {dataset.baseline.feature_name}")
        lines.append("")
        lines.append("```text")
        for row in dataset.baseline.confusion_matrix:
            lines.append(str(list(row)))
        lines.append("```")
        lines.append("")
        lines.append(f"### {dataset.dataset_id} / {dataset.cift.feature_name}")
        lines.append("")
        lines.append("```text")
        for row in dataset.cift.confusion_matrix:
            lines.append(str(list(row)))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def write_cift_probe_comparison_markdown(path: Path, report: CiftProbeComparisonReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cift_probe_comparison_markdown(report), encoding="utf-8")
