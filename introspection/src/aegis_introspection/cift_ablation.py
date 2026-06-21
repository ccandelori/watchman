from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
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
from aegis_introspection.probe import IntVector, JsonValue, encode_labels, tensor_to_float_matrix


CiftAblationRepresentation: TypeAlias = Literal[
    "diagonal_distance",
    "standardized_residual_concat",
    "absolute_standardized_residual_concat",
]
CiftAblationClassifierMode: TypeAlias = Literal["standard_scaled_logreg", "raw_logreg"]
FloatMatrix: TypeAlias = NDArray[np.float64]


class _Classifier(Protocol):
    def fit(self, matrix: FloatMatrix, labels: IntVector) -> "_Classifier":
        ...

    def predict(self, matrix: FloatMatrix) -> IntVector:
        ...


@dataclass(frozen=True)
class CiftAblationVariant:
    variant_id: str
    feature_name: str
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    representation: CiftAblationRepresentation
    classifier_mode: CiftAblationClassifierMode
    ridge: float


@dataclass(frozen=True)
class CiftAblationDataset:
    dataset_id: str
    artifact: ActivationArtifact


@dataclass(frozen=True)
class CiftAblationVariantReport:
    variant_id: str
    representation: CiftAblationRepresentation
    classifier_mode: CiftAblationClassifierMode
    feature_name: str
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    ridge: float
    method_name: str
    label_names: tuple[str, ...]
    example_count: int
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    folds: tuple[BinaryFoldMetrics, ...]


@dataclass(frozen=True)
class DatasetCiftAblationReport:
    dataset_id: str
    source_model_id: str
    source_revision: str
    source_selected_device: str
    baseline: BinaryMethodReport
    variants: tuple[CiftAblationVariantReport, ...]
    best_variant: CiftAblationVariantReport
    macro_f1_delta: float
    accuracy_delta: float
    winning_feature_key: str


@dataclass(frozen=True)
class CiftAblationReport:
    evaluation_strategy: EvaluationStrategy
    task_name: str
    task_description: str
    baseline_feature_key: str
    fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    dataset_count: int
    variant_count: int
    ablation_win_count: int
    baseline_win_count: int
    tie_count: int
    datasets: tuple[DatasetCiftAblationReport, ...]


@dataclass(frozen=True)
class _LayerCalibration:
    source_feature_key: str
    mean: torch.Tensor
    variance: torch.Tensor


@dataclass(frozen=True)
class _AblationCalibration:
    variant: CiftAblationVariant
    layers: tuple[_LayerCalibration, ...]


def _validate_variant(variant: CiftAblationVariant) -> None:
    if variant.variant_id == "":
        raise BinaryTaskError("CIFT ablation variant id must not be empty.")
    if variant.feature_name == "":
        raise BinaryTaskError("CIFT ablation feature name must not be empty.")
    if len(variant.source_feature_keys) == 0:
        raise BinaryTaskError(f"CIFT ablation variant '{variant.variant_id}' requires source features.")
    if len(set(variant.source_feature_keys)) != len(variant.source_feature_keys):
        raise BinaryTaskError(f"CIFT ablation variant '{variant.variant_id}' source features must be unique.")
    if len(variant.calibration_source_labels) == 0:
        raise BinaryTaskError(f"CIFT ablation variant '{variant.variant_id}' requires calibration labels.")
    if variant.ridge <= 0:
        raise BinaryTaskError(f"CIFT ablation variant '{variant.variant_id}' ridge must be greater than 0.")
    if variant.representation not in (
        "diagonal_distance",
        "standardized_residual_concat",
        "absolute_standardized_residual_concat",
    ):
        raise BinaryTaskError(
            f"CIFT ablation variant '{variant.variant_id}' has unsupported representation "
            f"'{variant.representation}'."
        )
    if variant.classifier_mode not in ("standard_scaled_logreg", "raw_logreg"):
        raise BinaryTaskError(
            f"CIFT ablation variant '{variant.variant_id}' has unsupported classifier mode "
            f"'{variant.classifier_mode}'."
        )


def _validate_variants(variants: tuple[CiftAblationVariant, ...]) -> None:
    if len(variants) == 0:
        raise BinaryTaskError("At least one CIFT ablation variant is required.")
    for variant in variants:
        _validate_variant(variant)
    if len({variant.variant_id for variant in variants}) != len(variants):
        raise BinaryTaskError("CIFT ablation variant ids must be unique.")
    if len({variant.feature_name for variant in variants}) != len(variants):
        raise BinaryTaskError("CIFT ablation feature names must be unique.")


def _task_definition(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise BinaryTaskError(f"Expected exactly one binary task named '{task_name}', found {len(matches)}.")
    return matches[0]


def _artifact_index_by_id(artifact: ActivationArtifact) -> dict[str, int]:
    return {example_id: index for index, example_id in enumerate(artifact["example_ids"])}


def _artifact_indices_for_dataset_rows(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    row_indices: tuple[int, ...],
) -> tuple[int, ...]:
    index_by_id = _artifact_index_by_id(artifact)
    artifact_indices: list[int] = []
    for row_index in row_indices:
        example_id = dataset.example_ids[row_index]
        artifact_index = index_by_id.get(example_id)
        if artifact_index is None:
            raise BinaryTaskError(f"Artifact does not contain binary task example '{example_id}'.")
        artifact_indices.append(artifact_index)
    return tuple(artifact_indices)


def _calibration_artifact_indices(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    train_indices: IntVector,
    variant: CiftAblationVariant,
) -> tuple[int, ...]:
    calibration_labels = set(variant.calibration_source_labels)
    task_example_ids = set(dataset.example_ids)
    train_example_ids = {dataset.example_ids[index] for index in train_indices.tolist()}
    calibration_indices: list[int] = []

    for artifact_index, example_id in enumerate(artifact["example_ids"]):
        label = artifact["labels"][artifact_index]
        if label not in calibration_labels:
            continue
        if example_id in task_example_ids and example_id not in train_example_ids:
            continue
        calibration_indices.append(artifact_index)

    if len(calibration_indices) == 0:
        raise BinaryTaskError(f"CIFT ablation variant '{variant.variant_id}' has no calibration rows.")
    return tuple(calibration_indices)


def _feature_tensor(artifact: ActivationArtifact, feature_key: str) -> torch.Tensor:
    feature_tensor = artifact["features"].get(feature_key)
    if feature_tensor is None:
        raise BinaryTaskError(f"CIFT ablation source feature '{feature_key}' is not present in the artifact.")
    return feature_tensor.float()


def _feature_rows(
    artifact: ActivationArtifact,
    feature_key: str,
    artifact_indices: tuple[int, ...],
) -> torch.Tensor:
    return _feature_tensor(artifact, feature_key)[list(artifact_indices)]


def _fit_calibration(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    train_indices: IntVector,
    variant: CiftAblationVariant,
) -> _AblationCalibration:
    calibration_indices = _calibration_artifact_indices(
        artifact=artifact,
        dataset=dataset,
        train_indices=train_indices,
        variant=variant,
    )
    layers: list[_LayerCalibration] = []
    for source_feature_key in variant.source_feature_keys:
        rows = _feature_rows(
            artifact=artifact,
            feature_key=source_feature_key,
            artifact_indices=calibration_indices,
        )
        layers.append(
            _LayerCalibration(
                source_feature_key=source_feature_key,
                mean=rows.mean(dim=0),
                variance=rows.var(dim=0, unbiased=False),
            )
        )
    return _AblationCalibration(variant=variant, layers=tuple(layers))


def _diagonal_distance_matrix(
    artifact: ActivationArtifact,
    artifact_indices: tuple[int, ...],
    calibration: _AblationCalibration,
) -> torch.Tensor:
    layer_scores: list[torch.Tensor] = []
    for layer in calibration.layers:
        rows = _feature_rows(
            artifact=artifact,
            feature_key=layer.source_feature_key,
            artifact_indices=artifact_indices,
        )
        denominator = layer.variance + calibration.variant.ridge
        squared_distance = ((rows - layer.mean) ** 2) / denominator
        layer_scores.append(torch.sqrt(squared_distance.sum(dim=1)))
    return torch.stack(layer_scores, dim=1)


def _standardized_residual_matrix(
    artifact: ActivationArtifact,
    artifact_indices: tuple[int, ...],
    calibration: _AblationCalibration,
) -> torch.Tensor:
    layer_residuals: list[torch.Tensor] = []
    for layer in calibration.layers:
        rows = _feature_rows(
            artifact=artifact,
            feature_key=layer.source_feature_key,
            artifact_indices=artifact_indices,
        )
        denominator = torch.sqrt(layer.variance + calibration.variant.ridge)
        layer_residuals.append((rows - layer.mean) / denominator)
    return torch.cat(layer_residuals, dim=1)


def _absolute_standardized_residual_matrix(
    artifact: ActivationArtifact,
    artifact_indices: tuple[int, ...],
    calibration: _AblationCalibration,
) -> torch.Tensor:
    return torch.abs(
        _standardized_residual_matrix(
            artifact=artifact,
            artifact_indices=artifact_indices,
            calibration=calibration,
        )
    )


def _transform_variant(
    artifact: ActivationArtifact,
    artifact_indices: tuple[int, ...],
    calibration: _AblationCalibration,
) -> torch.Tensor:
    if len(artifact_indices) == 0:
        raise BinaryTaskError(f"CIFT ablation variant '{calibration.variant.variant_id}' has no transform rows.")
    if calibration.variant.representation == "diagonal_distance":
        return _diagonal_distance_matrix(
            artifact=artifact,
            artifact_indices=artifact_indices,
            calibration=calibration,
        )
    if calibration.variant.representation == "standardized_residual_concat":
        return _standardized_residual_matrix(
            artifact=artifact,
            artifact_indices=artifact_indices,
            calibration=calibration,
        )
    if calibration.variant.representation == "absolute_standardized_residual_concat":
        return _absolute_standardized_residual_matrix(
            artifact=artifact,
            artifact_indices=artifact_indices,
            calibration=calibration,
        )
    raise BinaryTaskError(
        f"CIFT ablation variant '{calibration.variant.variant_id}' has unsupported representation "
        f"'{calibration.variant.representation}'."
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


def _variant_report(variant: CiftAblationVariant, method: BinaryMethodReport) -> CiftAblationVariantReport:
    return CiftAblationVariantReport(
        variant_id=variant.variant_id,
        representation=variant.representation,
        classifier_mode=variant.classifier_mode,
        feature_name=variant.feature_name,
        source_feature_keys=variant.source_feature_keys,
        calibration_source_labels=variant.calibration_source_labels,
        ridge=variant.ridge,
        method_name=method.method_name,
        label_names=method.label_names,
        example_count=method.example_count,
        accuracy_mean=method.accuracy_mean,
        accuracy_std=method.accuracy_std,
        macro_f1_mean=method.macro_f1_mean,
        macro_f1_std=method.macro_f1_std,
        confusion_matrix=method.confusion_matrix,
        folds=method.folds,
    )


def _build_variant_classifier(config: BinaryTaskConfig, variant: CiftAblationVariant) -> _Classifier:
    if variant.classifier_mode == "standard_scaled_logreg":
        return build_activation_classifier(config)
    if variant.classifier_mode == "raw_logreg":
        return LogisticRegression(
            C=config.regularization_c,
            class_weight="balanced",
            max_iter=config.max_iter,
            random_state=config.random_seed,
        )
    raise BinaryTaskError(
        f"CIFT ablation variant '{variant.variant_id}' has unsupported classifier mode "
        f"'{variant.classifier_mode}'."
    )


def evaluate_grouped_cift_ablation_variant(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    binary_config: BinaryTaskConfig,
    variant: CiftAblationVariant,
) -> CiftAblationVariantReport:
    _validate_variant(variant)
    label_encoding = encode_labels(dataset.target_labels)
    encoded_labels = label_encoding.encoded_labels
    splits = stratified_group_splits(encoded_labels, dataset.families, binary_config)
    fold_predictions: list[tuple[int, IntVector, IntVector]] = []

    for split in splits:
        calibration = _fit_calibration(
            artifact=artifact,
            dataset=dataset,
            train_indices=split.train_indices,
            variant=variant,
        )
        train_artifact_indices = _artifact_indices_for_dataset_rows(
            artifact=artifact,
            dataset=dataset,
            row_indices=tuple(int(index) for index in split.train_indices.tolist()),
        )
        test_artifact_indices = _artifact_indices_for_dataset_rows(
            artifact=artifact,
            dataset=dataset,
            row_indices=tuple(int(index) for index in split.test_indices.tolist()),
        )
        train_matrix = tensor_to_float_matrix(
            _transform_variant(
                artifact=artifact,
                artifact_indices=train_artifact_indices,
                calibration=calibration,
            )
        )
        test_matrix = tensor_to_float_matrix(
            _transform_variant(
                artifact=artifact,
                artifact_indices=test_artifact_indices,
                calibration=calibration,
            )
        )
        classifier = _build_variant_classifier(binary_config, variant)
        classifier.fit(train_matrix, encoded_labels[split.train_indices])
        predictions = classifier.predict(test_matrix).astype(np.int64, copy=False)
        fold_predictions.append((split.fold_index, encoded_labels[split.test_indices], predictions))

    return _variant_report(
        variant=variant,
        method=_method_report(
            feature_name=variant.feature_name,
            label_names=label_encoding.label_names,
            true_labels=encoded_labels,
            fold_predictions=tuple(fold_predictions),
        ),
    )


def _best_variant(variants: tuple[CiftAblationVariantReport, ...]) -> CiftAblationVariantReport:
    if len(variants) == 0:
        raise BinaryTaskError("Cannot select a best CIFT ablation variant from an empty set.")
    return max(variants, key=lambda variant: (variant.macro_f1_mean, variant.accuracy_mean))


def _winning_feature_key(
    baseline: BinaryMethodReport,
    best_variant: CiftAblationVariantReport,
) -> str:
    baseline_score = (baseline.macro_f1_mean, baseline.accuracy_mean)
    variant_score = (best_variant.macro_f1_mean, best_variant.accuracy_mean)
    if variant_score > baseline_score:
        return best_variant.feature_name
    if baseline_score > variant_score:
        return baseline.feature_name
    return "tie"


def _compare_dataset(
    dataset: CiftAblationDataset,
    definition: BinaryTaskDefinition,
    baseline_feature_key: str,
    variants: tuple[CiftAblationVariant, ...],
    binary_config: BinaryTaskConfig,
) -> DatasetCiftAblationReport:
    task_dataset = build_binary_task_dataset(dataset.artifact, definition)
    baseline = evaluate_grouped_activation_method(
        artifact=dataset.artifact,
        dataset=task_dataset,
        config=replace(binary_config, activation_feature_key=baseline_feature_key),
    )
    variant_reports = tuple(
        evaluate_grouped_cift_ablation_variant(
            artifact=dataset.artifact,
            dataset=task_dataset,
            binary_config=binary_config,
            variant=variant,
        )
        for variant in variants
    )
    best_variant = _best_variant(variant_reports)
    metadata = dataset.artifact["metadata"]
    return DatasetCiftAblationReport(
        dataset_id=dataset.dataset_id,
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        baseline=baseline,
        variants=variant_reports,
        best_variant=best_variant,
        macro_f1_delta=best_variant.macro_f1_mean - baseline.macro_f1_mean,
        accuracy_delta=best_variant.accuracy_mean - baseline.accuracy_mean,
        winning_feature_key=_winning_feature_key(baseline, best_variant),
    )


def compare_grouped_cift_ablation(
    datasets: tuple[CiftAblationDataset, ...],
    task_name: str,
    baseline_feature_key: str,
    variants: tuple[CiftAblationVariant, ...],
    binary_config: BinaryTaskConfig,
) -> CiftAblationReport:
    if len(datasets) == 0:
        raise BinaryTaskError("At least one dataset is required for CIFT ablation.")
    if baseline_feature_key == "":
        raise BinaryTaskError("CIFT ablation baseline feature key must not be empty.")
    _validate_variants(variants)

    definition = _task_definition(task_name)
    dataset_reports = tuple(
        _compare_dataset(
            dataset=dataset,
            definition=definition,
            baseline_feature_key=baseline_feature_key,
            variants=variants,
            binary_config=binary_config,
        )
        for dataset in datasets
    )
    baseline_win_count = sum(1 for dataset in dataset_reports if dataset.winning_feature_key == baseline_feature_key)
    tie_count = sum(1 for dataset in dataset_reports if dataset.winning_feature_key == "tie")
    ablation_win_count = len(dataset_reports) - baseline_win_count - tie_count

    return CiftAblationReport(
        evaluation_strategy="stratified_group_kfold",
        task_name=definition.name,
        task_description=definition.description,
        baseline_feature_key=baseline_feature_key,
        fold_count=binary_config.fold_count,
        random_seed=binary_config.random_seed,
        regularization_c=binary_config.regularization_c,
        max_iter=binary_config.max_iter,
        dataset_count=len(dataset_reports),
        variant_count=len(variants),
        ablation_win_count=ablation_win_count,
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


def _variant_to_json(variant: CiftAblationVariantReport) -> dict[str, JsonValue]:
    return {
        "variant_id": variant.variant_id,
        "representation": variant.representation,
        "classifier_mode": variant.classifier_mode,
        "feature_name": variant.feature_name,
        "source_feature_keys": list(variant.source_feature_keys),
        "calibration_source_labels": list(variant.calibration_source_labels),
        "ridge": variant.ridge,
        "method_name": variant.method_name,
        "label_names": list(variant.label_names),
        "example_count": variant.example_count,
        "accuracy_mean": variant.accuracy_mean,
        "accuracy_std": variant.accuracy_std,
        "macro_f1_mean": variant.macro_f1_mean,
        "macro_f1_std": variant.macro_f1_std,
        "confusion_matrix": [list(row) for row in variant.confusion_matrix],
        "folds": [_fold_to_json(fold) for fold in variant.folds],
    }


def _dataset_to_json(dataset: DatasetCiftAblationReport) -> dict[str, JsonValue]:
    return {
        "dataset_id": dataset.dataset_id,
        "source_model_id": dataset.source_model_id,
        "source_revision": dataset.source_revision,
        "source_selected_device": dataset.source_selected_device,
        "baseline": _method_to_json(dataset.baseline),
        "variants": [_variant_to_json(variant) for variant in dataset.variants],
        "best_variant": _variant_to_json(dataset.best_variant),
        "macro_f1_delta": dataset.macro_f1_delta,
        "accuracy_delta": dataset.accuracy_delta,
        "winning_feature_key": dataset.winning_feature_key,
    }


def cift_ablation_report_to_json(report: CiftAblationReport) -> dict[str, JsonValue]:
    return {
        "evaluation_strategy": report.evaluation_strategy,
        "task_name": report.task_name,
        "task_description": report.task_description,
        "baseline_feature_key": report.baseline_feature_key,
        "fold_count": report.fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "dataset_count": report.dataset_count,
        "variant_count": report.variant_count,
        "ablation_win_count": report.ablation_win_count,
        "baseline_win_count": report.baseline_win_count,
        "tie_count": report.tie_count,
        "datasets": [_dataset_to_json(dataset) for dataset in report.datasets],
    }


def write_cift_ablation_json(path: Path, report: CiftAblationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cift_ablation_report_to_json(report), file, indent=2)
        file.write("\n")


def _joined(values: tuple[str, ...]) -> str:
    return "`, `".join(values)


def render_cift_ablation_markdown(report: CiftAblationReport) -> str:
    lines = [
        "# CIFT-Like Ablation",
        "",
        "## Source",
        "",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Baseline feature: `{report.baseline_feature_key}`",
        f"- Dataset count: `{report.dataset_count}`",
        f"- Variant count: `{report.variant_count}`",
        f"- Ablation wins: `{report.ablation_win_count}`",
        f"- Baseline wins: `{report.baseline_win_count}`",
        f"- Ties: `{report.tie_count}`",
        "",
        "## Best Variant by Dataset",
        "",
        "| Dataset | Baseline Macro F1 | Best Variant | Best Variant Macro F1 | Delta Macro F1 | Winner |",
        "|---|---:|---|---:|---:|---|",
    ]
    for dataset in report.datasets:
        lines.append(
            f"| `{dataset.dataset_id}` | "
            f"{dataset.baseline.macro_f1_mean:.4f} | "
            f"`{dataset.best_variant.feature_name}` | "
            f"{dataset.best_variant.macro_f1_mean:.4f} | "
            f"{dataset.macro_f1_delta:+.4f} | "
            f"`{dataset.winning_feature_key}` |"
        )

    lines.extend(
        [
            "",
            "## Aggregate by Variant",
            "",
            "| Variant | Representation | Classifier Mode | Calibration Labels | Mean Macro F1 | Min Macro F1 |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    first_dataset = report.datasets[0]
    for first_variant in first_dataset.variants:
        matching_variants = tuple(
            variant
            for dataset in report.datasets
            for variant in dataset.variants
            if variant.feature_name == first_variant.feature_name
        )
        macro_f1_scores = tuple(variant.macro_f1_mean for variant in matching_variants)
        lines.append(
            f"| `{first_variant.feature_name}` | "
            f"`{first_variant.representation}` | "
            f"`{first_variant.classifier_mode}` | "
            f"`{_joined(first_variant.calibration_source_labels)}` | "
            f"{_mean(macro_f1_scores):.4f} | "
            f"{min(macro_f1_scores):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Variant Results",
            "",
            "| Dataset | Variant | Representation | Classifier Mode | Calibration Labels | Macro F1 | Delta Macro F1 |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for dataset in report.datasets:
        for variant in dataset.variants:
            lines.append(
                f"| `{dataset.dataset_id}` | "
                f"`{variant.feature_name}` | "
                f"`{variant.representation}` | "
                f"`{variant.classifier_mode}` | "
                f"`{_joined(variant.calibration_source_labels)}` | "
                f"{variant.macro_f1_mean:.4f} | "
                f"{variant.macro_f1_mean - dataset.baseline.macro_f1_mean:+.4f} |"
            )

    lines.extend(
        [
            "",
            "## Variant Sources",
            "",
            "| Variant | Source Features | Ridge |",
            "|---|---|---:|",
        ]
    )
    seen_variants: set[str] = set()
    for dataset in report.datasets:
        for variant in dataset.variants:
            if variant.variant_id in seen_variants:
                continue
            seen_variants.add(variant.variant_id)
            lines.append(
                f"| `{variant.feature_name}` | "
                f"`{_joined(variant.source_feature_keys)}` | "
                f"{variant.ridge:.6g} |"
            )

    return "\n".join(lines)


def write_cift_ablation_markdown(path: Path, report: CiftAblationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cift_ablation_markdown(report), encoding="utf-8")
