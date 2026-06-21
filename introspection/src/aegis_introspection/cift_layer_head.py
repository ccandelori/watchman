from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

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


FloatMatrix: TypeAlias = NDArray[np.float64]


class _ProbabilisticClassifier(Protocol):
    classes_: IntVector

    def fit(self, matrix: FloatMatrix, labels: IntVector) -> "_ProbabilisticClassifier":
        ...

    def predict_proba(self, matrix: FloatMatrix) -> FloatMatrix:
        ...


@dataclass(frozen=True)
class CiftLayerHeadConfig:
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    ridge: float
    output_feature_key: str
    risk_label: str


@dataclass(frozen=True)
class CiftLayerHeadComparisonDataset:
    dataset_id: str
    artifact: ActivationArtifact


@dataclass(frozen=True)
class CiftLayerHeadWeightFold:
    fold_index: int
    source_feature_keys: tuple[str, ...]
    weights: tuple[float, ...]
    threshold: float


@dataclass(frozen=True)
class CiftLayerHeadMethodReport:
    method_name: str
    feature_name: str
    label_names: tuple[str, ...]
    example_count: int
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    folds: tuple[BinaryFoldMetrics, ...]
    weight_folds: tuple[CiftLayerHeadWeightFold, ...]


@dataclass(frozen=True)
class DatasetCiftLayerHeadComparison:
    dataset_id: str
    source_model_id: str
    source_revision: str
    source_selected_device: str
    baseline: BinaryMethodReport
    head: CiftLayerHeadMethodReport
    macro_f1_delta: float
    accuracy_delta: float
    winning_feature_key: str


@dataclass(frozen=True)
class CiftLayerHeadComparisonReport:
    evaluation_strategy: EvaluationStrategy
    task_name: str
    task_description: str
    baseline_feature_key: str
    head_feature_key: str
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    risk_label: str
    ridge: float
    fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    dataset_count: int
    head_win_count: int
    baseline_win_count: int
    tie_count: int
    datasets: tuple[DatasetCiftLayerHeadComparison, ...]


@dataclass(frozen=True)
class _LayerCalibration:
    source_feature_key: str
    mean: torch.Tensor
    variance: torch.Tensor


@dataclass(frozen=True)
class _HeadCalibration:
    config: CiftLayerHeadConfig
    layers: tuple[_LayerCalibration, ...]


@dataclass(frozen=True)
class _HeadWeights:
    weights: FloatMatrix
    threshold: float


def _validate_head_config(config: CiftLayerHeadConfig) -> None:
    if len(config.source_feature_keys) == 0:
        raise BinaryTaskError("CIFT layer head requires at least one source feature.")
    if len(set(config.source_feature_keys)) != len(config.source_feature_keys):
        raise BinaryTaskError("CIFT layer head source feature keys must be unique.")
    if len(config.calibration_source_labels) == 0:
        raise BinaryTaskError("CIFT layer head requires at least one calibration source label.")
    if config.ridge <= 0:
        raise BinaryTaskError("CIFT layer head ridge must be greater than 0.")
    if config.output_feature_key == "":
        raise BinaryTaskError("CIFT layer head output feature key must not be empty.")
    if config.risk_label == "":
        raise BinaryTaskError("CIFT layer head risk label must not be empty.")


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
    config: CiftLayerHeadConfig,
) -> tuple[int, ...]:
    calibration_labels = set(config.calibration_source_labels)
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
        raise BinaryTaskError("CIFT layer head has no calibration rows.")
    return tuple(calibration_indices)


def _feature_tensor(artifact: ActivationArtifact, feature_key: str) -> torch.Tensor:
    feature_tensor = artifact["features"].get(feature_key)
    if feature_tensor is None:
        raise BinaryTaskError(f"CIFT layer head source feature '{feature_key}' is not present in the artifact.")
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
    config: CiftLayerHeadConfig,
) -> _HeadCalibration:
    calibration_indices = _calibration_artifact_indices(
        artifact=artifact,
        dataset=dataset,
        train_indices=train_indices,
        config=config,
    )
    layers: list[_LayerCalibration] = []
    for source_feature_key in config.source_feature_keys:
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
    return _HeadCalibration(config=config, layers=tuple(layers))


def _residual_matrix(
    artifact: ActivationArtifact,
    artifact_indices: tuple[int, ...],
    layer: _LayerCalibration,
    ridge: float,
) -> torch.Tensor:
    rows = _feature_rows(
        artifact=artifact,
        feature_key=layer.source_feature_key,
        artifact_indices=artifact_indices,
    )
    denominator = torch.sqrt(layer.variance + ridge)
    return (rows - layer.mean) / denominator


def _risk_label_index(label_names: tuple[str, ...], config: CiftLayerHeadConfig) -> int:
    matches = tuple(index for index, label_name in enumerate(label_names) if label_name == config.risk_label)
    if len(matches) != 1:
        raise BinaryTaskError(f"CIFT layer head risk label '{config.risk_label}' is not in labels {label_names}.")
    return matches[0]


def _other_label_index(label_count: int, risk_label_index: int) -> int:
    if label_count != 2:
        raise BinaryTaskError("CIFT layer head requires exactly two encoded labels.")
    return 1 - risk_label_index


def _risk_probability_column(classifier: _ProbabilisticClassifier, risk_label_index: int) -> int:
    classes = tuple(int(label_index) for label_index in classifier.classes_.tolist())
    if risk_label_index not in classes:
        raise BinaryTaskError(f"CIFT layer classifier was not fitted with risk label index {risk_label_index}.")
    return classes.index(risk_label_index)


def _layer_risk_scores(
    artifact: ActivationArtifact,
    train_artifact_indices: tuple[int, ...],
    test_artifact_indices: tuple[int, ...],
    calibration: _HeadCalibration,
    train_labels: IntVector,
    binary_config: BinaryTaskConfig,
    risk_label_index: int,
) -> tuple[FloatMatrix, FloatMatrix]:
    train_scores = np.zeros((len(train_artifact_indices), len(calibration.layers)), dtype=np.float64)
    test_scores = np.zeros((len(test_artifact_indices), len(calibration.layers)), dtype=np.float64)

    for layer_index, layer in enumerate(calibration.layers):
        train_matrix = tensor_to_float_matrix(
            _residual_matrix(
                artifact=artifact,
                artifact_indices=train_artifact_indices,
                layer=layer,
                ridge=calibration.config.ridge,
            )
        )
        test_matrix = tensor_to_float_matrix(
            _residual_matrix(
                artifact=artifact,
                artifact_indices=test_artifact_indices,
                layer=layer,
                ridge=calibration.config.ridge,
            )
        )
        classifier = build_activation_classifier(binary_config)
        classifier.fit(train_matrix, train_labels)
        risk_column = _risk_probability_column(classifier, risk_label_index)
        train_scores[:, layer_index] = classifier.predict_proba(train_matrix)[:, risk_column]
        test_scores[:, layer_index] = classifier.predict_proba(test_matrix)[:, risk_column]

    return train_scores, test_scores


def _normalized_auc_weights(layer_scores: FloatMatrix, risk_targets: IntVector) -> FloatMatrix:
    raw_weights: list[float] = []
    for layer_index in range(layer_scores.shape[1]):
        try:
            auc = float(roc_auc_score(risk_targets, layer_scores[:, layer_index]))
        except ValueError:
            auc = 0.5
        raw_weights.append(max(0.0, auc - 0.5))

    weights = np.asarray(raw_weights, dtype=np.float64)
    weight_sum = float(weights.sum())
    if weight_sum == 0.0:
        return np.full(layer_scores.shape[1], 1.0 / layer_scores.shape[1], dtype=np.float64)
    return weights / weight_sum


def _predictions_from_threshold(
    risk_scores: FloatMatrix,
    threshold: float,
    risk_label_index: int,
    other_label_index: int,
) -> IntVector:
    return np.asarray(
        [risk_label_index if score >= threshold else other_label_index for score in risk_scores],
        dtype=np.int64,
    )


def _candidate_thresholds(weighted_scores: FloatMatrix) -> tuple[float, ...]:
    unique_scores = np.unique(weighted_scores)
    if unique_scores.shape[0] == 0:
        raise BinaryTaskError("Cannot learn CIFT layer head threshold from empty scores.")
    candidates: list[float] = [float(unique_scores[0] - 1e-12), float(unique_scores[-1] + 1e-12), 0.5]
    for left, right in zip(unique_scores[:-1], unique_scores[1:], strict=True):
        candidates.append(float((left + right) / 2.0))
    return tuple(sorted(set(candidates)))


def _learn_threshold(
    weighted_scores: FloatMatrix,
    encoded_labels: IntVector,
    risk_label_index: int,
    other_label_index: int,
) -> float:
    label_indices = np.arange(2, dtype=np.int64)
    candidates = _candidate_thresholds(weighted_scores)

    def score_threshold(threshold: float) -> tuple[float, float]:
        predictions = _predictions_from_threshold(
            risk_scores=weighted_scores,
            threshold=threshold,
            risk_label_index=risk_label_index,
            other_label_index=other_label_index,
        )
        return (
            float(f1_score(encoded_labels, predictions, average="macro", labels=label_indices, zero_division=0)),
            float(accuracy_score(encoded_labels, predictions)),
        )

    return max(candidates, key=score_threshold)


def _fit_head_weights(
    train_layer_scores: FloatMatrix,
    train_labels: IntVector,
    risk_label_index: int,
    other_label_index: int,
) -> _HeadWeights:
    risk_targets = np.asarray([1 if label == risk_label_index else 0 for label in train_labels], dtype=np.int64)
    weights = _normalized_auc_weights(train_layer_scores, risk_targets)
    weighted_scores = train_layer_scores @ weights
    threshold = _learn_threshold(
        weighted_scores=weighted_scores,
        encoded_labels=train_labels,
        risk_label_index=risk_label_index,
        other_label_index=other_label_index,
    )
    return _HeadWeights(weights=weights, threshold=threshold)


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
    weight_folds: tuple[CiftLayerHeadWeightFold, ...],
) -> CiftLayerHeadMethodReport:
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
    return CiftLayerHeadMethodReport(
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
        weight_folds=weight_folds,
    )


def evaluate_grouped_cift_layer_head_method(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    binary_config: BinaryTaskConfig,
    head_config: CiftLayerHeadConfig,
) -> CiftLayerHeadMethodReport:
    _validate_head_config(head_config)
    label_encoding = encode_labels(dataset.target_labels)
    encoded_labels = label_encoding.encoded_labels
    risk_index = _risk_label_index(label_encoding.label_names, head_config)
    other_index = _other_label_index(len(label_encoding.label_names), risk_index)
    splits = stratified_group_splits(encoded_labels, dataset.families, binary_config)
    fold_predictions: list[tuple[int, IntVector, IntVector]] = []
    weight_folds: list[CiftLayerHeadWeightFold] = []

    for split in splits:
        calibration = _fit_calibration(
            artifact=artifact,
            dataset=dataset,
            train_indices=split.train_indices,
            config=head_config,
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
        train_layer_scores, test_layer_scores = _layer_risk_scores(
            artifact=artifact,
            train_artifact_indices=train_artifact_indices,
            test_artifact_indices=test_artifact_indices,
            calibration=calibration,
            train_labels=encoded_labels[split.train_indices],
            binary_config=binary_config,
            risk_label_index=risk_index,
        )
        head_weights = _fit_head_weights(
            train_layer_scores=train_layer_scores,
            train_labels=encoded_labels[split.train_indices],
            risk_label_index=risk_index,
            other_label_index=other_index,
        )
        test_weighted_scores = test_layer_scores @ head_weights.weights
        predictions = _predictions_from_threshold(
            risk_scores=test_weighted_scores,
            threshold=head_weights.threshold,
            risk_label_index=risk_index,
            other_label_index=other_index,
        )
        fold_predictions.append((split.fold_index, encoded_labels[split.test_indices], predictions))
        weight_folds.append(
            CiftLayerHeadWeightFold(
                fold_index=split.fold_index,
                source_feature_keys=head_config.source_feature_keys,
                weights=tuple(float(weight) for weight in head_weights.weights.tolist()),
                threshold=head_weights.threshold,
            )
        )

    return _method_report(
        feature_name=head_config.output_feature_key,
        label_names=label_encoding.label_names,
        true_labels=encoded_labels,
        fold_predictions=tuple(fold_predictions),
        weight_folds=tuple(weight_folds),
    )


def _winning_feature_key(
    baseline: BinaryMethodReport,
    head: CiftLayerHeadMethodReport,
) -> str:
    baseline_score = (baseline.macro_f1_mean, baseline.accuracy_mean)
    head_score = (head.macro_f1_mean, head.accuracy_mean)
    if head_score > baseline_score:
        return head.feature_name
    if baseline_score > head_score:
        return baseline.feature_name
    return "tie"


def _compare_dataset(
    dataset: CiftLayerHeadComparisonDataset,
    definition: BinaryTaskDefinition,
    baseline_feature_key: str,
    head_config: CiftLayerHeadConfig,
    binary_config: BinaryTaskConfig,
) -> DatasetCiftLayerHeadComparison:
    task_dataset = build_binary_task_dataset(dataset.artifact, definition)
    baseline = evaluate_grouped_activation_method(
        artifact=dataset.artifact,
        dataset=task_dataset,
        config=replace(binary_config, activation_feature_key=baseline_feature_key),
    )
    head = evaluate_grouped_cift_layer_head_method(
        artifact=dataset.artifact,
        dataset=task_dataset,
        binary_config=binary_config,
        head_config=head_config,
    )
    metadata = dataset.artifact["metadata"]
    return DatasetCiftLayerHeadComparison(
        dataset_id=dataset.dataset_id,
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        baseline=baseline,
        head=head,
        macro_f1_delta=head.macro_f1_mean - baseline.macro_f1_mean,
        accuracy_delta=head.accuracy_mean - baseline.accuracy_mean,
        winning_feature_key=_winning_feature_key(baseline, head),
    )


def compare_grouped_cift_layer_head(
    datasets: tuple[CiftLayerHeadComparisonDataset, ...],
    task_name: str,
    baseline_feature_key: str,
    head_config: CiftLayerHeadConfig,
    binary_config: BinaryTaskConfig,
) -> CiftLayerHeadComparisonReport:
    if len(datasets) == 0:
        raise BinaryTaskError("At least one dataset is required for CIFT layer head comparison.")
    if baseline_feature_key == "":
        raise BinaryTaskError("CIFT layer head baseline feature key must not be empty.")
    _validate_head_config(head_config)

    definition = _task_definition(task_name)
    dataset_reports = tuple(
        _compare_dataset(
            dataset=dataset,
            definition=definition,
            baseline_feature_key=baseline_feature_key,
            head_config=head_config,
            binary_config=binary_config,
        )
        for dataset in datasets
    )
    head_win_count = sum(1 for dataset in dataset_reports if dataset.winning_feature_key == head_config.output_feature_key)
    baseline_win_count = sum(1 for dataset in dataset_reports if dataset.winning_feature_key == baseline_feature_key)
    tie_count = sum(1 for dataset in dataset_reports if dataset.winning_feature_key == "tie")

    return CiftLayerHeadComparisonReport(
        evaluation_strategy="stratified_group_kfold",
        task_name=definition.name,
        task_description=definition.description,
        baseline_feature_key=baseline_feature_key,
        head_feature_key=head_config.output_feature_key,
        source_feature_keys=head_config.source_feature_keys,
        calibration_source_labels=head_config.calibration_source_labels,
        risk_label=head_config.risk_label,
        ridge=head_config.ridge,
        fold_count=binary_config.fold_count,
        random_seed=binary_config.random_seed,
        regularization_c=binary_config.regularization_c,
        max_iter=binary_config.max_iter,
        dataset_count=len(dataset_reports),
        head_win_count=head_win_count,
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


def _weight_fold_to_json(fold: CiftLayerHeadWeightFold) -> dict[str, JsonValue]:
    return {
        "fold_index": fold.fold_index,
        "source_feature_keys": list(fold.source_feature_keys),
        "weights": list(fold.weights),
        "threshold": fold.threshold,
    }


def _head_to_json(head: CiftLayerHeadMethodReport) -> dict[str, JsonValue]:
    return {
        "method_name": head.method_name,
        "feature_name": head.feature_name,
        "label_names": list(head.label_names),
        "example_count": head.example_count,
        "accuracy_mean": head.accuracy_mean,
        "accuracy_std": head.accuracy_std,
        "macro_f1_mean": head.macro_f1_mean,
        "macro_f1_std": head.macro_f1_std,
        "confusion_matrix": [list(row) for row in head.confusion_matrix],
        "folds": [_fold_to_json(fold) for fold in head.folds],
        "weight_folds": [_weight_fold_to_json(fold) for fold in head.weight_folds],
    }


def _baseline_to_json(method: BinaryMethodReport) -> dict[str, JsonValue]:
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


def _dataset_to_json(dataset: DatasetCiftLayerHeadComparison) -> dict[str, JsonValue]:
    return {
        "dataset_id": dataset.dataset_id,
        "source_model_id": dataset.source_model_id,
        "source_revision": dataset.source_revision,
        "source_selected_device": dataset.source_selected_device,
        "baseline": _baseline_to_json(dataset.baseline),
        "head": _head_to_json(dataset.head),
        "macro_f1_delta": dataset.macro_f1_delta,
        "accuracy_delta": dataset.accuracy_delta,
        "winning_feature_key": dataset.winning_feature_key,
    }


def cift_layer_head_report_to_json(report: CiftLayerHeadComparisonReport) -> dict[str, JsonValue]:
    return {
        "evaluation_strategy": report.evaluation_strategy,
        "task_name": report.task_name,
        "task_description": report.task_description,
        "baseline_feature_key": report.baseline_feature_key,
        "head_feature_key": report.head_feature_key,
        "source_feature_keys": list(report.source_feature_keys),
        "calibration_source_labels": list(report.calibration_source_labels),
        "risk_label": report.risk_label,
        "ridge": report.ridge,
        "fold_count": report.fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "dataset_count": report.dataset_count,
        "head_win_count": report.head_win_count,
        "baseline_win_count": report.baseline_win_count,
        "tie_count": report.tie_count,
        "datasets": [_dataset_to_json(dataset) for dataset in report.datasets],
    }


def write_cift_layer_head_json(path: Path, report: CiftLayerHeadComparisonReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cift_layer_head_report_to_json(report), file, indent=2)
        file.write("\n")


def _joined(values: tuple[str, ...]) -> str:
    return "`, `".join(values)


def _mean_weight(dataset: DatasetCiftLayerHeadComparison, source_feature_index: int) -> float:
    weights = tuple(fold.weights[source_feature_index] for fold in dataset.head.weight_folds)
    return _mean(weights)


def render_cift_layer_head_markdown(report: CiftLayerHeadComparisonReport) -> str:
    lines = [
        "# CIFT Layer-Weighted Head",
        "",
        "## Source",
        "",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Baseline feature: `{report.baseline_feature_key}`",
        f"- Head feature: `{report.head_feature_key}`",
        f"- Source features: `{_joined(report.source_feature_keys)}`",
        f"- Calibration source labels: `{_joined(report.calibration_source_labels)}`",
        f"- Risk label: `{report.risk_label}`",
        f"- Dataset count: `{report.dataset_count}`",
        f"- Head wins: `{report.head_win_count}`",
        f"- Baseline wins: `{report.baseline_win_count}`",
        f"- Ties: `{report.tie_count}`",
        "",
        "## Dataset Comparison",
        "",
        "| Dataset | Baseline Macro F1 | Head Macro F1 | Delta Macro F1 | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in report.datasets:
        lines.append(
            f"| `{dataset.dataset_id}` | "
            f"{dataset.baseline.macro_f1_mean:.4f} | "
            f"{dataset.head.macro_f1_mean:.4f} | "
            f"{dataset.macro_f1_delta:+.4f} | "
            f"`{dataset.winning_feature_key}` |"
        )

    lines.extend(
        [
            "",
            "## Mean Layer Weights",
            "",
            "| Dataset | Source Feature | Mean Weight |",
            "|---|---|---:|",
        ]
    )
    for dataset in report.datasets:
        for source_feature_index, source_feature_key in enumerate(report.source_feature_keys):
            lines.append(
                f"| `{dataset.dataset_id}` | "
                f"`{source_feature_key}` | "
                f"{_mean_weight(dataset, source_feature_index):.4f} |"
            )

    return "\n".join(lines)


def write_cift_layer_head_markdown(path: Path, report: CiftLayerHeadComparisonReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cift_layer_head_markdown(report), encoding="utf-8")
