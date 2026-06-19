from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

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
from aegis_introspection.error_analysis import (
    BinaryExamplePrediction,
    BinaryMethodErrorAnalysis,
    summarize_family_predictions,
)
from aegis_introspection.probe import IntVector, JsonValue, encode_labels, tensor_to_float_matrix


FloatMatrix: TypeAlias = NDArray[np.float64]


class _ProbabilisticClassifier(Protocol):
    classes_: IntVector

    def fit(self, matrix: FloatMatrix, labels: IntVector) -> "_ProbabilisticClassifier":
        ...

    def predict(self, matrix: FloatMatrix) -> IntVector:
        ...

    def predict_proba(self, matrix: FloatMatrix) -> FloatMatrix:
        ...


@dataclass(frozen=True)
class CiftMetaHeadVariant:
    variant_id: str
    feature_name: str
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    ridge: float
    risk_label: str
    inner_fold_count: int


@dataclass(frozen=True)
class CiftMetaHeadComparisonDataset:
    dataset_id: str
    artifact: ActivationArtifact


@dataclass(frozen=True)
class CiftMetaHeadFold:
    fold_index: int
    source_feature_keys: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float


@dataclass(frozen=True)
class CiftMetaHeadVariantReport:
    variant_id: str
    feature_name: str
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    ridge: float
    risk_label: str
    inner_fold_count: int
    method_name: str
    label_names: tuple[str, ...]
    example_count: int
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    folds: tuple[BinaryFoldMetrics, ...]
    meta_folds: tuple[CiftMetaHeadFold, ...]


@dataclass(frozen=True)
class DatasetCiftMetaHeadComparison:
    dataset_id: str
    source_model_id: str
    source_revision: str
    source_selected_device: str
    baseline: BinaryMethodReport
    variants: tuple[CiftMetaHeadVariantReport, ...]
    best_variant: CiftMetaHeadVariantReport
    macro_f1_delta: float
    accuracy_delta: float
    winning_feature_key: str


@dataclass(frozen=True)
class CiftMetaHeadComparisonReport:
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
    meta_head_win_count: int
    baseline_win_count: int
    tie_count: int
    datasets: tuple[DatasetCiftMetaHeadComparison, ...]


@dataclass(frozen=True)
class _SourceCalibration:
    source_feature_key: str
    mean: torch.Tensor
    variance: torch.Tensor


@dataclass(frozen=True)
class _OuterFoldScores:
    train_scores: FloatMatrix
    test_scores: FloatMatrix


@dataclass(frozen=True)
class _MetaHeadFit:
    classifier: Pipeline
    coefficients: tuple[float, ...]
    intercept: float


def _validate_variant(variant: CiftMetaHeadVariant) -> None:
    if variant.variant_id == "":
        raise BinaryTaskError("CIFT meta-head variant id must not be empty.")
    if variant.feature_name == "":
        raise BinaryTaskError(f"CIFT meta-head variant '{variant.variant_id}' feature name must not be empty.")
    if len(variant.source_feature_keys) == 0:
        raise BinaryTaskError(f"CIFT meta-head variant '{variant.variant_id}' requires source features.")
    if len(set(variant.source_feature_keys)) != len(variant.source_feature_keys):
        raise BinaryTaskError(f"CIFT meta-head variant '{variant.variant_id}' source features must be unique.")
    if len(variant.calibration_source_labels) == 0:
        raise BinaryTaskError(f"CIFT meta-head variant '{variant.variant_id}' requires calibration labels.")
    if variant.ridge <= 0:
        raise BinaryTaskError(f"CIFT meta-head variant '{variant.variant_id}' ridge must be greater than 0.")
    if variant.risk_label == "":
        raise BinaryTaskError(f"CIFT meta-head variant '{variant.variant_id}' risk label must not be empty.")
    if variant.inner_fold_count < 2:
        raise BinaryTaskError(f"CIFT meta-head variant '{variant.variant_id}' inner_fold_count must be at least 2.")


def _validate_variants(variants: tuple[CiftMetaHeadVariant, ...]) -> None:
    if len(variants) == 0:
        raise BinaryTaskError("At least one CIFT meta-head variant is required.")
    for variant in variants:
        _validate_variant(variant)
    if len({variant.variant_id for variant in variants}) != len(variants):
        raise BinaryTaskError("CIFT meta-head variant ids must be unique.")
    if len({variant.feature_name for variant in variants}) != len(variants):
        raise BinaryTaskError("CIFT meta-head feature names must be unique.")


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
    variant: CiftMetaHeadVariant,
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
        raise BinaryTaskError(f"CIFT meta-head variant '{variant.variant_id}' has no calibration rows.")
    return tuple(calibration_indices)


def _feature_tensor(artifact: ActivationArtifact, feature_key: str, variant: CiftMetaHeadVariant) -> torch.Tensor:
    feature_tensor = artifact["features"].get(feature_key)
    if feature_tensor is None:
        raise BinaryTaskError(
            f"CIFT meta-head variant '{variant.variant_id}' source feature '{feature_key}' is not present."
        )
    return feature_tensor.float()


def _feature_rows(
    artifact: ActivationArtifact,
    feature_key: str,
    artifact_indices: tuple[int, ...],
    variant: CiftMetaHeadVariant,
) -> torch.Tensor:
    return _feature_tensor(artifact, feature_key, variant)[list(artifact_indices)]


def _fit_source_calibration(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    train_indices: IntVector,
    variant: CiftMetaHeadVariant,
    source_feature_key: str,
) -> _SourceCalibration:
    calibration_indices = _calibration_artifact_indices(
        artifact=artifact,
        dataset=dataset,
        train_indices=train_indices,
        variant=variant,
    )
    rows = _feature_rows(
        artifact=artifact,
        feature_key=source_feature_key,
        artifact_indices=calibration_indices,
        variant=variant,
    )
    return _SourceCalibration(
        source_feature_key=source_feature_key,
        mean=rows.mean(dim=0),
        variance=rows.var(dim=0, unbiased=False),
    )


def _residual_matrix(
    artifact: ActivationArtifact,
    artifact_indices: tuple[int, ...],
    calibration: _SourceCalibration,
    variant: CiftMetaHeadVariant,
) -> torch.Tensor:
    rows = _feature_rows(
        artifact=artifact,
        feature_key=calibration.source_feature_key,
        artifact_indices=artifact_indices,
        variant=variant,
    )
    denominator = torch.sqrt(calibration.variance + variant.ridge)
    return (rows - calibration.mean) / denominator


def _risk_label_index(label_names: tuple[str, ...], variant: CiftMetaHeadVariant) -> int:
    matches = tuple(index for index, label_name in enumerate(label_names) if label_name == variant.risk_label)
    if len(matches) != 1:
        raise BinaryTaskError(f"CIFT meta-head risk label '{variant.risk_label}' is not in labels {label_names}.")
    return matches[0]


def _risk_probability_column(classifier: _ProbabilisticClassifier, risk_label_index: int) -> int:
    classes = tuple(int(label_index) for label_index in classifier.classes_.tolist())
    if risk_label_index not in classes:
        raise BinaryTaskError(f"CIFT meta-head classifier was not fitted with risk label index {risk_label_index}.")
    return classes.index(risk_label_index)


def _inner_config(binary_config: BinaryTaskConfig, variant: CiftMetaHeadVariant) -> BinaryTaskConfig:
    return replace(binary_config, fold_count=variant.inner_fold_count)


def _predict_source_risk_scores(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    source_feature_key: str,
    train_indices: IntVector,
    predict_indices: IntVector,
    encoded_labels: IntVector,
    binary_config: BinaryTaskConfig,
    variant: CiftMetaHeadVariant,
    risk_label_index: int,
) -> FloatMatrix:
    calibration = _fit_source_calibration(
        artifact=artifact,
        dataset=dataset,
        train_indices=train_indices,
        variant=variant,
        source_feature_key=source_feature_key,
    )
    train_artifact_indices = _artifact_indices_for_dataset_rows(
        artifact=artifact,
        dataset=dataset,
        row_indices=tuple(int(index) for index in train_indices.tolist()),
    )
    predict_artifact_indices = _artifact_indices_for_dataset_rows(
        artifact=artifact,
        dataset=dataset,
        row_indices=tuple(int(index) for index in predict_indices.tolist()),
    )
    train_matrix = tensor_to_float_matrix(
        _residual_matrix(
            artifact=artifact,
            artifact_indices=train_artifact_indices,
            calibration=calibration,
            variant=variant,
        )
    ).astype(np.float64, copy=False)
    predict_matrix = tensor_to_float_matrix(
        _residual_matrix(
            artifact=artifact,
            artifact_indices=predict_artifact_indices,
            calibration=calibration,
            variant=variant,
        )
    ).astype(np.float64, copy=False)
    classifier = cast(_ProbabilisticClassifier, build_activation_classifier(binary_config))
    classifier.fit(train_matrix, encoded_labels[train_indices])
    risk_column = _risk_probability_column(classifier, risk_label_index)
    return classifier.predict_proba(predict_matrix)[:, risk_column].astype(np.float64, copy=False)


def _source_oof_and_test_scores(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    source_feature_key: str,
    outer_train_indices: IntVector,
    outer_test_indices: IntVector,
    encoded_labels: IntVector,
    binary_config: BinaryTaskConfig,
    variant: CiftMetaHeadVariant,
    risk_label_index: int,
) -> tuple[FloatMatrix, FloatMatrix]:
    inner_binary_config = _inner_config(binary_config, variant)
    train_labels = encoded_labels[outer_train_indices]
    train_groups = tuple(dataset.families[int(index)] for index in outer_train_indices.tolist())
    inner_splits = stratified_group_splits(train_labels, train_groups, inner_binary_config)
    train_scores = np.zeros(outer_train_indices.shape[0], dtype=np.float64)

    for inner_split in inner_splits:
        inner_train_indices = outer_train_indices[inner_split.train_indices]
        inner_validation_indices = outer_train_indices[inner_split.test_indices]
        train_scores[inner_split.test_indices] = _predict_source_risk_scores(
            artifact=artifact,
            dataset=dataset,
            source_feature_key=source_feature_key,
            train_indices=inner_train_indices,
            predict_indices=inner_validation_indices,
            encoded_labels=encoded_labels,
            binary_config=binary_config,
            variant=variant,
            risk_label_index=risk_label_index,
        )

    test_scores = _predict_source_risk_scores(
        artifact=artifact,
        dataset=dataset,
        source_feature_key=source_feature_key,
        train_indices=outer_train_indices,
        predict_indices=outer_test_indices,
        encoded_labels=encoded_labels,
        binary_config=binary_config,
        variant=variant,
        risk_label_index=risk_label_index,
    )
    return train_scores, test_scores


def _outer_fold_scores(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    outer_train_indices: IntVector,
    outer_test_indices: IntVector,
    encoded_labels: IntVector,
    binary_config: BinaryTaskConfig,
    variant: CiftMetaHeadVariant,
    risk_label_index: int,
) -> _OuterFoldScores:
    train_scores = np.zeros((outer_train_indices.shape[0], len(variant.source_feature_keys)), dtype=np.float64)
    test_scores = np.zeros((outer_test_indices.shape[0], len(variant.source_feature_keys)), dtype=np.float64)

    for source_index, source_feature_key in enumerate(variant.source_feature_keys):
        source_train_scores, source_test_scores = _source_oof_and_test_scores(
            artifact=artifact,
            dataset=dataset,
            source_feature_key=source_feature_key,
            outer_train_indices=outer_train_indices,
            outer_test_indices=outer_test_indices,
            encoded_labels=encoded_labels,
            binary_config=binary_config,
            variant=variant,
            risk_label_index=risk_label_index,
        )
        train_scores[:, source_index] = source_train_scores
        test_scores[:, source_index] = source_test_scores

    return _OuterFoldScores(train_scores=train_scores, test_scores=test_scores)


def _logistic_regression_from_pipeline(classifier: Pipeline) -> LogisticRegression:
    estimator = classifier.named_steps.get("logisticregression")
    if not isinstance(estimator, LogisticRegression):
        raise BinaryTaskError("CIFT meta-head classifier does not contain a logisticregression step.")
    return estimator


def _risk_oriented_coefficients(classifier: Pipeline, risk_label_index: int) -> tuple[tuple[float, ...], float]:
    estimator = _logistic_regression_from_pipeline(classifier)
    classes = tuple(int(label_index) for label_index in estimator.classes_.tolist())
    if len(classes) != 2:
        raise BinaryTaskError("CIFT meta-head requires a binary logistic regression estimator.")
    positive_class = classes[-1]
    direction = 1.0 if positive_class == risk_label_index else -1.0
    coefficients = tuple(float(direction * value) for value in estimator.coef_[0].tolist())
    intercept = float(direction * estimator.intercept_[0])
    return coefficients, intercept


def _fit_meta_head(
    train_scores: FloatMatrix,
    train_labels: IntVector,
    binary_config: BinaryTaskConfig,
    risk_label_index: int,
) -> _MetaHeadFit:
    classifier = build_activation_classifier(binary_config)
    classifier.fit(train_scores, train_labels)
    coefficients, intercept = _risk_oriented_coefficients(classifier, risk_label_index)
    return _MetaHeadFit(classifier=classifier, coefficients=coefficients, intercept=intercept)


def _matrix_to_tuple(matrix: NDArray[np.int64]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _mean(values: tuple[float, ...]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: tuple[float, ...]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64)))


def _method_report(
    variant: CiftMetaHeadVariant,
    label_names: tuple[str, ...],
    true_labels: IntVector,
    fold_predictions: tuple[tuple[int, IntVector, IntVector], ...],
    meta_folds: tuple[CiftMetaHeadFold, ...],
) -> CiftMetaHeadVariantReport:
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
    return CiftMetaHeadVariantReport(
        variant_id=variant.variant_id,
        feature_name=variant.feature_name,
        source_feature_keys=variant.source_feature_keys,
        calibration_source_labels=variant.calibration_source_labels,
        ridge=variant.ridge,
        risk_label=variant.risk_label,
        inner_fold_count=variant.inner_fold_count,
        method_name="activation_probe",
        label_names=label_names,
        example_count=int(true_labels.shape[0]),
        accuracy_mean=_mean(accuracies),
        accuracy_std=_std(accuracies),
        macro_f1_mean=_mean(macro_f1_scores),
        macro_f1_std=_std(macro_f1_scores),
        confusion_matrix=_matrix_to_tuple(confusion_total),
        folds=tuple(folds),
        meta_folds=meta_folds,
    )


def _example_predictions_from_encoded_labels(
    dataset: BinaryTaskDataset,
    label_names: tuple[str, ...],
    fold_index: int,
    test_indices: IntVector,
    predictions: IntVector,
) -> tuple[BinaryExamplePrediction, ...]:
    rows: list[BinaryExamplePrediction] = []
    for row_index, predicted_index in zip(test_indices.tolist(), predictions.tolist(), strict=True):
        predicted_label = label_names[int(predicted_index)]
        true_label = dataset.target_labels[row_index]
        rows.append(
            BinaryExamplePrediction(
                fold_index=fold_index,
                example_id=dataset.example_ids[row_index],
                family=dataset.families[row_index],
                source_label=dataset.source_labels[row_index],
                true_label=true_label,
                predicted_label=predicted_label,
                is_correct=predicted_label == true_label,
            )
        )
    return tuple(rows)


def _method_error_analysis(
    variant: CiftMetaHeadVariant,
    label_names: tuple[str, ...],
    predictions: tuple[BinaryExamplePrediction, ...],
) -> BinaryMethodErrorAnalysis:
    if len(predictions) == 0:
        raise BinaryTaskError(f"CIFT meta-head variant '{variant.variant_id}' produced no predictions.")
    correct_count = sum(1 for prediction in predictions if prediction.is_correct)
    prediction_count = len(predictions)
    return BinaryMethodErrorAnalysis(
        method_name="activation_probe",
        feature_name=variant.feature_name,
        label_names=label_names,
        prediction_count=prediction_count,
        correct_count=correct_count,
        error_count=prediction_count - correct_count,
        accuracy=float(correct_count / prediction_count),
        family_summaries=summarize_family_predictions(predictions),
        predictions=predictions,
    )


def evaluate_grouped_cift_meta_head_variant(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    binary_config: BinaryTaskConfig,
    variant: CiftMetaHeadVariant,
) -> CiftMetaHeadVariantReport:
    _validate_variant(variant)
    label_encoding = encode_labels(dataset.target_labels)
    encoded_labels = label_encoding.encoded_labels
    risk_index = _risk_label_index(label_encoding.label_names, variant)
    splits = stratified_group_splits(encoded_labels, dataset.families, binary_config)
    fold_predictions: list[tuple[int, IntVector, IntVector]] = []
    meta_folds: list[CiftMetaHeadFold] = []

    for split in splits:
        scores = _outer_fold_scores(
            artifact=artifact,
            dataset=dataset,
            outer_train_indices=split.train_indices,
            outer_test_indices=split.test_indices,
            encoded_labels=encoded_labels,
            binary_config=binary_config,
            variant=variant,
            risk_label_index=risk_index,
        )
        meta_head = _fit_meta_head(
            train_scores=scores.train_scores,
            train_labels=encoded_labels[split.train_indices],
            binary_config=binary_config,
            risk_label_index=risk_index,
        )
        predictions = meta_head.classifier.predict(scores.test_scores).astype(np.int64, copy=False)
        fold_predictions.append((split.fold_index, encoded_labels[split.test_indices], predictions))
        meta_folds.append(
            CiftMetaHeadFold(
                fold_index=split.fold_index,
                source_feature_keys=variant.source_feature_keys,
                coefficients=meta_head.coefficients,
                intercept=meta_head.intercept,
            )
        )

    return _method_report(
        variant=variant,
        label_names=label_encoding.label_names,
        true_labels=encoded_labels,
        fold_predictions=tuple(fold_predictions),
        meta_folds=tuple(meta_folds),
    )


def collect_grouped_cift_meta_head_predictions(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    binary_config: BinaryTaskConfig,
    variant: CiftMetaHeadVariant,
) -> BinaryMethodErrorAnalysis:
    _validate_variant(variant)
    label_encoding = encode_labels(dataset.target_labels)
    encoded_labels = label_encoding.encoded_labels
    risk_index = _risk_label_index(label_encoding.label_names, variant)
    splits = stratified_group_splits(encoded_labels, dataset.families, binary_config)
    predictions: list[BinaryExamplePrediction] = []

    for split in splits:
        scores = _outer_fold_scores(
            artifact=artifact,
            dataset=dataset,
            outer_train_indices=split.train_indices,
            outer_test_indices=split.test_indices,
            encoded_labels=encoded_labels,
            binary_config=binary_config,
            variant=variant,
            risk_label_index=risk_index,
        )
        meta_head = _fit_meta_head(
            train_scores=scores.train_scores,
            train_labels=encoded_labels[split.train_indices],
            binary_config=binary_config,
            risk_label_index=risk_index,
        )
        predicted_labels = meta_head.classifier.predict(scores.test_scores).astype(np.int64, copy=False)
        predictions.extend(
            _example_predictions_from_encoded_labels(
                dataset=dataset,
                label_names=label_encoding.label_names,
                fold_index=split.fold_index,
                test_indices=split.test_indices,
                predictions=predicted_labels,
            )
        )

    return _method_error_analysis(
        variant=variant,
        label_names=label_encoding.label_names,
        predictions=tuple(predictions),
    )


def _best_variant(variants: tuple[CiftMetaHeadVariantReport, ...]) -> CiftMetaHeadVariantReport:
    if len(variants) == 0:
        raise BinaryTaskError("Cannot select a best CIFT meta-head variant from an empty set.")
    return max(variants, key=lambda variant: (variant.macro_f1_mean, variant.accuracy_mean))


def _winning_feature_key(
    baseline: BinaryMethodReport,
    best_variant: CiftMetaHeadVariantReport,
) -> str:
    baseline_score = (baseline.macro_f1_mean, baseline.accuracy_mean)
    variant_score = (best_variant.macro_f1_mean, best_variant.accuracy_mean)
    if variant_score > baseline_score:
        return best_variant.feature_name
    if baseline_score > variant_score:
        return baseline.feature_name
    return "tie"


def _compare_dataset(
    dataset: CiftMetaHeadComparisonDataset,
    definition: BinaryTaskDefinition,
    baseline_feature_key: str,
    variants: tuple[CiftMetaHeadVariant, ...],
    binary_config: BinaryTaskConfig,
) -> DatasetCiftMetaHeadComparison:
    task_dataset = build_binary_task_dataset(dataset.artifact, definition)
    baseline = evaluate_grouped_activation_method(
        artifact=dataset.artifact,
        dataset=task_dataset,
        config=replace(binary_config, activation_feature_key=baseline_feature_key),
    )
    variant_reports = tuple(
        evaluate_grouped_cift_meta_head_variant(
            artifact=dataset.artifact,
            dataset=task_dataset,
            binary_config=binary_config,
            variant=variant,
        )
        for variant in variants
    )
    best_variant = _best_variant(variant_reports)
    metadata = dataset.artifact["metadata"]
    return DatasetCiftMetaHeadComparison(
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


def compare_grouped_cift_meta_head(
    datasets: tuple[CiftMetaHeadComparisonDataset, ...],
    task_name: str,
    baseline_feature_key: str,
    variants: tuple[CiftMetaHeadVariant, ...],
    binary_config: BinaryTaskConfig,
) -> CiftMetaHeadComparisonReport:
    if len(datasets) == 0:
        raise BinaryTaskError("At least one dataset is required for CIFT meta-head comparison.")
    if baseline_feature_key == "":
        raise BinaryTaskError("CIFT meta-head baseline feature key must not be empty.")
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
    meta_head_win_count = len(dataset_reports) - baseline_win_count - tie_count

    return CiftMetaHeadComparisonReport(
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
        meta_head_win_count=meta_head_win_count,
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


def _meta_fold_to_json(fold: CiftMetaHeadFold) -> dict[str, JsonValue]:
    return {
        "fold_index": fold.fold_index,
        "source_feature_keys": list(fold.source_feature_keys),
        "coefficients": list(fold.coefficients),
        "intercept": fold.intercept,
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


def _variant_to_json(variant: CiftMetaHeadVariantReport) -> dict[str, JsonValue]:
    return {
        "variant_id": variant.variant_id,
        "feature_name": variant.feature_name,
        "source_feature_keys": list(variant.source_feature_keys),
        "calibration_source_labels": list(variant.calibration_source_labels),
        "ridge": variant.ridge,
        "risk_label": variant.risk_label,
        "inner_fold_count": variant.inner_fold_count,
        "method_name": variant.method_name,
        "label_names": list(variant.label_names),
        "example_count": variant.example_count,
        "accuracy_mean": variant.accuracy_mean,
        "accuracy_std": variant.accuracy_std,
        "macro_f1_mean": variant.macro_f1_mean,
        "macro_f1_std": variant.macro_f1_std,
        "confusion_matrix": [list(row) for row in variant.confusion_matrix],
        "folds": [_fold_to_json(fold) for fold in variant.folds],
        "meta_folds": [_meta_fold_to_json(fold) for fold in variant.meta_folds],
    }


def _dataset_to_json(dataset: DatasetCiftMetaHeadComparison) -> dict[str, JsonValue]:
    return {
        "dataset_id": dataset.dataset_id,
        "source_model_id": dataset.source_model_id,
        "source_revision": dataset.source_revision,
        "source_selected_device": dataset.source_selected_device,
        "baseline": _baseline_to_json(dataset.baseline),
        "variants": [_variant_to_json(variant) for variant in dataset.variants],
        "best_variant": _variant_to_json(dataset.best_variant),
        "macro_f1_delta": dataset.macro_f1_delta,
        "accuracy_delta": dataset.accuracy_delta,
        "winning_feature_key": dataset.winning_feature_key,
    }


def cift_meta_head_report_to_json(report: CiftMetaHeadComparisonReport) -> dict[str, JsonValue]:
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
        "meta_head_win_count": report.meta_head_win_count,
        "baseline_win_count": report.baseline_win_count,
        "tie_count": report.tie_count,
        "datasets": [_dataset_to_json(dataset) for dataset in report.datasets],
    }


def write_cift_meta_head_json(path: Path, report: CiftMetaHeadComparisonReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cift_meta_head_report_to_json(report), file, indent=2)
        file.write("\n")


def _joined(values: tuple[str, ...]) -> str:
    return "`, `".join(values)


def _mean_coefficient(variant: CiftMetaHeadVariantReport, source_feature_index: int) -> float:
    coefficients = tuple(fold.coefficients[source_feature_index] for fold in variant.meta_folds)
    return _mean(coefficients)


def render_cift_meta_head_markdown(report: CiftMetaHeadComparisonReport) -> str:
    lines = [
        "# CIFT OOF Meta-Head",
        "",
        "## Source",
        "",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Baseline feature: `{report.baseline_feature_key}`",
        f"- Dataset count: `{report.dataset_count}`",
        f"- Variant count: `{report.variant_count}`",
        f"- Meta-head wins: `{report.meta_head_win_count}`",
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
            "| Variant | Source Count | Inner Folds | Mean Macro F1 | Min Macro F1 |",
            "|---|---:|---:|---:|---:|",
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
            f"{len(first_variant.source_feature_keys)} | "
            f"{first_variant.inner_fold_count} | "
            f"{_mean(macro_f1_scores):.4f} | "
            f"{min(macro_f1_scores):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Mean Risk-Oriented Coefficients",
            "",
            "| Dataset | Variant | Source Feature | Mean Coefficient |",
            "|---|---|---|---:|",
        ]
    )
    for dataset in report.datasets:
        for variant in dataset.variants:
            for source_feature_index, source_feature_key in enumerate(variant.source_feature_keys):
                lines.append(
                    f"| `{dataset.dataset_id}` | "
                    f"`{variant.feature_name}` | "
                    f"`{source_feature_key}` | "
                    f"{_mean_coefficient(variant, source_feature_index):+.4f} |"
                )

    lines.extend(
        [
            "",
            "## Variant Results",
            "",
            "| Dataset | Variant | Source Count | Macro F1 | Delta Macro F1 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for dataset in report.datasets:
        for variant in dataset.variants:
            lines.append(
                f"| `{dataset.dataset_id}` | "
                f"`{variant.feature_name}` | "
                f"{len(variant.source_feature_keys)} | "
                f"{variant.macro_f1_mean:.4f} | "
                f"{variant.macro_f1_mean - dataset.baseline.macro_f1_mean:+.4f} |"
            )

    return "\n".join(lines)


def write_cift_meta_head_markdown(path: Path, report: CiftMetaHeadComparisonReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cift_meta_head_markdown(report), encoding="utf-8")
