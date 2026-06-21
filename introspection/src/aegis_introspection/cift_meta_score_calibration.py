from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import (
    BinaryMethodName,
    BinaryTaskConfig,
    BinaryTaskDataset,
    BinaryTaskDefinition,
    BinaryTaskError,
    EvaluationStrategy,
    build_binary_task_dataset,
    default_binary_task_definitions,
)
from aegis_introspection.cift_meta_head import (
    CiftMetaHeadSourceScoreFold,
    CiftMetaHeadVariant,
    collect_grouped_cift_meta_head_source_score_folds,
)
from aegis_introspection.error_analysis import (
    BinaryErrorAnalysisReport,
    BinaryExamplePrediction,
    BinaryMethodErrorAnalysis,
    BinaryTaskErrorAnalysis,
    collect_grouped_activation_predictions,
    summarize_family_predictions,
)
from aegis_introspection.probe import JsonValue
from aegis_introspection.residual_error_comparison import (
    DatasetResidualErrorComparison,
    compare_binary_error_residuals,
)


CiftMetaScoreCalibrationRule: TypeAlias = Literal[
    "raw_probability",
    "clipped_logit",
    "platt_probability",
]


@dataclass(frozen=True)
class CiftMetaScoreCalibrationDataset:
    dataset_id: str
    artifact: ActivationArtifact


@dataclass(frozen=True)
class CiftMetaScoreCalibrationVariant:
    variant_id: str
    feature_name: str
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    ridge: float
    risk_label: str
    inner_fold_count: int
    meta_regularization_c: float
    score_calibration_rule: CiftMetaScoreCalibrationRule


@dataclass(frozen=True)
class _CalibratedScoreFold:
    fold_index: int
    source_feature_keys: tuple[str, ...]
    label_names: tuple[str, ...]
    test_indices: np.ndarray
    train_scores: np.ndarray
    test_scores: np.ndarray
    train_labels: np.ndarray


@dataclass(frozen=True)
class CiftMetaScoreCalibrationDatasetVariantReport:
    dataset_id: str
    variant_id: str
    feature_name: str
    score_calibration_rule: CiftMetaScoreCalibrationRule
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    meta_regularization_c: float
    reference_error_count: int
    candidate_error_count: int
    fixed_error_count: int
    persistent_error_count: int
    introduced_error_count: int
    net_error_delta: int
    reference_accuracy: float
    candidate_accuracy: float


@dataclass(frozen=True)
class CiftMetaScoreCalibrationVariantSummary:
    variant_id: str
    feature_name: str
    score_calibration_rule: CiftMetaScoreCalibrationRule
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    meta_regularization_c: float
    dataset_count: int
    reference_error_count: int
    candidate_error_count: int
    fixed_error_count: int
    persistent_error_count: int
    introduced_error_count: int
    net_error_delta: int
    mean_candidate_accuracy: float
    min_candidate_accuracy: float


@dataclass(frozen=True)
class CiftMetaScoreCalibrationReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    evaluation_strategy: EvaluationStrategy
    fold_count: int
    inner_fold_count: int
    source_regularization_c: float
    meta_regularization_c: float
    random_seed: int
    max_iter: int
    task_name: str
    method_name: BinaryMethodName
    baseline_feature_key: str
    dataset_count: int
    variant_count: int
    best_variant_summary: CiftMetaScoreCalibrationVariantSummary
    variant_summaries: tuple[CiftMetaScoreCalibrationVariantSummary, ...]
    dataset_variants: tuple[CiftMetaScoreCalibrationDatasetVariantReport, ...]


def _validate_variant(variant: CiftMetaScoreCalibrationVariant) -> None:
    if variant.variant_id == "":
        raise BinaryTaskError("CIFT score calibration variant id must not be empty.")
    if variant.feature_name == "":
        raise BinaryTaskError(f"CIFT score calibration variant '{variant.variant_id}' feature name must not be empty.")
    if len(variant.source_feature_keys) == 0:
        raise BinaryTaskError(f"CIFT score calibration variant '{variant.variant_id}' requires source features.")
    if len(set(variant.source_feature_keys)) != len(variant.source_feature_keys):
        raise BinaryTaskError(f"CIFT score calibration variant '{variant.variant_id}' source features must be unique.")
    if len(variant.calibration_source_labels) == 0:
        raise BinaryTaskError(f"CIFT score calibration variant '{variant.variant_id}' requires calibration labels.")
    if variant.ridge <= 0:
        raise BinaryTaskError(f"CIFT score calibration variant '{variant.variant_id}' ridge must be greater than 0.")
    if variant.risk_label == "":
        raise BinaryTaskError(f"CIFT score calibration variant '{variant.variant_id}' risk label must not be empty.")
    if variant.inner_fold_count < 2:
        raise BinaryTaskError(f"CIFT score calibration variant '{variant.variant_id}' inner_fold_count must be at least 2.")
    if variant.meta_regularization_c <= 0:
        raise BinaryTaskError(
            f"CIFT score calibration variant '{variant.variant_id}' meta_regularization_c must be greater than 0."
        )
    if variant.score_calibration_rule not in ("raw_probability", "clipped_logit", "platt_probability"):
        raise BinaryTaskError(
            f"CIFT score calibration variant '{variant.variant_id}' has unsupported score calibration rule "
            f"'{variant.score_calibration_rule}'."
        )


def _validate_inputs(
    datasets: tuple[CiftMetaScoreCalibrationDataset, ...],
    baseline_feature_key: str,
    variants: tuple[CiftMetaScoreCalibrationVariant, ...],
) -> None:
    if len(datasets) == 0:
        raise BinaryTaskError("At least one CIFT score calibration dataset is required.")
    if baseline_feature_key == "":
        raise BinaryTaskError("CIFT score calibration baseline feature key must not be empty.")
    for index, dataset in enumerate(datasets):
        if dataset.dataset_id == "":
            raise BinaryTaskError(f"CIFT score calibration dataset {index} has an empty dataset id.")
    if len(variants) == 0:
        raise BinaryTaskError("At least one CIFT score calibration variant is required.")
    for variant in variants:
        _validate_variant(variant)
    if len({variant.variant_id for variant in variants}) != len(variants):
        raise BinaryTaskError("CIFT score calibration variant ids must be unique.")
    if len({variant.feature_name for variant in variants}) != len(variants):
        raise BinaryTaskError("CIFT score calibration feature names must be unique.")
    if len({variant.meta_regularization_c for variant in variants}) != 1:
        raise BinaryTaskError("CIFT score calibration variants must share one meta_regularization_c.")


def _task_definition(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise BinaryTaskError(f"Expected exactly one binary task named '{task_name}', found {len(matches)}.")
    return matches[0]


def _head_variant(variant: CiftMetaScoreCalibrationVariant) -> CiftMetaHeadVariant:
    return CiftMetaHeadVariant(
        variant_id=variant.variant_id,
        feature_name=variant.feature_name,
        source_feature_keys=variant.source_feature_keys,
        calibration_source_labels=variant.calibration_source_labels,
        ridge=variant.ridge,
        risk_label=variant.risk_label,
        inner_fold_count=variant.inner_fold_count,
        decision_rule="logistic_default",
    )


def _risk_label_index(label_names: tuple[str, ...], risk_label: str) -> int:
    matches = tuple(index for index, label_name in enumerate(label_names) if label_name == risk_label)
    if len(matches) != 1:
        raise BinaryTaskError(f"CIFT score calibration risk label '{risk_label}' is not in labels {label_names}.")
    return matches[0]


def _other_label(dataset: BinaryTaskDataset, risk_label: str) -> str:
    labels = tuple(sorted(set(dataset.target_labels)))
    other_labels = tuple(label for label in labels if label != risk_label)
    if len(other_labels) != 1:
        raise BinaryTaskError("CIFT score calibration requires exactly one non-risk target label.")
    return other_labels[0]


def _build_meta_classifier(
    variant: CiftMetaScoreCalibrationVariant,
    binary_config: BinaryTaskConfig,
) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=variant.meta_regularization_c,
            class_weight="balanced",
            max_iter=binary_config.max_iter,
            random_state=binary_config.random_seed,
        ),
    )


def _build_platt_classifier(binary_config: BinaryTaskConfig) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=binary_config.max_iter,
            random_state=binary_config.random_seed,
        ),
    )


def _risk_probability_column(classifier: Pipeline, risk_label_index: int) -> int:
    estimator = classifier.named_steps.get("logisticregression")
    if not isinstance(estimator, LogisticRegression):
        raise BinaryTaskError("CIFT score calibration classifier does not contain a logisticregression step.")
    classes = tuple(int(label_index) for label_index in estimator.classes_.tolist())
    if risk_label_index not in classes:
        raise BinaryTaskError(f"CIFT score calibration classifier was not fitted with risk label index {risk_label_index}.")
    return classes.index(risk_label_index)


def _clipped_logit(scores: np.ndarray) -> np.ndarray:
    clipped_scores = np.clip(scores.astype(np.float64, copy=False), 1e-4, 1.0 - 1e-4)
    return np.log(clipped_scores / (1.0 - clipped_scores)).astype(np.float64, copy=False)


def _platt_calibrated_scores(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    train_labels: np.ndarray,
    risk_label_index: int,
    binary_config: BinaryTaskConfig,
) -> tuple[np.ndarray, np.ndarray]:
    unique_labels = np.unique(train_labels)
    if unique_labels.shape[0] != 2:
        raise BinaryTaskError("CIFT score calibration Platt scaling requires both classes in the train fold.")
    calibrated_train = np.zeros_like(train_scores, dtype=np.float64)
    calibrated_test = np.zeros_like(test_scores, dtype=np.float64)
    for source_index in range(train_scores.shape[1]):
        classifier = _build_platt_classifier(binary_config)
        classifier.fit(train_scores[:, source_index].reshape(-1, 1), train_labels)
        risk_column = _risk_probability_column(classifier=classifier, risk_label_index=risk_label_index)
        calibrated_train[:, source_index] = classifier.predict_proba(
            train_scores[:, source_index].reshape(-1, 1)
        )[:, risk_column]
        calibrated_test[:, source_index] = classifier.predict_proba(
            test_scores[:, source_index].reshape(-1, 1)
        )[:, risk_column]
    return calibrated_train, calibrated_test


def _calibrated_fold(
    fold: CiftMetaHeadSourceScoreFold,
    variant: CiftMetaScoreCalibrationVariant,
    binary_config: BinaryTaskConfig,
) -> _CalibratedScoreFold:
    risk_index = _risk_label_index(label_names=fold.label_names, risk_label=variant.risk_label)
    if variant.score_calibration_rule == "raw_probability":
        train_scores = fold.train_scores.astype(np.float64, copy=False)
        test_scores = fold.test_scores.astype(np.float64, copy=False)
    elif variant.score_calibration_rule == "clipped_logit":
        train_scores = _clipped_logit(fold.train_scores)
        test_scores = _clipped_logit(fold.test_scores)
    elif variant.score_calibration_rule == "platt_probability":
        train_scores, test_scores = _platt_calibrated_scores(
            train_scores=fold.train_scores.astype(np.float64, copy=False),
            test_scores=fold.test_scores.astype(np.float64, copy=False),
            train_labels=fold.train_labels,
            risk_label_index=risk_index,
            binary_config=binary_config,
        )
    else:
        raise BinaryTaskError(f"Unsupported CIFT score calibration rule '{variant.score_calibration_rule}'.")
    return _CalibratedScoreFold(
        fold_index=fold.fold_index,
        source_feature_keys=fold.source_feature_keys,
        label_names=fold.label_names,
        test_indices=fold.test_indices,
        train_scores=train_scores,
        test_scores=test_scores,
        train_labels=fold.train_labels,
    )


def _predict_fold(
    dataset: BinaryTaskDataset,
    fold: CiftMetaHeadSourceScoreFold,
    variant: CiftMetaScoreCalibrationVariant,
    binary_config: BinaryTaskConfig,
) -> tuple[BinaryExamplePrediction, ...]:
    calibrated_fold = _calibrated_fold(fold=fold, variant=variant, binary_config=binary_config)
    classifier = _build_meta_classifier(variant=variant, binary_config=binary_config)
    classifier.fit(calibrated_fold.train_scores, calibrated_fold.train_labels)
    predicted_indices = classifier.predict(calibrated_fold.test_scores).astype(np.int64, copy=False)
    risk_index = _risk_label_index(label_names=calibrated_fold.label_names, risk_label=variant.risk_label)
    other_label = _other_label(dataset=dataset, risk_label=variant.risk_label)
    predictions: list[BinaryExamplePrediction] = []
    for row_index, predicted_index in zip(calibrated_fold.test_indices.tolist(), predicted_indices.tolist(), strict=True):
        predicted_label = variant.risk_label if int(predicted_index) == risk_index else other_label
        true_label = dataset.target_labels[row_index]
        predictions.append(
            BinaryExamplePrediction(
                fold_index=calibrated_fold.fold_index,
                example_id=dataset.example_ids[row_index],
                family=dataset.families[row_index],
                source_label=dataset.source_labels[row_index],
                true_label=true_label,
                predicted_label=predicted_label,
                is_correct=predicted_label == true_label,
            )
        )
    return tuple(predictions)


def _method_error_analysis(
    variant: CiftMetaScoreCalibrationVariant,
    label_names: tuple[str, ...],
    predictions: tuple[BinaryExamplePrediction, ...],
) -> BinaryMethodErrorAnalysis:
    if len(predictions) == 0:
        raise BinaryTaskError(f"CIFT score calibration variant '{variant.variant_id}' produced no predictions.")
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


def _collect_score_calibrated_meta_predictions(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    binary_config: BinaryTaskConfig,
    variant: CiftMetaScoreCalibrationVariant,
) -> BinaryMethodErrorAnalysis:
    folds = collect_grouped_cift_meta_head_source_score_folds(
        artifact=artifact,
        dataset=dataset,
        binary_config=binary_config,
        variant=_head_variant(variant),
    )
    predictions = tuple(
        prediction
        for fold in folds
        for prediction in _predict_fold(
            dataset=dataset,
            fold=fold,
            variant=variant,
            binary_config=binary_config,
        )
    )
    return _method_error_analysis(
        variant=variant,
        label_names=folds[0].label_names,
        predictions=predictions,
    )


def _error_analysis_report(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    method: BinaryMethodErrorAnalysis,
    config: BinaryTaskConfig,
) -> BinaryErrorAnalysisReport:
    metadata = artifact["metadata"]
    return BinaryErrorAnalysisReport(
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        evaluation_strategy="stratified_group_kfold",
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        regularization_c=config.regularization_c,
        max_iter=config.max_iter,
        activation_feature_key=method.feature_name,
        tasks=(
            BinaryTaskErrorAnalysis(
                task_name=dataset.name,
                description=dataset.description,
                label_names=method.label_names,
                methods=(method,),
            ),
        ),
    )


def _compare_dataset_variant(
    dataset: CiftMetaScoreCalibrationDataset,
    definition: BinaryTaskDefinition,
    baseline_feature_key: str,
    variant: CiftMetaScoreCalibrationVariant,
    binary_config: BinaryTaskConfig,
) -> DatasetResidualErrorComparison:
    task_dataset = build_binary_task_dataset(dataset.artifact, definition)
    baseline_config = replace(binary_config, activation_feature_key=baseline_feature_key)
    baseline_method = collect_grouped_activation_predictions(
        artifact=dataset.artifact,
        dataset=task_dataset,
        config=baseline_config,
    )
    candidate_method = _collect_score_calibrated_meta_predictions(
        artifact=dataset.artifact,
        dataset=task_dataset,
        binary_config=binary_config,
        variant=variant,
    )
    baseline_report = _error_analysis_report(
        artifact=dataset.artifact,
        dataset=task_dataset,
        method=baseline_method,
        config=baseline_config,
    )
    candidate_report = _error_analysis_report(
        artifact=dataset.artifact,
        dataset=task_dataset,
        method=candidate_method,
        config=binary_config,
    )
    return DatasetResidualErrorComparison(
        dataset_id=dataset.dataset_id,
        comparison=compare_binary_error_residuals(
            reference_report=baseline_report,
            candidate_report=candidate_report,
            task_name=definition.name,
            method_name="activation_probe",
        ),
    )


def _dataset_variant_report(
    variant: CiftMetaScoreCalibrationVariant,
    comparison: DatasetResidualErrorComparison,
) -> CiftMetaScoreCalibrationDatasetVariantReport:
    residual = comparison.comparison
    return CiftMetaScoreCalibrationDatasetVariantReport(
        dataset_id=comparison.dataset_id,
        variant_id=variant.variant_id,
        feature_name=variant.feature_name,
        score_calibration_rule=variant.score_calibration_rule,
        source_feature_keys=variant.source_feature_keys,
        calibration_source_labels=variant.calibration_source_labels,
        meta_regularization_c=variant.meta_regularization_c,
        reference_error_count=residual.reference_error_count,
        candidate_error_count=residual.candidate_error_count,
        fixed_error_count=residual.fixed_error_count,
        persistent_error_count=residual.persistent_error_count,
        introduced_error_count=residual.introduced_error_count,
        net_error_delta=residual.introduced_error_count - residual.fixed_error_count,
        reference_accuracy=residual.reference_accuracy,
        candidate_accuracy=residual.candidate_accuracy,
    )


def _mean(values: tuple[float, ...]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _variant_summary(
    variant: CiftMetaScoreCalibrationVariant,
    dataset_reports: tuple[CiftMetaScoreCalibrationDatasetVariantReport, ...],
) -> CiftMetaScoreCalibrationVariantSummary:
    candidate_accuracies = tuple(report.candidate_accuracy for report in dataset_reports)
    fixed_error_count = sum(report.fixed_error_count for report in dataset_reports)
    introduced_error_count = sum(report.introduced_error_count for report in dataset_reports)
    return CiftMetaScoreCalibrationVariantSummary(
        variant_id=variant.variant_id,
        feature_name=variant.feature_name,
        score_calibration_rule=variant.score_calibration_rule,
        source_feature_keys=variant.source_feature_keys,
        calibration_source_labels=variant.calibration_source_labels,
        meta_regularization_c=variant.meta_regularization_c,
        dataset_count=len({report.dataset_id for report in dataset_reports}),
        reference_error_count=sum(report.reference_error_count for report in dataset_reports),
        candidate_error_count=sum(report.candidate_error_count for report in dataset_reports),
        fixed_error_count=fixed_error_count,
        persistent_error_count=sum(report.persistent_error_count for report in dataset_reports),
        introduced_error_count=introduced_error_count,
        net_error_delta=introduced_error_count - fixed_error_count,
        mean_candidate_accuracy=_mean(candidate_accuracies),
        min_candidate_accuracy=min(candidate_accuracies),
    )


def _best_summary(
    summaries: tuple[CiftMetaScoreCalibrationVariantSummary, ...],
) -> CiftMetaScoreCalibrationVariantSummary:
    return min(
        summaries,
        key=lambda summary: (
            summary.net_error_delta,
            summary.introduced_error_count,
            summary.candidate_error_count,
            -summary.fixed_error_count,
            -summary.mean_candidate_accuracy,
        ),
    )


def compare_cift_meta_score_calibration(
    datasets: tuple[CiftMetaScoreCalibrationDataset, ...],
    task_name: str,
    baseline_feature_key: str,
    variants: tuple[CiftMetaScoreCalibrationVariant, ...],
    binary_config: BinaryTaskConfig,
) -> CiftMetaScoreCalibrationReport:
    _validate_inputs(datasets=datasets, baseline_feature_key=baseline_feature_key, variants=variants)
    definition = _task_definition(task_name)
    dataset_variant_reports: list[CiftMetaScoreCalibrationDatasetVariantReport] = []
    summaries: list[CiftMetaScoreCalibrationVariantSummary] = []

    for variant in variants:
        comparisons = tuple(
            _compare_dataset_variant(
                dataset=dataset,
                definition=definition,
                baseline_feature_key=baseline_feature_key,
                variant=variant,
                binary_config=binary_config,
            )
            for dataset in datasets
        )
        reports = tuple(_dataset_variant_report(variant=variant, comparison=comparison) for comparison in comparisons)
        dataset_variant_reports.extend(reports)
        summaries.append(_variant_summary(variant=variant, dataset_reports=reports))

    first_metadata = datasets[0].artifact["metadata"]
    summary_tuple = tuple(summaries)
    return CiftMetaScoreCalibrationReport(
        source_model_id=first_metadata["model_id"],
        source_revision=first_metadata["revision"],
        source_selected_device=first_metadata["selected_device"],
        evaluation_strategy="stratified_group_kfold",
        fold_count=binary_config.fold_count,
        inner_fold_count=variants[0].inner_fold_count,
        source_regularization_c=binary_config.regularization_c,
        meta_regularization_c=variants[0].meta_regularization_c,
        random_seed=binary_config.random_seed,
        max_iter=binary_config.max_iter,
        task_name=definition.name,
        method_name="activation_probe",
        baseline_feature_key=baseline_feature_key,
        dataset_count=len({dataset.dataset_id for dataset in datasets}),
        variant_count=len(variants),
        best_variant_summary=_best_summary(summary_tuple),
        variant_summaries=summary_tuple,
        dataset_variants=tuple(dataset_variant_reports),
    )


def _summary_to_json(summary: CiftMetaScoreCalibrationVariantSummary) -> dict[str, JsonValue]:
    return {
        "variant_id": summary.variant_id,
        "feature_name": summary.feature_name,
        "score_calibration_rule": summary.score_calibration_rule,
        "source_feature_keys": list(summary.source_feature_keys),
        "calibration_source_labels": list(summary.calibration_source_labels),
        "meta_regularization_c": summary.meta_regularization_c,
        "dataset_count": summary.dataset_count,
        "reference_error_count": summary.reference_error_count,
        "candidate_error_count": summary.candidate_error_count,
        "fixed_error_count": summary.fixed_error_count,
        "persistent_error_count": summary.persistent_error_count,
        "introduced_error_count": summary.introduced_error_count,
        "net_error_delta": summary.net_error_delta,
        "mean_candidate_accuracy": summary.mean_candidate_accuracy,
        "min_candidate_accuracy": summary.min_candidate_accuracy,
    }


def _dataset_variant_to_json(report: CiftMetaScoreCalibrationDatasetVariantReport) -> dict[str, JsonValue]:
    return {
        "dataset_id": report.dataset_id,
        "variant_id": report.variant_id,
        "feature_name": report.feature_name,
        "score_calibration_rule": report.score_calibration_rule,
        "source_feature_keys": list(report.source_feature_keys),
        "calibration_source_labels": list(report.calibration_source_labels),
        "meta_regularization_c": report.meta_regularization_c,
        "reference_error_count": report.reference_error_count,
        "candidate_error_count": report.candidate_error_count,
        "fixed_error_count": report.fixed_error_count,
        "persistent_error_count": report.persistent_error_count,
        "introduced_error_count": report.introduced_error_count,
        "net_error_delta": report.net_error_delta,
        "reference_accuracy": report.reference_accuracy,
        "candidate_accuracy": report.candidate_accuracy,
    }


def cift_meta_score_calibration_to_json(report: CiftMetaScoreCalibrationReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "evaluation_strategy": report.evaluation_strategy,
        "fold_count": report.fold_count,
        "inner_fold_count": report.inner_fold_count,
        "source_regularization_c": report.source_regularization_c,
        "meta_regularization_c": report.meta_regularization_c,
        "random_seed": report.random_seed,
        "max_iter": report.max_iter,
        "task_name": report.task_name,
        "method_name": report.method_name,
        "baseline_feature_key": report.baseline_feature_key,
        "dataset_count": report.dataset_count,
        "variant_count": report.variant_count,
        "best_variant_summary": _summary_to_json(report.best_variant_summary),
        "variant_summaries": [_summary_to_json(summary) for summary in report.variant_summaries],
        "dataset_variants": [_dataset_variant_to_json(dataset_variant) for dataset_variant in report.dataset_variants],
    }


def write_cift_meta_score_calibration_json(path: Path, report: CiftMetaScoreCalibrationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cift_meta_score_calibration_to_json(report), file, indent=2)
        file.write("\n")


def _joined(values: tuple[str, ...]) -> str:
    return "`, `".join(values)


def render_cift_meta_score_calibration_markdown(report: CiftMetaScoreCalibrationReport) -> str:
    lines = [
        "# CIFT Meta-Head Source-Score Calibration",
        "",
        "## Source",
        "",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Method: `{report.method_name}`",
        f"- Baseline feature: `{report.baseline_feature_key}`",
        f"- Source-head C: `{report.source_regularization_c}`",
        f"- Meta-head C: `{report.meta_regularization_c}`",
        f"- Dataset count: `{report.dataset_count}`",
        f"- Variant count: `{report.variant_count}`",
        f"- Best variant: `{report.best_variant_summary.variant_id}`",
        "",
        "## Variant Summary",
        "",
        (
            "| Variant | Calibration Rule | Meta C | Source Count | Calibration Labels | Candidate Errors | "
            "Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |"
        ),
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in report.variant_summaries:
        lines.append(
            f"| `{summary.variant_id}` | "
            f"`{summary.score_calibration_rule}` | "
            f"{summary.meta_regularization_c:.4g} | "
            f"{len(summary.source_feature_keys)} | "
            f"`{_joined(summary.calibration_source_labels)}` | "
            f"{summary.candidate_error_count} | "
            f"{summary.fixed_error_count} | "
            f"{summary.persistent_error_count} | "
            f"{summary.introduced_error_count} | "
            f"{summary.net_error_delta} | "
            f"{summary.mean_candidate_accuracy:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Dataset Variant Results",
            "",
            "| Dataset | Variant | Candidate Errors | Fixed | Persistent | Introduced | Candidate Accuracy |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_variant in report.dataset_variants:
        lines.append(
            f"| `{dataset_variant.dataset_id}` | "
            f"`{dataset_variant.variant_id}` | "
            f"{dataset_variant.candidate_error_count} | "
            f"{dataset_variant.fixed_error_count} | "
            f"{dataset_variant.persistent_error_count} | "
            f"{dataset_variant.introduced_error_count} | "
            f"{dataset_variant.candidate_accuracy:.4f} |"
        )
    return "\n".join(lines)


def write_cift_meta_score_calibration_markdown(path: Path, report: CiftMetaScoreCalibrationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cift_meta_score_calibration_markdown(report), encoding="utf-8")
