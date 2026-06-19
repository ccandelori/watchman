from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.optimize import Bounds, LinearConstraint
from sklearn.metrics import accuracy_score, f1_score

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
    CiftMetaHeadExampleDiagnostic,
    CiftMetaHeadSourceScoreFold,
    CiftMetaHeadVariant,
    collect_grouped_cift_meta_head_source_score_folds,
    collect_grouped_cift_meta_head_diagnostics,
    collect_grouped_cift_meta_head_predictions,
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


CiftMetaCombinerRule: TypeAlias = Literal[
    "logistic_meta_head",
    "mean_score",
    "max_score",
    "top_two_mean",
    "majority_vote",
    "positive_logistic",
    "simplex_logistic",
    "positive_logistic_train_f1_threshold",
    "simplex_logistic_train_f1_threshold",
]

FloatVector: TypeAlias = NDArray[np.float64]
FloatMatrix: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class CiftMetaCombinerDataset:
    dataset_id: str
    artifact: ActivationArtifact


@dataclass(frozen=True)
class CiftMetaCombinerVariant:
    variant_id: str
    feature_name: str
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    ridge: float
    risk_label: str
    inner_fold_count: int
    combiner_rule: CiftMetaCombinerRule


@dataclass(frozen=True)
class _ConstrainedLogisticFit:
    intercept: float
    coefficients: FloatVector
    feature_mean: FloatVector
    feature_scale: FloatVector
    threshold: float


@dataclass(frozen=True)
class CiftMetaCombinerDatasetVariantReport:
    dataset_id: str
    variant_id: str
    feature_name: str
    combiner_rule: CiftMetaCombinerRule
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    reference_error_count: int
    candidate_error_count: int
    fixed_error_count: int
    persistent_error_count: int
    introduced_error_count: int
    net_error_delta: int
    reference_accuracy: float
    candidate_accuracy: float


@dataclass(frozen=True)
class CiftMetaCombinerVariantSummary:
    variant_id: str
    feature_name: str
    combiner_rule: CiftMetaCombinerRule
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
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
class CiftMetaCombinerAblationReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    evaluation_strategy: EvaluationStrategy
    fold_count: int
    inner_fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    task_name: str
    method_name: BinaryMethodName
    baseline_feature_key: str
    dataset_count: int
    variant_count: int
    best_variant_summary: CiftMetaCombinerVariantSummary
    variant_summaries: tuple[CiftMetaCombinerVariantSummary, ...]
    dataset_variants: tuple[CiftMetaCombinerDatasetVariantReport, ...]


def _validate_variant(variant: CiftMetaCombinerVariant) -> None:
    if variant.variant_id == "":
        raise BinaryTaskError("CIFT combiner variant id must not be empty.")
    if variant.feature_name == "":
        raise BinaryTaskError(f"CIFT combiner variant '{variant.variant_id}' feature name must not be empty.")
    if len(variant.source_feature_keys) == 0:
        raise BinaryTaskError(f"CIFT combiner variant '{variant.variant_id}' requires source features.")
    if len(set(variant.source_feature_keys)) != len(variant.source_feature_keys):
        raise BinaryTaskError(f"CIFT combiner variant '{variant.variant_id}' source features must be unique.")
    if len(variant.calibration_source_labels) == 0:
        raise BinaryTaskError(f"CIFT combiner variant '{variant.variant_id}' requires calibration labels.")
    if variant.ridge <= 0:
        raise BinaryTaskError(f"CIFT combiner variant '{variant.variant_id}' ridge must be greater than 0.")
    if variant.risk_label == "":
        raise BinaryTaskError(f"CIFT combiner variant '{variant.variant_id}' risk label must not be empty.")
    if variant.inner_fold_count < 2:
        raise BinaryTaskError(f"CIFT combiner variant '{variant.variant_id}' inner_fold_count must be at least 2.")
    if variant.combiner_rule not in (
        "logistic_meta_head",
        "mean_score",
        "max_score",
        "top_two_mean",
        "majority_vote",
        "positive_logistic",
        "simplex_logistic",
        "positive_logistic_train_f1_threshold",
        "simplex_logistic_train_f1_threshold",
    ):
        raise BinaryTaskError(
            f"CIFT combiner variant '{variant.variant_id}' has unsupported combiner rule "
            f"'{variant.combiner_rule}'."
        )


def _validate_inputs(
    datasets: tuple[CiftMetaCombinerDataset, ...],
    baseline_feature_key: str,
    variants: tuple[CiftMetaCombinerVariant, ...],
) -> None:
    if len(datasets) == 0:
        raise BinaryTaskError("At least one CIFT combiner dataset is required.")
    if baseline_feature_key == "":
        raise BinaryTaskError("CIFT combiner baseline feature key must not be empty.")
    for index, dataset in enumerate(datasets):
        if dataset.dataset_id == "":
            raise BinaryTaskError(f"CIFT combiner dataset {index} has an empty dataset id.")
    if len(variants) == 0:
        raise BinaryTaskError("At least one CIFT combiner variant is required.")
    for variant in variants:
        _validate_variant(variant)
    if len({variant.variant_id for variant in variants}) != len(variants):
        raise BinaryTaskError("CIFT combiner variant ids must be unique.")
    if len({variant.feature_name for variant in variants}) != len(variants):
        raise BinaryTaskError("CIFT combiner feature names must be unique.")


def _task_definition(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise BinaryTaskError(f"Expected exactly one binary task named '{task_name}', found {len(matches)}.")
    return matches[0]


def _head_variant(variant: CiftMetaCombinerVariant) -> CiftMetaHeadVariant:
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


def _other_label(target_labels: tuple[str, ...], risk_label: str) -> str:
    labels = tuple(sorted(set(target_labels)))
    if risk_label not in labels:
        raise BinaryTaskError(f"CIFT combiner risk label '{risk_label}' is not in labels {labels}.")
    other_labels = tuple(label for label in labels if label != risk_label)
    if len(other_labels) != 1:
        raise BinaryTaskError("CIFT combiner requires exactly one non-risk target label.")
    return other_labels[0]


def _combine_source_scores(
    diagnostics: CiftMetaHeadExampleDiagnostic,
    combiner_rule: CiftMetaCombinerRule,
) -> float:
    source_scores = np.asarray(tuple(source.risk_score for source in diagnostics.sources), dtype=np.float64)
    if source_scores.shape[0] == 0:
        raise BinaryTaskError(f"CIFT combiner diagnostic for '{diagnostics.example_id}' has no source scores.")
    if combiner_rule == "mean_score":
        return float(np.mean(source_scores))
    if combiner_rule == "max_score":
        return float(np.max(source_scores))
    if combiner_rule == "top_two_mean":
        selected_count = min(2, source_scores.shape[0])
        return float(np.mean(np.sort(source_scores)[-selected_count:]))
    if combiner_rule == "majority_vote":
        votes = np.asarray(source_scores >= 0.5, dtype=np.float64)
        return float(np.mean(votes))
    raise BinaryTaskError(f"Combiner rule '{combiner_rule}' does not use source-score aggregation.")


def _sigmoid(values: FloatVector) -> FloatVector:
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float64, copy=False)


def _risk_label_index(label_names: tuple[str, ...], risk_label: str) -> int:
    matches = tuple(index for index, label_name in enumerate(label_names) if label_name == risk_label)
    if len(matches) != 1:
        raise BinaryTaskError(f"CIFT combiner risk label '{risk_label}' is not in labels {label_names}.")
    return matches[0]


def _balanced_sample_weights(labels: FloatVector) -> FloatVector:
    positive_count = float(np.sum(labels))
    negative_count = float(labels.shape[0] - positive_count)
    if positive_count == 0.0 or negative_count == 0.0:
        raise BinaryTaskError("CIFT constrained combiner requires both classes in each training fold.")
    positive_weight = labels.shape[0] / (2.0 * positive_count)
    negative_weight = labels.shape[0] / (2.0 * negative_count)
    return np.asarray(
        [positive_weight if label == 1.0 else negative_weight for label in labels.tolist()],
        dtype=np.float64,
    )


def _standardized_train_scores(scores: FloatMatrix) -> tuple[FloatMatrix, FloatVector, FloatVector]:
    feature_mean = scores.mean(axis=0).astype(np.float64, copy=False)
    feature_scale = scores.std(axis=0).astype(np.float64, copy=False)
    safe_scale = np.where(feature_scale == 0.0, 1.0, feature_scale).astype(np.float64, copy=False)
    standardized = ((scores - feature_mean) / safe_scale).astype(np.float64, copy=False)
    return standardized, feature_mean, safe_scale


def _standardize_scores(scores: FloatMatrix, feature_mean: FloatVector, feature_scale: FloatVector) -> FloatMatrix:
    return ((scores - feature_mean) / feature_scale).astype(np.float64, copy=False)


def _candidate_thresholds(risk_scores: FloatVector) -> tuple[float, ...]:
    unique_scores = np.unique(risk_scores)
    if unique_scores.shape[0] == 0:
        raise BinaryTaskError("Cannot learn CIFT constrained combiner threshold from empty scores.")
    candidates: list[float] = [float(unique_scores[0] - 1e-12), float(unique_scores[-1] + 1e-12), 0.5]
    for left, right in zip(unique_scores[:-1], unique_scores[1:], strict=True):
        candidates.append(float((left + right) / 2.0))
    return tuple(sorted(set(candidates)))


def _learn_threshold(
    risk_scores: FloatVector,
    binary_labels: FloatVector,
) -> float:
    encoded_labels = binary_labels.astype(np.int64, copy=False)
    label_indices = np.arange(2, dtype=np.int64)

    def score_threshold(threshold: float) -> tuple[float, float]:
        predictions = np.asarray([1 if score >= threshold else 0 for score in risk_scores], dtype=np.int64)
        return (
            float(f1_score(encoded_labels, predictions, average="macro", labels=label_indices, zero_division=0)),
            float(accuracy_score(encoded_labels, predictions)),
        )

    return max(_candidate_thresholds(risk_scores), key=score_threshold)


def _fit_constrained_logistic(
    train_scores: FloatMatrix,
    train_labels: NDArray[np.int64],
    label_names: tuple[str, ...],
    variant: CiftMetaCombinerVariant,
    binary_config: BinaryTaskConfig,
) -> _ConstrainedLogisticFit:
    risk_index = _risk_label_index(label_names=label_names, risk_label=variant.risk_label)
    binary_labels = (train_labels == risk_index).astype(np.float64, copy=False)
    sample_weights = _balanced_sample_weights(binary_labels)
    standardized_scores, feature_mean, feature_scale = _standardized_train_scores(train_scores)
    feature_count = standardized_scores.shape[1]
    initial = np.zeros(feature_count + 1, dtype=np.float64)
    initial[1:] = 1.0 / float(feature_count)
    regularization = 1.0 / binary_config.regularization_c

    def objective(parameters: FloatVector) -> float:
        intercept = parameters[0]
        coefficients = parameters[1:]
        logits = intercept + standardized_scores @ coefficients
        losses = np.logaddexp(0.0, logits) - binary_labels * logits
        weighted_loss = float(np.mean(sample_weights * losses))
        penalty = float(0.5 * regularization * np.sum(coefficients * coefficients) / float(feature_count))
        return weighted_loss + penalty

    bounds = Bounds(
        lb=np.asarray([-np.inf] + [0.0 for _ in range(feature_count)], dtype=np.float64),
        ub=np.asarray([np.inf] + [np.inf for _ in range(feature_count)], dtype=np.float64),
    )
    constraints: tuple[LinearConstraint, ...] = ()
    method = "L-BFGS-B"
    if variant.combiner_rule in ("simplex_logistic", "simplex_logistic_train_f1_threshold"):
        constraint_row = np.asarray([0.0] + [1.0 for _ in range(feature_count)], dtype=np.float64)
        constraints = (LinearConstraint(constraint_row, lb=1.0, ub=1.0),)
        method = "SLSQP"

    result = minimize(
        objective,
        initial,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": binary_config.max_iter},
    )
    if not result.success:
        raise BinaryTaskError(
            f"CIFT constrained combiner '{variant.variant_id}' optimization failed: {result.message}."
        )
    parameters = np.asarray(result.x, dtype=np.float64)
    train_risk_scores = _sigmoid(parameters[0] + standardized_scores @ parameters[1:])
    if variant.combiner_rule in ("positive_logistic_train_f1_threshold", "simplex_logistic_train_f1_threshold"):
        threshold = _learn_threshold(risk_scores=train_risk_scores, binary_labels=binary_labels)
    else:
        threshold = 0.5
    return _ConstrainedLogisticFit(
        intercept=float(parameters[0]),
        coefficients=parameters[1:].astype(np.float64, copy=False),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        threshold=threshold,
    )


def _predict_constrained_logistic_scores(
    fit: _ConstrainedLogisticFit,
    test_scores: FloatMatrix,
) -> FloatVector:
    standardized_scores = _standardize_scores(
        scores=test_scores,
        feature_mean=fit.feature_mean,
        feature_scale=fit.feature_scale,
    )
    logits = fit.intercept + standardized_scores @ fit.coefficients
    return _sigmoid(logits)


def _predictions_from_source_score_folds(
    dataset: BinaryTaskDataset,
    variant: CiftMetaCombinerVariant,
    folds: tuple[CiftMetaHeadSourceScoreFold, ...],
    binary_config: BinaryTaskConfig,
) -> BinaryMethodErrorAnalysis:
    predictions: list[BinaryExamplePrediction] = []
    for fold in folds:
        fit = _fit_constrained_logistic(
            train_scores=fold.train_scores,
            train_labels=fold.train_labels,
            label_names=fold.label_names,
            variant=variant,
            binary_config=binary_config,
        )
        risk_scores = _predict_constrained_logistic_scores(fit=fit, test_scores=fold.test_scores)
        other_label = _other_label(dataset.target_labels, variant.risk_label)
        for row_index, risk_score in zip(fold.test_indices.tolist(), risk_scores.tolist(), strict=True):
            predicted_label = variant.risk_label if risk_score >= fit.threshold else other_label
            true_label = dataset.target_labels[row_index]
            predictions.append(
                BinaryExamplePrediction(
                    fold_index=fold.fold_index,
                    example_id=dataset.example_ids[row_index],
                    family=dataset.families[row_index],
                    source_label=dataset.source_labels[row_index],
                    true_label=true_label,
                    predicted_label=predicted_label,
                    is_correct=predicted_label == true_label,
                )
            )
    return _method_error_analysis(
        feature_name=variant.feature_name,
        label_names=folds[0].label_names,
        predictions=tuple(predictions),
    )


def _method_error_analysis(
    feature_name: str,
    label_names: tuple[str, ...],
    predictions: tuple[BinaryExamplePrediction, ...],
) -> BinaryMethodErrorAnalysis:
    if len(predictions) == 0:
        raise BinaryTaskError(f"CIFT combiner feature '{feature_name}' produced no predictions.")
    correct_count = sum(1 for prediction in predictions if prediction.is_correct)
    prediction_count = len(predictions)
    return BinaryMethodErrorAnalysis(
        method_name="activation_probe",
        feature_name=feature_name,
        label_names=label_names,
        prediction_count=prediction_count,
        correct_count=correct_count,
        error_count=prediction_count - correct_count,
        accuracy=float(correct_count / prediction_count),
        family_summaries=summarize_family_predictions(predictions),
        predictions=predictions,
    )


def _collect_constrained_combiner_predictions(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    binary_config: BinaryTaskConfig,
    variant: CiftMetaCombinerVariant,
) -> BinaryMethodErrorAnalysis:
    diagnostics = collect_grouped_cift_meta_head_diagnostics(
        artifact=artifact,
        dataset=dataset,
        binary_config=binary_config,
        variant=_head_variant(variant),
    )
    other_label = _other_label(dataset.target_labels, variant.risk_label)
    predictions: list[BinaryExamplePrediction] = []
    for diagnostic in diagnostics:
        combined_score = _combine_source_scores(diagnostics=diagnostic, combiner_rule=variant.combiner_rule)
        predicted_label = variant.risk_label if combined_score >= 0.5 else other_label
        predictions.append(
            BinaryExamplePrediction(
                fold_index=diagnostic.fold_index,
                example_id=diagnostic.example_id,
                family=diagnostic.family,
                source_label=diagnostic.source_label,
                true_label=diagnostic.true_label,
                predicted_label=predicted_label,
                is_correct=predicted_label == diagnostic.true_label,
            )
        )
    return _method_error_analysis(
        feature_name=variant.feature_name,
        label_names=tuple(sorted(set(dataset.target_labels))),
        predictions=tuple(predictions),
    )


def _collect_candidate_predictions(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    binary_config: BinaryTaskConfig,
    variant: CiftMetaCombinerVariant,
) -> BinaryMethodErrorAnalysis:
    if variant.combiner_rule == "logistic_meta_head":
        return collect_grouped_cift_meta_head_predictions(
            artifact=artifact,
            dataset=dataset,
            binary_config=binary_config,
            variant=_head_variant(variant),
        )
    if variant.combiner_rule in (
        "positive_logistic",
        "simplex_logistic",
        "positive_logistic_train_f1_threshold",
        "simplex_logistic_train_f1_threshold",
    ):
        folds = collect_grouped_cift_meta_head_source_score_folds(
            artifact=artifact,
            dataset=dataset,
            binary_config=binary_config,
            variant=_head_variant(variant),
        )
        return _predictions_from_source_score_folds(
            dataset=dataset,
            variant=variant,
            folds=folds,
            binary_config=binary_config,
        )
    return _collect_constrained_combiner_predictions(
        artifact=artifact,
        dataset=dataset,
        binary_config=binary_config,
        variant=variant,
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
    dataset: CiftMetaCombinerDataset,
    definition: BinaryTaskDefinition,
    baseline_feature_key: str,
    variant: CiftMetaCombinerVariant,
    binary_config: BinaryTaskConfig,
) -> DatasetResidualErrorComparison:
    task_dataset = build_binary_task_dataset(dataset.artifact, definition)
    baseline_config = replace(binary_config, activation_feature_key=baseline_feature_key)
    baseline_method = collect_grouped_activation_predictions(
        artifact=dataset.artifact,
        dataset=task_dataset,
        config=baseline_config,
    )
    candidate_method = _collect_candidate_predictions(
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
    variant: CiftMetaCombinerVariant,
    comparison: DatasetResidualErrorComparison,
) -> CiftMetaCombinerDatasetVariantReport:
    residual = comparison.comparison
    return CiftMetaCombinerDatasetVariantReport(
        dataset_id=comparison.dataset_id,
        variant_id=variant.variant_id,
        feature_name=variant.feature_name,
        combiner_rule=variant.combiner_rule,
        source_feature_keys=variant.source_feature_keys,
        calibration_source_labels=variant.calibration_source_labels,
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
    variant: CiftMetaCombinerVariant,
    dataset_reports: tuple[CiftMetaCombinerDatasetVariantReport, ...],
) -> CiftMetaCombinerVariantSummary:
    candidate_accuracies = tuple(report.candidate_accuracy for report in dataset_reports)
    fixed_error_count = sum(report.fixed_error_count for report in dataset_reports)
    introduced_error_count = sum(report.introduced_error_count for report in dataset_reports)
    return CiftMetaCombinerVariantSummary(
        variant_id=variant.variant_id,
        feature_name=variant.feature_name,
        combiner_rule=variant.combiner_rule,
        source_feature_keys=variant.source_feature_keys,
        calibration_source_labels=variant.calibration_source_labels,
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
    summaries: tuple[CiftMetaCombinerVariantSummary, ...],
) -> CiftMetaCombinerVariantSummary:
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


def compare_cift_meta_combiner_ablation(
    datasets: tuple[CiftMetaCombinerDataset, ...],
    task_name: str,
    baseline_feature_key: str,
    variants: tuple[CiftMetaCombinerVariant, ...],
    binary_config: BinaryTaskConfig,
) -> CiftMetaCombinerAblationReport:
    _validate_inputs(datasets=datasets, baseline_feature_key=baseline_feature_key, variants=variants)
    definition = _task_definition(task_name)
    dataset_variant_reports: list[CiftMetaCombinerDatasetVariantReport] = []
    summaries: list[CiftMetaCombinerVariantSummary] = []

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
    return CiftMetaCombinerAblationReport(
        source_model_id=first_metadata["model_id"],
        source_revision=first_metadata["revision"],
        source_selected_device=first_metadata["selected_device"],
        evaluation_strategy="stratified_group_kfold",
        fold_count=binary_config.fold_count,
        inner_fold_count=variants[0].inner_fold_count,
        random_seed=binary_config.random_seed,
        regularization_c=binary_config.regularization_c,
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


def _summary_to_json(summary: CiftMetaCombinerVariantSummary) -> dict[str, JsonValue]:
    return {
        "variant_id": summary.variant_id,
        "feature_name": summary.feature_name,
        "combiner_rule": summary.combiner_rule,
        "source_feature_keys": list(summary.source_feature_keys),
        "calibration_source_labels": list(summary.calibration_source_labels),
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


def _dataset_variant_to_json(report: CiftMetaCombinerDatasetVariantReport) -> dict[str, JsonValue]:
    return {
        "dataset_id": report.dataset_id,
        "variant_id": report.variant_id,
        "feature_name": report.feature_name,
        "combiner_rule": report.combiner_rule,
        "source_feature_keys": list(report.source_feature_keys),
        "calibration_source_labels": list(report.calibration_source_labels),
        "reference_error_count": report.reference_error_count,
        "candidate_error_count": report.candidate_error_count,
        "fixed_error_count": report.fixed_error_count,
        "persistent_error_count": report.persistent_error_count,
        "introduced_error_count": report.introduced_error_count,
        "net_error_delta": report.net_error_delta,
        "reference_accuracy": report.reference_accuracy,
        "candidate_accuracy": report.candidate_accuracy,
    }


def cift_meta_combiner_ablation_report_to_json(report: CiftMetaCombinerAblationReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "evaluation_strategy": report.evaluation_strategy,
        "fold_count": report.fold_count,
        "inner_fold_count": report.inner_fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
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


def write_cift_meta_combiner_ablation_json(path: Path, report: CiftMetaCombinerAblationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cift_meta_combiner_ablation_report_to_json(report), file, indent=2)
        file.write("\n")


def _joined(values: tuple[str, ...]) -> str:
    return "`, `".join(values)


def render_cift_meta_combiner_ablation_markdown(report: CiftMetaCombinerAblationReport) -> str:
    lines = [
        "# CIFT Meta-Head Combiner Ablation",
        "",
        "## Source",
        "",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Method: `{report.method_name}`",
        f"- Baseline feature: `{report.baseline_feature_key}`",
        f"- Dataset count: `{report.dataset_count}`",
        f"- Variant count: `{report.variant_count}`",
        f"- Best variant: `{report.best_variant_summary.variant_id}`",
        "",
        "## Variant Summary",
        "",
        (
            "| Variant | Combiner Rule | Source Count | Calibration Labels | Candidate Errors | Fixed | "
            "Persistent | Introduced | Net Error Delta | Mean Accuracy |"
        ),
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in report.variant_summaries:
        lines.append(
            f"| `{summary.variant_id}` | "
            f"`{summary.combiner_rule}` | "
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


def write_cift_meta_combiner_ablation_markdown(path: Path, report: CiftMetaCombinerAblationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cift_meta_combiner_ablation_markdown(report), encoding="utf-8")
