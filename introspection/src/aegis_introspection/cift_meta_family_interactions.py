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


CiftMetaFamilyInteractionRule: TypeAlias = Literal[
    "raw_scores",
    "family_means",
    "family_mean_gaps",
]


@dataclass(frozen=True)
class CiftMetaFamilyInteractionDataset:
    dataset_id: str
    artifact: ActivationArtifact


@dataclass(frozen=True)
class CiftMetaFamilyInteractionVariant:
    variant_id: str
    feature_name: str
    final_token_source_feature_keys: tuple[str, ...]
    mean_pool_source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    ridge: float
    risk_label: str
    inner_fold_count: int
    meta_regularization_c: float
    interaction_rule: CiftMetaFamilyInteractionRule


@dataclass(frozen=True)
class _InteractionScoreFold:
    fold_index: int
    source_feature_keys: tuple[str, ...]
    label_names: tuple[str, ...]
    test_indices: np.ndarray
    train_scores: np.ndarray
    test_scores: np.ndarray
    train_labels: np.ndarray


@dataclass(frozen=True)
class CiftMetaFamilyInteractionDatasetVariantReport:
    dataset_id: str
    variant_id: str
    feature_name: str
    interaction_rule: CiftMetaFamilyInteractionRule
    final_token_source_feature_keys: tuple[str, ...]
    mean_pool_source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    source_feature_count: int
    added_feature_count: int
    meta_feature_count: int
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
class CiftMetaFamilyInteractionVariantSummary:
    variant_id: str
    feature_name: str
    interaction_rule: CiftMetaFamilyInteractionRule
    final_token_source_feature_keys: tuple[str, ...]
    mean_pool_source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    source_feature_count: int
    added_feature_count: int
    meta_feature_count: int
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
class CiftMetaFamilyInteractionReport:
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
    best_variant_summary: CiftMetaFamilyInteractionVariantSummary
    variant_summaries: tuple[CiftMetaFamilyInteractionVariantSummary, ...]
    dataset_variants: tuple[CiftMetaFamilyInteractionDatasetVariantReport, ...]


def _source_feature_keys(variant: CiftMetaFamilyInteractionVariant) -> tuple[str, ...]:
    return variant.final_token_source_feature_keys + variant.mean_pool_source_feature_keys


def _source_feature_count(variant: CiftMetaFamilyInteractionVariant) -> int:
    return len(_source_feature_keys(variant))


def _added_feature_count(variant: CiftMetaFamilyInteractionVariant) -> int:
    if variant.interaction_rule == "raw_scores":
        return 0
    if variant.interaction_rule == "family_means":
        return 2
    if variant.interaction_rule == "family_mean_gaps":
        return 4
    raise BinaryTaskError(
        f"CIFT family-interaction variant '{variant.variant_id}' has unsupported interaction rule "
        f"'{variant.interaction_rule}'."
    )


def _meta_feature_count(variant: CiftMetaFamilyInteractionVariant) -> int:
    return _source_feature_count(variant) + _added_feature_count(variant)


def _validate_variant(variant: CiftMetaFamilyInteractionVariant) -> None:
    if variant.variant_id == "":
        raise BinaryTaskError("CIFT family-interaction variant id must not be empty.")
    if variant.feature_name == "":
        raise BinaryTaskError(f"CIFT family-interaction variant '{variant.variant_id}' feature name must not be empty.")
    if len(variant.final_token_source_feature_keys) == 0:
        raise BinaryTaskError(
            f"CIFT family-interaction variant '{variant.variant_id}' requires final-token source features."
        )
    if len(variant.mean_pool_source_feature_keys) == 0:
        raise BinaryTaskError(
            f"CIFT family-interaction variant '{variant.variant_id}' requires mean-pool source features."
        )
    if len(set(_source_feature_keys(variant))) != _source_feature_count(variant):
        raise BinaryTaskError(
            f"CIFT family-interaction variant '{variant.variant_id}' source features must be unique."
        )
    if len(variant.calibration_source_labels) == 0:
        raise BinaryTaskError(
            f"CIFT family-interaction variant '{variant.variant_id}' requires calibration labels."
        )
    if variant.ridge <= 0:
        raise BinaryTaskError(f"CIFT family-interaction variant '{variant.variant_id}' ridge must be greater than 0.")
    if variant.risk_label == "":
        raise BinaryTaskError(f"CIFT family-interaction variant '{variant.variant_id}' risk label must not be empty.")
    if variant.inner_fold_count < 2:
        raise BinaryTaskError(
            f"CIFT family-interaction variant '{variant.variant_id}' inner_fold_count must be at least 2."
        )
    if variant.meta_regularization_c <= 0:
        raise BinaryTaskError(
            f"CIFT family-interaction variant '{variant.variant_id}' meta_regularization_c must be greater than 0."
        )
    if variant.interaction_rule not in ("raw_scores", "family_means", "family_mean_gaps"):
        raise BinaryTaskError(
            f"CIFT family-interaction variant '{variant.variant_id}' has unsupported interaction rule "
            f"'{variant.interaction_rule}'."
        )


def _validate_inputs(
    datasets: tuple[CiftMetaFamilyInteractionDataset, ...],
    baseline_feature_key: str,
    variants: tuple[CiftMetaFamilyInteractionVariant, ...],
) -> None:
    if len(datasets) == 0:
        raise BinaryTaskError("At least one CIFT family-interaction dataset is required.")
    if baseline_feature_key == "":
        raise BinaryTaskError("CIFT family-interaction baseline feature key must not be empty.")
    for index, dataset in enumerate(datasets):
        if dataset.dataset_id == "":
            raise BinaryTaskError(f"CIFT family-interaction dataset {index} has an empty dataset id.")
    if len(variants) == 0:
        raise BinaryTaskError("At least one CIFT family-interaction variant is required.")
    for variant in variants:
        _validate_variant(variant)
    if len({variant.variant_id for variant in variants}) != len(variants):
        raise BinaryTaskError("CIFT family-interaction variant ids must be unique.")
    if len({variant.feature_name for variant in variants}) != len(variants):
        raise BinaryTaskError("CIFT family-interaction feature names must be unique.")
    if len({variant.meta_regularization_c for variant in variants}) != 1:
        raise BinaryTaskError("CIFT family-interaction variants must share one meta_regularization_c.")


def _task_definition(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise BinaryTaskError(f"Expected exactly one binary task named '{task_name}', found {len(matches)}.")
    return matches[0]


def _head_variant(variant: CiftMetaFamilyInteractionVariant) -> CiftMetaHeadVariant:
    return CiftMetaHeadVariant(
        variant_id=variant.variant_id,
        feature_name=variant.feature_name,
        source_feature_keys=_source_feature_keys(variant),
        calibration_source_labels=variant.calibration_source_labels,
        ridge=variant.ridge,
        risk_label=variant.risk_label,
        inner_fold_count=variant.inner_fold_count,
        decision_rule="logistic_default",
    )


def _risk_label_index(label_names: tuple[str, ...], risk_label: str) -> int:
    matches = tuple(index for index, label_name in enumerate(label_names) if label_name == risk_label)
    if len(matches) != 1:
        raise BinaryTaskError(f"CIFT family-interaction risk label '{risk_label}' is not in labels {label_names}.")
    return matches[0]


def _other_label(dataset: BinaryTaskDataset, risk_label: str) -> str:
    labels = tuple(sorted(set(dataset.target_labels)))
    other_labels = tuple(label for label in labels if label != risk_label)
    if len(other_labels) != 1:
        raise BinaryTaskError("CIFT family-interaction requires exactly one non-risk target label.")
    return other_labels[0]


def _build_meta_classifier(
    variant: CiftMetaFamilyInteractionVariant,
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


def _family_mean_columns(scores: np.ndarray, variant: CiftMetaFamilyInteractionVariant) -> tuple[np.ndarray, np.ndarray]:
    final_token_count = len(variant.final_token_source_feature_keys)
    final_token_scores = scores[:, :final_token_count]
    mean_pool_scores = scores[:, final_token_count:]
    final_token_mean = final_token_scores.mean(axis=1, keepdims=True)
    mean_pool_mean = mean_pool_scores.mean(axis=1, keepdims=True)
    return final_token_mean, mean_pool_mean


def _interaction_scores(scores: np.ndarray, variant: CiftMetaFamilyInteractionVariant) -> np.ndarray:
    source_scores = scores.astype(np.float64, copy=False)
    if variant.interaction_rule == "raw_scores":
        return source_scores

    final_token_mean, mean_pool_mean = _family_mean_columns(scores=source_scores, variant=variant)
    if variant.interaction_rule == "family_means":
        return np.concatenate((source_scores, final_token_mean, mean_pool_mean), axis=1)
    if variant.interaction_rule == "family_mean_gaps":
        mean_gap = mean_pool_mean - final_token_mean
        absolute_mean_gap = np.abs(mean_gap)
        return np.concatenate(
            (source_scores, final_token_mean, mean_pool_mean, mean_gap, absolute_mean_gap),
            axis=1,
        )
    raise BinaryTaskError(
        f"CIFT family-interaction variant '{variant.variant_id}' has unsupported interaction rule "
        f"'{variant.interaction_rule}'."
    )


def _interaction_fold(
    fold: CiftMetaHeadSourceScoreFold,
    variant: CiftMetaFamilyInteractionVariant,
) -> _InteractionScoreFold:
    return _InteractionScoreFold(
        fold_index=fold.fold_index,
        source_feature_keys=fold.source_feature_keys,
        label_names=fold.label_names,
        test_indices=fold.test_indices,
        train_scores=_interaction_scores(scores=fold.train_scores, variant=variant),
        test_scores=_interaction_scores(scores=fold.test_scores, variant=variant),
        train_labels=fold.train_labels,
    )


def _predict_fold(
    dataset: BinaryTaskDataset,
    fold: CiftMetaHeadSourceScoreFold,
    variant: CiftMetaFamilyInteractionVariant,
    binary_config: BinaryTaskConfig,
) -> tuple[BinaryExamplePrediction, ...]:
    interaction_fold = _interaction_fold(fold=fold, variant=variant)
    classifier = _build_meta_classifier(variant=variant, binary_config=binary_config)
    classifier.fit(interaction_fold.train_scores, interaction_fold.train_labels)
    predicted_indices = classifier.predict(interaction_fold.test_scores).astype(np.int64, copy=False)
    risk_index = _risk_label_index(label_names=interaction_fold.label_names, risk_label=variant.risk_label)
    other_label = _other_label(dataset=dataset, risk_label=variant.risk_label)
    predictions: list[BinaryExamplePrediction] = []
    for row_index, predicted_index in zip(interaction_fold.test_indices.tolist(), predicted_indices.tolist(), strict=True):
        predicted_label = variant.risk_label if int(predicted_index) == risk_index else other_label
        true_label = dataset.target_labels[row_index]
        predictions.append(
            BinaryExamplePrediction(
                fold_index=interaction_fold.fold_index,
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
    variant: CiftMetaFamilyInteractionVariant,
    label_names: tuple[str, ...],
    predictions: tuple[BinaryExamplePrediction, ...],
) -> BinaryMethodErrorAnalysis:
    if len(predictions) == 0:
        raise BinaryTaskError(f"CIFT family-interaction variant '{variant.variant_id}' produced no predictions.")
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


def _collect_family_interaction_meta_predictions(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    binary_config: BinaryTaskConfig,
    variant: CiftMetaFamilyInteractionVariant,
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
    dataset: CiftMetaFamilyInteractionDataset,
    definition: BinaryTaskDefinition,
    baseline_feature_key: str,
    variant: CiftMetaFamilyInteractionVariant,
    binary_config: BinaryTaskConfig,
) -> DatasetResidualErrorComparison:
    task_dataset = build_binary_task_dataset(dataset.artifact, definition)
    baseline_config = replace(binary_config, activation_feature_key=baseline_feature_key)
    baseline_method = collect_grouped_activation_predictions(
        artifact=dataset.artifact,
        dataset=task_dataset,
        config=baseline_config,
    )
    candidate_method = _collect_family_interaction_meta_predictions(
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
    variant: CiftMetaFamilyInteractionVariant,
    comparison: DatasetResidualErrorComparison,
) -> CiftMetaFamilyInteractionDatasetVariantReport:
    residual = comparison.comparison
    return CiftMetaFamilyInteractionDatasetVariantReport(
        dataset_id=comparison.dataset_id,
        variant_id=variant.variant_id,
        feature_name=variant.feature_name,
        interaction_rule=variant.interaction_rule,
        final_token_source_feature_keys=variant.final_token_source_feature_keys,
        mean_pool_source_feature_keys=variant.mean_pool_source_feature_keys,
        calibration_source_labels=variant.calibration_source_labels,
        source_feature_count=_source_feature_count(variant),
        added_feature_count=_added_feature_count(variant),
        meta_feature_count=_meta_feature_count(variant),
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
    variant: CiftMetaFamilyInteractionVariant,
    dataset_reports: tuple[CiftMetaFamilyInteractionDatasetVariantReport, ...],
) -> CiftMetaFamilyInteractionVariantSummary:
    candidate_accuracies = tuple(report.candidate_accuracy for report in dataset_reports)
    fixed_error_count = sum(report.fixed_error_count for report in dataset_reports)
    introduced_error_count = sum(report.introduced_error_count for report in dataset_reports)
    return CiftMetaFamilyInteractionVariantSummary(
        variant_id=variant.variant_id,
        feature_name=variant.feature_name,
        interaction_rule=variant.interaction_rule,
        final_token_source_feature_keys=variant.final_token_source_feature_keys,
        mean_pool_source_feature_keys=variant.mean_pool_source_feature_keys,
        calibration_source_labels=variant.calibration_source_labels,
        source_feature_count=_source_feature_count(variant),
        added_feature_count=_added_feature_count(variant),
        meta_feature_count=_meta_feature_count(variant),
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
    summaries: tuple[CiftMetaFamilyInteractionVariantSummary, ...],
) -> CiftMetaFamilyInteractionVariantSummary:
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


def compare_cift_meta_family_interactions(
    datasets: tuple[CiftMetaFamilyInteractionDataset, ...],
    task_name: str,
    baseline_feature_key: str,
    variants: tuple[CiftMetaFamilyInteractionVariant, ...],
    binary_config: BinaryTaskConfig,
) -> CiftMetaFamilyInteractionReport:
    _validate_inputs(datasets=datasets, baseline_feature_key=baseline_feature_key, variants=variants)
    definition = _task_definition(task_name)
    dataset_variant_reports: list[CiftMetaFamilyInteractionDatasetVariantReport] = []
    summaries: list[CiftMetaFamilyInteractionVariantSummary] = []

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
    return CiftMetaFamilyInteractionReport(
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


def _summary_to_json(summary: CiftMetaFamilyInteractionVariantSummary) -> dict[str, JsonValue]:
    return {
        "variant_id": summary.variant_id,
        "feature_name": summary.feature_name,
        "interaction_rule": summary.interaction_rule,
        "final_token_source_feature_keys": list(summary.final_token_source_feature_keys),
        "mean_pool_source_feature_keys": list(summary.mean_pool_source_feature_keys),
        "calibration_source_labels": list(summary.calibration_source_labels),
        "source_feature_count": summary.source_feature_count,
        "added_feature_count": summary.added_feature_count,
        "meta_feature_count": summary.meta_feature_count,
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


def _dataset_variant_to_json(report: CiftMetaFamilyInteractionDatasetVariantReport) -> dict[str, JsonValue]:
    return {
        "dataset_id": report.dataset_id,
        "variant_id": report.variant_id,
        "feature_name": report.feature_name,
        "interaction_rule": report.interaction_rule,
        "final_token_source_feature_keys": list(report.final_token_source_feature_keys),
        "mean_pool_source_feature_keys": list(report.mean_pool_source_feature_keys),
        "calibration_source_labels": list(report.calibration_source_labels),
        "source_feature_count": report.source_feature_count,
        "added_feature_count": report.added_feature_count,
        "meta_feature_count": report.meta_feature_count,
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


def cift_meta_family_interactions_to_json(report: CiftMetaFamilyInteractionReport) -> dict[str, JsonValue]:
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


def write_cift_meta_family_interactions_json(path: Path, report: CiftMetaFamilyInteractionReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cift_meta_family_interactions_to_json(report), file, indent=2)
        file.write("\n")


def _joined(values: tuple[str, ...]) -> str:
    return "`, `".join(values)


def render_cift_meta_family_interactions_markdown(report: CiftMetaFamilyInteractionReport) -> str:
    lines = [
        "# CIFT Meta-Head Family Interactions",
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
            "| Variant | Interaction Rule | Meta C | Source Count | Added Features | Meta Features | "
            "Calibration Labels | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | "
            "Mean Accuracy |"
        ),
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in report.variant_summaries:
        lines.append(
            f"| `{summary.variant_id}` | "
            f"`{summary.interaction_rule}` | "
            f"{summary.meta_regularization_c:.4g} | "
            f"{summary.source_feature_count} | "
            f"{summary.added_feature_count} | "
            f"{summary.meta_feature_count} | "
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


def write_cift_meta_family_interactions_markdown(path: Path, report: CiftMetaFamilyInteractionReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cift_meta_family_interactions_markdown(report), encoding="utf-8")
