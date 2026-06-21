from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import (
    BinaryTaskConfig,
    BinaryTaskDefinition,
    BinaryTaskError,
    EvaluationStrategy,
    build_binary_task_dataset,
    default_binary_task_definitions,
)
from aegis_introspection.cift_meta_head import (
    CiftMetaHeadExampleDiagnostic,
    CiftMetaHeadSourceDiagnostic,
    CiftMetaHeadVariant,
    collect_grouped_cift_meta_head_diagnostics,
)
from aegis_introspection.error_analysis import BinaryExamplePrediction, collect_grouped_activation_predictions
from aegis_introspection.probe import JsonValue


@dataclass(frozen=True)
class CiftMetaScoreDiagnosticDataset:
    dataset_id: str
    artifact: ActivationArtifact


@dataclass(frozen=True)
class CiftMetaScoreSourceSummary:
    source_feature_key: str
    example_count: int
    mean_risk_score: float
    mean_standardized_score: float
    mean_coefficient: float
    mean_logit_contribution: float
    max_abs_logit_contribution: float


@dataclass(frozen=True)
class CiftMetaIntroducedErrorDiagnostic:
    fold_index: int
    example_id: str
    family: str
    source_label: str
    true_label: str
    reference_predicted_label: str
    candidate_predicted_label: str
    meta_risk_score: float
    meta_risk_logit: float
    decision_threshold: float
    intercept: float
    sources: tuple[CiftMetaHeadSourceDiagnostic, ...]


@dataclass(frozen=True)
class CiftMetaScoreDiagnosticReport:
    dataset_id: str
    source_model_id: str
    source_revision: str
    source_selected_device: str
    evaluation_strategy: EvaluationStrategy
    fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    task_name: str
    reference_feature_key: str
    candidate_feature_key: str
    prediction_count: int
    reference_error_count: int
    candidate_error_count: int
    reference_accuracy: float
    candidate_accuracy: float
    introduced_error_count: int
    introduced_errors: tuple[CiftMetaIntroducedErrorDiagnostic, ...]
    source_summaries: tuple[CiftMetaScoreSourceSummary, ...]


def _task_definition(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise BinaryTaskError(f"Expected exactly one binary task named '{task_name}', found {len(matches)}.")
    return matches[0]


def _prediction_index(
    predictions: tuple[BinaryExamplePrediction, ...],
) -> dict[str, BinaryExamplePrediction]:
    indexed: dict[str, BinaryExamplePrediction] = {}
    for prediction in predictions:
        if prediction.example_id in indexed:
            raise BinaryTaskError(f"Duplicate reference prediction for example '{prediction.example_id}'.")
        indexed[prediction.example_id] = prediction
    return indexed


def _diagnostic_index(
    diagnostics: tuple[CiftMetaHeadExampleDiagnostic, ...],
) -> dict[str, CiftMetaHeadExampleDiagnostic]:
    indexed: dict[str, CiftMetaHeadExampleDiagnostic] = {}
    for diagnostic in diagnostics:
        if diagnostic.example_id in indexed:
            raise BinaryTaskError(f"Duplicate candidate diagnostic for example '{diagnostic.example_id}'.")
        indexed[diagnostic.example_id] = diagnostic
    return indexed


def _introduced_error_diagnostic(
    reference: BinaryExamplePrediction,
    candidate: CiftMetaHeadExampleDiagnostic,
) -> CiftMetaIntroducedErrorDiagnostic:
    if reference.true_label != candidate.true_label:
        raise BinaryTaskError(
            f"Example '{reference.example_id}' has mismatched true labels: "
            f"reference='{reference.true_label}', candidate='{candidate.true_label}'."
        )
    if reference.family != candidate.family:
        raise BinaryTaskError(
            f"Example '{reference.example_id}' has mismatched families: "
            f"reference='{reference.family}', candidate='{candidate.family}'."
        )
    return CiftMetaIntroducedErrorDiagnostic(
        fold_index=candidate.fold_index,
        example_id=candidate.example_id,
        family=candidate.family,
        source_label=candidate.source_label,
        true_label=candidate.true_label,
        reference_predicted_label=reference.predicted_label,
        candidate_predicted_label=candidate.predicted_label,
        meta_risk_score=candidate.meta_risk_score,
        meta_risk_logit=candidate.meta_risk_logit,
        decision_threshold=candidate.decision_threshold,
        intercept=candidate.intercept,
        sources=candidate.sources,
    )


def _mean(values: tuple[float, ...]) -> float:
    if len(values) == 0:
        raise BinaryTaskError("Cannot summarize an empty numeric sequence.")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _max_abs(values: tuple[float, ...]) -> float:
    if len(values) == 0:
        raise BinaryTaskError("Cannot summarize an empty numeric sequence.")
    return float(np.max(np.abs(np.asarray(values, dtype=np.float64))))


def _source_summaries(
    introduced_errors: tuple[CiftMetaIntroducedErrorDiagnostic, ...],
) -> tuple[CiftMetaScoreSourceSummary, ...]:
    if len(introduced_errors) == 0:
        return ()
    source_feature_keys = tuple(source.source_feature_key for source in introduced_errors[0].sources)
    summaries: list[CiftMetaScoreSourceSummary] = []
    for source_feature_key in source_feature_keys:
        matching_sources = tuple(
            source
            for error in introduced_errors
            for source in error.sources
            if source.source_feature_key == source_feature_key
        )
        if len(matching_sources) != len(introduced_errors):
            raise BinaryTaskError(f"Introduced-error diagnostics are missing source '{source_feature_key}'.")
        summaries.append(
            CiftMetaScoreSourceSummary(
                source_feature_key=source_feature_key,
                example_count=len(matching_sources),
                mean_risk_score=_mean(tuple(source.risk_score for source in matching_sources)),
                mean_standardized_score=_mean(tuple(source.standardized_score for source in matching_sources)),
                mean_coefficient=_mean(tuple(source.coefficient for source in matching_sources)),
                mean_logit_contribution=_mean(tuple(source.logit_contribution for source in matching_sources)),
                max_abs_logit_contribution=_max_abs(
                    tuple(source.logit_contribution for source in matching_sources)
                ),
            )
        )
    return tuple(
        sorted(
            summaries,
            key=lambda summary: abs(summary.mean_logit_contribution),
            reverse=True,
        )
    )


def diagnose_cift_meta_introduced_errors(
    dataset: CiftMetaScoreDiagnosticDataset,
    task_name: str,
    baseline_feature_key: str,
    variant: CiftMetaHeadVariant,
    binary_config: BinaryTaskConfig,
) -> CiftMetaScoreDiagnosticReport:
    if dataset.dataset_id == "":
        raise BinaryTaskError("CIFT meta score diagnostic dataset id must not be empty.")
    if baseline_feature_key == "":
        raise BinaryTaskError("CIFT meta score diagnostic baseline feature key must not be empty.")

    task_dataset = build_binary_task_dataset(dataset.artifact, _task_definition(task_name))
    reference_method = collect_grouped_activation_predictions(
        artifact=dataset.artifact,
        dataset=task_dataset,
        config=replace(binary_config, activation_feature_key=baseline_feature_key),
    )
    candidate_diagnostics = collect_grouped_cift_meta_head_diagnostics(
        artifact=dataset.artifact,
        dataset=task_dataset,
        binary_config=binary_config,
        variant=variant,
    )
    reference_predictions = _prediction_index(reference_method.predictions)
    candidate_predictions = _diagnostic_index(candidate_diagnostics)

    if set(reference_predictions.keys()) != set(candidate_predictions.keys()):
        raise BinaryTaskError("Reference predictions and candidate diagnostics have different example ids.")

    introduced_errors: list[CiftMetaIntroducedErrorDiagnostic] = []
    for example_id in sorted(reference_predictions.keys()):
        reference = reference_predictions[example_id]
        candidate = candidate_predictions[example_id]
        if reference.is_correct and not candidate.is_correct:
            introduced_errors.append(_introduced_error_diagnostic(reference=reference, candidate=candidate))

    introduced_error_tuple = tuple(
        sorted(
            introduced_errors,
            key=lambda error: (error.family, error.example_id),
        )
    )
    candidate_correct_count = sum(1 for diagnostic in candidate_diagnostics if diagnostic.is_correct)
    metadata = dataset.artifact["metadata"]
    return CiftMetaScoreDiagnosticReport(
        dataset_id=dataset.dataset_id,
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        evaluation_strategy="stratified_group_kfold",
        fold_count=binary_config.fold_count,
        random_seed=binary_config.random_seed,
        regularization_c=binary_config.regularization_c,
        max_iter=binary_config.max_iter,
        task_name=task_name,
        reference_feature_key=baseline_feature_key,
        candidate_feature_key=variant.feature_name,
        prediction_count=reference_method.prediction_count,
        reference_error_count=reference_method.error_count,
        candidate_error_count=len(candidate_diagnostics) - candidate_correct_count,
        reference_accuracy=reference_method.accuracy,
        candidate_accuracy=float(candidate_correct_count / len(candidate_diagnostics)),
        introduced_error_count=len(introduced_error_tuple),
        introduced_errors=introduced_error_tuple,
        source_summaries=_source_summaries(introduced_error_tuple),
    )


def _source_diagnostic_to_json(source: CiftMetaHeadSourceDiagnostic) -> dict[str, JsonValue]:
    return {
        "source_feature_key": source.source_feature_key,
        "risk_score": source.risk_score,
        "standardized_score": source.standardized_score,
        "coefficient": source.coefficient,
        "logit_contribution": source.logit_contribution,
    }


def _introduced_error_to_json(error: CiftMetaIntroducedErrorDiagnostic) -> dict[str, JsonValue]:
    return {
        "fold_index": error.fold_index,
        "example_id": error.example_id,
        "family": error.family,
        "source_label": error.source_label,
        "true_label": error.true_label,
        "reference_predicted_label": error.reference_predicted_label,
        "candidate_predicted_label": error.candidate_predicted_label,
        "meta_risk_score": error.meta_risk_score,
        "meta_risk_logit": error.meta_risk_logit,
        "decision_threshold": error.decision_threshold,
        "intercept": error.intercept,
        "sources": [_source_diagnostic_to_json(source) for source in error.sources],
    }


def _source_summary_to_json(summary: CiftMetaScoreSourceSummary) -> dict[str, JsonValue]:
    return {
        "source_feature_key": summary.source_feature_key,
        "example_count": summary.example_count,
        "mean_risk_score": summary.mean_risk_score,
        "mean_standardized_score": summary.mean_standardized_score,
        "mean_coefficient": summary.mean_coefficient,
        "mean_logit_contribution": summary.mean_logit_contribution,
        "max_abs_logit_contribution": summary.max_abs_logit_contribution,
    }


def cift_meta_score_diagnostics_to_json(report: CiftMetaScoreDiagnosticReport) -> dict[str, JsonValue]:
    return {
        "dataset_id": report.dataset_id,
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "evaluation_strategy": report.evaluation_strategy,
        "fold_count": report.fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "task_name": report.task_name,
        "reference_feature_key": report.reference_feature_key,
        "candidate_feature_key": report.candidate_feature_key,
        "prediction_count": report.prediction_count,
        "reference_error_count": report.reference_error_count,
        "candidate_error_count": report.candidate_error_count,
        "reference_accuracy": report.reference_accuracy,
        "candidate_accuracy": report.candidate_accuracy,
        "introduced_error_count": report.introduced_error_count,
        "introduced_errors": [_introduced_error_to_json(error) for error in report.introduced_errors],
        "source_summaries": [_source_summary_to_json(summary) for summary in report.source_summaries],
    }


def write_cift_meta_score_diagnostics_json(path: Path, report: CiftMetaScoreDiagnosticReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cift_meta_score_diagnostics_to_json(report), file, indent=2)
        file.write("\n")


def _format_probability(value: float) -> str:
    return f"{value:.4f}"


def _format_signed(value: float) -> str:
    return f"{value:+.4f}"


def render_cift_meta_score_diagnostics_markdown(report: CiftMetaScoreDiagnosticReport) -> str:
    lines = [
        "# CIFT Meta-Head Score Diagnostics",
        "",
        "## Source",
        "",
        f"- Model: `{report.source_model_id}`",
        f"- Revision: `{report.source_revision}`",
        f"- Extraction device: `{report.source_selected_device}`",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Dataset: `{report.dataset_id}`",
        f"- Reference feature: `{report.reference_feature_key}`",
        f"- Candidate feature: `{report.candidate_feature_key}`",
        f"- Fold count: `{report.fold_count}`",
        "",
        "## Summary",
        "",
        "| Reference Errors | Candidate Errors | Introduced Errors | Reference Accuracy | Candidate Accuracy |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {report.reference_error_count} | {report.candidate_error_count} | "
            f"{report.introduced_error_count} | {report.reference_accuracy:.4f} | "
            f"{report.candidate_accuracy:.4f} |"
        ),
        "",
        "## Source Summary",
        "",
    ]

    if len(report.source_summaries) == 0:
        lines.extend(["No introduced errors to summarize.", ""])
    else:
        lines.extend(
            [
                (
                    "| Source Feature | Mean Risk Score | Mean Standardized Score | Mean Coefficient | "
                    "Mean Logit Contribution | Max Abs Contribution |"
                ),
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for summary in report.source_summaries:
            lines.append(
                f"| `{summary.source_feature_key}` | "
                f"{_format_probability(summary.mean_risk_score)} | "
                f"{_format_signed(summary.mean_standardized_score)} | "
                f"{_format_signed(summary.mean_coefficient)} | "
                f"{_format_signed(summary.mean_logit_contribution)} | "
                f"{summary.max_abs_logit_contribution:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Introduced Errors",
            "",
        ]
    )
    if len(report.introduced_errors) == 0:
        lines.extend(["No introduced errors.", ""])
        return "\n".join(lines)

    lines.extend(
        [
            (
                "| Example | Family | True Label | Reference Prediction | Candidate Prediction | "
                "Meta Risk | Threshold |"
            ),
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for error in report.introduced_errors:
        lines.append(
            f"| `{error.example_id}` | `{error.family}` | `{error.true_label}` | "
            f"`{error.reference_predicted_label}` | `{error.candidate_predicted_label}` | "
            f"{_format_probability(error.meta_risk_score)} | {_format_probability(error.decision_threshold)} |"
        )

    lines.extend(["", "## Introduced Error Source Evidence", ""])
    for error in report.introduced_errors:
        lines.extend(
            [
                f"### `{error.example_id}`",
                "",
                f"- Fold: `{error.fold_index}`",
                f"- Meta risk score: `{_format_probability(error.meta_risk_score)}`",
                f"- Meta risk logit: `{_format_signed(error.meta_risk_logit)}`",
                f"- Decision threshold: `{_format_probability(error.decision_threshold)}`",
                f"- Intercept: `{_format_signed(error.intercept)}`",
                "",
                "| Source Feature | Risk Score | Standardized Score | Coefficient | Logit Contribution |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for source in sorted(error.sources, key=lambda item: abs(item.logit_contribution), reverse=True):
            lines.append(
                f"| `{source.source_feature_key}` | "
                f"{_format_probability(source.risk_score)} | "
                f"{_format_signed(source.standardized_score)} | "
                f"{_format_signed(source.coefficient)} | "
                f"{_format_signed(source.logit_contribution)} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_cift_meta_score_diagnostics_markdown(path: Path, report: CiftMetaScoreDiagnosticReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cift_meta_score_diagnostics_markdown(report), encoding="utf-8")
