from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aegis_introspection.binary_tasks import BinaryMethodName, BinaryTaskError, EvaluationStrategy
from aegis_introspection.error_analysis import (
    BinaryErrorAnalysisReport,
    BinaryExamplePrediction,
    BinaryMethodErrorAnalysis,
    BinaryTaskErrorAnalysis,
)
from aegis_introspection.probe import JsonValue


@dataclass(frozen=True)
class ResidualErrorExample:
    example_id: str
    family: str
    source_label: str
    true_label: str
    reference_predicted_label: str
    candidate_predicted_label: str
    reference_fold_index: int
    candidate_fold_index: int


@dataclass(frozen=True)
class ResidualErrorFamilySummary:
    family: str
    fixed_error_count: int
    persistent_error_count: int
    introduced_error_count: int


@dataclass(frozen=True)
class ResidualErrorComparisonReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    evaluation_strategy: EvaluationStrategy
    fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    task_name: str
    method_name: BinaryMethodName
    reference_feature_key: str
    candidate_feature_key: str
    prediction_count: int
    reference_error_count: int
    candidate_error_count: int
    reference_accuracy: float
    candidate_accuracy: float
    fixed_error_count: int
    persistent_error_count: int
    introduced_error_count: int
    fixed_errors: tuple[ResidualErrorExample, ...]
    persistent_errors: tuple[ResidualErrorExample, ...]
    introduced_errors: tuple[ResidualErrorExample, ...]
    family_summaries: tuple[ResidualErrorFamilySummary, ...]


@dataclass(frozen=True)
class ResidualErrorSuiteInput:
    dataset_id: str
    reference_report: BinaryErrorAnalysisReport
    candidate_report: BinaryErrorAnalysisReport


@dataclass(frozen=True)
class DatasetResidualErrorComparison:
    dataset_id: str
    comparison: ResidualErrorComparisonReport


@dataclass(frozen=True)
class ResidualErrorSuiteFeatureSummary:
    reference_feature_key: str
    comparison_count: int
    reference_error_count: int
    candidate_error_count: int
    fixed_error_count: int
    persistent_error_count: int
    introduced_error_count: int
    net_error_delta: int


@dataclass(frozen=True)
class ResidualErrorSuiteReport:
    evaluation_strategy: EvaluationStrategy
    task_name: str
    method_name: BinaryMethodName
    candidate_feature_key: str
    reference_feature_keys: tuple[str, ...]
    dataset_count: int
    comparison_count: int
    feature_summaries: tuple[ResidualErrorSuiteFeatureSummary, ...]
    comparisons: tuple[DatasetResidualErrorComparison, ...]


def _task_by_name(report: BinaryErrorAnalysisReport, task_name: str) -> BinaryTaskErrorAnalysis:
    matches = tuple(task for task in report.tasks if task.task_name == task_name)
    if len(matches) != 1:
        raise BinaryTaskError(f"Expected exactly one task named '{task_name}', found {len(matches)}.")
    return matches[0]


def _method_by_name(task: BinaryTaskErrorAnalysis, method_name: BinaryMethodName) -> BinaryMethodErrorAnalysis:
    matches = tuple(method for method in task.methods if method.method_name == method_name)
    if len(matches) != 1:
        raise BinaryTaskError(f"Expected exactly one method named '{method_name}', found {len(matches)}.")
    return matches[0]


def _predictions_by_example(
    method: BinaryMethodErrorAnalysis,
) -> dict[str, BinaryExamplePrediction]:
    indexed: dict[str, BinaryExamplePrediction] = {}
    for prediction in method.predictions:
        if prediction.example_id in indexed:
            raise BinaryTaskError(
                f"Method '{method.method_name}' has duplicate prediction for example '{prediction.example_id}'."
            )
        indexed[prediction.example_id] = prediction
    return indexed


def _validate_compatible_reports(
    reference_report: BinaryErrorAnalysisReport,
    candidate_report: BinaryErrorAnalysisReport,
) -> None:
    mismatches: list[str] = []
    if reference_report.source_model_id != candidate_report.source_model_id:
        mismatches.append("source_model_id")
    if reference_report.source_revision != candidate_report.source_revision:
        mismatches.append("source_revision")
    if reference_report.evaluation_strategy != candidate_report.evaluation_strategy:
        mismatches.append("evaluation_strategy")
    if reference_report.fold_count != candidate_report.fold_count:
        mismatches.append("fold_count")
    if reference_report.random_seed != candidate_report.random_seed:
        mismatches.append("random_seed")
    if reference_report.regularization_c != candidate_report.regularization_c:
        mismatches.append("regularization_c")
    if reference_report.max_iter != candidate_report.max_iter:
        mismatches.append("max_iter")
    if len(mismatches) > 0:
        raise BinaryTaskError(f"Reports are not comparable; mismatched fields: {', '.join(mismatches)}.")


def _residual_error_example(
    reference_prediction: BinaryExamplePrediction,
    candidate_prediction: BinaryExamplePrediction,
) -> ResidualErrorExample:
    if reference_prediction.true_label != candidate_prediction.true_label:
        raise BinaryTaskError(
            f"Example '{reference_prediction.example_id}' has mismatched true labels: "
            f"reference='{reference_prediction.true_label}', candidate='{candidate_prediction.true_label}'."
        )
    if reference_prediction.family != candidate_prediction.family:
        raise BinaryTaskError(
            f"Example '{reference_prediction.example_id}' has mismatched families: "
            f"reference='{reference_prediction.family}', candidate='{candidate_prediction.family}'."
        )
    return ResidualErrorExample(
        example_id=reference_prediction.example_id,
        family=reference_prediction.family,
        source_label=reference_prediction.source_label,
        true_label=reference_prediction.true_label,
        reference_predicted_label=reference_prediction.predicted_label,
        candidate_predicted_label=candidate_prediction.predicted_label,
        reference_fold_index=reference_prediction.fold_index,
        candidate_fold_index=candidate_prediction.fold_index,
    )


def _sorted_errors(errors: tuple[ResidualErrorExample, ...]) -> tuple[ResidualErrorExample, ...]:
    return tuple(sorted(errors, key=lambda error: (error.family, error.example_id)))


def _family_summaries(
    fixed_errors: tuple[ResidualErrorExample, ...],
    persistent_errors: tuple[ResidualErrorExample, ...],
    introduced_errors: tuple[ResidualErrorExample, ...],
) -> tuple[ResidualErrorFamilySummary, ...]:
    families = sorted(
        {
            error.family
            for error in fixed_errors + persistent_errors + introduced_errors
        }
    )
    return tuple(
        ResidualErrorFamilySummary(
            family=family,
            fixed_error_count=sum(1 for error in fixed_errors if error.family == family),
            persistent_error_count=sum(1 for error in persistent_errors if error.family == family),
            introduced_error_count=sum(1 for error in introduced_errors if error.family == family),
        )
        for family in families
    )


def compare_binary_error_residuals(
    reference_report: BinaryErrorAnalysisReport,
    candidate_report: BinaryErrorAnalysisReport,
    task_name: str,
    method_name: BinaryMethodName,
) -> ResidualErrorComparisonReport:
    _validate_compatible_reports(reference_report, candidate_report)
    reference_task = _task_by_name(reference_report, task_name)
    candidate_task = _task_by_name(candidate_report, task_name)
    reference_method = _method_by_name(reference_task, method_name)
    candidate_method = _method_by_name(candidate_task, method_name)
    reference_predictions = _predictions_by_example(reference_method)
    candidate_predictions = _predictions_by_example(candidate_method)

    if set(reference_predictions.keys()) != set(candidate_predictions.keys()):
        raise BinaryTaskError("Reports are not comparable; prediction example ids differ.")

    fixed_errors: list[ResidualErrorExample] = []
    persistent_errors: list[ResidualErrorExample] = []
    introduced_errors: list[ResidualErrorExample] = []

    for example_id in sorted(reference_predictions.keys()):
        reference_prediction = reference_predictions[example_id]
        candidate_prediction = candidate_predictions[example_id]
        residual = _residual_error_example(reference_prediction, candidate_prediction)

        if not reference_prediction.is_correct and candidate_prediction.is_correct:
            fixed_errors.append(residual)
        elif not reference_prediction.is_correct and not candidate_prediction.is_correct:
            persistent_errors.append(residual)
        elif reference_prediction.is_correct and not candidate_prediction.is_correct:
            introduced_errors.append(residual)

    fixed_error_tuple = _sorted_errors(tuple(fixed_errors))
    persistent_error_tuple = _sorted_errors(tuple(persistent_errors))
    introduced_error_tuple = _sorted_errors(tuple(introduced_errors))

    return ResidualErrorComparisonReport(
        source_model_id=reference_report.source_model_id,
        source_revision=reference_report.source_revision,
        source_selected_device=reference_report.source_selected_device,
        evaluation_strategy=reference_report.evaluation_strategy,
        fold_count=reference_report.fold_count,
        random_seed=reference_report.random_seed,
        regularization_c=reference_report.regularization_c,
        max_iter=reference_report.max_iter,
        task_name=task_name,
        method_name=method_name,
        reference_feature_key=reference_report.activation_feature_key,
        candidate_feature_key=candidate_report.activation_feature_key,
        prediction_count=reference_method.prediction_count,
        reference_error_count=reference_method.error_count,
        candidate_error_count=candidate_method.error_count,
        reference_accuracy=reference_method.accuracy,
        candidate_accuracy=candidate_method.accuracy,
        fixed_error_count=len(fixed_error_tuple),
        persistent_error_count=len(persistent_error_tuple),
        introduced_error_count=len(introduced_error_tuple),
        fixed_errors=fixed_error_tuple,
        persistent_errors=persistent_error_tuple,
        introduced_errors=introduced_error_tuple,
        family_summaries=_family_summaries(fixed_error_tuple, persistent_error_tuple, introduced_error_tuple),
    )


def _ordered_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return tuple(ordered)


def _validate_suite_inputs(inputs: tuple[ResidualErrorSuiteInput, ...]) -> None:
    if len(inputs) == 0:
        raise BinaryTaskError("At least one residual suite input is required.")
    for index, suite_input in enumerate(inputs):
        if suite_input.dataset_id == "":
            raise BinaryTaskError(f"Residual suite input {index} has an empty dataset id.")


def _feature_summary(
    reference_feature_key: str,
    comparisons: tuple[DatasetResidualErrorComparison, ...],
) -> ResidualErrorSuiteFeatureSummary:
    matching_comparisons = tuple(
        item.comparison for item in comparisons if item.comparison.reference_feature_key == reference_feature_key
    )
    if len(matching_comparisons) == 0:
        raise BinaryTaskError(f"Residual suite has no comparisons for reference feature '{reference_feature_key}'.")
    fixed_error_count = sum(comparison.fixed_error_count for comparison in matching_comparisons)
    introduced_error_count = sum(comparison.introduced_error_count for comparison in matching_comparisons)
    return ResidualErrorSuiteFeatureSummary(
        reference_feature_key=reference_feature_key,
        comparison_count=len(matching_comparisons),
        reference_error_count=sum(comparison.reference_error_count for comparison in matching_comparisons),
        candidate_error_count=sum(comparison.candidate_error_count for comparison in matching_comparisons),
        fixed_error_count=fixed_error_count,
        persistent_error_count=sum(comparison.persistent_error_count for comparison in matching_comparisons),
        introduced_error_count=introduced_error_count,
        net_error_delta=introduced_error_count - fixed_error_count,
    )


def compare_binary_error_residual_suite(
    inputs: tuple[ResidualErrorSuiteInput, ...],
    task_name: str,
    method_name: BinaryMethodName,
) -> ResidualErrorSuiteReport:
    _validate_suite_inputs(inputs)
    comparisons = tuple(
        DatasetResidualErrorComparison(
            dataset_id=suite_input.dataset_id,
            comparison=compare_binary_error_residuals(
                reference_report=suite_input.reference_report,
                candidate_report=suite_input.candidate_report,
                task_name=task_name,
                method_name=method_name,
            ),
        )
        for suite_input in inputs
    )
    candidate_feature_keys = _ordered_unique(
        tuple(item.comparison.candidate_feature_key for item in comparisons)
    )
    if len(candidate_feature_keys) != 1:
        raise BinaryTaskError(
            f"Residual suite requires one candidate feature, found: {', '.join(candidate_feature_keys)}."
        )
    comparison_keys = tuple(
        (item.dataset_id, item.comparison.reference_feature_key, item.comparison.candidate_feature_key)
        for item in comparisons
    )
    if len(set(comparison_keys)) != len(comparison_keys):
        raise BinaryTaskError("Residual suite contains duplicate dataset/reference/candidate comparisons.")

    reference_feature_keys = _ordered_unique(tuple(item.comparison.reference_feature_key for item in comparisons))
    return ResidualErrorSuiteReport(
        evaluation_strategy=comparisons[0].comparison.evaluation_strategy,
        task_name=task_name,
        method_name=method_name,
        candidate_feature_key=candidate_feature_keys[0],
        reference_feature_keys=reference_feature_keys,
        dataset_count=len(set(item.dataset_id for item in comparisons)),
        comparison_count=len(comparisons),
        feature_summaries=tuple(_feature_summary(feature_key, comparisons) for feature_key in reference_feature_keys),
        comparisons=comparisons,
    )


def _error_to_json(error: ResidualErrorExample) -> dict[str, JsonValue]:
    return {
        "example_id": error.example_id,
        "family": error.family,
        "source_label": error.source_label,
        "true_label": error.true_label,
        "reference_predicted_label": error.reference_predicted_label,
        "candidate_predicted_label": error.candidate_predicted_label,
        "reference_fold_index": error.reference_fold_index,
        "candidate_fold_index": error.candidate_fold_index,
    }


def _family_summary_to_json(summary: ResidualErrorFamilySummary) -> dict[str, JsonValue]:
    return {
        "family": summary.family,
        "fixed_error_count": summary.fixed_error_count,
        "persistent_error_count": summary.persistent_error_count,
        "introduced_error_count": summary.introduced_error_count,
    }


def residual_error_comparison_report_to_json(report: ResidualErrorComparisonReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "evaluation_strategy": report.evaluation_strategy,
        "fold_count": report.fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "task_name": report.task_name,
        "method_name": report.method_name,
        "reference_feature_key": report.reference_feature_key,
        "candidate_feature_key": report.candidate_feature_key,
        "prediction_count": report.prediction_count,
        "reference_error_count": report.reference_error_count,
        "candidate_error_count": report.candidate_error_count,
        "reference_accuracy": report.reference_accuracy,
        "candidate_accuracy": report.candidate_accuracy,
        "fixed_error_count": report.fixed_error_count,
        "persistent_error_count": report.persistent_error_count,
        "introduced_error_count": report.introduced_error_count,
        "fixed_errors": [_error_to_json(error) for error in report.fixed_errors],
        "persistent_errors": [_error_to_json(error) for error in report.persistent_errors],
        "introduced_errors": [_error_to_json(error) for error in report.introduced_errors],
        "family_summaries": [_family_summary_to_json(summary) for summary in report.family_summaries],
    }


def _suite_feature_summary_to_json(summary: ResidualErrorSuiteFeatureSummary) -> dict[str, JsonValue]:
    return {
        "reference_feature_key": summary.reference_feature_key,
        "comparison_count": summary.comparison_count,
        "reference_error_count": summary.reference_error_count,
        "candidate_error_count": summary.candidate_error_count,
        "fixed_error_count": summary.fixed_error_count,
        "persistent_error_count": summary.persistent_error_count,
        "introduced_error_count": summary.introduced_error_count,
        "net_error_delta": summary.net_error_delta,
    }


def _suite_comparison_to_json(comparison: DatasetResidualErrorComparison) -> dict[str, JsonValue]:
    return {
        "dataset_id": comparison.dataset_id,
        "comparison": residual_error_comparison_report_to_json(comparison.comparison),
    }


def residual_error_suite_report_to_json(report: ResidualErrorSuiteReport) -> dict[str, JsonValue]:
    return {
        "evaluation_strategy": report.evaluation_strategy,
        "task_name": report.task_name,
        "method_name": report.method_name,
        "candidate_feature_key": report.candidate_feature_key,
        "reference_feature_keys": list(report.reference_feature_keys),
        "dataset_count": report.dataset_count,
        "comparison_count": report.comparison_count,
        "feature_summaries": [_suite_feature_summary_to_json(summary) for summary in report.feature_summaries],
        "comparisons": [_suite_comparison_to_json(comparison) for comparison in report.comparisons],
    }


def write_residual_error_comparison_json(path: Path, report: ResidualErrorComparisonReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(residual_error_comparison_report_to_json(report), file, indent=2)
        file.write("\n")


def write_residual_error_suite_json(path: Path, report: ResidualErrorSuiteReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(residual_error_suite_report_to_json(report), file, indent=2)
        file.write("\n")


def _render_error_table(title: str, errors: tuple[ResidualErrorExample, ...]) -> list[str]:
    lines = [
        f"## {title}",
        "",
    ]
    if len(errors) == 0:
        lines.extend(["No examples.", ""])
        return lines

    lines.extend(
        [
            "| Example | Family | True Label | Reference Prediction | Candidate Prediction |",
            "|---|---|---|---|---|",
        ]
    )
    for error in errors:
        lines.append(
            f"| `{error.example_id}` | `{error.family}` | `{error.true_label}` | "
            f"`{error.reference_predicted_label}` | `{error.candidate_predicted_label}` |"
        )
    lines.append("")
    return lines


def render_residual_error_comparison_markdown(report: ResidualErrorComparisonReport) -> str:
    lines = [
        "# Residual Error Comparison",
        "",
        "## Source",
        "",
        f"- Model: `{report.source_model_id}`",
        f"- Revision: `{report.source_revision}`",
        f"- Extraction device: `{report.source_selected_device}`",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Method: `{report.method_name}`",
        f"- Reference feature: `{report.reference_feature_key}`",
        f"- Candidate feature: `{report.candidate_feature_key}`",
        f"- Fold count: `{report.fold_count}`",
        "",
        "## Summary",
        "",
        "| Reference Errors | Candidate Errors | Fixed Errors | Persistent Errors | Introduced Errors |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {report.reference_error_count} | {report.candidate_error_count} | "
            f"{report.fixed_error_count} | {report.persistent_error_count} | "
            f"{report.introduced_error_count} |"
        ),
        "",
        "| Reference Accuracy | Candidate Accuracy | Predictions |",
        "|---:|---:|---:|",
        f"| {report.reference_accuracy:.4f} | {report.candidate_accuracy:.4f} | {report.prediction_count} |",
        "",
        "## Family Summary",
        "",
    ]

    if len(report.family_summaries) == 0:
        lines.extend(["No residual error changes.", ""])
    else:
        lines.extend(
            [
                "| Family | Fixed Errors | Persistent Errors | Introduced Errors |",
                "|---|---:|---:|---:|",
            ]
        )
        for summary in report.family_summaries:
            lines.append(
                f"| `{summary.family}` | {summary.fixed_error_count} | "
                f"{summary.persistent_error_count} | {summary.introduced_error_count} |"
            )
        lines.append("")

    lines.extend(_render_error_table("Fixed Errors", report.fixed_errors))
    lines.extend(_render_error_table("Persistent Errors", report.persistent_errors))
    lines.extend(_render_error_table("Introduced Errors", report.introduced_errors))
    return "\n".join(lines)


def render_residual_error_suite_markdown(report: ResidualErrorSuiteReport) -> str:
    lines = [
        "# Residual Error Suite",
        "",
        "## Source",
        "",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Method: `{report.method_name}`",
        f"- Candidate feature: `{report.candidate_feature_key}`",
        f"- Dataset count: `{report.dataset_count}`",
        f"- Comparison count: `{report.comparison_count}`",
        "",
        "## Aggregate by Reference Feature",
        "",
        "| Reference Feature | Comparisons | Reference Errors | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in report.feature_summaries:
        lines.append(
            f"| `{summary.reference_feature_key}` | {summary.comparison_count} | "
            f"{summary.reference_error_count} | {summary.candidate_error_count} | "
            f"{summary.fixed_error_count} | {summary.persistent_error_count} | "
            f"{summary.introduced_error_count} | {summary.net_error_delta} |"
        )

    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| Dataset | Reference Feature | Reference Errors | Candidate Errors | Fixed | Persistent | Introduced | Reference Accuracy | Candidate Accuracy |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report.comparisons:
        comparison = item.comparison
        lines.append(
            f"| `{item.dataset_id}` | `{comparison.reference_feature_key}` | "
            f"{comparison.reference_error_count} | {comparison.candidate_error_count} | "
            f"{comparison.fixed_error_count} | {comparison.persistent_error_count} | "
            f"{comparison.introduced_error_count} | {comparison.reference_accuracy:.4f} | "
            f"{comparison.candidate_accuracy:.4f} |"
        )

    lines.extend(["", "## Family Deltas", ""])
    for item in report.comparisons:
        comparison = item.comparison
        lines.extend(
            [
                f"### {item.dataset_id} / {comparison.reference_feature_key}",
                "",
            ]
        )
        if len(comparison.family_summaries) == 0:
            lines.extend(["No residual error changes.", ""])
            continue
        lines.extend(
            [
                "| Family | Fixed | Persistent | Introduced |",
                "|---|---:|---:|---:|",
            ]
        )
        for summary in comparison.family_summaries:
            lines.append(
                f"| `{summary.family}` | {summary.fixed_error_count} | "
                f"{summary.persistent_error_count} | {summary.introduced_error_count} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_residual_error_comparison_markdown(path: Path, report: ResidualErrorComparisonReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_residual_error_comparison_markdown(report), encoding="utf-8")


def write_residual_error_suite_markdown(path: Path, report: ResidualErrorSuiteReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_residual_error_suite_markdown(report), encoding="utf-8")
