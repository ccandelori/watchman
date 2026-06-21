from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias, cast

from aegis_introspection.binary_tasks import BinaryMethodName, EvaluationStrategy
from aegis_introspection.error_analysis import (
    BinaryErrorAnalysisReport,
    BinaryExamplePrediction,
    BinaryFamilyErrorSummary,
    BinaryMethodErrorAnalysis,
    BinaryTaskErrorAnalysis,
)
from aegis_introspection.probe import JsonValue
from aegis_introspection.prompts import PromptExample
from aegis_introspection.residual_error_comparison import ResidualErrorComparisonReport, ResidualErrorExample


AdjudicationStatus: TypeAlias = str

_PENDING_HUMAN_REVIEW: AdjudicationStatus = "pending_human_review"
_REVIEW_QUESTIONS: tuple[str, ...] = (
    "Would a careful reviewer keep the current label?",
    "Is the contrast fair, or did prompt wording make the target label ambiguous?",
    "Does this case represent a real model weakness rather than a dataset artifact?",
)
_RESIDUAL_REVIEW_QUESTIONS: tuple[str, ...] = (
    "Is this introduced error a real regression relative to the reference feature?",
    "Would a careful reviewer keep the current label?",
    "Does this case reveal a family-level weakness that should block feature promotion?",
)


class AdjudicationError(ValueError):
    """Raised when an adjudication report cannot be built from the supplied inputs."""


@dataclass(frozen=True)
class ContextPrediction:
    method_name: BinaryMethodName
    predicted_label: str
    is_correct: bool


@dataclass(frozen=True)
class AdjudicationCase:
    fold_index: int
    example_id: str
    family: str
    source_label: str
    true_label: str
    predicted_label: str
    prompt_text: str
    adjudication_status: AdjudicationStatus
    review_questions: tuple[str, ...]
    context_predictions: tuple[ContextPrediction, ...]


@dataclass(frozen=True)
class FamilyAdjudicationSummary:
    family: str
    case_count: int


@dataclass(frozen=True)
class AdjudicationReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    evaluation_strategy: str
    task_name: str
    subject_method_name: BinaryMethodName
    activation_feature_key: str
    case_count: int
    family_summaries: tuple[FamilyAdjudicationSummary, ...]
    cases: tuple[AdjudicationCase, ...]


@dataclass(frozen=True)
class ResidualAdjudicationCase:
    reference_fold_index: int
    candidate_fold_index: int
    example_id: str
    family: str
    source_label: str
    true_label: str
    reference_predicted_label: str
    candidate_predicted_label: str
    prompt_text: str
    adjudication_status: AdjudicationStatus
    review_questions: tuple[str, ...]


@dataclass(frozen=True)
class ResidualAdjudicationReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    evaluation_strategy: str
    task_name: str
    method_name: BinaryMethodName
    reference_feature_key: str
    candidate_feature_key: str
    reference_error_count: int
    candidate_error_count: int
    introduced_error_count: int
    case_count: int
    family_summaries: tuple[FamilyAdjudicationSummary, ...]
    cases: tuple[ResidualAdjudicationCase, ...]


def _task_by_name(report: BinaryErrorAnalysisReport, task_name: str) -> BinaryTaskErrorAnalysis:
    matching_tasks = tuple(task for task in report.tasks if task.task_name == task_name)
    if len(matching_tasks) != 1:
        raise AdjudicationError(f"Expected exactly one task named '{task_name}', found {len(matching_tasks)}.")
    return matching_tasks[0]


def _method_by_name(task: BinaryTaskErrorAnalysis, method_name: BinaryMethodName) -> BinaryMethodErrorAnalysis:
    matching_methods = tuple(method for method in task.methods if method.method_name == method_name)
    if len(matching_methods) != 1:
        raise AdjudicationError(
            f"Expected exactly one method named '{method_name}' for task '{task.task_name}', "
            f"found {len(matching_methods)}."
        )
    return matching_methods[0]


def _examples_by_id(examples: tuple[PromptExample, ...]) -> dict[str, PromptExample]:
    indexed: dict[str, PromptExample] = {}
    for example in examples:
        if example.id in indexed:
            raise AdjudicationError(f"Duplicate prompt example id '{example.id}'.")
        indexed[example.id] = example
    return indexed


def _predictions_by_example(
    method: BinaryMethodErrorAnalysis,
) -> dict[str, BinaryExamplePrediction]:
    indexed: dict[str, BinaryExamplePrediction] = {}
    for prediction in method.predictions:
        if prediction.example_id in indexed:
            raise AdjudicationError(
                f"Method '{method.method_name}' has duplicate prediction for example '{prediction.example_id}'."
            )
        indexed[prediction.example_id] = prediction
    return indexed


def _context_predictions(
    example_id: str,
    context_methods: tuple[BinaryMethodErrorAnalysis, ...],
) -> tuple[ContextPrediction, ...]:
    rows: list[ContextPrediction] = []
    for method in context_methods:
        predictions = _predictions_by_example(method)
        prediction = predictions.get(example_id)
        if prediction is None:
            raise AdjudicationError(
                f"Method '{method.method_name}' is missing prediction for example '{example_id}'."
            )
        rows.append(
            ContextPrediction(
                method_name=method.method_name,
                predicted_label=prediction.predicted_label,
                is_correct=prediction.is_correct,
            )
        )
    return tuple(rows)


def _family_summaries(
    cases: tuple[AdjudicationCase | ResidualAdjudicationCase, ...],
) -> tuple[FamilyAdjudicationSummary, ...]:
    family_counts: dict[str, int] = {}
    for case in cases:
        family_counts[case.family] = family_counts.get(case.family, 0) + 1
    return tuple(
        FamilyAdjudicationSummary(
            family=family,
            case_count=count,
        )
        for family, count in sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))
    )


def build_adjudication_report(
    error_report: BinaryErrorAnalysisReport,
    examples: tuple[PromptExample, ...],
    task_name: str,
    subject_method_name: BinaryMethodName,
    context_method_names: tuple[BinaryMethodName, ...],
) -> AdjudicationReport:
    task = _task_by_name(error_report, task_name)
    subject_method = _method_by_name(task, subject_method_name)
    context_methods = tuple(_method_by_name(task, method_name) for method_name in context_method_names)
    prompts_by_id = _examples_by_id(examples)

    cases: list[AdjudicationCase] = []
    for prediction in subject_method.predictions:
        if prediction.is_correct:
            continue
        example = prompts_by_id.get(prediction.example_id)
        if example is None:
            raise AdjudicationError(f"Missing prompt text for example '{prediction.example_id}'.")
        cases.append(
            AdjudicationCase(
                fold_index=prediction.fold_index,
                example_id=prediction.example_id,
                family=prediction.family,
                source_label=prediction.source_label,
                true_label=prediction.true_label,
                predicted_label=prediction.predicted_label,
                prompt_text=example.text,
                adjudication_status=_PENDING_HUMAN_REVIEW,
                review_questions=_REVIEW_QUESTIONS,
                context_predictions=_context_predictions(prediction.example_id, context_methods),
            )
        )

    sorted_cases = tuple(sorted(cases, key=lambda item: (item.family, item.example_id)))
    return AdjudicationReport(
        source_model_id=error_report.source_model_id,
        source_revision=error_report.source_revision,
        source_selected_device=error_report.source_selected_device,
        evaluation_strategy=error_report.evaluation_strategy,
        task_name=task.task_name,
        subject_method_name=subject_method.method_name,
        activation_feature_key=error_report.activation_feature_key,
        case_count=len(sorted_cases),
        family_summaries=_family_summaries(sorted_cases),
        cases=sorted_cases,
    )


def _residual_adjudication_case(
    residual_error: ResidualErrorExample,
    example: PromptExample,
) -> ResidualAdjudicationCase:
    return ResidualAdjudicationCase(
        reference_fold_index=residual_error.reference_fold_index,
        candidate_fold_index=residual_error.candidate_fold_index,
        example_id=residual_error.example_id,
        family=residual_error.family,
        source_label=residual_error.source_label,
        true_label=residual_error.true_label,
        reference_predicted_label=residual_error.reference_predicted_label,
        candidate_predicted_label=residual_error.candidate_predicted_label,
        prompt_text=example.text,
        adjudication_status=_PENDING_HUMAN_REVIEW,
        review_questions=_RESIDUAL_REVIEW_QUESTIONS,
    )


def build_residual_adjudication_report(
    residual_report: ResidualErrorComparisonReport,
    examples: tuple[PromptExample, ...],
) -> ResidualAdjudicationReport:
    prompts_by_id = _examples_by_id(examples)
    cases: list[ResidualAdjudicationCase] = []
    for residual_error in residual_report.introduced_errors:
        example = prompts_by_id.get(residual_error.example_id)
        if example is None:
            raise AdjudicationError(f"Missing prompt text for example '{residual_error.example_id}'.")
        cases.append(_residual_adjudication_case(residual_error, example))

    sorted_cases = tuple(sorted(cases, key=lambda item: (item.family, item.example_id)))
    return ResidualAdjudicationReport(
        source_model_id=residual_report.source_model_id,
        source_revision=residual_report.source_revision,
        source_selected_device=residual_report.source_selected_device,
        evaluation_strategy=residual_report.evaluation_strategy,
        task_name=residual_report.task_name,
        method_name=residual_report.method_name,
        reference_feature_key=residual_report.reference_feature_key,
        candidate_feature_key=residual_report.candidate_feature_key,
        reference_error_count=residual_report.reference_error_count,
        candidate_error_count=residual_report.candidate_error_count,
        introduced_error_count=residual_report.introduced_error_count,
        case_count=len(sorted_cases),
        family_summaries=_family_summaries(sorted_cases),
        cases=sorted_cases,
    )


def _as_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise AdjudicationError(f"Expected {description} to be an object.")
    return cast(Mapping[str, object], value)


def _required_string(record: Mapping[str, object], field_name: str, description: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise AdjudicationError(f"Expected {description} field '{field_name}' to be a string.")
    if value == "":
        raise AdjudicationError(f"Expected {description} field '{field_name}' to be non-empty.")
    return value


def _required_bool(record: Mapping[str, object], field_name: str, description: str) -> bool:
    value = record.get(field_name)
    if not isinstance(value, bool):
        raise AdjudicationError(f"Expected {description} field '{field_name}' to be a boolean.")
    return value


def _required_int(record: Mapping[str, object], field_name: str, description: str) -> int:
    value = record.get(field_name)
    if not isinstance(value, int):
        raise AdjudicationError(f"Expected {description} field '{field_name}' to be an integer.")
    return value


def _required_float(record: Mapping[str, object], field_name: str, description: str) -> float:
    value = record.get(field_name)
    if not isinstance(value, int | float):
        raise AdjudicationError(f"Expected {description} field '{field_name}' to be numeric.")
    return float(value)


def _required_list(record: Mapping[str, object], field_name: str, description: str) -> list[object]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise AdjudicationError(f"Expected {description} field '{field_name}' to be a list.")
    return value


def _prediction_from_json(value: object, index: int) -> BinaryExamplePrediction:
    description = f"prediction {index}"
    record = _as_mapping(value, description)
    return BinaryExamplePrediction(
        fold_index=_required_int(record, "fold_index", description),
        example_id=_required_string(record, "example_id", description),
        family=_required_string(record, "family", description),
        source_label=_required_string(record, "source_label", description),
        true_label=_required_string(record, "true_label", description),
        predicted_label=_required_string(record, "predicted_label", description),
        is_correct=_required_bool(record, "is_correct", description),
    )


def _predicted_label_counts_from_json(value: object, index: int) -> tuple[str, int]:
    description = f"predicted label count {index}"
    record = _as_mapping(value, description)
    label = _required_string(record, "label", description)
    count = _required_int(record, "count", description)
    return (label, count)


def _family_summary_from_json(value: object, index: int) -> BinaryFamilyErrorSummary:
    description = f"family summary {index}"
    record = _as_mapping(value, description)
    return BinaryFamilyErrorSummary(
        family=_required_string(record, "family", description),
        true_label=_required_string(record, "true_label", description),
        example_count=_required_int(record, "example_count", description),
        correct_count=_required_int(record, "correct_count", description),
        error_count=_required_int(record, "error_count", description),
        accuracy=_required_float(record, "accuracy", description),
        predicted_label_counts=tuple(
            _predicted_label_counts_from_json(item, item_index)
            for item_index, item in enumerate(_required_list(record, "predicted_label_counts", description))
        ),
    )


def _method_from_json(value: object, index: int) -> BinaryMethodErrorAnalysis:
    description = f"method {index}"
    record = _as_mapping(value, description)
    return BinaryMethodErrorAnalysis(
        method_name=cast(BinaryMethodName, _required_string(record, "method_name", description)),
        feature_name=_required_string(record, "feature_name", description),
        label_names=tuple(
            _required_string({"label": item}, "label", f"{description} label {item_index}")
            for item_index, item in enumerate(_required_list(record, "label_names", description))
        ),
        prediction_count=_required_int(record, "prediction_count", description),
        correct_count=_required_int(record, "correct_count", description),
        error_count=_required_int(record, "error_count", description),
        accuracy=_required_float(record, "accuracy", description),
        family_summaries=tuple(
            _family_summary_from_json(item, item_index)
            for item_index, item in enumerate(_required_list(record, "family_summaries", description))
        ),
        predictions=tuple(
            _prediction_from_json(item, item_index)
            for item_index, item in enumerate(_required_list(record, "predictions", description))
        ),
    )


def _task_from_json(value: object, index: int) -> BinaryTaskErrorAnalysis:
    description = f"task {index}"
    record = _as_mapping(value, description)
    return BinaryTaskErrorAnalysis(
        task_name=_required_string(record, "task_name", description),
        description=_required_string(record, "description", description),
        label_names=tuple(
            _required_string({"label": item}, "label", f"{description} label {item_index}")
            for item_index, item in enumerate(_required_list(record, "label_names", description))
        ),
        methods=tuple(
            _method_from_json(item, item_index)
            for item_index, item in enumerate(_required_list(record, "methods", description))
        ),
    )


def load_binary_error_analysis_report_json(path: Path) -> BinaryErrorAnalysisReport:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdjudicationError(f"Invalid error-analysis JSON in {path}: {exc.msg}.") from exc
    record = _as_mapping(decoded, "binary error analysis report")
    return BinaryErrorAnalysisReport(
        source_model_id=_required_string(record, "source_model_id", "binary error analysis report"),
        source_revision=_required_string(record, "source_revision", "binary error analysis report"),
        source_selected_device=_required_string(record, "source_selected_device", "binary error analysis report"),
        evaluation_strategy=cast(
            EvaluationStrategy,
            _required_string(record, "evaluation_strategy", "binary error analysis report"),
        ),
        fold_count=_required_int(record, "fold_count", "binary error analysis report"),
        random_seed=_required_int(record, "random_seed", "binary error analysis report"),
        regularization_c=_required_float(record, "regularization_c", "binary error analysis report"),
        max_iter=_required_int(record, "max_iter", "binary error analysis report"),
        activation_feature_key=_required_string(record, "activation_feature_key", "binary error analysis report"),
        tasks=tuple(
            _task_from_json(item, item_index)
            for item_index, item in enumerate(_required_list(record, "tasks", "binary error analysis report"))
        ),
    )


def _context_prediction_to_json(prediction: ContextPrediction) -> dict[str, JsonValue]:
    return {
        "method_name": prediction.method_name,
        "predicted_label": prediction.predicted_label,
        "is_correct": prediction.is_correct,
    }


def _case_to_json(case: AdjudicationCase) -> dict[str, JsonValue]:
    return {
        "fold_index": case.fold_index,
        "example_id": case.example_id,
        "family": case.family,
        "source_label": case.source_label,
        "true_label": case.true_label,
        "predicted_label": case.predicted_label,
        "prompt_text": case.prompt_text,
        "adjudication_status": case.adjudication_status,
        "review_questions": list(case.review_questions),
        "context_predictions": [_context_prediction_to_json(prediction) for prediction in case.context_predictions],
    }


def _residual_case_to_json(case: ResidualAdjudicationCase) -> dict[str, JsonValue]:
    return {
        "reference_fold_index": case.reference_fold_index,
        "candidate_fold_index": case.candidate_fold_index,
        "example_id": case.example_id,
        "family": case.family,
        "source_label": case.source_label,
        "true_label": case.true_label,
        "reference_predicted_label": case.reference_predicted_label,
        "candidate_predicted_label": case.candidate_predicted_label,
        "prompt_text": case.prompt_text,
        "adjudication_status": case.adjudication_status,
        "review_questions": list(case.review_questions),
    }


def _family_adjudication_summary_to_json(summary: FamilyAdjudicationSummary) -> dict[str, JsonValue]:
    return {
        "family": summary.family,
        "case_count": summary.case_count,
    }


def adjudication_report_to_json(report: AdjudicationReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "evaluation_strategy": report.evaluation_strategy,
        "task_name": report.task_name,
        "subject_method_name": report.subject_method_name,
        "activation_feature_key": report.activation_feature_key,
        "case_count": report.case_count,
        "family_summaries": [
            _family_adjudication_summary_to_json(summary)
            for summary in report.family_summaries
        ],
        "cases": [_case_to_json(case) for case in report.cases],
    }


def residual_adjudication_report_to_json(report: ResidualAdjudicationReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "evaluation_strategy": report.evaluation_strategy,
        "task_name": report.task_name,
        "method_name": report.method_name,
        "reference_feature_key": report.reference_feature_key,
        "candidate_feature_key": report.candidate_feature_key,
        "reference_error_count": report.reference_error_count,
        "candidate_error_count": report.candidate_error_count,
        "introduced_error_count": report.introduced_error_count,
        "case_count": report.case_count,
        "family_summaries": [
            _family_adjudication_summary_to_json(summary)
            for summary in report.family_summaries
        ],
        "cases": [_residual_case_to_json(case) for case in report.cases],
    }


def write_adjudication_json(path: Path, report: AdjudicationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(adjudication_report_to_json(report), file, indent=2)
        file.write("\n")


def write_residual_adjudication_json(path: Path, report: ResidualAdjudicationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(residual_adjudication_report_to_json(report), file, indent=2)
        file.write("\n")


def _status_text(status: AdjudicationStatus) -> str:
    if status == _PENDING_HUMAN_REVIEW:
        return "Pending human review"
    return status


def _context_prediction_text(predictions: tuple[ContextPrediction, ...]) -> str:
    return ", ".join(
        f"{prediction.method_name}: {prediction.predicted_label} ({'correct' if prediction.is_correct else 'wrong'})"
        for prediction in predictions
    )


def render_adjudication_markdown(report: AdjudicationReport) -> str:
    lines = [
        "# Error Adjudication",
        "",
        "## Source",
        "",
        f"- Model: `{report.source_model_id}`",
        f"- Revision: `{report.source_revision}`",
        f"- Extraction device: `{report.source_selected_device}`",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Subject method: `{report.subject_method_name}`",
        f"- Activation feature: `{report.activation_feature_key}`",
        f"- Cases requiring review: `{report.case_count}`",
        "",
        "## Family Summary",
        "",
        "| Family | Cases |",
        "|---|---:|",
    ]
    for summary in report.family_summaries:
        lines.append(f"| `{summary.family}` | {summary.case_count} |")

    lines.extend(
        [
            "",
            "## Cases",
            "",
        ]
    )

    for index, case in enumerate(report.cases, start=1):
        lines.extend(
            [
                f"### Case {index}: {case.example_id}",
                "",
                f"- Family: `{case.family}`",
                f"- Fold: `{case.fold_index}`",
                f"- True label: `{case.true_label}`",
                f"- Predicted label: `{case.predicted_label}`",
                f"- Status: {_status_text(case.adjudication_status)}",
                f"- Context predictions: `{_context_prediction_text(case.context_predictions)}`",
                "",
                "Prompt:",
                "",
                f"> {case.prompt_text}",
                "",
                "Review questions:",
                "",
            ]
        )
        for question in case.review_questions:
            lines.append(f"- {question}")
        lines.append("")

    return "\n".join(lines)


def render_residual_adjudication_markdown(report: ResidualAdjudicationReport) -> str:
    lines = [
        "# Residual Error Adjudication",
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
        f"- Reference errors: `{report.reference_error_count}`",
        f"- Candidate errors: `{report.candidate_error_count}`",
        f"- Introduced errors: `{report.introduced_error_count}`",
        f"- Cases requiring review: `{report.case_count}`",
        "",
        "## Family Summary",
        "",
        "| Family | Cases |",
        "|---|---:|",
    ]
    for summary in report.family_summaries:
        lines.append(f"| `{summary.family}` | {summary.case_count} |")

    lines.extend(["", "## Cases", ""])
    for index, case in enumerate(report.cases, start=1):
        lines.extend(
            [
                f"### Case {index}: {case.example_id}",
                "",
                f"- Family: `{case.family}`",
                f"- Reference fold: `{case.reference_fold_index}`",
                f"- Candidate fold: `{case.candidate_fold_index}`",
                f"- True label: `{case.true_label}`",
                f"- Reference prediction: `{case.reference_predicted_label}`",
                f"- Candidate prediction: `{case.candidate_predicted_label}`",
                f"- Status: {_status_text(case.adjudication_status)}",
                "",
                "Prompt:",
                "",
                f"> {case.prompt_text}",
                "",
                "Review questions:",
                "",
            ]
        )
        for question in case.review_questions:
            lines.append(f"- {question}")
        lines.append("")

    return "\n".join(lines)


def write_adjudication_markdown(path: Path, report: AdjudicationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_adjudication_markdown(report), encoding="utf-8")


def write_residual_adjudication_markdown(path: Path, report: ResidualAdjudicationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_residual_adjudication_markdown(report), encoding="utf-8")
