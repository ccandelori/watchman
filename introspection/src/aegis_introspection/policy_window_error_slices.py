from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, TypeAlias, cast

from aegis_introspection.binary_tasks import BinaryMethodName
from aegis_introspection.error_analysis import (
    BinaryErrorAnalysisReport,
    BinaryExamplePrediction,
    BinaryMethodErrorAnalysis,
)
from aegis_introspection.probe import JsonValue


PayloadCondition: TypeAlias = Literal["payload", "no_payload"]
SelectedField: TypeAlias = Literal["credential_value", "summary_value"]
SelectedMode: TypeAlias = Literal["mode_a", "mode_b"]
SelectedAction: TypeAlias = Literal["copy", "mask"]
ErrorSliceDimension: TypeAlias = Literal[
    "family",
    "source_label",
    "credential_type",
    "payload_condition",
    "selected_field",
    "selected_mode",
    "selected_action",
]


class PolicyWindowErrorSliceError(ValueError):
    """Raised when policy-window error slices cannot be derived."""


@dataclass(frozen=True)
class PromptPolicyMetadata:
    example_id: str
    family: str
    credential_type: str
    payload_condition: PayloadCondition
    selected_field: SelectedField
    selected_mode: SelectedMode
    selected_action: SelectedAction


@dataclass(frozen=True)
class ErrorSliceSummary:
    dimension: ErrorSliceDimension
    value: str
    true_label: str
    example_count: int
    correct_count: int
    error_count: int
    accuracy: float
    predicted_label_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PolicyWindowErrorSliceReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    evaluation_strategy: str
    fold_count: int
    activation_feature_key: str
    task_name: str
    method_name: BinaryMethodName
    feature_name: str
    dimensions: tuple[ErrorSliceDimension, ...]
    summaries: tuple[ErrorSliceSummary, ...]


def load_prompt_policy_metadata(path: Path) -> dict[str, PromptPolicyMetadata]:
    metadata_by_id: dict[str, PromptPolicyMetadata] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            decoded = json.loads(line)
            record = _as_mapping(decoded, f"line {line_number}")
            metadata = _prompt_policy_metadata(record=record, line_number=line_number)
            if metadata.example_id in metadata_by_id:
                raise PolicyWindowErrorSliceError(f"Line {line_number}: duplicate example id '{metadata.example_id}'.")
            metadata_by_id[metadata.example_id] = metadata
    if len(metadata_by_id) == 0:
        raise PolicyWindowErrorSliceError(f"No prompt metadata found in {path}.")
    return metadata_by_id


def build_policy_window_error_slice_report(
    report: BinaryErrorAnalysisReport,
    metadata_by_id: Mapping[str, PromptPolicyMetadata],
    task_name: str,
    method_name: BinaryMethodName,
    dimensions: tuple[ErrorSliceDimension, ...],
) -> PolicyWindowErrorSliceReport:
    method = _method_for_task(report=report, task_name=task_name, method_name=method_name)
    summaries = _summaries_for_dimensions(
        predictions=method.predictions,
        metadata_by_id=metadata_by_id,
        dimensions=dimensions,
    )
    return PolicyWindowErrorSliceReport(
        source_model_id=report.source_model_id,
        source_revision=report.source_revision,
        source_selected_device=report.source_selected_device,
        evaluation_strategy=report.evaluation_strategy,
        fold_count=report.fold_count,
        activation_feature_key=report.activation_feature_key,
        task_name=task_name,
        method_name=method_name,
        feature_name=method.feature_name,
        dimensions=dimensions,
        summaries=summaries,
    )


def policy_window_error_slice_report_to_json(report: PolicyWindowErrorSliceReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "evaluation_strategy": report.evaluation_strategy,
        "fold_count": report.fold_count,
        "activation_feature_key": report.activation_feature_key,
        "task_name": report.task_name,
        "method_name": report.method_name,
        "feature_name": report.feature_name,
        "dimensions": list(report.dimensions),
        "summaries": [_summary_to_json(summary) for summary in report.summaries],
    }


def write_policy_window_error_slice_json(path: Path, report: PolicyWindowErrorSliceReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(policy_window_error_slice_report_to_json(report), file, indent=2)
        file.write("\n")


def render_policy_window_error_slice_markdown(report: PolicyWindowErrorSliceReport) -> str:
    lines = [
        "# Policy-Window Error Slices",
        "",
        "## Source",
        "",
        f"- Model: `{report.source_model_id}`",
        f"- Revision: `{report.source_revision}`",
        f"- Extraction device: `{report.source_selected_device}`",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Activation feature: `{report.activation_feature_key}`",
        f"- Fold count: `{report.fold_count}`",
        f"- Task: `{report.task_name}`",
        f"- Method: `{report.method_name}`",
        "",
        "## Error Slices",
        "",
    ]
    summaries_with_errors = tuple(summary for summary in report.summaries if summary.error_count > 0)
    if len(summaries_with_errors) == 0:
        lines.append("No metadata-slice errors.")
        return "\n".join(lines) + "\n"
    lines.extend(
        (
            "| Dimension | Value | True Label | Examples | Errors | Accuracy | Predicted Labels |",
            "|---|---|---|---:|---:|---:|---|",
        )
    )
    for summary in summaries_with_errors:
        predicted_label_counts = ", ".join(f"{label}={count}" for label, count in summary.predicted_label_counts)
        lines.append(
            f"| `{summary.dimension}` | `{summary.value}` | `{summary.true_label}` | "
            f"{summary.example_count} | {summary.error_count} | {summary.accuracy:.4f} | "
            f"`{predicted_label_counts}` |"
        )
    return "\n".join(lines) + "\n"


def write_policy_window_error_slice_markdown(path: Path, report: PolicyWindowErrorSliceReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_policy_window_error_slice_markdown(report), encoding="utf-8")


def _prompt_policy_metadata(record: Mapping[str, object], line_number: int) -> PromptPolicyMetadata:
    return PromptPolicyMetadata(
        example_id=_required_string(record=record, field_name="example_id", line_number=line_number),
        family=_required_string(record=record, field_name="family", line_number=line_number),
        credential_type=_required_string(record=record, field_name="credential_type", line_number=line_number),
        payload_condition=_payload_condition(record=record),
        selected_field=_selected_field(
            record=record,
            field_name="policy_window_selected_field",
            line_number=line_number,
        ),
        selected_mode=_selected_mode(
            record=record,
            field_name="policy_window_selected_mode",
            line_number=line_number,
        ),
        selected_action=_selected_action(
            record=record,
            field_name="policy_window_selected_action",
            line_number=line_number,
        ),
    )


def _summaries_for_dimensions(
    predictions: tuple[BinaryExamplePrediction, ...],
    metadata_by_id: Mapping[str, PromptPolicyMetadata],
    dimensions: tuple[ErrorSliceDimension, ...],
) -> tuple[ErrorSliceSummary, ...]:
    grouped_predictions: dict[tuple[ErrorSliceDimension, str, str], list[BinaryExamplePrediction]] = {}
    for prediction in predictions:
        metadata = metadata_by_id.get(prediction.example_id)
        if metadata is None:
            raise PolicyWindowErrorSliceError(f"Missing prompt metadata for example '{prediction.example_id}'.")
        for dimension in dimensions:
            key = (dimension, _dimension_value(metadata=metadata, prediction=prediction, dimension=dimension), prediction.true_label)
            grouped_predictions.setdefault(key, []).append(prediction)

    summaries = tuple(
        _summary_from_predictions(
            dimension=dimension,
            value=value,
            true_label=true_label,
            predictions=tuple(grouped),
        )
        for (dimension, value, true_label), grouped in grouped_predictions.items()
    )
    return tuple(
        sorted(
            summaries,
            key=lambda summary: (
                summary.dimension,
                -summary.error_count,
                summary.value,
                summary.true_label,
            ),
        )
    )


def _summary_from_predictions(
    dimension: ErrorSliceDimension,
    value: str,
    true_label: str,
    predictions: tuple[BinaryExamplePrediction, ...],
) -> ErrorSliceSummary:
    correct_count = sum(1 for prediction in predictions if prediction.is_correct)
    example_count = len(predictions)
    predicted_label_counts = tuple(
        (label, sum(1 for prediction in predictions if prediction.predicted_label == label))
        for label in sorted({prediction.predicted_label for prediction in predictions})
    )
    return ErrorSliceSummary(
        dimension=dimension,
        value=value,
        true_label=true_label,
        example_count=example_count,
        correct_count=correct_count,
        error_count=example_count - correct_count,
        accuracy=float(correct_count / example_count),
        predicted_label_counts=predicted_label_counts,
    )


def _method_for_task(
    report: BinaryErrorAnalysisReport,
    task_name: str,
    method_name: BinaryMethodName,
) -> BinaryMethodErrorAnalysis:
    for task in report.tasks:
        if task.task_name != task_name:
            continue
        for method in task.methods:
            if method.method_name == method_name:
                return method
        raise PolicyWindowErrorSliceError(f"Task '{task_name}' does not contain method '{method_name}'.")
    raise PolicyWindowErrorSliceError(f"Error analysis report does not contain task '{task_name}'.")


def _dimension_value(
    metadata: PromptPolicyMetadata,
    prediction: BinaryExamplePrediction,
    dimension: ErrorSliceDimension,
) -> str:
    if dimension == "family":
        return metadata.family
    if dimension == "source_label":
        return prediction.source_label
    if dimension == "credential_type":
        return metadata.credential_type
    if dimension == "payload_condition":
        return metadata.payload_condition
    if dimension == "selected_field":
        return metadata.selected_field
    if dimension == "selected_mode":
        return metadata.selected_mode
    if dimension == "selected_action":
        return metadata.selected_action
    raise PolicyWindowErrorSliceError(f"Unsupported error-slice dimension '{dimension}'.")


def _payload_condition(record: Mapping[str, object]) -> PayloadCondition:
    if record.get("payload_token_span") is None:
        return "no_payload"
    return "payload"


def _required_string(record: Mapping[str, object], field_name: str, line_number: int) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise PolicyWindowErrorSliceError(f"Line {line_number}: field '{field_name}' must be a string.")
    if value == "":
        raise PolicyWindowErrorSliceError(f"Line {line_number}: field '{field_name}' must not be empty.")
    return value


def _selected_field(record: Mapping[str, object], field_name: str, line_number: int) -> SelectedField:
    value = _required_string(record=record, field_name=field_name, line_number=line_number)
    if value not in ("credential_value", "summary_value"):
        raise PolicyWindowErrorSliceError(f"Line {line_number}: field '{field_name}' has invalid value '{value}'.")
    return cast(SelectedField, value)


def _selected_mode(record: Mapping[str, object], field_name: str, line_number: int) -> SelectedMode:
    value = _required_string(record=record, field_name=field_name, line_number=line_number)
    if value not in ("mode_a", "mode_b"):
        raise PolicyWindowErrorSliceError(f"Line {line_number}: field '{field_name}' has invalid value '{value}'.")
    return cast(SelectedMode, value)


def _selected_action(record: Mapping[str, object], field_name: str, line_number: int) -> SelectedAction:
    value = _required_string(record=record, field_name=field_name, line_number=line_number)
    if value not in ("copy", "mask"):
        raise PolicyWindowErrorSliceError(f"Line {line_number}: field '{field_name}' has invalid value '{value}'.")
    return cast(SelectedAction, value)


def _as_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise PolicyWindowErrorSliceError(f"{description}: expected a JSON object.")
    return cast(Mapping[str, object], value)


def _summary_to_json(summary: ErrorSliceSummary) -> dict[str, JsonValue]:
    return {
        "dimension": summary.dimension,
        "value": summary.value,
        "true_label": summary.true_label,
        "example_count": summary.example_count,
        "correct_count": summary.correct_count,
        "error_count": summary.error_count,
        "accuracy": summary.accuracy,
        "predicted_label_counts": [
            {"label": label, "count": count} for label, count in summary.predicted_label_counts
        ],
    }
