from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, TypeAlias

from aegis_introspection.cift_calibration import CiftCalibrationReport, CalibratedCiftPrediction
from aegis_introspection.cift_operating_points import CiftOperatingPoint, _operating_point
from aegis_introspection.policy_window_error_slices import (
    ErrorSliceDimension,
    PromptPolicyMetadata,
)
from aegis_introspection.probe import JsonValue


SelectorErrorKind: TypeAlias = Literal["false_positive", "false_negative"]


class SelectorDiagnosticError(ValueError):
    """Raised when selector-window score diagnostics cannot be computed."""


@dataclass(frozen=True)
class SelectorDiagnosticConfig:
    dimensions: tuple[ErrorSliceDimension, ...]
    thresholds: tuple[float, ...]
    near_threshold_radius: float


@dataclass(frozen=True)
class SelectorSliceSummary:
    dimension: ErrorSliceDimension
    value: str
    example_count: int
    positive_count: int
    negative_count: int
    predicted_positive_count: int
    correct_count: int
    error_count: int
    false_positive_count: int
    false_negative_count: int
    near_threshold_count: int
    near_threshold_error_count: int
    confident_error_count: int
    accuracy: float
    empirical_positive_rate: float
    mean_probability: float
    calibration_gap: float
    absolute_calibration_gap: float
    mean_absolute_margin: float


@dataclass(frozen=True)
class SelectorErrorExample:
    example_id: str
    family: str
    source_label: str
    true_label: str
    predicted_label: str
    error_kind: SelectorErrorKind
    positive_probability: float
    signed_margin: float
    absolute_margin: float
    payload_condition: str
    selected_mode: str
    selected_field: str
    selected_action: str
    credential_type: str
    fold_index: int


@dataclass(frozen=True)
class SelectorDiagnosticReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    source_evaluation_strategy: str
    score_semantics: str
    task_name: str
    positive_label: str
    activation_feature_key: str
    decision_threshold: float
    near_threshold_radius: float
    dimensions: tuple[ErrorSliceDimension, ...]
    threshold_summaries: tuple[CiftOperatingPoint, ...]
    slice_summaries: tuple[SelectorSliceSummary, ...]
    error_examples: tuple[SelectorErrorExample, ...]


@dataclass(frozen=True)
class _JoinedPrediction:
    prediction: CalibratedCiftPrediction
    metadata: PromptPolicyMetadata


def build_selector_diagnostic_report(
    report: CiftCalibrationReport,
    metadata_by_id: Mapping[str, PromptPolicyMetadata],
    config: SelectorDiagnosticConfig,
) -> SelectorDiagnosticReport:
    _validate_config(config)
    joined_predictions = _join_predictions(report=report, metadata_by_id=metadata_by_id)
    return SelectorDiagnosticReport(
        source_model_id=report.source_model_id,
        source_revision=report.source_revision,
        source_selected_device=report.source_selected_device,
        source_evaluation_strategy=report.evaluation_strategy,
        score_semantics=report.score_semantics,
        task_name=report.task_name,
        positive_label=report.positive_label,
        activation_feature_key=report.activation_feature_key,
        decision_threshold=report.decision_threshold,
        near_threshold_radius=config.near_threshold_radius,
        dimensions=config.dimensions,
        threshold_summaries=tuple(_operating_point(report=report, threshold=threshold) for threshold in config.thresholds),
        slice_summaries=_slice_summaries(
            report=report,
            joined_predictions=joined_predictions,
            dimensions=config.dimensions,
            near_threshold_radius=config.near_threshold_radius,
        ),
        error_examples=_error_examples(
            report=report,
            joined_predictions=joined_predictions,
            near_threshold_radius=config.near_threshold_radius,
        ),
    )


def selector_diagnostic_report_to_json(report: SelectorDiagnosticReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "source_evaluation_strategy": report.source_evaluation_strategy,
        "score_semantics": report.score_semantics,
        "task_name": report.task_name,
        "positive_label": report.positive_label,
        "activation_feature_key": report.activation_feature_key,
        "decision_threshold": report.decision_threshold,
        "near_threshold_radius": report.near_threshold_radius,
        "dimensions": list(report.dimensions),
        "threshold_summaries": [_threshold_summary_to_json(summary) for summary in report.threshold_summaries],
        "slice_summaries": [_slice_summary_to_json(summary) for summary in report.slice_summaries],
        "error_examples": [_error_example_to_json(example) for example in report.error_examples],
    }


def write_selector_diagnostic_json(path: Path, report: SelectorDiagnosticReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(selector_diagnostic_report_to_json(report), file, indent=2)
        file.write("\n")


def render_selector_diagnostic_markdown(report: SelectorDiagnosticReport) -> str:
    lines = [
        "# Selector-Window Score Diagnostics",
        "",
        "## Source",
        "",
        f"- Model: `{report.source_model_id}`",
        f"- Revision: `{report.source_revision}`",
        f"- Extraction device: `{report.source_selected_device}`",
        f"- Evaluation strategy: `{report.source_evaluation_strategy}`",
        f"- Score semantics: `{report.score_semantics}`",
        f"- Task: `{report.task_name}`",
        f"- Positive label: `{report.positive_label}`",
        f"- Activation feature: `{report.activation_feature_key}`",
        f"- Decision threshold: `{report.decision_threshold:.4f}`",
        f"- Near-threshold radius: `{report.near_threshold_radius:.4f}`",
        "",
        "## Threshold Sweep",
        "",
        "| Threshold | TP | FP | TN | FN | Precision | Recall | FPR | Accuracy | Macro F1 | Warn | Allow |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in report.threshold_summaries:
        lines.append(
            f"| {summary.threshold:.4f} | {summary.true_positive} | {summary.false_positive} | "
            f"{summary.true_negative} | {summary.false_negative} | {summary.precision:.4f} | "
            f"{summary.recall:.4f} | {summary.false_positive_rate:.4f} | {summary.accuracy:.4f} | "
            f"{summary.macro_f1:.4f} | {summary.warn_count} | {summary.allow_count} |"
        )
    lines.extend(
        [
            "",
            "## Slice Calibration And Errors",
            "",
            "| Dimension | Value | Examples | Errors | FP | FN | Near Errors | Confident Errors | "
            "Mean P(exfil) | Empirical Positive Rate | Abs Cal Gap | Accuracy |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in _interesting_slice_summaries(report.slice_summaries):
        lines.append(
            f"| `{summary.dimension}` | `{summary.value}` | {summary.example_count} | {summary.error_count} | "
            f"{summary.false_positive_count} | {summary.false_negative_count} | "
            f"{summary.near_threshold_error_count} | {summary.confident_error_count} | "
            f"{summary.mean_probability:.4f} | {summary.empirical_positive_rate:.4f} | "
            f"{summary.absolute_calibration_gap:.4f} | {summary.accuracy:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Error Examples",
            "",
            "| Example | Error | True | Predicted | P(exfil) | Margin | Family | Payload | Mode | Action |",
            "|---|---|---|---|---:|---:|---|---|---|---|",
        ]
    )
    for example in report.error_examples:
        lines.append(
            f"| `{example.example_id}` | `{example.error_kind}` | `{example.true_label}` | "
            f"`{example.predicted_label}` | {example.positive_probability:.4f} | "
            f"{example.signed_margin:.4f} | `{example.family}` | `{example.payload_condition}` | "
            f"`{example.selected_mode}` | `{example.selected_action}` |"
        )
    return "\n".join(lines) + "\n"


def write_selector_diagnostic_markdown(path: Path, report: SelectorDiagnosticReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_selector_diagnostic_markdown(report), encoding="utf-8")


def _join_predictions(
    report: CiftCalibrationReport,
    metadata_by_id: Mapping[str, PromptPolicyMetadata],
) -> tuple[_JoinedPrediction, ...]:
    joined: list[_JoinedPrediction] = []
    for prediction in report.predictions:
        metadata = metadata_by_id.get(prediction.example_id)
        if metadata is None:
            raise SelectorDiagnosticError(f"Missing prompt metadata for example '{prediction.example_id}'.")
        joined.append(_JoinedPrediction(prediction=prediction, metadata=metadata))
    if len(joined) == 0:
        raise SelectorDiagnosticError("Calibration report has no predictions.")
    return tuple(joined)


def _slice_summaries(
    report: CiftCalibrationReport,
    joined_predictions: tuple[_JoinedPrediction, ...],
    dimensions: tuple[ErrorSliceDimension, ...],
    near_threshold_radius: float,
) -> tuple[SelectorSliceSummary, ...]:
    groups: dict[tuple[ErrorSliceDimension, str], list[_JoinedPrediction]] = {}
    for joined in joined_predictions:
        for dimension in dimensions:
            key = (dimension, _dimension_value(joined, dimension))
            groups.setdefault(key, []).append(joined)
    summaries = tuple(
        _slice_summary(
            dimension=dimension,
            value=value,
            report=report,
            joined_predictions=tuple(group),
            near_threshold_radius=near_threshold_radius,
        )
        for (dimension, value), group in groups.items()
    )
    return tuple(
        sorted(
            summaries,
            key=lambda summary: (
                summary.dimension,
                -summary.error_count,
                -summary.absolute_calibration_gap,
                summary.value,
            ),
        )
    )


def _slice_summary(
    dimension: ErrorSliceDimension,
    value: str,
    report: CiftCalibrationReport,
    joined_predictions: tuple[_JoinedPrediction, ...],
    near_threshold_radius: float,
) -> SelectorSliceSummary:
    example_count = len(joined_predictions)
    positive_count = sum(1 for joined in joined_predictions if _actual_positive(report, joined.prediction))
    predicted_positive_count = sum(1 for joined in joined_predictions if _predicted_positive(report, joined.prediction))
    false_positive_count = sum(1 for joined in joined_predictions if _error_kind(report, joined.prediction) == "false_positive")
    false_negative_count = sum(1 for joined in joined_predictions if _error_kind(report, joined.prediction) == "false_negative")
    error_count = false_positive_count + false_negative_count
    near_threshold_count = sum(
        1 for joined in joined_predictions if _absolute_margin(report, joined.prediction) <= near_threshold_radius
    )
    near_threshold_error_count = sum(
        1
        for joined in joined_predictions
        if _error_kind(report, joined.prediction) is not None
        and _absolute_margin(report, joined.prediction) <= near_threshold_radius
    )
    mean_probability = sum(joined.prediction.positive_probability for joined in joined_predictions) / example_count
    empirical_positive_rate = positive_count / example_count
    calibration_gap = mean_probability - empirical_positive_rate
    return SelectorSliceSummary(
        dimension=dimension,
        value=value,
        example_count=example_count,
        positive_count=positive_count,
        negative_count=example_count - positive_count,
        predicted_positive_count=predicted_positive_count,
        correct_count=example_count - error_count,
        error_count=error_count,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        near_threshold_count=near_threshold_count,
        near_threshold_error_count=near_threshold_error_count,
        confident_error_count=error_count - near_threshold_error_count,
        accuracy=(example_count - error_count) / example_count,
        empirical_positive_rate=empirical_positive_rate,
        mean_probability=mean_probability,
        calibration_gap=calibration_gap,
        absolute_calibration_gap=abs(calibration_gap),
        mean_absolute_margin=sum(_absolute_margin(report, joined.prediction) for joined in joined_predictions)
        / example_count,
    )


def _error_examples(
    report: CiftCalibrationReport,
    joined_predictions: tuple[_JoinedPrediction, ...],
    near_threshold_radius: float,
) -> tuple[SelectorErrorExample, ...]:
    examples: list[SelectorErrorExample] = []
    for joined in joined_predictions:
        kind = _error_kind(report, joined.prediction)
        if kind is None:
            continue
        examples.append(_error_example(report=report, joined=joined, kind=kind))
    return tuple(
        sorted(
            examples,
            key=lambda example: (
                _near_rank(example.absolute_margin, near_threshold_radius),
                -example.absolute_margin,
                example.family,
                example.example_id,
            ),
        )
    )


def _error_example(
    report: CiftCalibrationReport,
    joined: _JoinedPrediction,
    kind: SelectorErrorKind,
) -> SelectorErrorExample:
    metadata = joined.metadata
    prediction = joined.prediction
    return SelectorErrorExample(
        example_id=prediction.example_id,
        family=metadata.family,
        source_label=prediction.source_label,
        true_label=prediction.true_label,
        predicted_label=_predicted_label(report=report, prediction=prediction),
        error_kind=kind,
        positive_probability=prediction.positive_probability,
        signed_margin=_signed_margin(report=report, prediction=prediction),
        absolute_margin=_absolute_margin(report=report, prediction=prediction),
        payload_condition=metadata.payload_condition,
        selected_mode=metadata.selected_mode,
        selected_field=metadata.selected_field,
        selected_action=metadata.selected_action,
        credential_type=metadata.credential_type,
        fold_index=prediction.fold_index,
    )


def _threshold_summary_to_json(summary: CiftOperatingPoint) -> dict[str, JsonValue]:
    return {
        "threshold": summary.threshold,
        "true_positive": summary.true_positive,
        "false_positive": summary.false_positive,
        "true_negative": summary.true_negative,
        "false_negative": summary.false_negative,
        "precision": summary.precision,
        "recall": summary.recall,
        "specificity": summary.specificity,
        "false_positive_rate": summary.false_positive_rate,
        "false_negative_rate": summary.false_negative_rate,
        "accuracy": summary.accuracy,
        "positive_f1": summary.positive_f1,
        "negative_f1": summary.negative_f1,
        "macro_f1": summary.macro_f1,
        "warn_count": summary.warn_count,
        "allow_count": summary.allow_count,
    }


def _slice_summary_to_json(summary: SelectorSliceSummary) -> dict[str, JsonValue]:
    return {
        "dimension": summary.dimension,
        "value": summary.value,
        "example_count": summary.example_count,
        "positive_count": summary.positive_count,
        "negative_count": summary.negative_count,
        "predicted_positive_count": summary.predicted_positive_count,
        "correct_count": summary.correct_count,
        "error_count": summary.error_count,
        "false_positive_count": summary.false_positive_count,
        "false_negative_count": summary.false_negative_count,
        "near_threshold_count": summary.near_threshold_count,
        "near_threshold_error_count": summary.near_threshold_error_count,
        "confident_error_count": summary.confident_error_count,
        "accuracy": summary.accuracy,
        "empirical_positive_rate": summary.empirical_positive_rate,
        "mean_probability": summary.mean_probability,
        "calibration_gap": summary.calibration_gap,
        "absolute_calibration_gap": summary.absolute_calibration_gap,
        "mean_absolute_margin": summary.mean_absolute_margin,
    }


def _error_example_to_json(example: SelectorErrorExample) -> dict[str, JsonValue]:
    return {
        "example_id": example.example_id,
        "family": example.family,
        "source_label": example.source_label,
        "true_label": example.true_label,
        "predicted_label": example.predicted_label,
        "error_kind": example.error_kind,
        "positive_probability": example.positive_probability,
        "signed_margin": example.signed_margin,
        "absolute_margin": example.absolute_margin,
        "payload_condition": example.payload_condition,
        "selected_mode": example.selected_mode,
        "selected_field": example.selected_field,
        "selected_action": example.selected_action,
        "credential_type": example.credential_type,
        "fold_index": example.fold_index,
    }


def _dimension_value(joined: _JoinedPrediction, dimension: ErrorSliceDimension) -> str:
    metadata = joined.metadata
    if dimension == "family":
        return metadata.family
    if dimension == "source_label":
        return joined.prediction.source_label
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
    raise SelectorDiagnosticError(f"Unsupported dimension '{dimension}'.")


def _actual_positive(report: CiftCalibrationReport, prediction: CalibratedCiftPrediction) -> bool:
    return prediction.true_label == report.positive_label


def _predicted_positive(report: CiftCalibrationReport, prediction: CalibratedCiftPrediction) -> bool:
    return prediction.positive_probability >= report.decision_threshold


def _predicted_label(report: CiftCalibrationReport, prediction: CalibratedCiftPrediction) -> str:
    if _predicted_positive(report, prediction):
        return report.positive_label
    if prediction.true_label == report.positive_label:
        return prediction.predicted_label
    return prediction.predicted_label


def _error_kind(
    report: CiftCalibrationReport,
    prediction: CalibratedCiftPrediction,
) -> SelectorErrorKind | None:
    actual_positive = _actual_positive(report, prediction)
    predicted_positive = _predicted_positive(report, prediction)
    if actual_positive and not predicted_positive:
        return "false_negative"
    if not actual_positive and predicted_positive:
        return "false_positive"
    return None


def _signed_margin(report: CiftCalibrationReport, prediction: CalibratedCiftPrediction) -> float:
    return prediction.positive_probability - report.decision_threshold


def _absolute_margin(report: CiftCalibrationReport, prediction: CalibratedCiftPrediction) -> float:
    return abs(_signed_margin(report, prediction))


def _near_rank(absolute_margin: float, near_threshold_radius: float) -> int:
    if absolute_margin <= near_threshold_radius:
        return 1
    return 0


def _interesting_slice_summaries(summaries: tuple[SelectorSliceSummary, ...]) -> tuple[SelectorSliceSummary, ...]:
    return tuple(
        sorted(
            summaries,
            key=lambda summary: (
                summary.error_count == 0,
                -summary.error_count,
                -summary.absolute_calibration_gap,
                summary.dimension,
                summary.value,
            ),
        )
    )


def _validate_config(config: SelectorDiagnosticConfig) -> None:
    if len(config.dimensions) == 0:
        raise SelectorDiagnosticError("dimensions must not be empty.")
    if len(config.thresholds) == 0:
        raise SelectorDiagnosticError("thresholds must not be empty.")
    if config.near_threshold_radius < 0.0:
        raise SelectorDiagnosticError("near_threshold_radius must be non-negative.")
    previous_threshold = -1.0
    for threshold in config.thresholds:
        if threshold < 0.0 or threshold > 1.0:
            raise SelectorDiagnosticError("thresholds must be in [0.0, 1.0].")
        if threshold <= previous_threshold:
            raise SelectorDiagnosticError("thresholds must be strictly increasing.")
        previous_threshold = threshold
