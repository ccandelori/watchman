from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aegis_introspection.cift_calibration import CiftCalibrationReport
from aegis_introspection.probe import JsonValue


class CiftOperatingPointError(ValueError):
    """Raised when CIFT operating points cannot be computed."""


@dataclass(frozen=True)
class CiftOperatingPointConfig:
    thresholds: tuple[float, ...]


@dataclass(frozen=True)
class CiftOperatingPoint:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    accuracy: float
    positive_f1: float
    negative_f1: float
    macro_f1: float
    warn_count: int
    allow_count: int


@dataclass(frozen=True)
class CiftOperatingPointReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    source_evaluation_strategy: str
    score_semantics: str
    task_name: str
    positive_label: str
    activation_feature_key: str
    source_decision_threshold: float
    best_macro_f1_threshold: float
    high_recall_threshold: float
    operating_points: tuple[CiftOperatingPoint, ...]


def build_cift_operating_point_report(
    report: CiftCalibrationReport,
    config: CiftOperatingPointConfig,
) -> CiftOperatingPointReport:
    _validate_config(config)
    operating_points = tuple(_operating_point(report=report, threshold=threshold) for threshold in config.thresholds)
    best_macro_f1 = max(operating_points, key=lambda point: (point.macro_f1, point.recall, -point.threshold))
    high_recall = max(operating_points, key=lambda point: (point.recall, point.precision, -point.threshold))
    return CiftOperatingPointReport(
        source_model_id=report.source_model_id,
        source_revision=report.source_revision,
        source_selected_device=report.source_selected_device,
        source_evaluation_strategy=report.evaluation_strategy,
        score_semantics=report.score_semantics,
        task_name=report.task_name,
        positive_label=report.positive_label,
        activation_feature_key=report.activation_feature_key,
        source_decision_threshold=report.decision_threshold,
        best_macro_f1_threshold=best_macro_f1.threshold,
        high_recall_threshold=high_recall.threshold,
        operating_points=operating_points,
    )


def cift_operating_point_report_to_json(report: CiftOperatingPointReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "source_evaluation_strategy": report.source_evaluation_strategy,
        "score_semantics": report.score_semantics,
        "task_name": report.task_name,
        "positive_label": report.positive_label,
        "activation_feature_key": report.activation_feature_key,
        "source_decision_threshold": report.source_decision_threshold,
        "best_macro_f1_threshold": report.best_macro_f1_threshold,
        "high_recall_threshold": report.high_recall_threshold,
        "operating_points": [_operating_point_to_json(point) for point in report.operating_points],
    }


def write_cift_operating_point_json(path: Path, report: CiftOperatingPointReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cift_operating_point_report_to_json(report), file, indent=2)
        file.write("\n")


def render_cift_operating_point_markdown(report: CiftOperatingPointReport) -> str:
    lines = [
        "# CIFT Detector Operating Points",
        "",
        "## Source",
        "",
        f"- Model: `{report.source_model_id}`",
        f"- Revision: `{report.source_revision}`",
        f"- Extraction device: `{report.source_selected_device}`",
        f"- Source evaluation strategy: `{report.source_evaluation_strategy}`",
        f"- Score semantics: `{report.score_semantics}`",
        f"- Task: `{report.task_name}`",
        f"- Positive label: `{report.positive_label}`",
        f"- Activation feature: `{report.activation_feature_key}`",
        f"- Source decision threshold: `{report.source_decision_threshold:.4f}`",
        f"- Best macro-F1 threshold: `{report.best_macro_f1_threshold:.4f}`",
        f"- High-recall threshold: `{report.high_recall_threshold:.4f}`",
        "",
        "## Threshold Sweep",
        "",
        "| Threshold | TP | FP | TN | FN | Precision | Recall | FPR | Accuracy | Macro F1 | Warn | Allow |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in report.operating_points:
        lines.append(
            f"| {point.threshold:.4f} | "
            f"{point.true_positive} | {point.false_positive} | {point.true_negative} | {point.false_negative} | "
            f"{point.precision:.4f} | {point.recall:.4f} | {point.false_positive_rate:.4f} | "
            f"{point.accuracy:.4f} | {point.macro_f1:.4f} | {point.warn_count} | {point.allow_count} |"
        )
    return "\n".join(lines) + "\n"


def write_cift_operating_point_markdown(path: Path, report: CiftOperatingPointReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cift_operating_point_markdown(report), encoding="utf-8")


def _operating_point(report: CiftCalibrationReport, threshold: float) -> CiftOperatingPoint:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for prediction in report.predictions:
        actual_positive = prediction.true_label == report.positive_label
        predicted_positive = prediction.positive_probability >= threshold
        if actual_positive and predicted_positive:
            true_positive += 1
        elif not actual_positive and predicted_positive:
            false_positive += 1
        elif not actual_positive and not predicted_positive:
            true_negative += 1
        else:
            false_negative += 1

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    false_positive_rate = _safe_divide(false_positive, false_positive + true_negative)
    false_negative_rate = _safe_divide(false_negative, false_negative + true_positive)
    accuracy = _safe_divide(true_positive + true_negative, len(report.predictions))
    positive_f1 = _f1(precision, recall)
    negative_precision = _safe_divide(true_negative, true_negative + false_negative)
    negative_recall = specificity
    negative_f1 = _f1(negative_precision, negative_recall)
    return CiftOperatingPoint(
        threshold=threshold,
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        specificity=specificity,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        accuracy=accuracy,
        positive_f1=positive_f1,
        negative_f1=negative_f1,
        macro_f1=(positive_f1 + negative_f1) / 2.0,
        warn_count=true_positive + false_positive,
        allow_count=true_negative + false_negative,
    )


def _operating_point_to_json(point: CiftOperatingPoint) -> dict[str, JsonValue]:
    return {
        "threshold": point.threshold,
        "true_positive": point.true_positive,
        "false_positive": point.false_positive,
        "true_negative": point.true_negative,
        "false_negative": point.false_negative,
        "precision": point.precision,
        "recall": point.recall,
        "specificity": point.specificity,
        "false_positive_rate": point.false_positive_rate,
        "false_negative_rate": point.false_negative_rate,
        "accuracy": point.accuracy,
        "positive_f1": point.positive_f1,
        "negative_f1": point.negative_f1,
        "macro_f1": point.macro_f1,
        "warn_count": point.warn_count,
        "allow_count": point.allow_count,
    }


def _validate_config(config: CiftOperatingPointConfig) -> None:
    if len(config.thresholds) == 0:
        raise CiftOperatingPointError("thresholds must not be empty.")
    previous_threshold = -1.0
    for threshold in config.thresholds:
        if threshold < 0.0 or threshold > 1.0:
            raise CiftOperatingPointError("thresholds must be in [0.0, 1.0].")
        if threshold <= previous_threshold:
            raise CiftOperatingPointError("thresholds must be strictly increasing.")
        previous_threshold = threshold


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator == 0.0:
        return 0.0
    return 2.0 * precision * recall / denominator
