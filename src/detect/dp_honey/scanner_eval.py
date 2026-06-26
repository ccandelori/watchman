"""Deterministic DP-HONEY scanner evidence reports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

from . import scanner
from .errors import DPHoneyError
from .formats import REGISTRY_VERSION, list_formats
from .grammar import FormatSpec

SCANNER_EVAL_SCHEMA_VERSION = "detect.dp_honey.scanner_eval/v1"
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
_CONFIDENCE_SCORES = {"low": 0.35, "medium": 0.65, "high": 0.95}


class DPHoneyScannerEvalError(DPHoneyError):
    """Raised when scanner evaluation input is invalid."""


@dataclass(frozen=True)
class DPHoneyScannerEvalConfig:
    positive_per_format: int
    seed: int
    target_alpha: float
    negative_count: int
    calibration_count: int

    def __post_init__(self) -> None:
        if self.positive_per_format < 1:
            raise DPHoneyScannerEvalError("positive_per_format must be positive.")
        if self.negative_count < 1:
            raise DPHoneyScannerEvalError("negative_count must be positive.")
        if self.calibration_count < 1:
            raise DPHoneyScannerEvalError("calibration_count must be positive.")
        if not math.isfinite(self.target_alpha) or self.target_alpha <= 0.0 or self.target_alpha >= 1.0:
            raise DPHoneyScannerEvalError("target_alpha must be finite and in (0.0, 1.0).")


@dataclass(frozen=True)
class DPHoneyScannerEvalCounts:
    true_positive: int
    false_negative: int
    true_negative: int
    false_positive: int

    def to_dict(self) -> dict[str, int]:
        return {
            "true_positive": self.true_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
        }


@dataclass(frozen=True)
class DPHoneyScannerFormatMetric:
    format_slug: str
    positive_examples: int
    true_positive: int
    false_negative: int
    false_negative_rate: float

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "format_slug": self.format_slug,
            "positive_examples": self.positive_examples,
            "true_positive": self.true_positive,
            "false_negative": self.false_negative,
            "false_negative_rate": self.false_negative_rate,
        }


def build_scanner_eval_report(config: DPHoneyScannerEvalConfig) -> dict[str, JsonValue]:
    scannable_specs = tuple(spec for spec in list_formats() if spec.scannable)
    if len(scannable_specs) == 0:
        raise DPHoneyScannerEvalError("no scannable DP-HONEY formats are registered.")
    calibration = _calibrate_confidence_threshold(config)
    threshold = _required_numeric(calibration["threshold"], "conformal threshold")
    positive_metrics = tuple(_evaluate_format(spec, config, threshold) for spec in scannable_specs)
    negative_texts = _negative_texts(config.negative_count, config.seed, "eval")
    negative_counts = _evaluate_negatives(negative_texts, threshold)
    true_positive = sum(metric.true_positive for metric in positive_metrics)
    false_negative = sum(metric.false_negative for metric in positive_metrics)
    counts = DPHoneyScannerEvalCounts(
        true_positive=true_positive,
        false_negative=false_negative,
        true_negative=negative_counts.true_negative,
        false_positive=negative_counts.false_positive,
    )
    positive_count = counts.true_positive + counts.false_negative
    negative_count = counts.true_negative + counts.false_positive
    detected_count = counts.true_positive + counts.false_positive
    return {
        "schema_version": SCANNER_EVAL_SCHEMA_VERSION,
        "scanner_kind": "registry_regex_validate_plus_unknown_entropy",
        "registry_version": REGISTRY_VERSION,
        "seed": config.seed,
        "target_alpha": config.target_alpha,
        "target_coverage": 1.0 - config.target_alpha,
        "positive_per_format": config.positive_per_format,
        "negative_requested_count": config.negative_count,
        "calibration_requested_count": config.calibration_count,
        "scannable_format_count": len(scannable_specs),
        "positive_example_count": positive_count,
        "negative_example_count": negative_count,
        "counts": counts.to_dict(),
        "precision": _safe_rate(counts.true_positive, detected_count),
        "recall": _safe_rate(counts.true_positive, positive_count),
        "false_positive_rate": _safe_rate(counts.false_positive, negative_count),
        "false_negative_rate": _safe_rate(counts.false_negative, positive_count),
        "one_token_detection": counts.false_negative == 0,
        "format_metrics": [metric.to_dict() for metric in positive_metrics],
        "conformal_calibration": calibration,
        "audit_safety": {
            "raw_secret_values_in_report": False,
            "finding_payload_redacted": True,
        },
    }


def render_scanner_eval_report_json(report: dict[str, JsonValue]) -> str:
    return json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"


def write_scanner_eval_report(path: Path, report: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_scanner_eval_report_json(report), encoding="utf-8")


def _calibrate_confidence_threshold(config: DPHoneyScannerEvalConfig) -> dict[str, JsonValue]:
    calibration_texts = _calibration_negative_texts(config)
    scores = tuple(_max_confidence_score(scanner.scan(text)) for text in calibration_texts)
    threshold = _conformal_quantile(scores, config.target_alpha)
    accepted_scores = tuple(score for score in scores if score > threshold)
    empirical_false_positive_rate = len(accepted_scores) / len(scores)
    return {
        "implemented": True,
        "status": "split_conformal_confidence_threshold",
        "target_alpha": config.target_alpha,
        "target_coverage": 1.0 - config.target_alpha,
        "calibration_benign_count": len(scores),
        "calibration_score_max": max(scores),
        "calibration_score_values": sorted(set(scores)),
        "quantile_rank": _conformal_rank(len(scores), config.target_alpha),
        "threshold": threshold,
        "score_rule": "detect if max_confidence_score > threshold",
        "recommended_min_confidence": _recommended_min_confidence(threshold),
        "empirical_calibration_false_positive_rate": empirical_false_positive_rate,
        "empirical_calibration_coverage": 1.0 - empirical_false_positive_rate,
        "paper_faithfulness_status": (
            "split conformal confidence threshold implemented for scanner findings; generator realism still requires "
            "stronger statistical distinguisher evidence"
        ),
    }


def _evaluate_format(
    spec: FormatSpec,
    config: DPHoneyScannerEvalConfig,
    threshold: float,
) -> DPHoneyScannerFormatMetric:
    rng = np.random.default_rng(config.seed + _stable_slug_offset(spec.slug))
    true_positive = 0
    false_negative = 0
    for index in range(config.positive_per_format):
        token = spec.random_example(rng)
        text = f"credential candidate {index}: {token}"
        findings = _accepted_findings(scanner.scan(text), threshold)
        if any(finding.get("format") == spec.slug for finding in findings):
            true_positive += 1
        else:
            false_negative += 1
    return DPHoneyScannerFormatMetric(
        format_slug=spec.slug,
        positive_examples=config.positive_per_format,
        true_positive=true_positive,
        false_negative=false_negative,
        false_negative_rate=_required_rate(false_negative, config.positive_per_format),
    )


def _evaluate_negatives(negative_texts: tuple[str, ...], threshold: float) -> DPHoneyScannerEvalCounts:
    true_negative = 0
    false_positive = 0
    for text in negative_texts:
        if len(_accepted_findings(scanner.scan(text), threshold)) == 0:
            true_negative += 1
        else:
            false_positive += 1
    return DPHoneyScannerEvalCounts(
        true_positive=0,
        false_negative=0,
        true_negative=true_negative,
        false_positive=false_positive,
    )


def _calibration_negative_texts(config: DPHoneyScannerEvalConfig) -> tuple[str, ...]:
    return _negative_texts(config.calibration_count, config.seed + 65_537, "calibration")


def _negative_texts(count: int, seed: int, corpus_role: str) -> tuple[str, ...]:
    return tuple(_negative_text(index, seed, corpus_role) for index in range(count))


def _negative_text(index: int, seed: int, corpus_role: str) -> str:
    opaque = _opaque_id(index, seed)
    templates = (
        f"{corpus_role} benign task {index}: review repository policy text with request id req_{opaque}.",
        f"{corpus_role} benign ticket {index}: token budget is {24000 + index}, no credential value is present.",
        f"{corpus_role} benign trace {index}: feature_flag_canary_mode=false and safe-canary is only prose.",
        f"{corpus_role} benign tool note {index}: client_ref_{opaque} is an opaque case identifier.",
        f"{corpus_role} benign release {index}: digest rel_{opaque} documents build metadata.",
        f"{corpus_role} benign analytics {index}: metric_{opaque} is a dashboard dimension.",
        f"{corpus_role} benign blob {index}: c2FmZV9ub25fc2VjcmV0X3RleHQ= is sample documentation.",
        f"{corpus_role} benign near-neighbor {index}: ghq_{opaque} is an invalid prefix fixture.",
        f"{corpus_role} benign near-neighbor {index}: debug_live_{opaque} is a non-secret local label.",
        f"{corpus_role} benign near-neighbor {index}: xoxc-{opaque} is not a Slack credential class.",
    )
    return templates[index % len(templates)]


def _opaque_id(index: int, seed: int) -> str:
    value = ((seed + 1) * 1_315_423_911 + (index + 17) * 2_654_435_761) % 16**16
    return f"{value:016x}"


def _stable_slug_offset(slug: str) -> int:
    total = 0
    for index, character in enumerate(slug):
        total += (index + 1) * ord(character)
    return total


def _accepted_findings(findings: list[dict[str, int | str]], threshold: float) -> tuple[dict[str, int | str], ...]:
    return tuple(finding for finding in findings if _confidence_score(str(finding["confidence"])) > threshold)


def _max_confidence_score(findings: list[dict[str, int | str]]) -> float:
    if len(findings) == 0:
        return 0.0
    return max(_confidence_score(str(finding["confidence"])) for finding in findings)


def _confidence_score(confidence: str) -> float:
    return _CONFIDENCE_SCORES.get(confidence, 0.0)


def _conformal_quantile(scores: tuple[float, ...], target_alpha: float) -> float:
    if len(scores) == 0:
        raise DPHoneyScannerEvalError("calibration scores must not be empty.")
    rank = _conformal_rank(len(scores), target_alpha)
    return sorted(scores)[rank - 1]


def _conformal_rank(score_count: int, target_alpha: float) -> int:
    if score_count < 1:
        raise DPHoneyScannerEvalError("score_count must be positive.")
    rank = math.ceil((score_count + 1) * (1.0 - target_alpha))
    return min(max(rank, 1), score_count)


def _recommended_min_confidence(threshold: float) -> str:
    for confidence in ("low", "medium", "high"):
        if _CONFIDENCE_SCORES[confidence] > threshold:
            return confidence
    return "above_high"


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _required_numeric(value: JsonValue, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DPHoneyScannerEvalError(f"{field_name} must be numeric.")
    if not math.isfinite(float(value)):
        raise DPHoneyScannerEvalError(f"{field_name} must be finite.")
    return float(value)


def _required_rate(numerator: int, denominator: int) -> float:
    rate = _safe_rate(numerator, denominator)
    if rate is None or not math.isfinite(rate):
        raise DPHoneyScannerEvalError("rate denominator must be positive.")
    return rate
