"""Aggregate DP-HONEY generation realism evidence.

This report compares generated honeytokens against same-format synthetic
reference examples with aggregate validity, duplicate-rate, entropy, and
model-likelihood metrics. It deliberately does not serialize token values and
does not claim the paper's full statistical-distinguisher suite.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

from .bigram import DEFAULT_CLIP, DEFAULT_CORPUS_SIZE, DEFAULT_EPSILON, DEFAULT_TRAIN_SEED, build_model
from .errors import DPHoneyError
from .formats import REGISTRY_VERSION, list_formats
from .grammar import FormatSpec
from .realism import compute_report

GENERATION_REALISM_EVAL_SCHEMA_VERSION = "detect.dp_honey.generation_realism_eval/v1"
GENERATION_REALISM_EVAL_MAX_PER_FORMAT = 1000
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class DPHoneyGenerationRealismEvalError(DPHoneyError):
    """Raised when DP-HONEY generation realism evaluation input is invalid."""


@dataclass(frozen=True)
class DPHoneyGenerationRealismEvalConfig:
    count_per_format: int
    seed: int

    def __post_init__(self) -> None:
        if self.count_per_format < 1:
            raise DPHoneyGenerationRealismEvalError("count_per_format must be positive.")
        if self.count_per_format > GENERATION_REALISM_EVAL_MAX_PER_FORMAT:
            raise DPHoneyGenerationRealismEvalError(
                f"count_per_format must be <= {GENERATION_REALISM_EVAL_MAX_PER_FORMAT}."
            )


@dataclass(frozen=True)
class DPHoneyGenerationRealismMetric:
    format_slug: str
    scannable: bool
    generated_count: int
    reference_count: int
    generated_validity_rate: float
    reference_validity_rate: float
    generated_duplicate_rate: float
    reference_duplicate_rate: float
    generated_char_entropy_bits: float
    reference_char_entropy_bits: float
    char_entropy_delta_bits: float
    generated_avg_log_likelihood: float
    reference_avg_log_likelihood: float
    avg_log_likelihood_delta: float
    finite_metrics: bool
    bounded_sanity_gate_passed: bool

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "format_slug": self.format_slug,
            "scannable": self.scannable,
            "generated_count": self.generated_count,
            "reference_count": self.reference_count,
            "generated_validity_rate": self.generated_validity_rate,
            "reference_validity_rate": self.reference_validity_rate,
            "generated_duplicate_rate": self.generated_duplicate_rate,
            "reference_duplicate_rate": self.reference_duplicate_rate,
            "generated_char_entropy_bits": self.generated_char_entropy_bits,
            "reference_char_entropy_bits": self.reference_char_entropy_bits,
            "char_entropy_delta_bits": self.char_entropy_delta_bits,
            "generated_avg_log_likelihood": self.generated_avg_log_likelihood,
            "reference_avg_log_likelihood": self.reference_avg_log_likelihood,
            "avg_log_likelihood_delta": self.avg_log_likelihood_delta,
            "finite_metrics": self.finite_metrics,
            "bounded_sanity_gate_passed": self.bounded_sanity_gate_passed,
        }


def build_generation_realism_eval_report(config: DPHoneyGenerationRealismEvalConfig) -> dict[str, JsonValue]:
    specs = tuple(list_formats())
    if len(specs) == 0:
        raise DPHoneyGenerationRealismEvalError("no DP-HONEY formats are registered.")
    metrics = tuple(_evaluate_format_generation_realism(spec, config) for spec in specs)
    return {
        "schema_version": GENERATION_REALISM_EVAL_SCHEMA_VERSION,
        "status": "bounded_generated_vs_reference_sanity_metrics",
        "registry_version": REGISTRY_VERSION,
        "seed": config.seed,
        "count_per_format": config.count_per_format,
        "format_count": len(specs),
        "scannable_format_count": sum(1 for spec in specs if spec.scannable),
        "generator_parameters": {
            "epsilon": DEFAULT_EPSILON,
            "clip": DEFAULT_CLIP,
            "corpus_size": DEFAULT_CORPUS_SIZE,
            "train_seed": DEFAULT_TRAIN_SEED,
        },
        "metric_families": [
            "format_validity",
            "duplicate_rate",
            "character_entropy",
            "model_avg_log_likelihood",
        ],
        "all_generated_tokens_valid": all(metric.generated_validity_rate == 1.0 for metric in metrics),
        "all_reference_tokens_valid": all(metric.reference_validity_rate == 1.0 for metric in metrics),
        "all_metrics_finite": all(metric.finite_metrics for metric in metrics),
        "bounded_sanity_gate_passed": all(metric.bounded_sanity_gate_passed for metric in metrics),
        "paper_faithful_statistical_distinguisher": False,
        "format_metrics": [metric.to_dict() for metric in metrics],
        "audit_safety": {
            "raw_secret_values_in_report": False,
            "finding_payload_redacted": True,
        },
        "limits": [
            "No raw generated or reference token values are serialized.",
            "This is a bounded aggregate sanity gate, not a trained discriminator benchmark.",
            "paper_faithful_statistical_distinguisher remains false until character-entropy tests, bigram "
            "likelihood thresholds, numeric-substring tests, and discriminator-MLP tests are evaluated under "
            "sealed evidence.",
        ],
    }


def render_generation_realism_eval_report_json(report: dict[str, JsonValue]) -> str:
    return json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"


def write_generation_realism_eval_report(path: Path, report: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_generation_realism_eval_report_json(report), encoding="utf-8")


def _evaluate_format_generation_realism(
    spec: FormatSpec,
    config: DPHoneyGenerationRealismEvalConfig,
) -> DPHoneyGenerationRealismMetric:
    slug_offset = _stable_slug_offset(spec.slug)
    model = build_model(spec)
    generated_tokens = model.sample(config.count_per_format, seed=config.seed + 10_000 + slug_offset)
    reference_rng = np.random.default_rng(config.seed + 20_000 + slug_offset)
    reference_tokens = [spec.random_example(reference_rng) for _ in range(config.count_per_format)]
    generated_report = compute_report(generated_tokens, model)
    reference_report = compute_report(reference_tokens, model)
    generated_validity = _report_float(generated_report, "validity_rate", spec.slug)
    reference_validity = _report_float(reference_report, "validity_rate", spec.slug)
    generated_duplicate_rate = _report_float(generated_report, "duplicate_rate", spec.slug)
    reference_duplicate_rate = _report_float(reference_report, "duplicate_rate", spec.slug)
    generated_entropy = _report_float(generated_report, "char_entropy_bits", spec.slug)
    reference_entropy = _report_float(reference_report, "char_entropy_bits", spec.slug)
    generated_likelihood = _report_float(generated_report, "avg_log_likelihood", spec.slug)
    reference_likelihood = _report_float(reference_report, "avg_log_likelihood", spec.slug)
    finite_metrics = all(
        math.isfinite(value)
        for value in (
            generated_validity,
            reference_validity,
            generated_duplicate_rate,
            reference_duplicate_rate,
            generated_entropy,
            reference_entropy,
            generated_likelihood,
            reference_likelihood,
        )
    )
    bounded_sanity_gate_passed = (
        len(generated_tokens) == config.count_per_format
        and len(reference_tokens) == config.count_per_format
        and generated_validity == 1.0
        and reference_validity == 1.0
        and generated_entropy > 0.0
        and reference_entropy > 0.0
        and finite_metrics
    )
    return DPHoneyGenerationRealismMetric(
        format_slug=spec.slug,
        scannable=spec.scannable,
        generated_count=len(generated_tokens),
        reference_count=len(reference_tokens),
        generated_validity_rate=generated_validity,
        reference_validity_rate=reference_validity,
        generated_duplicate_rate=generated_duplicate_rate,
        reference_duplicate_rate=reference_duplicate_rate,
        generated_char_entropy_bits=generated_entropy,
        reference_char_entropy_bits=reference_entropy,
        char_entropy_delta_bits=generated_entropy - reference_entropy,
        generated_avg_log_likelihood=generated_likelihood,
        reference_avg_log_likelihood=reference_likelihood,
        avg_log_likelihood_delta=generated_likelihood - reference_likelihood,
        finite_metrics=finite_metrics,
        bounded_sanity_gate_passed=bounded_sanity_gate_passed,
    )


def _stable_slug_offset(slug: str) -> int:
    total = 0
    for index, character in enumerate(slug):
        total += (index + 1) * ord(character)
    return total


def _report_float(report: dict[str, object], field_name: str, format_slug: str) -> float:
    value = report.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DPHoneyGenerationRealismEvalError(f"{format_slug}.{field_name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise DPHoneyGenerationRealismEvalError(f"{format_slug}.{field_name} must be finite.")
    return numeric
