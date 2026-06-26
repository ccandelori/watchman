from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import itertools
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from aegis.core.contracts import JsonValue
from aegis.replay.nimbus_training import (
    INFO_NCE_NEGATIVE_COUNT,
    NIMBUS_TRAINING_SCHEMA_VERSION,
    NimbusLeakageLabel,
    NimbusSecretContext,
    NimbusTrainingCorpusError,
    NimbusTrainingTurnRecord,
    read_nimbus_training_records_jsonl,
    validate_nimbus_training_record,
)

NIMBUS_INFONCE_MODEL_SCHEMA_VERSION = "aegis.nimbus_infonce_model/v0"
NIMBUS_INFONCE_EVAL_SCHEMA_VERSION = "aegis.nimbus_infonce_eval/v0"
NIMBUS_INFONCE_GROUPED_CV_SCHEMA_VERSION = "aegis.nimbus_infonce_grouped_cv/v0"
NIMBUS_INFONCE_MODEL_ID = "nimbus-infonce-lexical-v0"
NIMBUS_INFONCE_FEATURE_NAMES = ("output_token_overlap", "decoded_output_token_overlap", "state_token_overlap")
NIMBUS_INFONCE_PROMOTION_STATUS = "not_promotable_offline_scaffold"

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_BASE64_CANDIDATE_PATTERN = re.compile(r"\b[A-Za-z0-9+/]{12,}={0,2}\b")
_SAFE_PUBLIC_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_CREDENTIAL_LIKE_PREFIXES = ("ghp_", "github_pat_", "sk_", "hny_", "aws_", "AKIA")


class NimbusInfoNCEError(ValueError):
    """Raised when offline NIMBUS InfoNCE training or evaluation fails."""


class NimbusInfoNCEEvalFormat(StrEnum):
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass(frozen=True)
class NimbusInfoNCERunConfig:
    max_weight: int
    weight_step: int

    def __post_init__(self) -> None:
        if self.max_weight < 0:
            raise NimbusInfoNCEError("max_weight must be non-negative.")
        if self.weight_step < 1:
            raise NimbusInfoNCEError("weight_step must be positive.")


@dataclass(frozen=True)
class NimbusInfoNCEEvalConfig:
    allow_training_eval: bool


@dataclass(frozen=True)
class NimbusInfoNCEModel:
    schema_version: str
    model_id: str
    training_schema_version: str
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    negative_count: int
    positive_context_index: int
    training_record_count: int
    training_split_group_count: int
    source_corpus_sha256: str
    label_distribution: dict[str, int]
    mean_nce_loss_bits: float
    attack_top1_accuracy: float
    mean_estimated_leakage_bits: float
    promotion_status: str
    paper_faithful_learned_critic: bool

    def to_dict(self) -> dict[str, JsonValue]:
        label_distribution: dict[str, JsonValue] = {label: count for label, count in self.label_distribution.items()}
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "training_schema_version": self.training_schema_version,
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "negative_count": self.negative_count,
            "positive_context_index": self.positive_context_index,
            "training_record_count": self.training_record_count,
            "training_split_group_count": self.training_split_group_count,
            "source_corpus_sha256": self.source_corpus_sha256,
            "label_distribution": label_distribution,
            "mean_nce_loss_bits": self.mean_nce_loss_bits,
            "attack_top1_accuracy": self.attack_top1_accuracy,
            "mean_estimated_leakage_bits": self.mean_estimated_leakage_bits,
            "promotion_status": self.promotion_status,
            "paper_faithful_learned_critic": self.paper_faithful_learned_critic,
        }


@dataclass(frozen=True)
class NimbusInfoNCETurnMetric:
    example_id: str
    scenario_name: str
    split_group_key: str
    turn_index: int
    leakage_label: str
    leakage_expected: bool
    leakage_detected: bool
    classification_outcome: str
    target_turn_leakage_bits: float
    positive_probability: float
    nce_loss_bits: float
    estimated_leakage_bits: float
    absolute_error_bits: float
    target_cumulative_leakage_bits: float
    positive_rank: int

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "example_id": self.example_id,
            "scenario_name": self.scenario_name,
            "split_group_key": self.split_group_key,
            "turn_index": self.turn_index,
            "leakage_label": self.leakage_label,
            "leakage_expected": self.leakage_expected,
            "leakage_detected": self.leakage_detected,
            "classification_outcome": self.classification_outcome,
            "target_turn_leakage_bits": self.target_turn_leakage_bits,
            "positive_probability": self.positive_probability,
            "nce_loss_bits": self.nce_loss_bits,
            "estimated_leakage_bits": self.estimated_leakage_bits,
            "absolute_error_bits": self.absolute_error_bits,
            "target_cumulative_leakage_bits": self.target_cumulative_leakage_bits,
            "positive_rank": self.positive_rank,
        }


@dataclass(frozen=True)
class NimbusInfoNCESessionMetric:
    split_group_key: str
    leakage_expected: bool
    leakage_detected: bool
    classification_outcome: str
    turn_count: int
    attack_turn_count: int
    target_cumulative_leakage_bits: float
    estimated_cumulative_leakage_bits: float
    max_estimated_turn_leakage_bits: float

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "split_group_key": self.split_group_key,
            "leakage_expected": self.leakage_expected,
            "leakage_detected": self.leakage_detected,
            "classification_outcome": self.classification_outcome,
            "turn_count": self.turn_count,
            "attack_turn_count": self.attack_turn_count,
            "target_cumulative_leakage_bits": self.target_cumulative_leakage_bits,
            "estimated_cumulative_leakage_bits": self.estimated_cumulative_leakage_bits,
            "max_estimated_turn_leakage_bits": self.max_estimated_turn_leakage_bits,
        }


@dataclass(frozen=True)
class NimbusInfoNCELabelMetric:
    leakage_label: str
    count: int
    top1_accuracy: float
    mean_target_turn_leakage_bits: float
    mean_estimated_leakage_bits: float
    mean_absolute_error_bits: float

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "leakage_label": self.leakage_label,
            "count": self.count,
            "top1_accuracy": self.top1_accuracy,
            "mean_target_turn_leakage_bits": self.mean_target_turn_leakage_bits,
            "mean_estimated_leakage_bits": self.mean_estimated_leakage_bits,
            "mean_absolute_error_bits": self.mean_absolute_error_bits,
        }


@dataclass(frozen=True)
class NimbusInfoNCEEvalReport:
    schema_version: str
    model_id: str
    record_count: int
    split_group_count: int
    eval_corpus_sha256: str
    training_eval_reused: bool
    training_eval_allowed: bool
    attack_top1_accuracy: float | None
    mean_nce_loss_bits: float
    mean_estimated_leakage_bits: float
    mean_absolute_error_bits: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    false_positive_rate: float | None
    false_negative_rate: float | None
    session_true_positive: int
    session_true_negative: int
    session_false_positive: int
    session_false_negative: int
    session_false_positive_rate: float | None
    session_false_negative_rate: float | None
    promotion_status: str
    paper_faithful_learned_critic: bool
    label_metrics: tuple[NimbusInfoNCELabelMetric, ...]
    turn_metrics: tuple[NimbusInfoNCETurnMetric, ...]
    session_metrics: tuple[NimbusInfoNCESessionMetric, ...]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "record_count": self.record_count,
            "split_group_count": self.split_group_count,
            "eval_corpus_sha256": self.eval_corpus_sha256,
            "training_eval_reused": self.training_eval_reused,
            "training_eval_allowed": self.training_eval_allowed,
            "attack_top1_accuracy": self.attack_top1_accuracy,
            "mean_nce_loss_bits": self.mean_nce_loss_bits,
            "mean_estimated_leakage_bits": self.mean_estimated_leakage_bits,
            "mean_absolute_error_bits": self.mean_absolute_error_bits,
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "session_true_positive": self.session_true_positive,
            "session_true_negative": self.session_true_negative,
            "session_false_positive": self.session_false_positive,
            "session_false_negative": self.session_false_negative,
            "session_false_positive_rate": self.session_false_positive_rate,
            "session_false_negative_rate": self.session_false_negative_rate,
            "promotion_status": self.promotion_status,
            "paper_faithful_learned_critic": self.paper_faithful_learned_critic,
            "label_metrics": [metric.to_dict() for metric in self.label_metrics],
            "turn_metrics": [metric.to_dict() for metric in self.turn_metrics],
            "session_metrics": [metric.to_dict() for metric in self.session_metrics],
        }


@dataclass(frozen=True)
class NimbusInfoNCEGroupedCVFoldMetric:
    fold_index: int
    heldout_split_group_key: str
    training_record_count: int
    training_split_group_count: int
    eval_record_count: int
    eval_session_count: int
    attack_top1_accuracy: float | None
    mean_nce_loss_bits: float
    mean_estimated_leakage_bits: float
    mean_absolute_error_bits: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    false_positive_rate: float | None
    false_negative_rate: float | None
    session_true_positive: int
    session_true_negative: int
    session_false_positive: int
    session_false_negative: int
    session_false_positive_rate: float | None
    session_false_negative_rate: float | None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "fold_index": self.fold_index,
            "heldout_split_group_key": self.heldout_split_group_key,
            "training_record_count": self.training_record_count,
            "training_split_group_count": self.training_split_group_count,
            "eval_record_count": self.eval_record_count,
            "eval_session_count": self.eval_session_count,
            "attack_top1_accuracy": self.attack_top1_accuracy,
            "mean_nce_loss_bits": self.mean_nce_loss_bits,
            "mean_estimated_leakage_bits": self.mean_estimated_leakage_bits,
            "mean_absolute_error_bits": self.mean_absolute_error_bits,
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "session_true_positive": self.session_true_positive,
            "session_true_negative": self.session_true_negative,
            "session_false_positive": self.session_false_positive,
            "session_false_negative": self.session_false_negative,
            "session_false_positive_rate": self.session_false_positive_rate,
            "session_false_negative_rate": self.session_false_negative_rate,
        }


@dataclass(frozen=True)
class NimbusInfoNCEGroupedCVReport:
    schema_version: str
    model_id: str
    record_count: int
    split_group_count: int
    fold_count: int
    source_corpus_sha256: str
    attack_top1_accuracy: float
    mean_nce_loss_bits: float
    mean_estimated_leakage_bits: float
    mean_absolute_error_bits: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    false_positive_rate: float | None
    false_negative_rate: float | None
    session_true_positive: int
    session_true_negative: int
    session_false_positive: int
    session_false_negative: int
    session_false_positive_rate: float | None
    session_false_negative_rate: float | None
    promotion_status: str
    paper_faithful_learned_critic: bool
    fold_metrics: tuple[NimbusInfoNCEGroupedCVFoldMetric, ...]
    session_metrics: tuple[NimbusInfoNCESessionMetric, ...]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "record_count": self.record_count,
            "split_group_count": self.split_group_count,
            "fold_count": self.fold_count,
            "source_corpus_sha256": self.source_corpus_sha256,
            "attack_top1_accuracy": self.attack_top1_accuracy,
            "mean_nce_loss_bits": self.mean_nce_loss_bits,
            "mean_estimated_leakage_bits": self.mean_estimated_leakage_bits,
            "mean_absolute_error_bits": self.mean_absolute_error_bits,
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "session_true_positive": self.session_true_positive,
            "session_true_negative": self.session_true_negative,
            "session_false_positive": self.session_false_positive,
            "session_false_negative": self.session_false_negative,
            "session_false_positive_rate": self.session_false_positive_rate,
            "session_false_negative_rate": self.session_false_negative_rate,
            "promotion_status": self.promotion_status,
            "paper_faithful_learned_critic": self.paper_faithful_learned_critic,
            "fold_metrics": [metric.to_dict() for metric in self.fold_metrics],
            "session_metrics": [metric.to_dict() for metric in self.session_metrics],
        }


def train_nimbus_infonce_model(
    records: tuple[NimbusTrainingTurnRecord, ...],
    config: NimbusInfoNCERunConfig,
) -> NimbusInfoNCEModel:
    _validate_training_records(records)
    best_weights = _select_weights(records, config)
    model = _model_from_weights(records, best_weights)
    _validate_model(model)
    return model


def evaluate_nimbus_infonce_model(
    model: NimbusInfoNCEModel,
    records: tuple[NimbusTrainingTurnRecord, ...],
    config: NimbusInfoNCEEvalConfig,
) -> NimbusInfoNCEEvalReport:
    _validate_model(model)
    _validate_eval_records(records)
    eval_corpus_sha256 = _corpus_sha256(records)
    training_eval_reused = eval_corpus_sha256 == model.source_corpus_sha256
    if training_eval_reused and not config.allow_training_eval:
        raise NimbusInfoNCEError(
            "evaluation corpus matches model.source_corpus_sha256; pass allow_training_eval only for training "
            "diagnostics, not holdout or production evidence."
        )
    turn_metrics = tuple(_metric_for_record(model, record) for record in records)
    session_metrics = _session_metrics(turn_metrics)
    counts = _classification_counts(turn_metrics)
    session_counts = _session_classification_counts(session_metrics)
    positive_count = counts["true_positive"] + counts["false_negative"]
    negative_count = counts["true_negative"] + counts["false_positive"]
    session_positive_count = session_counts["true_positive"] + session_counts["false_negative"]
    session_negative_count = session_counts["true_negative"] + session_counts["false_positive"]
    return NimbusInfoNCEEvalReport(
        schema_version=NIMBUS_INFONCE_EVAL_SCHEMA_VERSION,
        model_id=model.model_id,
        record_count=len(records),
        split_group_count=len({record.split_group_key for record in records}),
        eval_corpus_sha256=eval_corpus_sha256,
        training_eval_reused=training_eval_reused,
        training_eval_allowed=config.allow_training_eval,
        attack_top1_accuracy=_attack_top1_accuracy_or_none(turn_metrics),
        mean_nce_loss_bits=_mean(tuple(metric.nce_loss_bits for metric in turn_metrics)),
        mean_estimated_leakage_bits=_mean(tuple(metric.estimated_leakage_bits for metric in turn_metrics)),
        mean_absolute_error_bits=_mean(tuple(metric.absolute_error_bits for metric in turn_metrics)),
        true_positive=counts["true_positive"],
        true_negative=counts["true_negative"],
        false_positive=counts["false_positive"],
        false_negative=counts["false_negative"],
        false_positive_rate=_safe_rate(counts["false_positive"], negative_count),
        false_negative_rate=_safe_rate(counts["false_negative"], positive_count),
        session_true_positive=session_counts["true_positive"],
        session_true_negative=session_counts["true_negative"],
        session_false_positive=session_counts["false_positive"],
        session_false_negative=session_counts["false_negative"],
        session_false_positive_rate=_safe_rate(session_counts["false_positive"], session_negative_count),
        session_false_negative_rate=_safe_rate(session_counts["false_negative"], session_positive_count),
        promotion_status=NIMBUS_INFONCE_PROMOTION_STATUS,
        paper_faithful_learned_critic=False,
        label_metrics=_label_metrics(turn_metrics),
        turn_metrics=turn_metrics,
        session_metrics=session_metrics,
    )


def grouped_cross_validate_nimbus_infonce(
    records: tuple[NimbusTrainingTurnRecord, ...],
    config: NimbusInfoNCERunConfig,
) -> NimbusInfoNCEGroupedCVReport:
    _validate_training_records(records)
    split_group_keys = tuple(sorted({record.split_group_key for record in records}))
    if len(split_group_keys) < 2:
        raise NimbusInfoNCEError("grouped cross-validation requires at least two split groups.")
    fold_metrics: list[NimbusInfoNCEGroupedCVFoldMetric] = []
    all_turn_metrics: list[NimbusInfoNCETurnMetric] = []
    all_session_metrics: list[NimbusInfoNCESessionMetric] = []
    total_counts = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    total_session_counts = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    for fold_index, heldout_split_group_key in enumerate(split_group_keys):
        training_records = tuple(record for record in records if record.split_group_key != heldout_split_group_key)
        eval_records = tuple(record for record in records if record.split_group_key == heldout_split_group_key)
        model = train_nimbus_infonce_model(training_records, config)
        report = evaluate_nimbus_infonce_model(
            model,
            eval_records,
            NimbusInfoNCEEvalConfig(allow_training_eval=False),
        )
        for key in total_counts:
            total_counts[key] += getattr(report, key)
            total_session_counts[key] += getattr(report, f"session_{key}")
        all_turn_metrics.extend(report.turn_metrics)
        all_session_metrics.extend(report.session_metrics)
        fold_metrics.append(
            NimbusInfoNCEGroupedCVFoldMetric(
                fold_index=fold_index,
                heldout_split_group_key=heldout_split_group_key,
                training_record_count=len(training_records),
                training_split_group_count=len({record.split_group_key for record in training_records}),
                eval_record_count=len(eval_records),
                eval_session_count=len(report.session_metrics),
                attack_top1_accuracy=report.attack_top1_accuracy,
                mean_nce_loss_bits=report.mean_nce_loss_bits,
                mean_estimated_leakage_bits=report.mean_estimated_leakage_bits,
                mean_absolute_error_bits=report.mean_absolute_error_bits,
                true_positive=report.true_positive,
                true_negative=report.true_negative,
                false_positive=report.false_positive,
                false_negative=report.false_negative,
                false_positive_rate=report.false_positive_rate,
                false_negative_rate=report.false_negative_rate,
                session_true_positive=report.session_true_positive,
                session_true_negative=report.session_true_negative,
                session_false_positive=report.session_false_positive,
                session_false_negative=report.session_false_negative,
                session_false_positive_rate=report.session_false_positive_rate,
                session_false_negative_rate=report.session_false_negative_rate,
            )
        )
    positive_count = total_counts["true_positive"] + total_counts["false_negative"]
    negative_count = total_counts["true_negative"] + total_counts["false_positive"]
    session_positive_count = total_session_counts["true_positive"] + total_session_counts["false_negative"]
    session_negative_count = total_session_counts["true_negative"] + total_session_counts["false_positive"]
    return NimbusInfoNCEGroupedCVReport(
        schema_version=NIMBUS_INFONCE_GROUPED_CV_SCHEMA_VERSION,
        model_id=NIMBUS_INFONCE_MODEL_ID,
        record_count=len(records),
        split_group_count=len(split_group_keys),
        fold_count=len(fold_metrics),
        source_corpus_sha256=_corpus_sha256(records),
        attack_top1_accuracy=_require_present_float(
            _attack_top1_accuracy_or_none(tuple(all_turn_metrics)),
            "grouped cross-validation attack_top1_accuracy",
        ),
        mean_nce_loss_bits=_mean(tuple(metric.nce_loss_bits for metric in all_turn_metrics)),
        mean_estimated_leakage_bits=_mean(tuple(metric.estimated_leakage_bits for metric in all_turn_metrics)),
        mean_absolute_error_bits=_mean(tuple(metric.absolute_error_bits for metric in all_turn_metrics)),
        true_positive=total_counts["true_positive"],
        true_negative=total_counts["true_negative"],
        false_positive=total_counts["false_positive"],
        false_negative=total_counts["false_negative"],
        false_positive_rate=_safe_rate(total_counts["false_positive"], negative_count),
        false_negative_rate=_safe_rate(total_counts["false_negative"], positive_count),
        session_true_positive=total_session_counts["true_positive"],
        session_true_negative=total_session_counts["true_negative"],
        session_false_positive=total_session_counts["false_positive"],
        session_false_negative=total_session_counts["false_negative"],
        session_false_positive_rate=_safe_rate(total_session_counts["false_positive"], session_negative_count),
        session_false_negative_rate=_safe_rate(total_session_counts["false_negative"], session_positive_count),
        promotion_status=NIMBUS_INFONCE_PROMOTION_STATUS,
        paper_faithful_learned_critic=False,
        fold_metrics=tuple(fold_metrics),
        session_metrics=tuple(all_session_metrics),
    )


def save_nimbus_infonce_model(path: Path, model: NimbusInfoNCEModel) -> None:
    _validate_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_nimbus_infonce_model(path: Path) -> NimbusInfoNCEModel:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    model = _model_from_mapping(_as_mapping(decoded, str(path)))
    _validate_model(model)
    return model


def save_nimbus_infonce_eval_report(path: Path, report: NimbusInfoNCEEvalReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_nimbus_infonce_grouped_cv_report(path: Path, report: NimbusInfoNCEGroupedCVReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_nimbus_infonce_markdown_report(path: Path, report: NimbusInfoNCEEvalReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_nimbus_infonce_markdown(report), encoding="utf-8")


def render_nimbus_infonce_markdown(report: NimbusInfoNCEEvalReport) -> str:
    lines = [
        "# NIMBUS InfoNCE Evaluation",
        "",
        f"- Model: `{report.model_id}`",
        f"- Records: `{report.record_count}`",
        f"- Split groups: `{report.split_group_count}`",
        f"- Eval corpus SHA-256: `{report.eval_corpus_sha256}`",
        f"- Training eval reused: `{str(report.training_eval_reused).lower()}`",
        f"- Training eval allowed: `{str(report.training_eval_allowed).lower()}`",
        f"- Attack top-1 accuracy: `{_format_optional_float(report.attack_top1_accuracy)}`",
        f"- Mean NCE loss bits: `{_format_float(report.mean_nce_loss_bits)}`",
        f"- Mean estimated leakage bits: `{_format_float(report.mean_estimated_leakage_bits)}`",
        f"- Mean absolute error bits: `{_format_float(report.mean_absolute_error_bits)}`",
        f"- False positive rate: `{_format_optional_float(report.false_positive_rate)}`",
        f"- False negative rate: `{_format_optional_float(report.false_negative_rate)}`",
        f"- Session false positive rate: `{_format_optional_float(report.session_false_positive_rate)}`",
        f"- Session false negative rate: `{_format_optional_float(report.session_false_negative_rate)}`",
        f"- Promotion status: `{report.promotion_status}`",
        f"- Paper-faithful learned critic: `{str(report.paper_faithful_learned_critic).lower()}`",
        "",
        "| Label | Count | Top-1 accuracy | Mean target bits | Mean estimated bits | MAE bits |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in report.label_metrics:
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_cell(metric.leakage_label),
                    str(metric.count),
                    _format_float(metric.top1_accuracy),
                    _format_float(metric.mean_target_turn_leakage_bits),
                    _format_float(metric.mean_estimated_leakage_bits),
                    _format_float(metric.mean_absolute_error_bits),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_train_args(argv: Sequence[str]) -> tuple[Path, Path, NimbusInfoNCERunConfig]:
    parser = argparse.ArgumentParser(description="Train an offline lexical NIMBUS InfoNCE critic artifact.")
    parser.add_argument("--input", required=True, type=Path, help="Path to nimbus-training-turn/v0 JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Path for the trained model JSON artifact.")
    parser.add_argument("--max-weight", type=int, required=False, default=4, help="Maximum integer feature weight.")
    parser.add_argument("--weight-step", type=int, required=False, default=1, help="Integer feature weight step.")
    args = parser.parse_args(argv)
    return args.input, args.output, NimbusInfoNCERunConfig(max_weight=args.max_weight, weight_step=args.weight_step)


def parse_eval_args(
    argv: Sequence[str],
) -> tuple[Path, Path, Path, NimbusInfoNCEEvalFormat, NimbusInfoNCEEvalConfig, Path | None, NimbusInfoNCERunConfig]:
    parser = argparse.ArgumentParser(description="Evaluate an offline lexical NIMBUS InfoNCE critic artifact.")
    parser.add_argument("--input", required=True, type=Path, help="Path to nimbus-training-turn/v0 JSONL.")
    parser.add_argument("--model", required=True, type=Path, help="Path to the trained model JSON artifact.")
    parser.add_argument("--output", required=True, type=Path, help="Path for the evaluation report.")
    parser.add_argument(
        "--allow-training-eval",
        action="store_true",
        help="Allow evaluation on the same corpus hash used for training. Use only for diagnostics.",
    )
    parser.add_argument(
        "--grouped-cv-output",
        required=False,
        type=Path,
        help="Optional JSON path for leave-one-split-group-out cross-validation evidence.",
    )
    parser.add_argument("--max-weight", type=int, required=False, default=4, help="Maximum integer feature weight.")
    parser.add_argument("--weight-step", type=int, required=False, default=1, help="Integer feature weight step.")
    parser.add_argument(
        "--format",
        choices=tuple(item.value for item in NimbusInfoNCEEvalFormat),
        required=False,
        default=NimbusInfoNCEEvalFormat.JSON.value,
        help="Output format for the evaluation report.",
    )
    args = parser.parse_args(argv)
    return (
        args.input,
        args.model,
        args.output,
        NimbusInfoNCEEvalFormat(args.format),
        NimbusInfoNCEEvalConfig(allow_training_eval=bool(args.allow_training_eval)),
        args.grouped_cv_output,
        NimbusInfoNCERunConfig(max_weight=args.max_weight, weight_step=args.weight_step),
    )


def main_train() -> None:
    try:
        input_path, output_path, config = parse_train_args(tuple(sys.argv[1:]))
        records = read_nimbus_training_records_jsonl(input_path)
        model = train_nimbus_infonce_model(records, config)
        save_nimbus_infonce_model(output_path, model)
    except (NimbusInfoNCEError, NimbusTrainingCorpusError) as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc


def main_eval() -> None:
    try:
        parsed_args = parse_eval_args(tuple(sys.argv[1:]))
        input_path, model_path, output_path, output_format, eval_config, grouped_cv_output, run_config = parsed_args
        records = read_nimbus_training_records_jsonl(input_path)
        model = load_nimbus_infonce_model(model_path)
        report = evaluate_nimbus_infonce_model(model, records, eval_config)
        if output_format == NimbusInfoNCEEvalFormat.JSON:
            save_nimbus_infonce_eval_report(output_path, report)
        elif output_format == NimbusInfoNCEEvalFormat.MARKDOWN:
            save_nimbus_infonce_markdown_report(output_path, report)
        else:
            raise NimbusInfoNCEError(f"Unsupported output format '{output_format}'.")
        if grouped_cv_output is not None:
            grouped_cv_report = grouped_cross_validate_nimbus_infonce(records, run_config)
            save_nimbus_infonce_grouped_cv_report(grouped_cv_output, grouped_cv_report)
    except (NimbusInfoNCEError, NimbusTrainingCorpusError) as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc


def _select_weights(
    records: tuple[NimbusTrainingTurnRecord, ...],
    config: NimbusInfoNCERunConfig,
) -> tuple[float, ...]:
    candidate_values = tuple(range(0, config.max_weight + 1, config.weight_step))
    if len(candidate_values) == 0:
        raise NimbusInfoNCEError("weight search space must not be empty.")
    best_weights: tuple[float, ...] | None = None
    best_loss: float | None = None
    for weights in itertools.product(candidate_values, repeat=len(NIMBUS_INFONCE_FEATURE_NAMES)):
        float_weights = tuple(float(weight) for weight in weights)
        loss = _mean_nce_loss_bits(records, float_weights)
        if _is_better_weight_candidate(loss, float_weights, best_loss, best_weights):
            best_loss = loss
            best_weights = float_weights
    if best_weights is None:
        raise NimbusInfoNCEError("failed to select InfoNCE weights.")
    return best_weights


def _is_better_weight_candidate(
    loss: float,
    weights: tuple[float, ...],
    best_loss: float | None,
    best_weights: tuple[float, ...] | None,
) -> bool:
    if best_loss is None or best_weights is None:
        return True
    if loss < best_loss:
        return True
    return loss == best_loss and sum(weights) < sum(best_weights)


def _model_from_weights(
    records: tuple[NimbusTrainingTurnRecord, ...],
    weights: tuple[float, ...],
) -> NimbusInfoNCEModel:
    probe_model = NimbusInfoNCEModel(
        schema_version=NIMBUS_INFONCE_MODEL_SCHEMA_VERSION,
        model_id=NIMBUS_INFONCE_MODEL_ID,
        training_schema_version=NIMBUS_TRAINING_SCHEMA_VERSION,
        feature_names=NIMBUS_INFONCE_FEATURE_NAMES,
        weights=weights,
        negative_count=INFO_NCE_NEGATIVE_COUNT,
        positive_context_index=0,
        training_record_count=len(records),
        training_split_group_count=len({record.split_group_key for record in records}),
        source_corpus_sha256=_corpus_sha256(records),
        label_distribution=_label_distribution(records),
        mean_nce_loss_bits=0.0,
        attack_top1_accuracy=0.0,
        mean_estimated_leakage_bits=0.0,
        promotion_status=NIMBUS_INFONCE_PROMOTION_STATUS,
        paper_faithful_learned_critic=False,
    )
    report = evaluate_nimbus_infonce_model(
        probe_model,
        records,
        NimbusInfoNCEEvalConfig(allow_training_eval=True),
    )
    return NimbusInfoNCEModel(
        schema_version=NIMBUS_INFONCE_MODEL_SCHEMA_VERSION,
        model_id=NIMBUS_INFONCE_MODEL_ID,
        training_schema_version=NIMBUS_TRAINING_SCHEMA_VERSION,
        feature_names=NIMBUS_INFONCE_FEATURE_NAMES,
        weights=weights,
        negative_count=INFO_NCE_NEGATIVE_COUNT,
        positive_context_index=0,
        training_record_count=len(records),
        training_split_group_count=len({record.split_group_key for record in records}),
        source_corpus_sha256=_corpus_sha256(records),
        label_distribution=_label_distribution(records),
        mean_nce_loss_bits=report.mean_nce_loss_bits,
        attack_top1_accuracy=_require_present_float(report.attack_top1_accuracy, "training attack_top1_accuracy"),
        mean_estimated_leakage_bits=report.mean_estimated_leakage_bits,
        promotion_status=NIMBUS_INFONCE_PROMOTION_STATUS,
        paper_faithful_learned_critic=False,
    )


def _metric_for_record(model: NimbusInfoNCEModel, record: NimbusTrainingTurnRecord) -> NimbusInfoNCETurnMetric:
    scores = _candidate_scores(model, record)
    probability = _positive_probability(scores)
    nce_loss_bits = -math.log2(probability)
    estimated_leakage_bits = max(0.0, math.log2(len(scores)) - nce_loss_bits)
    if estimated_leakage_bits < 1e-12:
        estimated_leakage_bits = 0.0
    leakage_expected = record.leakage_label != NimbusLeakageLabel.BENIGN
    leakage_detected = estimated_leakage_bits > 0.0
    return NimbusInfoNCETurnMetric(
        example_id=record.example_id,
        scenario_name=record.scenario_name,
        split_group_key=record.split_group_key,
        turn_index=record.turn_index,
        leakage_label=record.leakage_label.value,
        leakage_expected=leakage_expected,
        leakage_detected=leakage_detected,
        classification_outcome=_classification_outcome(expected=leakage_expected, detected=leakage_detected),
        target_turn_leakage_bits=record.target_turn_leakage_bits,
        positive_probability=probability,
        nce_loss_bits=nce_loss_bits,
        estimated_leakage_bits=estimated_leakage_bits,
        absolute_error_bits=abs(estimated_leakage_bits - record.target_turn_leakage_bits),
        target_cumulative_leakage_bits=record.target_cumulative_leakage_bits,
        positive_rank=_positive_rank(scores),
    )


def _candidate_scores(model: NimbusInfoNCEModel, record: NimbusTrainingTurnRecord) -> tuple[float, ...]:
    contexts = (record.true_secret_context, *record.negative_secret_contexts)
    return tuple(_score_context(model, record, context) for context in contexts)


def _score_context(
    model: NimbusInfoNCEModel,
    record: NimbusTrainingTurnRecord,
    context: NimbusSecretContext,
) -> float:
    features = _features_for_context(record, context)
    return sum(weight * feature for weight, feature in zip(model.weights, features, strict=True))


def _features_for_context(
    record: NimbusTrainingTurnRecord,
    context: NimbusSecretContext,
) -> tuple[float, float, float]:
    context_tokens = _tokens(context.context_text)
    output_tokens = _tokens(record.output_text)
    decoded_tokens = _decoded_output_tokens(record.output_text)
    state_tokens = _tokens(" ".join(message.content for message in record.state_messages))
    return (
        float(len(output_tokens & context_tokens)),
        float(len(decoded_tokens & context_tokens)),
        float(len(state_tokens & context_tokens)),
    )


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _decoded_output_tokens(text: str) -> set[str]:
    decoded_texts: list[str] = []
    for candidate in _BASE64_CANDIDATE_PATTERN.findall(text):
        decoded = _decode_base64_candidate(candidate)
        if decoded is not None:
            decoded_texts.append(decoded)
    return _tokens(" ".join(decoded_texts))


def _decode_base64_candidate(candidate: str) -> str | None:
    padded = candidate + ("=" * ((4 - len(candidate) % 4) % 4))
    try:
        decoded = base64.b64decode(padded.encode("ascii"), validate=True)
        text = decoded.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if not _looks_like_text(text):
        return None
    return text


def _looks_like_text(text: str) -> bool:
    if text == "":
        return False
    printable_count = sum(1 for character in text if character.isprintable())
    return printable_count / len(text) >= 0.9


def _positive_probability(scores: tuple[float, ...]) -> float:
    max_score = max(scores)
    exp_scores = tuple(math.exp(score - max_score) for score in scores)
    denominator = sum(exp_scores)
    if denominator <= 0.0:
        raise NimbusInfoNCEError("softmax denominator must be positive.")
    return exp_scores[0] / denominator


def _positive_rank(scores: tuple[float, ...]) -> int:
    positive_score = scores[0]
    stronger_or_tied_count = sum(1 for score in scores[1:] if score >= positive_score)
    return stronger_or_tied_count + 1


def _mean_nce_loss_bits(records: tuple[NimbusTrainingTurnRecord, ...], weights: tuple[float, ...]) -> float:
    model = NimbusInfoNCEModel(
        schema_version=NIMBUS_INFONCE_MODEL_SCHEMA_VERSION,
        model_id=NIMBUS_INFONCE_MODEL_ID,
        training_schema_version=NIMBUS_TRAINING_SCHEMA_VERSION,
        feature_names=NIMBUS_INFONCE_FEATURE_NAMES,
        weights=weights,
        negative_count=INFO_NCE_NEGATIVE_COUNT,
        positive_context_index=0,
        training_record_count=len(records),
        training_split_group_count=len({record.split_group_key for record in records}),
        source_corpus_sha256=_corpus_sha256(records),
        label_distribution=_label_distribution(records),
        mean_nce_loss_bits=0.0,
        attack_top1_accuracy=0.0,
        mean_estimated_leakage_bits=0.0,
        promotion_status=NIMBUS_INFONCE_PROMOTION_STATUS,
        paper_faithful_learned_critic=False,
    )
    losses = tuple(_metric_for_record(model, record).nce_loss_bits for record in records)
    return _mean(losses)


def _attack_top1_accuracy_or_none(metrics: tuple[NimbusInfoNCETurnMetric, ...]) -> float | None:
    attack_metrics = tuple(metric for metric in metrics if metric.leakage_expected)
    if len(attack_metrics) == 0:
        return None
    correct_count = sum(1 for metric in attack_metrics if metric.positive_rank == 1)
    return correct_count / len(attack_metrics)


def _label_metrics(metrics: tuple[NimbusInfoNCETurnMetric, ...]) -> tuple[NimbusInfoNCELabelMetric, ...]:
    labels = tuple(sorted({metric.leakage_label for metric in metrics}))
    grouped_metrics: list[NimbusInfoNCELabelMetric] = []
    for label in labels:
        label_turns = tuple(metric for metric in metrics if metric.leakage_label == label)
        grouped_metrics.append(
            NimbusInfoNCELabelMetric(
                leakage_label=label,
                count=len(label_turns),
                top1_accuracy=_top1_accuracy(label_turns),
                mean_target_turn_leakage_bits=_mean(tuple(metric.target_turn_leakage_bits for metric in label_turns)),
                mean_estimated_leakage_bits=_mean(tuple(metric.estimated_leakage_bits for metric in label_turns)),
                mean_absolute_error_bits=_mean(tuple(metric.absolute_error_bits for metric in label_turns)),
            )
        )
    return tuple(grouped_metrics)


def _top1_accuracy(metrics: tuple[NimbusInfoNCETurnMetric, ...]) -> float:
    if len(metrics) == 0:
        raise NimbusInfoNCEError("cannot compute top1 accuracy for an empty metric group.")
    correct_count = sum(1 for metric in metrics if metric.positive_rank == 1)
    return correct_count / len(metrics)


def _classification_counts(metrics: tuple[NimbusInfoNCETurnMetric, ...]) -> dict[str, int]:
    counts = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    for metric in metrics:
        counts[metric.classification_outcome] += 1
    return counts


def _session_metrics(metrics: tuple[NimbusInfoNCETurnMetric, ...]) -> tuple[NimbusInfoNCESessionMetric, ...]:
    session_metrics: list[NimbusInfoNCESessionMetric] = []
    for split_group_key in sorted({metric.split_group_key for metric in metrics}):
        session_turns = tuple(
            sorted(
                (metric for metric in metrics if metric.split_group_key == split_group_key),
                key=lambda metric: metric.turn_index,
            )
        )
        if len(session_turns) == 0:
            raise NimbusInfoNCEError(f"split group '{split_group_key}' has no turn metrics.")
        leakage_expected = any(metric.leakage_expected for metric in session_turns)
        estimated_cumulative_leakage_bits = sum(metric.estimated_leakage_bits for metric in session_turns)
        leakage_detected = estimated_cumulative_leakage_bits > 0.0
        session_metrics.append(
            NimbusInfoNCESessionMetric(
                split_group_key=split_group_key,
                leakage_expected=leakage_expected,
                leakage_detected=leakage_detected,
                classification_outcome=_classification_outcome(
                    expected=leakage_expected,
                    detected=leakage_detected,
                ),
                turn_count=len(session_turns),
                attack_turn_count=sum(1 for metric in session_turns if metric.leakage_expected),
                target_cumulative_leakage_bits=max(
                    metric.target_cumulative_leakage_bits for metric in session_turns
                ),
                estimated_cumulative_leakage_bits=estimated_cumulative_leakage_bits,
                max_estimated_turn_leakage_bits=max(metric.estimated_leakage_bits for metric in session_turns),
            )
        )
    return tuple(session_metrics)


def _session_classification_counts(metrics: tuple[NimbusInfoNCESessionMetric, ...]) -> dict[str, int]:
    counts = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    for metric in metrics:
        counts[metric.classification_outcome] += 1
    return counts


def _classification_outcome(expected: bool, detected: bool) -> str:
    if expected and detected:
        return "true_positive"
    if expected and not detected:
        return "false_negative"
    if not expected and detected:
        return "false_positive"
    return "true_negative"


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _mean(values: tuple[float, ...]) -> float:
    if len(values) == 0:
        raise NimbusInfoNCEError("cannot compute mean of an empty sequence.")
    return sum(values) / len(values)


def _require_present_float(value: float | None, field_name: str) -> float:
    if value is None:
        raise NimbusInfoNCEError(f"{field_name} must be available.")
    return value


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return _format_float(value)


def _markdown_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|")


def _validate_training_records(records: tuple[NimbusTrainingTurnRecord, ...]) -> None:
    if len(records) == 0:
        raise NimbusInfoNCEError("records must not be empty.")
    example_ids: set[str] = set()
    has_non_benign = False
    for record in records:
        _validate_training_record(record)
        if record.example_id in example_ids:
            raise NimbusInfoNCEError(f"duplicate example_id '{record.example_id}'.")
        example_ids.add(record.example_id)
        if record.leakage_label != NimbusLeakageLabel.BENIGN:
            has_non_benign = True
        if len(record.negative_secret_contexts) != INFO_NCE_NEGATIVE_COUNT:
            raise NimbusInfoNCEError(f"{record.example_id}: expected {INFO_NCE_NEGATIVE_COUNT} negative contexts.")
    if not has_non_benign:
        raise NimbusInfoNCEError("records must include at least one non-benign leakage example.")


def _validate_eval_records(records: tuple[NimbusTrainingTurnRecord, ...]) -> None:
    if len(records) == 0:
        raise NimbusInfoNCEError("records must not be empty.")
    example_ids: set[str] = set()
    for record in records:
        _validate_training_record(record)
        if record.example_id in example_ids:
            raise NimbusInfoNCEError(f"duplicate example_id '{record.example_id}'.")
        example_ids.add(record.example_id)
        if len(record.negative_secret_contexts) != INFO_NCE_NEGATIVE_COUNT:
            raise NimbusInfoNCEError(f"{record.example_id}: expected {INFO_NCE_NEGATIVE_COUNT} negative contexts.")


def _validate_training_record(record: NimbusTrainingTurnRecord) -> None:
    try:
        validate_nimbus_training_record(record)
    except NimbusTrainingCorpusError as exc:
        raise NimbusInfoNCEError(f"{record.example_id}: {exc}") from exc
    _safe_public_identifier(record.example_id, "example_id")
    _safe_public_identifier(record.scenario_name, "scenario_name")
    _safe_public_identifier(record.split_group_key, "split_group_key")


def _safe_public_identifier(value: str, field_name: str) -> str:
    if _SAFE_PUBLIC_IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise NimbusInfoNCEError(f"{field_name} must be a safe public identifier.")
    if any(value.startswith(prefix) for prefix in _CREDENTIAL_LIKE_PREFIXES):
        raise NimbusInfoNCEError(f"{field_name} must not be credential-shaped.")
    return value


def _validate_model(model: NimbusInfoNCEModel) -> None:
    if model.schema_version != NIMBUS_INFONCE_MODEL_SCHEMA_VERSION:
        raise NimbusInfoNCEError(f"schema_version must be {NIMBUS_INFONCE_MODEL_SCHEMA_VERSION}.")
    if model.model_id != NIMBUS_INFONCE_MODEL_ID:
        raise NimbusInfoNCEError(f"model_id must be {NIMBUS_INFONCE_MODEL_ID}.")
    if model.training_schema_version != NIMBUS_TRAINING_SCHEMA_VERSION:
        raise NimbusInfoNCEError(f"training_schema_version must be {NIMBUS_TRAINING_SCHEMA_VERSION}.")
    if model.feature_names != NIMBUS_INFONCE_FEATURE_NAMES:
        raise NimbusInfoNCEError("feature_names do not match the v0 evaluator.")
    if len(model.weights) != len(model.feature_names):
        raise NimbusInfoNCEError("weights length must match feature_names length.")
    if model.negative_count != INFO_NCE_NEGATIVE_COUNT:
        raise NimbusInfoNCEError(f"negative_count must be {INFO_NCE_NEGATIVE_COUNT}.")
    if model.positive_context_index != 0:
        raise NimbusInfoNCEError("positive_context_index must be 0.")
    if model.training_record_count < 1:
        raise NimbusInfoNCEError("training_record_count must be positive.")
    if model.training_split_group_count < 1:
        raise NimbusInfoNCEError("training_split_group_count must be positive.")
    if not _looks_like_sha256(model.source_corpus_sha256):
        raise NimbusInfoNCEError("source_corpus_sha256 must be a lowercase SHA-256 hex digest.")
    if len(model.label_distribution) == 0:
        raise NimbusInfoNCEError("label_distribution must not be empty.")
    for label, count in model.label_distribution.items():
        if label == "":
            raise NimbusInfoNCEError("label_distribution labels must not be empty.")
        if count < 1:
            raise NimbusInfoNCEError("label_distribution counts must be positive.")
    _validate_probability(model.attack_top1_accuracy, "attack_top1_accuracy")
    _validate_non_negative_finite(model.mean_nce_loss_bits, "mean_nce_loss_bits")
    _validate_non_negative_finite(model.mean_estimated_leakage_bits, "mean_estimated_leakage_bits")
    for weight in model.weights:
        _validate_non_negative_finite(weight, "weights entry")
    if model.promotion_status != NIMBUS_INFONCE_PROMOTION_STATUS:
        raise NimbusInfoNCEError(f"promotion_status must be {NIMBUS_INFONCE_PROMOTION_STATUS}.")
    if model.paper_faithful_learned_critic:
        raise NimbusInfoNCEError("offline lexical InfoNCE scaffold must not claim paper-faithful learned critic.")


def _model_from_mapping(record: Mapping[str, object]) -> NimbusInfoNCEModel:
    return NimbusInfoNCEModel(
        schema_version=_required_string(record, "schema_version"),
        model_id=_required_string(record, "model_id"),
        training_schema_version=_required_string(record, "training_schema_version"),
        feature_names=tuple(
            _required_string_value(item, "feature_names item") for item in _required_sequence(record, "feature_names")
        ),
        weights=tuple(_required_float_value(item, "weights item") for item in _required_sequence(record, "weights")),
        negative_count=_required_int(record, "negative_count"),
        positive_context_index=_required_int(record, "positive_context_index"),
        training_record_count=_required_int(record, "training_record_count"),
        training_split_group_count=_required_int(record, "training_split_group_count"),
        source_corpus_sha256=_required_string(record, "source_corpus_sha256"),
        label_distribution=_label_distribution_from_mapping(_required_mapping(record, "label_distribution")),
        mean_nce_loss_bits=_required_float(record, "mean_nce_loss_bits"),
        attack_top1_accuracy=_required_float(record, "attack_top1_accuracy"),
        mean_estimated_leakage_bits=_required_float(record, "mean_estimated_leakage_bits"),
        promotion_status=_required_string(record, "promotion_status"),
        paper_faithful_learned_critic=_required_bool(record, "paper_faithful_learned_critic"),
    )


def _corpus_sha256(records: tuple[NimbusTrainingTurnRecord, ...]) -> str:
    payload = json.dumps(
        [record.to_dict() for record in records],
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _label_distribution(records: tuple[NimbusTrainingTurnRecord, ...]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for record in records:
        label = record.leakage_label.value
        distribution[label] = distribution.get(label, 0) + 1
    return dict(sorted(distribution.items()))


def _label_distribution_from_mapping(record: Mapping[str, object]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for label, raw_count in record.items():
        if not isinstance(label, str):
            raise NimbusInfoNCEError("label_distribution labels must be strings.")
        if not isinstance(raw_count, int) or isinstance(raw_count, bool):
            raise NimbusInfoNCEError("label_distribution counts must be integers.")
        distribution[label] = raw_count
    return distribution


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_probability(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise NimbusInfoNCEError(f"{field_name} must be in [0.0, 1.0].")


def _validate_non_negative_finite(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise NimbusInfoNCEError(f"{field_name} must be finite and non-negative.")


def _required_sequence(record: Mapping[str, object], field_name: str) -> tuple[object, ...]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise NimbusInfoNCEError(f"{field_name} must be a list.")
    return tuple(value)


def _required_mapping(record: Mapping[str, object], field_name: str) -> Mapping[str, object]:
    return _as_mapping(record.get(field_name), field_name)


def _required_string(record: Mapping[str, object], field_name: str) -> str:
    return _required_string_value(record.get(field_name), field_name)


def _required_string_value(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise NimbusInfoNCEError(f"{field_name} must be a string.")
    return value


def _required_bool(record: Mapping[str, object], field_name: str) -> bool:
    value = record.get(field_name)
    if not isinstance(value, bool):
        raise NimbusInfoNCEError(f"{field_name} must be a boolean.")
    return value


def _required_int(record: Mapping[str, object], field_name: str) -> int:
    value = record.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise NimbusInfoNCEError(f"{field_name} must be an integer.")
    return value


def _required_float(record: Mapping[str, object], field_name: str) -> float:
    return _required_float_value(record.get(field_name), field_name)


def _required_float_value(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise NimbusInfoNCEError(f"{field_name} must be numeric.")
    return float(value)


def _as_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise NimbusInfoNCEError(f"{context} must be an object.")
    return cast(Mapping[str, object], value)
