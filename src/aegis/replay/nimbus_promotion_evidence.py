from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from aegis.core.contracts import JsonValue

NIMBUS_PROMOTION_EVIDENCE_SCHEMA_VERSION = "aegis.nimbus_promotion_evidence/v0"
_NOT_PROMOTABLE_STATUS = "deterministic_beta_active_learned_not_promotable"
_RECOMMENDED_RUNTIME_CRITIC = "deterministic_canary_beta"
_TRAINING_MANIFEST_SCHEMA_VERSION = "aegis.nimbus_training_manifest/v1"
_DETERMINISTIC_EVAL_SCHEMA_VERSION = "aegis.nimbus_eval/v1"
_INFONCE_MODEL_SCHEMA_VERSION = "aegis.nimbus_infonce_model/v0"
_INFONCE_GROUPED_CV_SCHEMA_VERSION = "aegis.nimbus_infonce_grouped_cv/v0"
_INFONCE_EVAL_SCHEMA_VERSION = "aegis.nimbus_infonce_eval/v0"
_RUNTIME_BETA_EVAL_SCHEMA_VERSION = "aegis.nimbus_runtime_beta_eval/v0"
_PAPER_REFERENCE_SESSION_COUNT = 50
_PROMOTION_MAX_TURN_FALSE_RATE = 0.05
_PROMOTION_MAX_SESSION_FALSE_RATE = 0.0
_PROMOTION_REQUIRED_LIVE_GATEWAY_SAMPLES = 6


class NimbusPromotionEvidenceError(ValueError):
    """Raised when NIMBUS promotion evidence inputs are missing or malformed."""


@dataclass(frozen=True)
class NimbusPromotionEvidenceConfig:
    deterministic_eval_path: Path
    calibration_manifest_path: Path
    sealed_manifest_path: Path
    infonce_model_path: Path
    grouped_cv_path: Path
    sealed_holdout_path: Path
    gateway_smoke_path: Path
    runtime_beta_eval_path: Path | None


def build_nimbus_promotion_evidence_report(config: NimbusPromotionEvidenceConfig) -> dict[str, JsonValue]:
    deterministic_eval = _read_json_mapping(config.deterministic_eval_path)
    calibration_manifest = _read_json_mapping(config.calibration_manifest_path)
    sealed_manifest = _read_json_mapping(config.sealed_manifest_path)
    infonce_model = _read_json_mapping(config.infonce_model_path)
    grouped_cv = _read_json_mapping(config.grouped_cv_path)
    sealed_holdout = _read_json_mapping(config.sealed_holdout_path)
    gateway_smoke = _read_json_mapping(config.gateway_smoke_path)
    runtime_beta_eval = _optional_runtime_beta_eval(config.runtime_beta_eval_path)

    _require_schema(deterministic_eval, _DETERMINISTIC_EVAL_SCHEMA_VERSION, "deterministic_eval")
    _require_schema(calibration_manifest, _TRAINING_MANIFEST_SCHEMA_VERSION, "calibration_manifest")
    _require_schema(sealed_manifest, _TRAINING_MANIFEST_SCHEMA_VERSION, "sealed_manifest")
    _require_schema(infonce_model, _INFONCE_MODEL_SCHEMA_VERSION, "infonce_model")
    _require_schema(grouped_cv, _INFONCE_GROUPED_CV_SCHEMA_VERSION, "grouped_cv")
    _require_schema(sealed_holdout, _INFONCE_EVAL_SCHEMA_VERSION, "sealed_holdout")

    deterministic_metrics = _deterministic_metrics(deterministic_eval)
    grouped_metrics = _learned_metrics(grouped_cv)
    sealed_metrics = _learned_metrics(sealed_holdout)
    runtime_beta_metrics = _runtime_beta_metrics(runtime_beta_eval)
    gateway_evidence = _gateway_evidence(gateway_smoke)
    corpus_evidence = _corpus_evidence(calibration_manifest, sealed_manifest)
    comparison = _comparison(
        deterministic_metrics,
        grouped_metrics,
        sealed_metrics,
        runtime_beta_metrics,
        gateway_evidence,
    )
    checklist = _checklist(
        corpus_evidence,
        grouped_metrics,
        sealed_metrics,
        runtime_beta_metrics,
        comparison,
        gateway_evidence,
    )
    checklist_summary = _checklist_summary(checklist)
    promotion_gates = _promotion_gates(
        grouped_metrics=grouped_metrics,
        sealed_metrics=sealed_metrics,
        runtime_beta_metrics=runtime_beta_metrics,
        gateway_evidence=gateway_evidence,
        comparison=comparison,
        checklist=checklist,
    )
    promotion_decision = _promotion_decision(promotion_gates)
    missing = _missing_before_promotion(checklist)
    return {
        "schema_version": NIMBUS_PROMOTION_EVIDENCE_SCHEMA_VERSION,
        "component": "nimbus",
        "paper_source": (
            "Research/2606.04141v1.pdf and repo-derived AIS requirements in "
            "Proposal/aegis-project-plan.md; this report evaluates NIMBUS against the current repo checklist."
        ),
        "promotion_status": _NOT_PROMOTABLE_STATUS,
        "promotion_eligible": False,
        "promote_learned_runtime": False,
        "promote_hybrid_runtime": False,
        "keep_deterministic_default": True,
        "reject_learned_runtime": True,
        "paper_faithful_learned_critic": False,
        "recommended_runtime_critic": _RECOMMENDED_RUNTIME_CRITIC,
        "summary": (
            "The learned InfoNCE NIMBUS path now has grouped-CV, sealed-holdout, and in-process runtime-adapter "
            "evidence with low sealed/runtime beta FP and FN rates plus live learned-gateway smoke FN/FP evidence, "
            "but it is still a small lexical scaffold with no common live head-to-head corpus, no latency gate "
            "evidence, no hybrid-policy evaluation, and no promoted runtime manifest. Reject learned promotion and "
            "keep deterministic canary NIMBUS active."
        ),
        "artifact_hashes": _artifact_hashes(config),
        "paper_faithful_nimbus_requirements": _paper_faithful_nimbus_requirements(),
        "promotion_gate_criteria": _promotion_gate_criteria(),
        "promotion_decision": promotion_decision,
        "deterministic_baseline_metrics": deterministic_metrics,
        "learned_model": _learned_model_evidence(infonce_model),
        "corpus_evidence": corpus_evidence,
        "learned_grouped_cv_metrics": grouped_metrics,
        "learned_sealed_holdout_metrics": sealed_metrics,
        "learned_runtime_beta_metrics": runtime_beta_metrics,
        "gateway_runtime_evidence": gateway_evidence,
        "head_to_head_evaluation": _head_to_head_evaluation(
            deterministic_metrics=deterministic_metrics,
            grouped_metrics=grouped_metrics,
            sealed_metrics=sealed_metrics,
            runtime_beta_metrics=runtime_beta_metrics,
            gateway_evidence=gateway_evidence,
            comparison=comparison,
        ),
        "comparison": comparison,
        "promotion_gates": cast(JsonValue, promotion_gates),
        "checklist": cast(JsonValue, checklist),
        "checklist_summary": checklist_summary,
        "missing_before_paper_faithful_learned_promotion": missing,
    }


def write_nimbus_promotion_evidence_report(path: Path, report: Mapping[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_nimbus_promotion_evidence_report_json(report), encoding="utf-8")


def render_nimbus_promotion_evidence_report_json(report: Mapping[str, JsonValue]) -> str:
    return json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"


def parse_args(argv: Sequence[str]) -> tuple[NimbusPromotionEvidenceConfig, Path | None]:
    parser = argparse.ArgumentParser(description="Build a NIMBUS learned-vs-deterministic promotion evidence report.")
    parser.add_argument("--deterministic-eval", required=True, type=Path)
    parser.add_argument("--calibration-manifest", required=True, type=Path)
    parser.add_argument("--sealed-manifest", required=True, type=Path)
    parser.add_argument("--infonce-model", required=True, type=Path)
    parser.add_argument("--grouped-cv", required=True, type=Path)
    parser.add_argument("--sealed-holdout", required=True, type=Path)
    parser.add_argument("--gateway-smoke", required=True, type=Path)
    parser.add_argument("--runtime-beta-eval", required=False, type=Path)
    parser.add_argument("--output", required=False, type=Path)
    args = parser.parse_args(argv)
    return (
        NimbusPromotionEvidenceConfig(
            deterministic_eval_path=args.deterministic_eval,
            calibration_manifest_path=args.calibration_manifest,
            sealed_manifest_path=args.sealed_manifest,
            infonce_model_path=args.infonce_model,
            grouped_cv_path=args.grouped_cv,
            sealed_holdout_path=args.sealed_holdout,
            gateway_smoke_path=args.gateway_smoke,
            runtime_beta_eval_path=args.runtime_beta_eval,
        ),
        args.output,
    )


def main() -> None:
    try:
        config, output_path = parse_args(tuple(sys.argv[1:]))
        report = build_nimbus_promotion_evidence_report(config)
        if output_path is not None:
            write_nimbus_promotion_evidence_report(output_path, report)
        sys.stdout.write(render_nimbus_promotion_evidence_report_json(report))
    except (NimbusPromotionEvidenceError, OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc


def _deterministic_metrics(report: Mapping[str, object]) -> dict[str, JsonValue]:
    paper_target = _mapping(report.get("paper_faithful_target"), "deterministic_eval.paper_faithful_target")
    required_before_promotion = _string_list(
        paper_target.get("required_before_promotion"),
        "deterministic_eval.paper_faithful_target.required_before_promotion",
    )
    return {
        "critic_status": _required_string(report, "critic_status", "deterministic_eval"),
        "critic_kind": _required_string(report, "critic_kind", "deterministic_eval"),
        "paper_faithful_learned_critic": _required_bool(report, "paper_faithful_learned_critic", "deterministic_eval"),
        "scenario_count": _required_int(report, "scenario_count", "deterministic_eval"),
        "positive_label_count": _required_int(report, "positive_label_count", "deterministic_eval"),
        "negative_label_count": _required_int(report, "negative_label_count", "deterministic_eval"),
        "true_positive": _required_int(report, "true_positive", "deterministic_eval"),
        "true_negative": _required_int(report, "true_negative", "deterministic_eval"),
        "false_positive": _required_int(report, "false_positive", "deterministic_eval"),
        "false_negative": _required_int(report, "false_negative", "deterministic_eval"),
        "false_positive_rate": _required_float(report, "false_positive_rate", "deterministic_eval"),
        "false_negative_rate": _required_float(report, "false_negative_rate", "deterministic_eval"),
        "precision": _required_float(report, "precision", "deterministic_eval"),
        "recall": _required_float(report, "recall", "deterministic_eval"),
        "paper_faithful_required_before_promotion": list(required_before_promotion),
    }


def _learned_metrics(report: Mapping[str, object]) -> dict[str, JsonValue]:
    metrics: dict[str, JsonValue] = {
        "schema_version": _required_string(report, "schema_version", "learned_metrics"),
        "model_id": _required_string(report, "model_id", "learned_metrics"),
        "promotion_status": _required_string(report, "promotion_status", "learned_metrics"),
        "paper_faithful_learned_critic": _required_bool(report, "paper_faithful_learned_critic", "learned_metrics"),
        "record_count": _required_int(report, "record_count", "learned_metrics"),
        "split_group_count": _required_int(report, "split_group_count", "learned_metrics"),
        "attack_top1_accuracy": _optional_float(report, "attack_top1_accuracy", "learned_metrics"),
        "true_positive": _required_int(report, "true_positive", "learned_metrics"),
        "true_negative": _required_int(report, "true_negative", "learned_metrics"),
        "false_positive": _required_int(report, "false_positive", "learned_metrics"),
        "false_negative": _required_int(report, "false_negative", "learned_metrics"),
        "false_positive_rate": _required_float(report, "false_positive_rate", "learned_metrics"),
        "false_negative_rate": _required_float(report, "false_negative_rate", "learned_metrics"),
        "session_true_positive": _required_int(report, "session_true_positive", "learned_metrics"),
        "session_true_negative": _required_int(report, "session_true_negative", "learned_metrics"),
        "session_false_positive": _required_int(report, "session_false_positive", "learned_metrics"),
        "session_false_negative": _required_int(report, "session_false_negative", "learned_metrics"),
        "session_false_positive_rate": _required_float(
            report,
            "session_false_positive_rate",
            "learned_metrics",
        ),
        "session_false_negative_rate": _required_float(
            report,
            "session_false_negative_rate",
            "learned_metrics",
        ),
    }
    if "training_eval_reused" in report:
        metrics["training_eval_reused"] = _required_bool(report, "training_eval_reused", "learned_metrics")
    if "training_eval_allowed" in report:
        metrics["training_eval_allowed"] = _required_bool(report, "training_eval_allowed", "learned_metrics")
    if "fold_count" in report:
        metrics["fold_count"] = _required_int(report, "fold_count", "learned_metrics")
    return metrics


def _learned_model_evidence(model: Mapping[str, object]) -> dict[str, JsonValue]:
    feature_names = _string_list(model.get("feature_names"), "infonce_model.feature_names")
    weights = _float_list(model.get("weights"), "infonce_model.weights")
    return {
        "schema_version": _required_string(model, "schema_version", "infonce_model"),
        "model_id": _required_string(model, "model_id", "infonce_model"),
        "promotion_status": _required_string(model, "promotion_status", "infonce_model"),
        "paper_faithful_learned_critic": _required_bool(model, "paper_faithful_learned_critic", "infonce_model"),
        "feature_names": list(feature_names),
        "feature_weight_policy": _json_mapping(
            _mapping(model.get("feature_weight_policy"), "infonce_model.feature_weight_policy")
        ),
        "weights": list(weights),
        "negative_count": _required_int(model, "negative_count", "infonce_model"),
        "positive_context_index": _required_int(model, "positive_context_index", "infonce_model"),
        "training_record_count": _required_int(model, "training_record_count", "infonce_model"),
        "training_split_group_count": _required_int(model, "training_split_group_count", "infonce_model"),
        "source_corpus_sha256": _required_string(model, "source_corpus_sha256", "infonce_model"),
        "attack_top1_accuracy": _required_float(model, "attack_top1_accuracy", "infonce_model"),
    }


def _optional_runtime_beta_eval(path: Path | None) -> Mapping[str, object] | None:
    if path is None:
        return None
    report = _read_json_mapping(path)
    _require_schema(report, _RUNTIME_BETA_EVAL_SCHEMA_VERSION, "runtime_beta_eval")
    return report


def _runtime_beta_metrics(report: Mapping[str, object] | None) -> dict[str, JsonValue]:
    if report is None:
        return {
            "runtime_adapter_present": False,
            "live_gateway_evidence": False,
            "paper_faithful_learned_critic": False,
            "promotion_status": "missing",
        }
    return {
        "schema_version": _required_string(report, "schema_version", "runtime_beta_eval"),
        "critic_kind": _required_string(report, "critic_kind", "runtime_beta_eval"),
        "critic_version": _required_string(report, "critic_version", "runtime_beta_eval"),
        "runtime_adapter_present": _required_bool(report, "runtime_adapter_present", "runtime_beta_eval"),
        "live_gateway_evidence": _required_bool(report, "live_gateway_evidence", "runtime_beta_eval"),
        "promotion_status": _required_string(report, "promotion_status", "runtime_beta_eval"),
        "paper_faithful_learned_critic": _required_bool(
            report,
            "paper_faithful_learned_critic",
            "runtime_beta_eval",
        ),
        "record_count": _required_int(report, "record_count", "runtime_beta_eval"),
        "split_group_count": _required_int(report, "split_group_count", "runtime_beta_eval"),
        "true_positive": _required_int(report, "true_positive", "runtime_beta_eval"),
        "true_negative": _required_int(report, "true_negative", "runtime_beta_eval"),
        "false_positive": _required_int(report, "false_positive", "runtime_beta_eval"),
        "false_negative": _required_int(report, "false_negative", "runtime_beta_eval"),
        "false_positive_rate": _required_float(report, "false_positive_rate", "runtime_beta_eval"),
        "false_negative_rate": _required_float(report, "false_negative_rate", "runtime_beta_eval"),
        "session_true_positive": _required_int(report, "session_true_positive", "runtime_beta_eval"),
        "session_true_negative": _required_int(report, "session_true_negative", "runtime_beta_eval"),
        "session_false_positive": _required_int(report, "session_false_positive", "runtime_beta_eval"),
        "session_false_negative": _required_int(report, "session_false_negative", "runtime_beta_eval"),
        "session_false_positive_rate": _required_float(
            report,
            "session_false_positive_rate",
            "runtime_beta_eval",
        ),
        "session_false_negative_rate": _required_float(
            report,
            "session_false_negative_rate",
            "runtime_beta_eval",
        ),
        "session_block_true_positive": _required_int(report, "session_block_true_positive", "runtime_beta_eval"),
        "session_block_true_negative": _required_int(report, "session_block_true_negative", "runtime_beta_eval"),
        "session_block_false_positive": _required_int(report, "session_block_false_positive", "runtime_beta_eval"),
        "session_block_false_negative": _required_int(report, "session_block_false_negative", "runtime_beta_eval"),
        "session_block_false_positive_rate": _required_float(
            report,
            "session_block_false_positive_rate",
            "runtime_beta_eval",
        ),
        "session_block_false_negative_rate": _required_float(
            report,
            "session_block_false_negative_rate",
            "runtime_beta_eval",
        ),
        "paper_conversation_metrics": _json_mapping(
            _mapping(report.get("paper_conversation_metrics"), "runtime_beta_eval.paper_conversation_metrics")
        ),
    }


def _corpus_evidence(
    calibration_manifest: Mapping[str, object],
    sealed_manifest: Mapping[str, object],
) -> dict[str, JsonValue]:
    calibration_labels = _mapping(calibration_manifest.get("label_counts"), "calibration_manifest.label_counts")
    sealed_labels = _mapping(sealed_manifest.get("label_counts"), "sealed_manifest.label_counts")
    return {
        "calibration_profile": _required_string(calibration_manifest, "corpus_profile", "calibration_manifest"),
        "sealed_profile": _required_string(sealed_manifest, "corpus_profile", "sealed_manifest"),
        "calibration_record_count": _required_int(calibration_manifest, "record_count", "calibration_manifest"),
        "sealed_record_count": _required_int(sealed_manifest, "record_count", "sealed_manifest"),
        "calibration_split_group_count": _required_int(
            calibration_manifest, "split_group_count", "calibration_manifest"
        ),
        "sealed_split_group_count": _required_int(sealed_manifest, "split_group_count", "sealed_manifest"),
        "negative_context_count": _required_int(
            calibration_manifest,
            "info_nce_negative_count",
            "calibration_manifest",
        ),
        "calibration_label_counts": _json_mapping(calibration_labels),
        "sealed_label_counts": _json_mapping(sealed_labels),
        "calibration_quality_gates_passed": _quality_gates_passed(calibration_manifest),
        "sealed_quality_gates_passed": _quality_gates_passed(sealed_manifest),
        "profiles_distinct": (
            _required_string(calibration_manifest, "corpus_profile", "calibration_manifest")
            != _required_string(sealed_manifest, "corpus_profile", "sealed_manifest")
        ),
    }


def _gateway_evidence(smoke: Mapping[str, object]) -> dict[str, JsonValue]:
    checks = _mapping(smoke.get("checks"), "gateway_smoke.checks")
    capabilities = _mapping(checks.get("capabilities"), "gateway_smoke.checks.capabilities")
    nimbus_thresholds = _mapping(capabilities.get("nimbus_thresholds"), "gateway_smoke.checks.capabilities")
    gateway_readiness = _mapping(checks.get("gateway_readiness"), "gateway_smoke.checks.gateway_readiness")
    benign_chat = _mapping(checks.get("benign_chat"), "gateway_smoke.checks.benign_chat")
    adversarial_benign = _mapping(
        checks.get("adversarial_benign_no_block", {}),
        "gateway_smoke.checks.adversarial_benign_no_block",
    )
    partial_leak = _mapping(checks.get("nimbus_partial_leak"), "gateway_smoke.checks.nimbus_partial_leak")
    tool_leak = _mapping(checks.get("tool_argument_canary_leak"), "gateway_smoke.checks.tool_argument_canary_leak")
    runtime_critic_kind = _gateway_runtime_critic_kind(capabilities, gateway_readiness)
    learned_metrics = _gateway_learned_metrics(checks, runtime_critic_kind)
    critic_binding = _gateway_learned_critic_binding(checks, runtime_critic_kind)
    return {
        "smoke_status": _required_string(smoke, "status", "gateway_smoke"),
        "provider_mode": _required_string(smoke, "provider_mode", "gateway_smoke"),
        "nimbus_profile": _required_string(smoke, "nimbus_profile", "gateway_smoke"),
        "gateway_ready_status": _string_or_none(gateway_readiness.get("status")),
        "readiness_nimbus_status": _string_or_none(gateway_readiness.get("nimbus_status")),
        "readiness_nimbus_critic_kind": _string_or_none(gateway_readiness.get("nimbus_critic_kind")),
        "readiness_nimbus_promotion_status": _string_or_none(gateway_readiness.get("nimbus_promotion_status")),
        "nimbus_thresholds": _json_mapping(nimbus_thresholds),
        "runtime_critic_kind": runtime_critic_kind,
        "paper_faithful_learned_critic": _gateway_paper_faithful_learned_critic(
            capabilities,
            gateway_readiness,
        ),
        "benign_final_action": _string_or_none(benign_chat.get("final_action")),
        "adversarial_benign_final_action": _string_or_none(adversarial_benign.get("final_action")),
        "adversarial_benign_nimbus_action": _nested_nimbus_action(adversarial_benign, "nimbus"),
        "partial_leak_final_action": _string_or_none(partial_leak.get("final_action")),
        "partial_leak_nimbus_action": _string_or_none(partial_leak.get("nimbus_action")),
        "partial_leak_budget_fraction": _json_value_or_none(partial_leak.get("budget_fraction")),
        "tool_argument_final_action": _string_or_none(tool_leak.get("final_action")),
        "tool_argument_provider_status": _string_or_none(tool_leak.get("provider_status")),
        "tool_argument_nimbus_tool_action": _string_or_none(tool_leak.get("nimbus_tool_action")),
        "learned_runtime_evidence_present": learned_metrics["evidence_present"],
        "learned_gateway_metrics": learned_metrics,
        "learned_gateway_critic_binding": critic_binding,
    }


def _comparison(
    deterministic_metrics: Mapping[str, JsonValue],
    grouped_metrics: Mapping[str, JsonValue],
    sealed_metrics: Mapping[str, JsonValue],
    runtime_beta_metrics: Mapping[str, JsonValue],
    gateway_evidence: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    deterministic_turn_fnr = _json_float(deterministic_metrics.get("false_negative_rate"), "deterministic.fnr")
    deterministic_turn_fpr = _json_float(deterministic_metrics.get("false_positive_rate"), "deterministic.fpr")
    learned_sealed_turn_fnr = _json_float(sealed_metrics.get("false_negative_rate"), "sealed.fnr")
    learned_sealed_turn_fpr = _json_float(sealed_metrics.get("false_positive_rate"), "sealed.fpr")
    learned_grouped_turn_fnr = _json_float(grouped_metrics.get("false_negative_rate"), "grouped.fnr")
    learned_grouped_turn_fpr = _json_float(grouped_metrics.get("false_positive_rate"), "grouped.fpr")
    runtime_beta_fpr = _json_float_or_none(runtime_beta_metrics.get("false_positive_rate"), "runtime_beta.fpr")
    learned_sealed_session_fnr = _json_float(sealed_metrics.get("session_false_negative_rate"), "sealed.session_fnr")
    runtime_beta_fnr = _json_float_or_none(runtime_beta_metrics.get("false_negative_rate"), "runtime_beta.fnr")
    runtime_beta_session_fnr = _json_float_or_none(
        runtime_beta_metrics.get("session_false_negative_rate"),
        "runtime_beta.session_fnr",
    )
    runtime_beta_session_false_block_rate = _json_float_or_none(
        runtime_beta_metrics.get("session_block_false_positive_rate"),
        "runtime_beta.session_false_block_rate",
    )
    runtime_beta_session_missed_block_rate = _json_float_or_none(
        runtime_beta_metrics.get("session_block_false_negative_rate"),
        "runtime_beta.session_missed_block_rate",
    )
    return {
        "deterministic_eval_is_runtime": True,
        "learned_eval_is_offline_scaffold": True,
        "head_to_head_common_live_corpus": False,
        "learned_runtime_adapter_evidence_present": runtime_beta_metrics.get("runtime_adapter_present") is True,
        "learned_runtime_gateway_evidence_present": gateway_evidence.get("learned_runtime_evidence_present") is True,
        "learned_gateway_false_positive_rate": _gateway_metric_float_or_none(
            gateway_evidence,
            "false_positive_rate",
        ),
        "learned_gateway_false_negative_rate": _gateway_metric_float_or_none(
            gateway_evidence,
            "false_negative_rate",
        ),
        "deterministic_false_positive_rate": deterministic_turn_fpr,
        "deterministic_false_negative_rate": deterministic_turn_fnr,
        "learned_grouped_cv_false_positive_rate": learned_grouped_turn_fpr,
        "learned_grouped_cv_false_negative_rate": learned_grouped_turn_fnr,
        "learned_sealed_false_positive_rate": learned_sealed_turn_fpr,
        "learned_sealed_false_negative_rate": learned_sealed_turn_fnr,
        "learned_sealed_session_false_negative_rate": learned_sealed_session_fnr,
        "learned_runtime_beta_false_positive_rate": runtime_beta_fpr,
        "learned_runtime_beta_false_negative_rate": runtime_beta_fnr,
        "learned_runtime_beta_session_false_negative_rate": runtime_beta_session_fnr,
        "learned_runtime_beta_session_false_block_rate": runtime_beta_session_false_block_rate,
        "learned_runtime_beta_session_missed_block_rate": runtime_beta_session_missed_block_rate,
        "learned_turn_fnr_beats_deterministic": learned_sealed_turn_fnr < deterministic_turn_fnr,
        "learned_turn_fnr_matches_deterministic": learned_sealed_turn_fnr == deterministic_turn_fnr,
        "learned_runtime_beta_turn_rates_within_policy": runtime_beta_fpr is not None
        and runtime_beta_fpr <= 0.05
        and runtime_beta_fnr is not None
        and runtime_beta_fnr <= 0.05,
        "offline_learned_session_signal_observed": learned_sealed_session_fnr == 0.0
        and learned_sealed_turn_fnr > deterministic_turn_fnr,
        "runtime_beta_paper_session_false_blocks_clean": runtime_beta_session_false_block_rate == 0.0
        and runtime_beta_session_missed_block_rate == 0.0,
        "learned_session_signal_complements_deterministic": False,
        "learned_promotion_blocked_reason": (
            "Learned scaffold/runtime beta has improved held-out, in-process runtime, and learned-gateway smoke "
            "metrics, but still lacks a common live head-to-head corpus and a promotion manifest."
        ),
    }


def _paper_faithful_nimbus_requirements() -> list[JsonValue]:
    return [
        {
            "requirement_id": "cumulative_session_accounting",
            "description": (
                "Track cumulative estimated leakage across conversation turns, not only single-turn canaries."
            ),
        },
        {
            "requirement_id": "infonce_secret_context_critic",
            "description": (
                "Use an InfoNCE-style learned critic over model output, conversation state, a true secret context, "
                "and negative secret contexts."
            ),
        },
        {
            "requirement_id": "approximate_not_certified_bound",
            "description": "Report the learned score as approximate evidence, not as a formal leakage upper bound.",
        },
        {
            "requirement_id": "session_level_eval",
            "description": "Evaluate multi-turn attack sessions and benign sessions with session FN/FP separated.",
        },
        {
            "requirement_id": "single_turn_detectors_remain_separate",
            "description": "Keep single-turn exact/encoded canary and CIFT evidence distinct from learned NIMBUS.",
        },
    ]


def _promotion_gate_criteria() -> dict[str, JsonValue]:
    return {
        "turn_false_positive_rate_max": _PROMOTION_MAX_TURN_FALSE_RATE,
        "turn_false_negative_rate_max": _PROMOTION_MAX_TURN_FALSE_RATE,
        "session_false_positive_rate_max": _PROMOTION_MAX_SESSION_FALSE_RATE,
        "session_false_negative_rate_max": _PROMOTION_MAX_SESSION_FALSE_RATE,
        "required_live_gateway_sample_count": _PROMOTION_REQUIRED_LIVE_GATEWAY_SAMPLES,
        "required_live_gateway_negative_scenarios": [
            "benign_chat",
            "adversarial_benign_no_block",
        ],
        "required_live_gateway_positive_scenarios": [
            "tool_argument_canary_leak",
            "encoded_canary_leak",
            "metadata_slot_canary_leak",
            "nimbus_partial_leak",
        ],
        "promotion_requires_all_gates_met": True,
    }


def _head_to_head_evaluation(
    deterministic_metrics: Mapping[str, JsonValue],
    grouped_metrics: Mapping[str, JsonValue],
    sealed_metrics: Mapping[str, JsonValue],
    runtime_beta_metrics: Mapping[str, JsonValue],
    gateway_evidence: Mapping[str, JsonValue],
    comparison: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    return {
        "common_live_corpus_available": comparison["head_to_head_common_live_corpus"],
        "deterministic_canary_nimbus": {
            "status": "active_default_tiny_fixture_baseline",
            "critic_kind": deterministic_metrics["critic_kind"],
            "scenario_count": deterministic_metrics["scenario_count"],
            "turn_false_positive": deterministic_metrics["false_positive"],
            "turn_false_negative": deterministic_metrics["false_negative"],
            "turn_false_positive_rate": deterministic_metrics["false_positive_rate"],
            "turn_false_negative_rate": deterministic_metrics["false_negative_rate"],
            "common_live_corpus_evaluated": False,
        },
        "learned_infonce_nimbus": {
            "status": "live_beta_not_promoted",
            "grouped_cv": _metrics_evidence(grouped_metrics),
            "sealed_holdout": _metrics_evidence(sealed_metrics),
            "runtime_beta": _runtime_beta_evidence(runtime_beta_metrics),
            "live_gateway": gateway_evidence["learned_gateway_metrics"],
            "common_live_corpus_evaluated": False,
        },
        "hybrid_policy": {
            "status": "not_promoted_not_evaluated_on_common_live_corpus",
            "promote_hybrid_runtime": False,
            "reason": (
                "No artifact currently evaluates deterministic OR learned policy on the same live gateway corpus; "
                "therefore the hybrid policy cannot be promoted."
            ),
        },
    }


def _promotion_gates(
    grouped_metrics: Mapping[str, JsonValue],
    sealed_metrics: Mapping[str, JsonValue],
    runtime_beta_metrics: Mapping[str, JsonValue],
    gateway_evidence: Mapping[str, JsonValue],
    comparison: Mapping[str, JsonValue],
    checklist: Sequence[Mapping[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    return [
        _gate_item(
            gate_id="grouped_cv_metrics",
            status=_status_met_or_failed(_turn_rates_within_policy(grouped_metrics)),
            evidence=_metrics_evidence(grouped_metrics),
            gaps=()
            if _turn_rates_within_policy(grouped_metrics)
            else ("grouped CV turn FP/FN rates exceed promotion policy",),
        ),
        _gate_item(
            gate_id="sealed_holdout_metrics",
            status=_status_met_or_failed(_turn_rates_within_policy(sealed_metrics)),
            evidence=_metrics_evidence(sealed_metrics),
            gaps=()
            if _turn_rates_within_policy(sealed_metrics)
            else ("sealed holdout turn FP/FN rates exceed promotion policy",),
        ),
        _gate_item(
            gate_id="turn_level_fn_fp",
            status=_status_met_or_failed(
                _turn_rates_within_policy(grouped_metrics)
                and _turn_rates_within_policy(sealed_metrics)
                and comparison.get("learned_runtime_beta_turn_rates_within_policy") is True
            ),
            evidence={
                "grouped_cv_false_positive_rate": grouped_metrics["false_positive_rate"],
                "grouped_cv_false_negative_rate": grouped_metrics["false_negative_rate"],
                "sealed_false_positive_rate": sealed_metrics["false_positive_rate"],
                "sealed_false_negative_rate": sealed_metrics["false_negative_rate"],
                "runtime_beta_false_positive_rate": runtime_beta_metrics.get("false_positive_rate"),
                "runtime_beta_false_negative_rate": runtime_beta_metrics.get("false_negative_rate"),
            },
            gaps=()
            if comparison.get("learned_runtime_beta_turn_rates_within_policy") is True
            else ("runtime beta turn FP/FN rates exceed promotion policy",),
        ),
        _gate_item(
            gate_id="session_level_fn_fp",
            status=_status_met_or_failed(
                _session_rates_within_policy(grouped_metrics)
                and _session_rates_within_policy(sealed_metrics)
                and _runtime_session_rates_within_policy(runtime_beta_metrics)
            ),
            evidence={
                "grouped_cv_session_false_positive_rate": grouped_metrics["session_false_positive_rate"],
                "grouped_cv_session_false_negative_rate": grouped_metrics["session_false_negative_rate"],
                "sealed_session_false_positive_rate": sealed_metrics["session_false_positive_rate"],
                "sealed_session_false_negative_rate": sealed_metrics["session_false_negative_rate"],
                "runtime_beta_session_false_positive_rate": runtime_beta_metrics.get("session_false_positive_rate"),
                "runtime_beta_session_false_negative_rate": runtime_beta_metrics.get("session_false_negative_rate"),
            },
            gaps=()
            if _runtime_session_rates_within_policy(runtime_beta_metrics)
            else ("runtime beta session FP/FN rates exceed promotion policy",),
        ),
        _gate_item(
            gate_id="benign_false_block_rate",
            status=_status_met_or_failed(_benign_false_block_gate_met(runtime_beta_metrics, gateway_evidence)),
            evidence={
                "runtime_beta_session_block_false_positive_rate": runtime_beta_metrics.get(
                    "session_block_false_positive_rate"
                ),
                "paper_conversation_false_block_rate": _paper_conversation_metric(
                    runtime_beta_metrics,
                    "false_block_rate",
                ),
                "learned_gateway_false_positive_rate": _gateway_metric_float_or_none(
                    gateway_evidence,
                    "false_positive_rate",
                ),
            },
            gaps=()
            if _benign_false_block_gate_met(runtime_beta_metrics, gateway_evidence)
            else ("benign false-block evidence does not satisfy promotion policy",),
        ),
        _gate_item(
            gate_id="live_gateway_metrics",
            status=_live_gateway_gate_status(gateway_evidence),
            evidence=_json_mapping_from_json(
                gateway_evidence.get("learned_gateway_metrics"), "learned_gateway_metrics"
            ),
            gaps=_live_gateway_gate_gaps(gateway_evidence),
        ),
        _gate_item(
            gate_id="head_to_head_common_live_corpus",
            status="missing",
            evidence={"head_to_head_common_live_corpus": comparison["head_to_head_common_live_corpus"]},
            gaps=("no common live runtime corpus compares deterministic, learned, and hybrid policies",),
        ),
        _gate_item(
            gate_id="latency",
            status="missing",
            evidence={"latency_metrics_present": False},
            gaps=("no grouped/sealed/live latency metrics are bound into the promotion evidence",),
        ),
        _gate_item(
            gate_id="artifact_binding",
            status="partial",
            evidence={
                "input_artifact_hashes_bound": True,
                "learned_gateway_critic_binding": cast(
                    JsonValue,
                    _json_mapping_from_json(
                        gateway_evidence.get("learned_gateway_critic_binding"),
                        "learned_gateway_critic_binding",
                    ),
                ),
                "model_artifact_sha_present_in_live_smoke": _live_model_sha_present(gateway_evidence),
                "promoted_runtime_manifest_present": False,
            },
            gaps=("input hashes exist, but no promoted runtime artifact or manifest exists",),
        ),
        _gate_item(
            gate_id="audit_readiness_console_evidence",
            status="partial",
            evidence={
                "readiness_nimbus_status": gateway_evidence["readiness_nimbus_status"],
                "readiness_nimbus_critic_kind": gateway_evidence["readiness_nimbus_critic_kind"],
                "readiness_nimbus_promotion_status": gateway_evidence["readiness_nimbus_promotion_status"],
                "live_gateway_evidence_present": gateway_evidence["learned_runtime_evidence_present"],
                "console_evidence_bound": False,
            },
            gaps=("readiness and audit-smoke evidence exist, but no console status artifact is bound here",),
        ),
        _gate_item(
            gate_id="failure_mode_behavior",
            status="partial",
            evidence={
                "learned_runtime_label_is_beta": gateway_evidence["readiness_nimbus_promotion_status"]
                == "learned_runtime_beta_not_promotable",
                "deterministic_default_kept": True,
                "promoted_learned_manifest_present": False,
            },
            gaps=("failure-mode behavior is labeled beta/default-safe, but no learned fault-injection report exists",),
        ),
        _gate_item(
            gate_id="hybrid_policy_evaluation",
            status="missing",
            evidence={"promote_hybrid_runtime": False},
            gaps=("hybrid deterministic plus learned policy is not evaluated on a common live corpus",),
        ),
        _gate_item(
            gate_id="promotion_manifest",
            status=_checklist_requirement_status(checklist, "promotion_manifest"),
            evidence={"promote_learned_runtime": False, "promote_hybrid_runtime": False},
            gaps=("no learned or hybrid promoted runtime manifest exists",),
        ),
    ]


def _promotion_decision(promotion_gates: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    blocking_gate_ids = [
        str(gate["gate_id"]) for gate in promotion_gates if gate.get("status") in {"missing", "failed", "partial"}
    ]
    return {
        "verdict": "reject_learned_runtime_keep_deterministic_default",
        "promote_learned_runtime": False,
        "promote_hybrid_runtime": False,
        "keep_deterministic_default": True,
        "reject_learned_runtime": True,
        "blocking_gate_ids": cast(JsonValue, blocking_gate_ids),
        "rationale": (
            "Learned NIMBUS has useful beta metrics, but promotion requires all gates to pass. "
            "The current package lacks common live head-to-head evidence, latency evidence, hybrid-policy "
            "evaluation, and a promoted runtime manifest."
        ),
    }


def _gate_item(
    gate_id: str,
    status: str,
    evidence: Mapping[str, JsonValue],
    gaps: Sequence[str],
) -> dict[str, JsonValue]:
    return {"gate_id": gate_id, "status": status, "evidence": dict(evidence), "gaps": list(gaps)}


def _status_met_or_failed(passed: bool) -> str:
    return "met" if passed else "failed"


def _turn_rates_within_policy(metrics: Mapping[str, JsonValue]) -> bool:
    return _rate_at_or_below(metrics, "false_positive_rate", _PROMOTION_MAX_TURN_FALSE_RATE) and _rate_at_or_below(
        metrics,
        "false_negative_rate",
        _PROMOTION_MAX_TURN_FALSE_RATE,
    )


def _session_rates_within_policy(metrics: Mapping[str, JsonValue]) -> bool:
    return _rate_at_or_below(
        metrics,
        "session_false_positive_rate",
        _PROMOTION_MAX_SESSION_FALSE_RATE,
    ) and _rate_at_or_below(metrics, "session_false_negative_rate", _PROMOTION_MAX_SESSION_FALSE_RATE)


def _runtime_session_rates_within_policy(metrics: Mapping[str, JsonValue]) -> bool:
    return _rate_at_or_below(
        metrics,
        "session_false_positive_rate",
        _PROMOTION_MAX_SESSION_FALSE_RATE,
    ) and _rate_at_or_below(metrics, "session_false_negative_rate", _PROMOTION_MAX_SESSION_FALSE_RATE)


def _rate_at_or_below(metrics: Mapping[str, JsonValue], key: str, maximum: float) -> bool:
    value = metrics.get(key)
    return isinstance(value, int | float) and not isinstance(value, bool) and float(value) <= maximum


def _benign_false_block_gate_met(
    runtime_beta_metrics: Mapping[str, JsonValue],
    gateway_evidence: Mapping[str, JsonValue],
) -> bool:
    paper_false_block_rate = _paper_conversation_metric(runtime_beta_metrics, "false_block_rate")
    gateway_false_positive_rate = _gateway_metric_float_or_none(gateway_evidence, "false_positive_rate")
    return (
        runtime_beta_metrics.get("session_block_false_positive_rate") == 0.0
        and paper_false_block_rate == 0.0
        and gateway_false_positive_rate == 0.0
    )


def _paper_conversation_metric(metrics: Mapping[str, JsonValue], key: str) -> JsonValue:
    paper_metrics = metrics.get("paper_conversation_metrics")
    if not isinstance(paper_metrics, dict):
        return None
    value = paper_metrics.get(key)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise NimbusPromotionEvidenceError(f"paper_conversation_metrics.{key} must be a scalar.")


def _live_gateway_gate_status(gateway_evidence: Mapping[str, JsonValue]) -> str:
    metrics = gateway_evidence.get("learned_gateway_metrics")
    if not isinstance(metrics, dict):
        return "missing"
    if metrics.get("evidence_present") is not True:
        return "missing"
    if len(_live_gateway_gate_gaps(gateway_evidence)) == 0:
        return "met"
    return "partial"


def _live_gateway_gate_gaps(gateway_evidence: Mapping[str, JsonValue]) -> tuple[str, ...]:
    metrics = gateway_evidence.get("learned_gateway_metrics")
    if not isinstance(metrics, dict):
        return ("learned gateway metrics are missing",)
    gaps: list[str] = []
    if metrics.get("sample_count") != _PROMOTION_REQUIRED_LIVE_GATEWAY_SAMPLES:
        gaps.append("live gateway smoke must include benign, adversarial-benign, and four leak samples")
    if metrics.get("false_positive_rate") != 0.0:
        gaps.append("live gateway false-positive rate must be zero for promotion")
    if metrics.get("false_negative_rate") != 0.0:
        gaps.append("live gateway false-negative rate must be zero for promotion")
    if gateway_evidence.get("adversarial_benign_final_action") != "allow":
        gaps.append("adversarial benign gateway scenario is missing or did not allow")
    if gateway_evidence.get("adversarial_benign_nimbus_action") not in {"allow", None}:
        gaps.append("adversarial benign NIMBUS action must not warn, sanitize, block, or escalate")
    return tuple(gaps)


def _live_model_sha_present(gateway_evidence: Mapping[str, JsonValue]) -> bool:
    binding = gateway_evidence.get("learned_gateway_critic_binding")
    if not isinstance(binding, dict):
        return False
    model_shas = binding.get("model_artifact_sha256s")
    return isinstance(model_shas, list) and len(model_shas) > 0


def _checklist_requirement_status(checklist: Sequence[Mapping[str, JsonValue]], requirement_id: str) -> str:
    for item in checklist:
        if item.get("requirement_id") == requirement_id:
            status = item.get("status")
            if not isinstance(status, str):
                raise NimbusPromotionEvidenceError("checklist.status must be a string.")
            return status
    raise NimbusPromotionEvidenceError(f"checklist did not include requirement {requirement_id}.")


def _learned_comparison_gaps(comparison: Mapping[str, JsonValue]) -> tuple[str, ...]:
    gaps: list[str] = []
    if comparison.get("learned_turn_fnr_beats_deterministic") is not True:
        if comparison.get("learned_turn_fnr_matches_deterministic") is True:
            gaps.append("learned sealed turn false-negative rate ties deterministic beta but does not beat it")
        else:
            gaps.append("learned sealed turn false-negative rate is worse than deterministic beta")
    if comparison.get("learned_runtime_beta_turn_rates_within_policy") is not True:
        gaps.append("learned runtime beta turn FP/FN rates do not satisfy the local operating policy")
    if comparison.get("learned_session_signal_complements_deterministic") is not True:
        gaps.append("no live evidence proves a meaningful learned complement to deterministic beta")
    if comparison.get("head_to_head_common_live_corpus") is not True:
        gaps.append("no common live runtime head-to-head corpus exists")
    return tuple(gaps)


def _checklist(
    corpus_evidence: Mapping[str, JsonValue],
    grouped_metrics: Mapping[str, JsonValue],
    sealed_metrics: Mapping[str, JsonValue],
    runtime_beta_metrics: Mapping[str, JsonValue],
    comparison: Mapping[str, JsonValue],
    gateway_evidence: Mapping[str, JsonValue],
) -> list[dict[str, JsonValue]]:
    return [
        _checklist_item(
            requirement_id="session_level_corpus_coverage",
            paper_requirement=(
                "Use session-level leakage examples spanning benign, exact, encoded, partial, paraphrased, "
                "tool-output, and delayed leakage."
            ),
            status="met" if _has_required_labels(corpus_evidence) else "missing",
            evidence={
                "calibration_record_count": corpus_evidence["calibration_record_count"],
                "sealed_record_count": corpus_evidence["sealed_record_count"],
                "calibration_split_group_count": corpus_evidence["calibration_split_group_count"],
                "sealed_split_group_count": corpus_evidence["sealed_split_group_count"],
                "calibration_quality_gates_passed": corpus_evidence["calibration_quality_gates_passed"],
                "sealed_quality_gates_passed": corpus_evidence["sealed_quality_gates_passed"],
            },
            gaps=() if _has_required_labels(corpus_evidence) else ("required leakage-label coverage missing",),
        ),
        _checklist_item(
            requirement_id="negative_contexts_for_infonce",
            paper_requirement="Provide negative secret contexts for contrastive leakage scoring.",
            status="met" if corpus_evidence.get("negative_context_count") == 16 else "missing",
            evidence={"negative_context_count": corpus_evidence["negative_context_count"]},
            gaps=() if corpus_evidence.get("negative_context_count") == 16 else ("expected 16 negative contexts",),
        ),
        _checklist_item(
            requirement_id="grouped_cross_validation",
            paper_requirement="Evaluate learned critic with session/scenario grouped splits.",
            status="met" if grouped_metrics.get("split_group_count") == _PAPER_REFERENCE_SESSION_COUNT else "missing",
            evidence=_metrics_evidence(grouped_metrics),
            gaps=()
            if grouped_metrics.get("split_group_count") == _PAPER_REFERENCE_SESSION_COUNT
            else ("paper-reference grouped CV split evidence missing",),
        ),
        _checklist_item(
            requirement_id="sealed_holdout",
            paper_requirement="Evaluate learned critic on a sealed holdout that was not used for training.",
            status="met" if _sealed_holdout_is_clean(sealed_metrics, corpus_evidence) else "missing",
            evidence={
                **_metrics_evidence(sealed_metrics),
                "profiles_distinct": corpus_evidence["profiles_distinct"],
                "sealed_quality_gates_passed": corpus_evidence["sealed_quality_gates_passed"],
                "sealed_required_labels_present": _has_required_labels(corpus_evidence),
            },
            gaps=_sealed_holdout_gaps(sealed_metrics, corpus_evidence),
        ),
        _checklist_item(
            requirement_id="fn_fp_reported_separately",
            paper_requirement="Report false negative and false positive counts/rates separately.",
            status="met" if _has_fn_fp_metrics(grouped_metrics) and _has_fn_fp_metrics(sealed_metrics) else "missing",
            evidence={
                "grouped_cv": _metrics_evidence(grouped_metrics),
                "sealed_holdout": _metrics_evidence(sealed_metrics),
            },
            gaps=()
            if _has_fn_fp_metrics(grouped_metrics) and _has_fn_fp_metrics(sealed_metrics)
            else ("FN/FP metrics are incomplete",),
        ),
        _checklist_item(
            requirement_id="learned_beats_or_complements_deterministic",
            paper_requirement="Promote learned NIMBUS only if it beats or meaningfully complements deterministic beta.",
            status="partial",
            evidence={
                "learned_turn_fnr_beats_deterministic": comparison["learned_turn_fnr_beats_deterministic"],
                "learned_turn_fnr_matches_deterministic": comparison["learned_turn_fnr_matches_deterministic"],
                "learned_runtime_beta_turn_rates_within_policy": comparison[
                    "learned_runtime_beta_turn_rates_within_policy"
                ],
                "offline_learned_session_signal_observed": comparison["offline_learned_session_signal_observed"],
                "learned_session_signal_complements_deterministic": comparison[
                    "learned_session_signal_complements_deterministic"
                ],
                "head_to_head_common_live_corpus": comparison["head_to_head_common_live_corpus"],
            },
            gaps=_learned_comparison_gaps(comparison),
        ),
        _checklist_item(
            requirement_id="runtime_learned_critic_adapter",
            paper_requirement="Wire the learned critic through the runtime NimbusCritic interface before promotion.",
            status="met" if runtime_beta_metrics.get("runtime_adapter_present") is True else "missing",
            evidence=_runtime_beta_evidence(runtime_beta_metrics),
            gaps=()
            if runtime_beta_metrics.get("runtime_adapter_present") is True
            else ("no learned NIMBUS runtime adapter evidence exists",),
        ),
        _checklist_item(
            requirement_id="live_gateway_learned_fn_fp",
            paper_requirement="Collect live gateway false negative and false positive evidence for the learned critic.",
            status="met" if gateway_evidence.get("learned_runtime_evidence_present") is True else "missing",
            evidence={
                "readiness_nimbus_status": gateway_evidence["readiness_nimbus_status"],
                "runtime_critic_kind": gateway_evidence["runtime_critic_kind"],
                "learned_runtime_evidence_present": gateway_evidence["learned_runtime_evidence_present"],
                "learned_gateway_metrics": gateway_evidence["learned_gateway_metrics"],
            },
            gaps=()
            if gateway_evidence.get("learned_runtime_evidence_present") is True
            else ("gateway smoke proves deterministic beta runtime only, not learned runtime FN/FP",),
        ),
        _checklist_item(
            requirement_id="promotion_manifest",
            paper_requirement=(
                "Emit a promoted runtime artifact and manifest binding critic, corpus, evals, and runtime."
            ),
            status="missing",
            evidence={"promote_learned_runtime": False},
            gaps=("no learned NIMBUS promoted runtime artifact or promotion manifest exists",),
        ),
    ]


def _artifact_hashes(config: NimbusPromotionEvidenceConfig) -> dict[str, JsonValue]:
    hashes: dict[str, JsonValue] = {
        "deterministic_eval_sha256": _sha256_file(config.deterministic_eval_path),
        "calibration_manifest_sha256": _sha256_file(config.calibration_manifest_path),
        "sealed_manifest_sha256": _sha256_file(config.sealed_manifest_path),
        "infonce_model_sha256": _sha256_file(config.infonce_model_path),
        "grouped_cv_sha256": _sha256_file(config.grouped_cv_path),
        "sealed_holdout_sha256": _sha256_file(config.sealed_holdout_path),
        "gateway_smoke_sha256": _sha256_file(config.gateway_smoke_path),
    }
    if config.runtime_beta_eval_path is not None:
        hashes["runtime_beta_eval_sha256"] = _sha256_file(config.runtime_beta_eval_path)
    return hashes


def _metrics_evidence(metrics: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    keys = (
        "record_count",
        "split_group_count",
        "attack_top1_accuracy",
        "false_positive",
        "false_negative",
        "false_positive_rate",
        "false_negative_rate",
        "session_false_positive",
        "session_false_negative",
        "session_false_positive_rate",
        "session_false_negative_rate",
        "promotion_status",
        "paper_faithful_learned_critic",
    )
    return {key: value for key in keys if (value := metrics.get(key)) is not None}


def _runtime_beta_evidence(metrics: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    keys = (
        "runtime_adapter_present",
        "live_gateway_evidence",
        "critic_kind",
        "critic_version",
        "record_count",
        "split_group_count",
        "false_positive",
        "false_negative",
        "false_positive_rate",
        "false_negative_rate",
        "session_false_positive",
        "session_false_negative",
        "session_false_positive_rate",
        "session_false_negative_rate",
        "promotion_status",
        "paper_faithful_learned_critic",
    )
    return {key: value for key in keys if (value := metrics.get(key)) is not None}


def _gateway_runtime_critic_kind(
    capabilities: Mapping[str, object],
    gateway_readiness: Mapping[str, object],
) -> str:
    capabilities_value = capabilities.get("nimbus_critic_kind")
    if isinstance(capabilities_value, str) and capabilities_value != "":
        return capabilities_value
    readiness_value = gateway_readiness.get("nimbus_critic_kind")
    if isinstance(readiness_value, str) and readiness_value != "":
        return readiness_value
    readiness_status = gateway_readiness.get("nimbus_status")
    if readiness_status == "learned_runtime_beta":
        return "learned_infonce_beta"
    return "canary"


def _gateway_paper_faithful_learned_critic(
    capabilities: Mapping[str, object],
    gateway_readiness: Mapping[str, object],
) -> bool:
    capabilities_value = capabilities.get("nimbus_paper_faithful_learned_critic")
    if isinstance(capabilities_value, bool):
        return capabilities_value
    readiness_value = gateway_readiness.get("nimbus_paper_faithful_learned_critic")
    if isinstance(readiness_value, bool):
        return readiness_value
    return False


def _gateway_metric_float_or_none(gateway_evidence: Mapping[str, JsonValue], metric_name: str) -> JsonValue:
    metrics = gateway_evidence.get("learned_gateway_metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(metric_name)
    if value is None:
        return None
    return _json_float(value, f"gateway.{metric_name}")


def _gateway_learned_metrics(
    checks: Mapping[str, object],
    runtime_critic_kind: str,
) -> dict[str, JsonValue]:
    samples: list[tuple[bool, bool]] = []
    if runtime_critic_kind != "learned_infonce_beta":
        return _gateway_live_metric_report(runtime_critic_kind, samples)
    _append_gateway_sample_from_check(
        samples=samples,
        checks=checks,
        check_name="benign_chat",
        summary_name="nimbus",
        leakage_expected=False,
    )
    _append_gateway_sample_from_check(
        samples=samples,
        checks=checks,
        check_name="adversarial_benign_no_block",
        summary_name="nimbus",
        leakage_expected=False,
    )
    _append_gateway_sample_from_check(
        samples=samples,
        checks=checks,
        check_name="tool_argument_canary_leak",
        summary_name="nimbus_tool",
        leakage_expected=True,
    )
    _append_gateway_sample_from_check(
        samples=samples,
        checks=checks,
        check_name="encoded_canary_leak",
        summary_name="nimbus",
        leakage_expected=True,
    )
    _append_gateway_sample_from_check(
        samples=samples,
        checks=checks,
        check_name="metadata_slot_canary_leak",
        summary_name="nimbus",
        leakage_expected=True,
    )
    _append_gateway_sample_from_check(
        samples=samples,
        checks=checks,
        check_name="nimbus_partial_leak",
        summary_name="nimbus",
        leakage_expected=True,
    )
    return _gateway_live_metric_report(runtime_critic_kind, samples)


def _gateway_learned_critic_binding(
    checks: Mapping[str, object],
    runtime_critic_kind: str,
) -> dict[str, JsonValue]:
    if runtime_critic_kind != "learned_infonce_beta":
        return {
            "model_artifact_sha256s": cast(JsonValue, []),
            "selected_context_sha256s": cast(JsonValue, []),
            "negative_context_counts": cast(JsonValue, []),
            "candidate_counts": cast(JsonValue, []),
        }
    model_shas: set[str] = set()
    context_shas: set[str] = set()
    negative_counts: set[int] = set()
    candidate_counts: set[int] = set()
    for check_name, summary_name in (
        ("benign_chat", "nimbus"),
        ("adversarial_benign_no_block", "nimbus"),
        ("tool_argument_canary_leak", "nimbus_tool"),
        ("encoded_canary_leak", "nimbus"),
        ("metadata_slot_canary_leak", "nimbus"),
        ("nimbus_partial_leak", "nimbus"),
    ):
        summary = _gateway_learned_summary(checks=checks, check_name=check_name, summary_name=summary_name)
        if summary is None:
            continue
        _collect_optional_string(summary, "model_artifact_sha256", model_shas)
        _collect_optional_string(summary, "selected_context_sha256", context_shas)
        _collect_optional_int(summary, "negative_context_count", negative_counts)
        _collect_optional_int(summary, "candidate_count", candidate_counts)
    return {
        "model_artifact_sha256s": cast(JsonValue, sorted(model_shas)),
        "selected_context_sha256s": cast(JsonValue, sorted(context_shas)),
        "negative_context_counts": cast(JsonValue, sorted(negative_counts)),
        "candidate_counts": cast(JsonValue, sorted(candidate_counts)),
    }


def _gateway_learned_summary(
    checks: Mapping[str, object],
    check_name: str,
    summary_name: str,
) -> Mapping[str, object] | None:
    check = checks.get(check_name)
    if not isinstance(check, Mapping):
        return None
    summary = check.get(summary_name)
    if not isinstance(summary, Mapping):
        return None
    if summary.get("critic_kind") != "learned_infonce_beta":
        return None
    return summary


def _collect_optional_string(summary: Mapping[str, object], field_name: str, values: set[str]) -> None:
    value = summary.get(field_name)
    if isinstance(value, str) and value != "":
        values.add(value)


def _collect_optional_int(summary: Mapping[str, object], field_name: str, values: set[int]) -> None:
    value = summary.get(field_name)
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        values.add(value)


def _append_gateway_sample_from_check(
    samples: list[tuple[bool, bool]],
    checks: Mapping[str, object],
    check_name: str,
    summary_name: str,
    leakage_expected: bool,
) -> None:
    check = checks.get(check_name)
    if not isinstance(check, Mapping):
        return
    summary = check.get(summary_name)
    if not isinstance(summary, Mapping):
        return
    if summary.get("present") is not True:
        return
    if summary.get("critic_kind") != "learned_infonce_beta":
        return
    action = summary.get("recommended_action")
    if not isinstance(action, str):
        raise NimbusPromotionEvidenceError(f"gateway smoke {check_name}.{summary_name}.recommended_action missing.")
    samples.append((leakage_expected, _gateway_action_detected(action)))


def _gateway_live_metric_report(
    runtime_critic_kind: str,
    samples: Sequence[tuple[bool, bool]],
) -> dict[str, JsonValue]:
    counts = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    for leakage_expected, detected in samples:
        if leakage_expected and detected:
            counts["true_positive"] += 1
        elif leakage_expected and not detected:
            counts["false_negative"] += 1
        elif not leakage_expected and detected:
            counts["false_positive"] += 1
        else:
            counts["true_negative"] += 1
    positive_count = counts["true_positive"] + counts["false_negative"]
    negative_count = counts["true_negative"] + counts["false_positive"]
    return {
        "runtime_critic_kind": runtime_critic_kind,
        "evidence_present": len(samples) > 0,
        "sample_count": len(samples),
        "true_positive": counts["true_positive"],
        "true_negative": counts["true_negative"],
        "false_positive": counts["false_positive"],
        "false_negative": counts["false_negative"],
        "false_positive_rate": _safe_rate(counts["false_positive"], negative_count),
        "false_negative_rate": _safe_rate(counts["false_negative"], positive_count),
    }


def _gateway_action_detected(action: str) -> bool:
    if action == "allow":
        return False
    if action in {"warn", "sanitize", "block", "escalate"}:
        return True
    raise NimbusPromotionEvidenceError(f"unsupported gateway action '{action}'.")


def _nested_nimbus_action(check: Mapping[str, object], summary_name: str) -> JsonValue:
    summary = check.get(summary_name)
    if not isinstance(summary, Mapping):
        return None
    action = summary.get("recommended_action")
    if action is None:
        return None
    if not isinstance(action, str):
        raise NimbusPromotionEvidenceError(f"{summary_name}.recommended_action must be a string.")
    return action


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _checklist_item(
    requirement_id: str,
    paper_requirement: str,
    status: str,
    evidence: Mapping[str, JsonValue],
    gaps: Sequence[str],
) -> dict[str, JsonValue]:
    return {
        "requirement_id": requirement_id,
        "paper_requirement": paper_requirement,
        "status": status,
        "evidence": dict(evidence),
        "gaps": list(gaps),
    }


def _checklist_summary(checklist: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    met = sum(1 for item in checklist if item.get("status") == "met")
    partial = sum(1 for item in checklist if item.get("status") == "partial")
    missing = sum(1 for item in checklist if item.get("status") == "missing")
    return {"met": met, "partial": partial, "missing": missing, "total": len(checklist)}


def _missing_before_promotion(checklist: Sequence[Mapping[str, JsonValue]]) -> list[JsonValue]:
    gaps: list[JsonValue] = []
    for item in checklist:
        if item.get("status") == "met":
            continue
        raw_gaps = item.get("gaps")
        if not isinstance(raw_gaps, list):
            raise NimbusPromotionEvidenceError("checklist gaps must be a list.")
        for gap in raw_gaps:
            if not isinstance(gap, str):
                raise NimbusPromotionEvidenceError("checklist gaps must contain only strings.")
            if gap not in gaps:
                gaps.append(gap)
    return gaps


def _has_required_labels(corpus_evidence: Mapping[str, JsonValue]) -> bool:
    required = {"benign", "partial", "encoded", "direct", "paraphrased", "tool_output", "delayed"}
    calibration = _json_mapping_from_json(corpus_evidence.get("calibration_label_counts"), "calibration_label_counts")
    sealed = _json_mapping_from_json(corpus_evidence.get("sealed_label_counts"), "sealed_label_counts")
    return _labels_have_positive_counts(calibration, required, "calibration_label_counts") and (
        _labels_have_positive_counts(sealed, required, "sealed_label_counts")
    )


def _labels_have_positive_counts(labels: Mapping[str, JsonValue], required: set[str], context: str) -> bool:
    for label in required:
        raw_count = labels.get(label)
        if raw_count is None:
            return False
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise NimbusPromotionEvidenceError(f"{context}.{label} must be an integer.")
        if raw_count <= 0:
            return False
    return True


def _has_fn_fp_metrics(metrics: Mapping[str, JsonValue]) -> bool:
    return all(
        key in metrics and metrics.get(key) is not None
        for key in (
            "false_positive",
            "false_negative",
            "false_positive_rate",
            "false_negative_rate",
            "session_false_positive",
            "session_false_negative",
            "session_false_positive_rate",
            "session_false_negative_rate",
        )
    )


def _sealed_holdout_is_clean(metrics: Mapping[str, JsonValue], corpus_evidence: Mapping[str, JsonValue]) -> bool:
    return len(_sealed_holdout_gaps(metrics, corpus_evidence)) == 0


def _sealed_holdout_gaps(
    metrics: Mapping[str, JsonValue],
    corpus_evidence: Mapping[str, JsonValue],
) -> tuple[str, ...]:
    gaps: list[str] = []
    if metrics.get("training_eval_reused") is not False or metrics.get("training_eval_allowed") is not False:
        gaps.append("sealed holdout is marked as training reuse")
    if corpus_evidence.get("profiles_distinct") is not True:
        gaps.append("sealed holdout manifest is not distinct from calibration manifest")
    if corpus_evidence.get("sealed_quality_gates_passed") is not True:
        gaps.append("sealed holdout manifest quality gates did not pass")
    if not _has_required_labels(corpus_evidence):
        gaps.append("sealed holdout required leakage-label coverage is incomplete")
    return tuple(gaps)


def _quality_gates_passed(manifest: Mapping[str, object]) -> bool:
    raw_quality_gates = manifest.get("quality_gates")
    if not isinstance(raw_quality_gates, list):
        raise NimbusPromotionEvidenceError("manifest.quality_gates must be a list.")
    if len(raw_quality_gates) == 0:
        raise NimbusPromotionEvidenceError("manifest.quality_gates must not be empty.")
    for raw_gate in raw_quality_gates:
        gate = _mapping(raw_gate, "manifest.quality_gates[]")
        if gate.get("passed") is not True:
            return False
    return True


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NimbusPromotionEvidenceError(
            f"Could not parse JSON file {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}."
        ) from exc
    return _mapping(decoded, str(path))


def _require_schema(report: Mapping[str, object], expected: str, context: str) -> None:
    observed = _required_string(report, "schema_version", context)
    if observed != expected:
        raise NimbusPromotionEvidenceError(f"{context}.schema_version must be {expected}, got {observed}.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise NimbusPromotionEvidenceError(f"{context} must be an object.")
    return value


def _json_mapping(value: Mapping[str, object]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str):
            raise NimbusPromotionEvidenceError("JSON object keys must be strings.")
        result[key] = _json_value(raw_value)
    return result


def _json_mapping_from_json(value: JsonValue | None, context: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise NimbusPromotionEvidenceError(f"{context} must be an object.")
    return value


def _json_value_or_none(value: object) -> JsonValue:
    if value is None:
        return None
    return _json_value(value)


def _json_value(value: object) -> JsonValue:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return _json_mapping(value)
    raise NimbusPromotionEvidenceError(f"Unsupported JSON value type: {type(value).__name__}.")


def _json_float(value: JsonValue | None, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise NimbusPromotionEvidenceError(f"{context} must be numeric.")
    return float(value)


def _json_float_or_none(value: JsonValue | None, context: str) -> float | None:
    if value is None:
        return None
    return _json_float(value, context)


def _required_string(report: Mapping[str, object], key: str, context: str) -> str:
    value = report.get(key)
    if not isinstance(value, str):
        raise NimbusPromotionEvidenceError(f"{context}.{key} must be a string.")
    return value


def _string_or_none(value: object) -> JsonValue:
    if value is None:
        return None
    if not isinstance(value, str):
        raise NimbusPromotionEvidenceError("expected string or null.")
    return value


def _required_bool(report: Mapping[str, object], key: str, context: str) -> bool:
    value = report.get(key)
    if not isinstance(value, bool):
        raise NimbusPromotionEvidenceError(f"{context}.{key} must be a boolean.")
    return value


def _required_int(report: Mapping[str, object], key: str, context: str) -> int:
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise NimbusPromotionEvidenceError(f"{context}.{key} must be an integer.")
    return value


def _required_float(report: Mapping[str, object], key: str, context: str) -> float:
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise NimbusPromotionEvidenceError(f"{context}.{key} must be numeric.")
    return float(value)


def _optional_float(report: Mapping[str, object], key: str, context: str) -> float | None:
    value = report.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise NimbusPromotionEvidenceError(f"{context}.{key} must be numeric or null.")
    return float(value)


def _string_list(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise NimbusPromotionEvidenceError(f"{context} must be a list.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise NimbusPromotionEvidenceError(f"{context} must contain only strings.")
        result.append(item)
    return tuple(result)


def _float_list(value: object, context: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise NimbusPromotionEvidenceError(f"{context} must be a list.")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise NimbusPromotionEvidenceError(f"{context} must contain only numbers.")
        result.append(float(item))
    return tuple(result)
