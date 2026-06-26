from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

DP_HONEY_PAPER_EVIDENCE_SCHEMA_VERSION = "aegis.dp_honey_paper_evidence/v1"
GENERATION_REALISM_EVAL_SCHEMA_VERSION = "detect.dp_honey.generation_realism_eval/v1"
GENERATION_REALISM_EVAL_STATUS = "bounded_generated_vs_reference_sanity_metrics"
STATISTICAL_DISTINGUISHER_EVAL_SCHEMA_VERSION = "detect.dp_honey.statistical_distinguisher_eval/v1"
STATISTICAL_DISTINGUISHER_EVAL_STATUS = "statistical_distinguisher_suite_evaluated"
REFERENCE_FEATURE_CORPUS_SCHEMA_VERSION = "detect.dp_honey.reference_feature_corpus/v1"
REFERENCE_FEATURE_NAMES = (
    "token_length",
    "digit_fraction",
    "alpha_fraction",
    "uppercase_fraction",
    "lowercase_fraction",
    "symbol_fraction",
    "single_token_entropy_bits",
    "avg_bigram_log_likelihood",
    "numeric_run_count",
    "numeric_run_avg_length",
    "numeric_run_max_length",
)
REQUIRED_STATISTICAL_DISTINGUISHER_TESTS = (
    "character_entropy_tests",
    "bigram_likelihood_tests",
    "numeric_substring_tests",
    "discriminator_mlp",
)
GENERATOR_PARAMETER_FIELDS = ("epsilon", "clip", "corpus_size", "train_seed")
_FORBIDDEN_AUDIT_MARKERS = ("{{CREDENTIAL:", "ghp_", "github_pat_", "sk_live_", "AKIA")
_MAX_PAPER_CONFORMAL_ALPHA = 0.01
_MIN_PAPER_BENIGN_CALIBRATION_COUNT = 1000
_MIN_PAPER_BENIGN_EVAL_COUNT = 1000
_MIN_PAPER_POSITIVE_EVAL_COUNT = 1000


class DPHoneyPaperEvidenceError(ValueError):
    """Raised when DP-HONEY paper evidence inputs are missing or malformed."""


@dataclass(frozen=True)
class DPHoneyPaperEvidenceConfig:
    scanner_eval_path: Path
    generation_realism_eval_path: Path
    statistical_distinguisher_eval_path: Path | None
    smoke_path: Path
    audit_jsonl_path: Path


def build_dp_honey_paper_evidence_report(config: DPHoneyPaperEvidenceConfig) -> dict[str, JsonValue]:
    scanner_eval = _read_json_mapping(config.scanner_eval_path)
    generation_realism_eval = _validate_generation_realism_eval(_read_json_mapping(config.generation_realism_eval_path))
    statistical_distinguisher_eval = _read_optional_statistical_distinguisher_eval(
        config.statistical_distinguisher_eval_path
    )
    smoke = _read_json_mapping(config.smoke_path)
    audit_records = _read_jsonl_mappings(config.audit_jsonl_path)
    audit_text = config.audit_jsonl_path.read_text(encoding="utf-8")
    generator_metadata = _first_dp_honey_metadata(audit_records)
    _validate_generator_parameter_binding(
        generator_metadata=generator_metadata,
        generation_realism_eval=generation_realism_eval,
        statistical_distinguisher_eval=statistical_distinguisher_eval,
    )
    checks = _mapping(smoke.get("checks"), "smoke.checks")

    checklist = (
        _dp_noised_bigram_check(generator_metadata),
        _format_fidelity_check(scanner_eval, generation_realism_eval),
        _statistical_realism_check(generation_realism_eval, statistical_distinguisher_eval),
        _conformal_check(scanner_eval),
        _scanner_fn_fp_check(scanner_eval),
        _gateway_substitution_check(checks, audit_records),
        _output_leak_detection_check(checks),
        _tool_argument_check(checks),
        _redacted_audit_check(audit_text, audit_records),
    )
    missing = tuple(
        gap
        for item in checklist
        if item["status"] != "met"
        for gap in _string_list(item.get("gaps"), f"{item['requirement_id']}.gaps")
    )
    met_count = sum(1 for item in checklist if item["status"] == "met")
    partial_count = sum(1 for item in checklist if item["status"] == "partial")
    missing_count = sum(1 for item in checklist if item["status"] == "missing")
    paper_faithful_plus = all(item["status"] == "met" for item in checklist)
    artifact_hashes: dict[str, JsonValue] = {
        "scanner_eval_sha256": _sha256_file(config.scanner_eval_path),
        "generation_realism_eval_sha256": _sha256_file(config.generation_realism_eval_path),
        "smoke_sha256": _sha256_file(config.smoke_path),
        "audit_jsonl_sha256": _sha256_file(config.audit_jsonl_path),
    }
    if config.statistical_distinguisher_eval_path is not None:
        artifact_hashes["statistical_distinguisher_eval_sha256"] = _sha256_file(
            config.statistical_distinguisher_eval_path
        )
    return {
        "schema_version": DP_HONEY_PAPER_EVIDENCE_SCHEMA_VERSION,
        "component": "dp_honey",
        "paper_source": "Research/2606.04141v1.pdf sections 4.3, 5.3, and 6",
        "promotion_status": (
            "paper_faithful_plus_candidate" if paper_faithful_plus else "paper_aligned_operational_beta"
        ),
        "paper_faithful_plus": paper_faithful_plus,
        "promotion_eligible": paper_faithful_plus,
        "summary": _summary(paper_faithful_plus, statistical_distinguisher_eval),
        "artifact_hashes": artifact_hashes,
        "scanner_metrics": _scanner_metrics(scanner_eval),
        "generation_realism_metrics": _generation_realism_metrics(generation_realism_eval),
        "statistical_distinguisher_metrics": _statistical_distinguisher_metrics(statistical_distinguisher_eval),
        "gateway_metrics": _gateway_metrics(checks, audit_records),
        "generator_metadata": _json_mapping_or_empty(generator_metadata),
        "checklist": [dict(item) for item in checklist],
        "checklist_summary": {
            "met": met_count,
            "partial": partial_count,
            "missing": missing_count,
            "total": len(checklist),
        },
        "missing_before_paper_faithful_plus": list(dict.fromkeys(missing)),
    }


def write_dp_honey_paper_evidence_report(path: Path, report: Mapping[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dp_honey_paper_evidence_report_json(report), encoding="utf-8")


def render_dp_honey_paper_evidence_report_json(report: Mapping[str, JsonValue]) -> str:
    return json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _summary(
    paper_faithful_plus: bool,
    statistical_distinguisher_eval: Mapping[str, object] | None,
) -> str:
    if paper_faithful_plus:
        reference_source = (
            statistical_distinguisher_eval.get("reference_source")
            if statistical_distinguisher_eval is not None
            else "unknown"
        )
        return (
            "DP-HONEY has DP-noised bigram provenance, split-conformal scanner calibration, held-out scanner "
            "FP/FN evidence, gateway substitution, canary leak detection, provider-egress blocking, redacted "
            "audit evidence, and a passed statistical distinguisher suite. This is paper-faithful+ candidate "
            f"evidence for the configured {reference_source} reference, not a production-secret "
            "indistinguishability proof."
        )
    if statistical_distinguisher_eval is None:
        statistical_status = "the full statistical distinguisher suite has not been supplied to this gate"
    else:
        suite = _mapping(statistical_distinguisher_eval.get("statistical_distinguisher_suite"), "statistical suite")
        failed_tests = tuple(
            test_name
            for test_name in REQUIRED_STATISTICAL_DISTINGUISHER_TESTS
            if _mapping(suite.get(test_name), f"statistical suite.{test_name}").get("status") != "passed"
        )
        if len(failed_tests) == 0:
            reference_source = _required_string(
                statistical_distinguisher_eval.get("reference_source"),
                "statistical_distinguisher_eval.reference_source",
            )
            statistical_status = (
                "the statistical distinguisher suite only uses reference_source="
                f"{reference_source}, not provider-like or real-credential-distribution evidence"
            )
        else:
            statistical_status = "the statistical distinguisher suite failed: " + ", ".join(failed_tests)
    return (
        "DP-HONEY has DP-noised bigram provenance, split-conformal scanner calibration, held-out scanner FP/FN "
        "evidence, gateway substitution, canary leak detection, provider-egress blocking, and redacted audit "
        f"evidence. It remains a paper-aligned operational beta because {statistical_status}."
    )


def parse_args(argv: Sequence[str]) -> tuple[DPHoneyPaperEvidenceConfig, Path | None]:
    parser = argparse.ArgumentParser(description="Build a DP-HONEY paper-faithfulness evidence report.")
    parser.add_argument("--scanner-eval", required=True, type=Path, help="Path to dp_honey_scanner_eval_v1.json.")
    parser.add_argument(
        "--generation-realism-eval",
        required=True,
        type=Path,
        help="Path to dp_honey_generation_realism_eval_v1.json.",
    )
    parser.add_argument(
        "--statistical-distinguisher-eval",
        required=False,
        type=Path,
        help="Optional path to dp_honey_statistical_distinguisher_eval_v1.json.",
    )
    parser.add_argument("--smoke", required=True, type=Path, help="Path to default mock-provider smoke JSON.")
    parser.add_argument("--audit-jsonl", required=True, type=Path, help="Path to matching smoke audit JSONL.")
    parser.add_argument("--output", required=False, type=Path, help="Optional JSON output path.")
    args = parser.parse_args(argv)
    return (
        DPHoneyPaperEvidenceConfig(
            scanner_eval_path=args.scanner_eval,
            generation_realism_eval_path=args.generation_realism_eval,
            statistical_distinguisher_eval_path=args.statistical_distinguisher_eval,
            smoke_path=args.smoke,
            audit_jsonl_path=args.audit_jsonl,
        ),
        args.output,
    )


def main() -> None:
    try:
        config, output_path = parse_args(tuple(sys.argv[1:]))
        report = build_dp_honey_paper_evidence_report(config)
        if output_path is not None:
            write_dp_honey_paper_evidence_report(output_path, report)
        sys.stdout.write(render_dp_honey_paper_evidence_report_json(report))
    except (DPHoneyPaperEvidenceError, OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc


def _dp_noised_bigram_check(generator_metadata: Mapping[str, object] | None) -> dict[str, JsonValue]:
    if generator_metadata is None:
        return _checklist_item(
            requirement_id="dp_noised_bigram_generation",
            paper_requirement=(
                "Generate honeytokens from format-specific character bigram models with Laplace-noised counts."
            ),
            status="missing",
            evidence={},
            gaps=("No DP-HONEY generator metadata was found in the runtime audit evidence.",),
        )
    required_fields = ("epsilon", "clip", "corpus_size", "registry_version", "spec_hash", "format_slug")
    missing_fields = tuple(field for field in required_fields if field not in generator_metadata)
    status = "met" if len(missing_fields) == 0 else "partial"
    return _checklist_item(
        requirement_id="dp_noised_bigram_generation",
        paper_requirement="Format-specific character bigram generation with Laplace noise and recorded DP parameters.",
        status=status,
        evidence={
            "format_slug": _json_value_or_none(generator_metadata.get("format_slug")),
            "epsilon": _json_value_or_none(generator_metadata.get("epsilon")),
            "clip": _json_value_or_none(generator_metadata.get("clip")),
            "corpus_size": _json_value_or_none(generator_metadata.get("corpus_size")),
            "registry_version": _json_value_or_none(generator_metadata.get("registry_version")),
            "spec_hash": _json_value_or_none(generator_metadata.get("spec_hash")),
        },
        gaps=tuple(f"missing generator metadata field: {field}" for field in missing_fields),
    )


def _format_fidelity_check(
    scanner_eval: Mapping[str, object],
    generation_realism_eval: Mapping[str, object],
) -> dict[str, JsonValue]:
    format_metrics = _mapping_list(scanner_eval.get("format_metrics"), "scanner_eval.format_metrics")
    all_formats_detected = len(format_metrics) > 0 and all(
        _int(metric.get("false_negative"), "false_negative") == 0 for metric in format_metrics
    )
    all_generated_valid = generation_realism_eval.get("all_generated_tokens_valid") is True
    all_reference_valid = generation_realism_eval.get("all_reference_tokens_valid") is True
    status = "met" if all_formats_detected and all_generated_valid and all_reference_valid else "missing"
    return _checklist_item(
        requirement_id="format_fidelity",
        paper_requirement="Validate generated honeytokens against the format grammar and scanner-recognized shapes.",
        status=status,
        evidence={
            "scannable_format_count": _json_value_or_none(scanner_eval.get("scannable_format_count")),
            "positive_example_count": _json_value_or_none(scanner_eval.get("positive_example_count")),
            "all_scannable_positive_examples_detected": all_formats_detected,
            "all_generated_tokens_valid": all_generated_valid,
            "all_reference_tokens_valid": all_reference_valid,
            "generation_realism_format_count": _json_value_or_none(generation_realism_eval.get("format_count")),
        },
        gaps=() if status == "met" else ("format validation or scanner-recognition evidence is incomplete",),
    )


def _statistical_realism_check(
    generation_realism_eval: Mapping[str, object],
    statistical_distinguisher_eval: Mapping[str, object] | None,
) -> dict[str, JsonValue]:
    bounded_gate = generation_realism_eval.get("bounded_sanity_gate_passed") is True
    statistical_suite_passed = _statistical_distinguisher_suite_passed(statistical_distinguisher_eval)
    paper_reference_source = _statistical_distinguisher_reference_is_paper_sufficient(statistical_distinguisher_eval)
    if bounded_gate and statistical_suite_passed and paper_reference_source:
        status = "met"
        gaps: tuple[str, ...] = ()
    elif bounded_gate and statistical_distinguisher_eval is None:
        status = "partial"
        gaps = (
            "Paper evaluates statistical distinguishers such as character-entropy tests, bigram likelihood, "
            "numeric-substring tests, and a discriminator MLP; this report provides bounded aggregate sanity "
            "metrics but not the full sealed distinguisher suite.",
        )
    elif bounded_gate and statistical_suite_passed:
        status = "partial"
        reference_source = _required_string(
            statistical_distinguisher_eval.get("reference_source"),
            "statistical_distinguisher_eval.reference_source",
        )
        gaps = (
            "The statistical distinguisher suite passed only against "
            f"{reference_source}; paper-faithful+ requires provider-like or real-credential-distribution "
            "reference evidence.",
        )
    elif bounded_gate:
        status = "partial"
        suite = _mapping(
            statistical_distinguisher_eval.get("statistical_distinguisher_suite"),
            "statistical_distinguisher_eval.statistical_distinguisher_suite",
        )
        failed_tests = tuple(
            test_name
            for test_name in REQUIRED_STATISTICAL_DISTINGUISHER_TESTS
            if _mapping(suite.get(test_name), f"statistical_distinguisher_eval.{test_name}").get("status")
            != "passed"
        )
        gaps = (
            "The statistical distinguisher suite ran but did not pass all required tests: "
            + ", ".join(failed_tests),
        )
    else:
        status = "missing"
        gaps = ("generation realism evidence is missing or failed its bounded sanity gate",)
    return _checklist_item(
        requirement_id="statistical_realism_distinguishers",
        paper_requirement=(
            "Evaluate generated honeytokens against statistical distinguishers rather than only structural validity."
        ),
        status=status,
        evidence={
            "generation_realism_schema_version": _json_value_or_none(generation_realism_eval.get("schema_version")),
            "generation_realism_status": _json_value_or_none(generation_realism_eval.get("status")),
            "bounded_sanity_gate_passed": bounded_gate,
            "synthetic_registry_statistical_distinguisher_passed": statistical_suite_passed,
            "paper_faithful_statistical_distinguisher": statistical_suite_passed and paper_reference_source,
            "declared_paper_faithful_statistical_distinguisher": _json_value_or_none(
                generation_realism_eval.get("paper_faithful_statistical_distinguisher")
            ),
            "metric_families": _json_value_or_none(generation_realism_eval.get("metric_families")),
            "all_metrics_finite": _json_value_or_none(generation_realism_eval.get("all_metrics_finite")),
            "count_per_format": _json_value_or_none(generation_realism_eval.get("count_per_format")),
            "format_count": _json_value_or_none(generation_realism_eval.get("format_count")),
            "statistical_distinguisher_eval_present": statistical_distinguisher_eval is not None,
            "statistical_distinguisher_eval_passed": statistical_suite_passed,
            "paper_sufficient_reference_source": paper_reference_source,
            "reference_source": _json_value_or_none(
                None
                if statistical_distinguisher_eval is None
                else statistical_distinguisher_eval.get("reference_source")
            ),
            "statistical_distinguisher_test_statuses": _statistical_distinguisher_test_statuses(
                statistical_distinguisher_eval
            ),
        },
        gaps=gaps,
    )


def _conformal_check(scanner_eval: Mapping[str, object]) -> dict[str, JsonValue]:
    calibration = _mapping(scanner_eval.get("conformal_calibration"), "scanner_eval.conformal_calibration")
    implemented = calibration.get("implemented") is True
    target_alpha = _number_or_none(calibration.get("target_alpha"), "scanner_eval.conformal_calibration.target_alpha")
    calibration_count = _int_or_none(
        calibration.get("calibration_benign_count"),
        "scanner_eval.conformal_calibration.calibration_benign_count",
    )
    paper_alpha = target_alpha is not None and target_alpha <= _MAX_PAPER_CONFORMAL_ALPHA
    enough_calibration = (
        calibration_count is not None and calibration_count >= _MIN_PAPER_BENIGN_CALIBRATION_COUNT
    )
    status = "met" if (
        implemented
        and calibration.get("status") == "split_conformal_confidence_threshold"
        and paper_alpha
        and enough_calibration
    ) else "missing"
    gaps: list[str] = []
    if not implemented or calibration.get("status") != "split_conformal_confidence_threshold":
        gaps.append("split conformal calibration evidence missing or malformed")
    if not paper_alpha:
        gaps.append("target_alpha must be <= 0.01 to match the paper's 0.99 coverage target")
    if not enough_calibration:
        gaps.append("benign calibration count must be >= 1000 for paper-shaped evidence")
    return _checklist_item(
        requirement_id="split_conformal_calibration",
        paper_requirement="Choose detector thresholds from held-out benign calibration scores at target alpha.",
        status=status,
        evidence={
            "target_alpha": _json_value_or_none(calibration.get("target_alpha")),
            "target_coverage": _json_value_or_none(calibration.get("target_coverage")),
            "calibration_benign_count": _json_value_or_none(calibration.get("calibration_benign_count")),
            "threshold": _json_value_or_none(calibration.get("threshold")),
            "recommended_min_confidence": _json_value_or_none(calibration.get("recommended_min_confidence")),
            "empirical_calibration_false_positive_rate": _json_value_or_none(
                calibration.get("empirical_calibration_false_positive_rate")
            ),
            "empirical_calibration_coverage": _json_value_or_none(
                calibration.get("empirical_calibration_coverage")
            ),
        },
        gaps=tuple(gaps),
    )


def _scanner_fn_fp_check(scanner_eval: Mapping[str, object]) -> dict[str, JsonValue]:
    counts = _mapping(scanner_eval.get("counts"), "scanner_eval.counts")
    has_counts = all(key in counts for key in ("true_positive", "true_negative", "false_positive", "false_negative"))
    has_rates = all(
        key in scanner_eval for key in ("false_positive_rate", "false_negative_rate", "precision", "recall")
    )
    positive_count = _int_or_none(scanner_eval.get("positive_example_count"), "scanner_eval.positive_example_count")
    negative_count = _int_or_none(scanner_eval.get("negative_example_count"), "scanner_eval.negative_example_count")
    enough_positive = positive_count is not None and positive_count >= _MIN_PAPER_POSITIVE_EVAL_COUNT
    enough_negative = negative_count is not None and negative_count >= _MIN_PAPER_BENIGN_EVAL_COUNT
    status = "met" if has_counts and has_rates and enough_positive and enough_negative else "missing"
    gaps: list[str] = []
    if not has_counts or not has_rates:
        gaps.append("scanner FP/FN evidence is incomplete")
    if not enough_positive:
        gaps.append("positive scanner eval count must be >= 1000")
    if not enough_negative:
        gaps.append("benign scanner eval count must be >= 1000")
    return _checklist_item(
        requirement_id="scanner_fn_fp",
        paper_requirement="Report canary detector false positives and false negatives separately.",
        status=status,
        evidence={
            "counts": _json_mapping_or_empty(counts),
            "positive_example_count": _json_value_or_none(scanner_eval.get("positive_example_count")),
            "negative_example_count": _json_value_or_none(scanner_eval.get("negative_example_count")),
            "precision": _json_value_or_none(scanner_eval.get("precision")),
            "recall": _json_value_or_none(scanner_eval.get("recall")),
            "false_positive_rate": _json_value_or_none(scanner_eval.get("false_positive_rate")),
            "false_negative_rate": _json_value_or_none(scanner_eval.get("false_negative_rate")),
        },
        gaps=tuple(gaps),
    )


def _gateway_substitution_check(
    checks: Mapping[str, object],
    audit_records: tuple[Mapping[str, object], ...],
) -> dict[str, JsonValue]:
    benign_chat = _mapping(checks.get("benign_chat"), "checks.benign_chat")
    audit_has_dp_honey_source = _audit_has_sensitive_span_source(audit_records, "dp_honey")
    substituted = benign_chat.get("credential_slot_status") == "honeytoken_substituted"
    active = benign_chat.get("dp_honey_status") == "active"
    status = "met" if substituted and active and audit_has_dp_honey_source else "missing"
    return _checklist_item(
        requirement_id="gateway_substitution_and_ledger",
        paper_requirement="Insert canary credentials into model-visible context while binding them to a ledger.",
        status=status,
        evidence={
            "credential_slot_status": _json_value_or_none(benign_chat.get("credential_slot_status")),
            "dp_honey_status": _json_value_or_none(benign_chat.get("dp_honey_status")),
            "audit_sensitive_span_source_dp_honey": audit_has_dp_honey_source,
        },
        gaps=() if status == "met" else ("gateway substitution or ledger binding evidence missing",),
    )


def _output_leak_detection_check(checks: Mapping[str, object]) -> dict[str, JsonValue]:
    metadata_slot = _mapping(checks.get("metadata_slot_canary_leak"), "checks.metadata_slot_canary_leak")
    encoded = _mapping(checks.get("encoded_canary_leak"), "checks.encoded_canary_leak")
    exact_ok = metadata_slot.get("final_action") in {"escalate", "block"}
    encoded_ok = encoded.get("final_action") in {"escalate", "block"}
    status = "met" if exact_ok and encoded_ok else "missing"
    return _checklist_item(
        requirement_id="output_leak_detection",
        paper_requirement="Detect direct or encoded canary leakage in model outputs.",
        status=status,
        evidence={
            "exact_or_registered_leak_final_action": _json_value_or_none(metadata_slot.get("final_action")),
            "encoded_leak_final_action": _json_value_or_none(encoded.get("final_action")),
        },
        gaps=() if status == "met" else ("exact and encoded output leak controls are not both proven",),
    )


def _tool_argument_check(checks: Mapping[str, object]) -> dict[str, JsonValue]:
    guard = _mapping(checks.get("provider_egress_guard_block"), "checks.provider_egress_guard_block")
    tool_canary = _mapping(checks.get("tool_argument_canary_leak"), "checks.tool_argument_canary_leak")
    blocks_raw_tool_secret = guard.get("provider_status") == "skipped" and guard.get("final_action") == "block"
    blocks_tool_canary = (
        tool_canary.get("provider_status") == "skipped"
        and tool_canary.get("final_action") in {"block", "escalate"}
        and tool_canary.get("tool_canary_action") in {"block", "escalate"}
        and tool_canary.get("nimbus_tool_action") == "block"
        and _positive_number(tool_canary.get("nimbus_tool_turn_estimated_leakage_bits"))
    )
    if blocks_raw_tool_secret and blocks_tool_canary:
        status = "met"
        gaps: tuple[str, ...] = ()
    elif blocks_raw_tool_secret or blocks_tool_canary:
        status = "partial"
        gaps = ("tool argument leakage evidence is incomplete across raw-secret and planted-canary paths",)
    else:
        status = "missing"
        gaps = ("tool argument leakage evidence is missing",)
    return _checklist_item(
        requirement_id="tool_argument_leakage",
        paper_requirement=(
            "Apply canary and leakage-accounting logic to serialized tool-call arguments before dispatch."
        ),
        status=status,
        evidence={
            "raw_tool_payload_blocked_before_provider": blocks_raw_tool_secret,
            "planted_canary_tool_payload_blocked_before_provider": blocks_tool_canary,
            "tool_canary_final_action": _json_value_or_none(tool_canary.get("final_action")),
            "tool_canary_detector_action": _json_value_or_none(tool_canary.get("tool_canary_action")),
            "nimbus_tool_detector_action": _json_value_or_none(tool_canary.get("nimbus_tool_action")),
            "nimbus_tool_turn_estimated_leakage_bits": _json_value_or_none(
                tool_canary.get("nimbus_tool_turn_estimated_leakage_bits")
            ),
            "tool_call_name": _first_match_field(guard, "tool_call_name"),
            "argument_path": _first_match_field(guard, "argument_path"),
        },
        gaps=gaps,
    )


def _redacted_audit_check(
    audit_text: str,
    audit_records: tuple[Mapping[str, object], ...],
) -> dict[str, JsonValue]:
    forbidden_present = tuple(marker for marker in _FORBIDDEN_AUDIT_MARKERS if marker in audit_text)
    redaction_present = "[REDACTED_SENSITIVE]" in audit_text
    runtime_evidence_present = all(
        _mapping(record.get("runtime_evidence"), "audit.runtime_evidence").get("schema_version")
        == "aegis.audit_runtime_evidence/v1"
        for record in audit_records
    )
    status = "met" if redaction_present and len(forbidden_present) == 0 and runtime_evidence_present else "missing"
    return _checklist_item(
        requirement_id="redacted_audit",
        paper_requirement="Produce audit evidence without copying raw production secrets or matched secret values.",
        status=status,
        evidence={
            "event_count": len(audit_records),
            "redaction_marker_present": redaction_present,
            "forbidden_audit_markers_present": list(forbidden_present),
            "runtime_evidence_present": runtime_evidence_present,
        },
        gaps=() if status == "met" else ("audit redaction or runtime evidence is incomplete",),
    )


def _scanner_metrics(scanner_eval: Mapping[str, object]) -> dict[str, JsonValue]:
    return {
        "target_alpha": _json_value_or_none(scanner_eval.get("target_alpha")),
        "target_coverage": _json_value_or_none(scanner_eval.get("target_coverage")),
        "positive_example_count": _json_value_or_none(scanner_eval.get("positive_example_count")),
        "negative_example_count": _json_value_or_none(scanner_eval.get("negative_example_count")),
        "scannable_format_count": _json_value_or_none(scanner_eval.get("scannable_format_count")),
        "precision": _json_value_or_none(scanner_eval.get("precision")),
        "recall": _json_value_or_none(scanner_eval.get("recall")),
        "false_positive_rate": _json_value_or_none(scanner_eval.get("false_positive_rate")),
        "false_negative_rate": _json_value_or_none(scanner_eval.get("false_negative_rate")),
    }


def _generation_realism_metrics(generation_realism_eval: Mapping[str, object]) -> dict[str, JsonValue]:
    return {
        "status": _json_value_or_none(generation_realism_eval.get("status")),
        "count_per_format": _json_value_or_none(generation_realism_eval.get("count_per_format")),
        "format_count": _json_value_or_none(generation_realism_eval.get("format_count")),
        "scannable_format_count": _json_value_or_none(generation_realism_eval.get("scannable_format_count")),
        "all_generated_tokens_valid": _json_value_or_none(generation_realism_eval.get("all_generated_tokens_valid")),
        "all_reference_tokens_valid": _json_value_or_none(generation_realism_eval.get("all_reference_tokens_valid")),
        "all_metrics_finite": _json_value_or_none(generation_realism_eval.get("all_metrics_finite")),
        "bounded_sanity_gate_passed": _json_value_or_none(generation_realism_eval.get("bounded_sanity_gate_passed")),
        "paper_faithful_statistical_distinguisher": _json_value_or_none(
            generation_realism_eval.get("paper_faithful_statistical_distinguisher")
        ),
        "full_statistical_distinguisher_suite_passed": _full_statistical_distinguisher_suite_passed(
            generation_realism_eval
        ),
    }


def _statistical_distinguisher_metrics(
    statistical_distinguisher_eval: Mapping[str, object] | None,
) -> dict[str, JsonValue]:
    if statistical_distinguisher_eval is None:
        return {
            "present": False,
            "all_required_tests_passed": False,
            "synthetic_registry_statistical_distinguisher_passed": False,
            "paper_faithful_statistical_distinguisher": False,
            "paper_sufficient_reference_source": False,
            "test_statuses": {},
        }
    reference_source = _required_string(
        statistical_distinguisher_eval.get("reference_source"),
        "statistical_distinguisher_eval.reference_source",
    )
    all_required_tests_passed = _bool(
        statistical_distinguisher_eval.get("all_required_tests_passed"),
        "statistical_distinguisher_eval.all_required_tests_passed",
    )
    return {
        "present": True,
        "status": _json_value_or_none(statistical_distinguisher_eval.get("status")),
        "reference_source": reference_source,
        "train_count_per_format": _json_value_or_none(statistical_distinguisher_eval.get("train_count_per_format")),
        "test_count_per_format": _json_value_or_none(statistical_distinguisher_eval.get("test_count_per_format")),
        "alpha": _json_value_or_none(statistical_distinguisher_eval.get("alpha")),
        "all_required_tests_passed": all_required_tests_passed,
        "synthetic_registry_statistical_distinguisher_passed": all_required_tests_passed,
        "paper_sufficient_reference_source": _reference_source_is_paper_sufficient(reference_source),
        "paper_faithful_statistical_distinguisher": _json_value_or_none(
            statistical_distinguisher_eval.get("paper_faithful_statistical_distinguisher")
        ),
        "test_statuses": _statistical_distinguisher_test_statuses(statistical_distinguisher_eval),
    }


def _gateway_metrics(
    checks: Mapping[str, object],
    audit_records: tuple[Mapping[str, object], ...],
) -> dict[str, JsonValue]:
    benign_chat = _mapping(checks.get("benign_chat"), "checks.benign_chat")
    slot_leak = _mapping(checks.get("metadata_slot_canary_leak"), "checks.metadata_slot_canary_leak")
    encoded_leak = _mapping(checks.get("encoded_canary_leak"), "checks.encoded_canary_leak")
    guard = _mapping(checks.get("provider_egress_guard_block"), "checks.provider_egress_guard_block")
    tool_canary = _mapping(checks.get("tool_argument_canary_leak"), "checks.tool_argument_canary_leak")
    return {
        "audit_event_count": len(audit_records),
        "benign_credential_slot_status": _json_value_or_none(benign_chat.get("credential_slot_status")),
        "benign_final_action": _json_value_or_none(benign_chat.get("final_action")),
        "registered_canary_leak_final_action": _json_value_or_none(slot_leak.get("final_action")),
        "encoded_canary_leak_final_action": _json_value_or_none(encoded_leak.get("final_action")),
        "tool_payload_block_final_action": _json_value_or_none(guard.get("final_action")),
        "tool_payload_provider_status": _json_value_or_none(guard.get("provider_status")),
        "tool_canary_leak_final_action": _json_value_or_none(tool_canary.get("final_action")),
        "tool_canary_leak_provider_status": _json_value_or_none(tool_canary.get("provider_status")),
        "tool_canary_nimbus_action": _json_value_or_none(tool_canary.get("nimbus_tool_action")),
    }


def _checklist_item(
    requirement_id: str,
    paper_requirement: str,
    status: str,
    evidence: Mapping[str, JsonValue],
    gaps: Sequence[str],
) -> dict[str, JsonValue]:
    if status not in {"met", "partial", "missing"}:
        raise DPHoneyPaperEvidenceError(f"invalid checklist status '{status}'.")
    return {
        "requirement_id": requirement_id,
        "paper_requirement": paper_requirement,
        "status": status,
        "evidence": dict(evidence),
        "gaps": list(gaps),
    }


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(decoded, str(path))


def _read_optional_statistical_distinguisher_eval(path: Path | None) -> Mapping[str, object] | None:
    if path is None:
        return None
    return _validate_statistical_distinguisher_eval(_read_json_mapping(path))


def _validate_generation_realism_eval(report: Mapping[str, object]) -> Mapping[str, object]:
    if report.get("schema_version") != GENERATION_REALISM_EVAL_SCHEMA_VERSION:
        raise DPHoneyPaperEvidenceError(
            f"generation_realism_eval.schema_version must be {GENERATION_REALISM_EVAL_SCHEMA_VERSION}."
        )
    if report.get("status") != GENERATION_REALISM_EVAL_STATUS:
        raise DPHoneyPaperEvidenceError(f"generation_realism_eval.status must be {GENERATION_REALISM_EVAL_STATUS}.")
    count_per_format = _positive_int(report.get("count_per_format"), "generation_realism_eval.count_per_format")
    format_count = _positive_int(report.get("format_count"), "generation_realism_eval.format_count")
    scannable_count = _nonnegative_int(
        report.get("scannable_format_count"), "generation_realism_eval.scannable_format_count"
    )
    if scannable_count > format_count:
        raise DPHoneyPaperEvidenceError("generation_realism_eval.scannable_format_count must be <= format_count.")
    metric_families = frozenset(_string_list(report.get("metric_families"), "generation_realism_eval.metric_families"))
    required_metric_families = frozenset(
        ("format_validity", "duplicate_rate", "character_entropy", "model_avg_log_likelihood")
    )
    missing_metric_families = required_metric_families - metric_families
    if len(missing_metric_families) > 0:
        missing = ", ".join(sorted(missing_metric_families))
        raise DPHoneyPaperEvidenceError(f"generation_realism_eval.metric_families missing: {missing}.")
    audit_safety = _mapping(report.get("audit_safety"), "generation_realism_eval.audit_safety")
    if audit_safety.get("raw_secret_values_in_report") is not False:
        raise DPHoneyPaperEvidenceError(
            "generation_realism_eval.audit_safety.raw_secret_values_in_report must be false."
        )
    if audit_safety.get("finding_payload_redacted") is not True:
        raise DPHoneyPaperEvidenceError("generation_realism_eval.audit_safety.finding_payload_redacted must be true.")
    _validate_generator_parameters(report.get("generator_parameters"), "generation_realism_eval.generator_parameters")
    metrics = _mapping_list(report.get("format_metrics"), "generation_realism_eval.format_metrics")
    if len(metrics) != format_count:
        raise DPHoneyPaperEvidenceError("generation_realism_eval.format_metrics length must equal format_count.")
    generated_valid_rates: list[float] = []
    reference_valid_rates: list[float] = []
    finite_metric_flags: list[bool] = []
    bounded_gate_flags: list[bool] = []
    for index, metric in enumerate(metrics):
        field_prefix = f"generation_realism_eval.format_metrics[{index}]"
        _required_string(metric.get("format_slug"), f"{field_prefix}.format_slug")
        generated_count = _int(metric.get("generated_count"), f"{field_prefix}.generated_count")
        reference_count = _int(metric.get("reference_count"), f"{field_prefix}.reference_count")
        if generated_count != count_per_format:
            raise DPHoneyPaperEvidenceError(f"{field_prefix}.generated_count must equal count_per_format.")
        if reference_count != count_per_format:
            raise DPHoneyPaperEvidenceError(f"{field_prefix}.reference_count must equal count_per_format.")
        generated_valid_rates.append(
            _rate_number(metric.get("generated_validity_rate"), f"{field_prefix}.generated_validity_rate")
        )
        reference_valid_rates.append(
            _rate_number(metric.get("reference_validity_rate"), f"{field_prefix}.reference_validity_rate")
        )
        _rate_number(metric.get("generated_duplicate_rate"), f"{field_prefix}.generated_duplicate_rate")
        _rate_number(metric.get("reference_duplicate_rate"), f"{field_prefix}.reference_duplicate_rate")
        _finite_number(metric.get("generated_char_entropy_bits"), f"{field_prefix}.generated_char_entropy_bits")
        _finite_number(metric.get("reference_char_entropy_bits"), f"{field_prefix}.reference_char_entropy_bits")
        _finite_number(metric.get("char_entropy_delta_bits"), f"{field_prefix}.char_entropy_delta_bits")
        _finite_number(metric.get("generated_avg_log_likelihood"), f"{field_prefix}.generated_avg_log_likelihood")
        _finite_number(metric.get("reference_avg_log_likelihood"), f"{field_prefix}.reference_avg_log_likelihood")
        _finite_number(metric.get("avg_log_likelihood_delta"), f"{field_prefix}.avg_log_likelihood_delta")
        finite_metric_flags.append(_bool(metric.get("finite_metrics"), f"{field_prefix}.finite_metrics"))
        bounded_gate_flags.append(
            _bool(metric.get("bounded_sanity_gate_passed"), f"{field_prefix}.bounded_sanity_gate_passed")
        )
    _consistent_bool(
        report.get("all_generated_tokens_valid"),
        all(rate == 1.0 for rate in generated_valid_rates),
        "generation_realism_eval.all_generated_tokens_valid",
    )
    _consistent_bool(
        report.get("all_reference_tokens_valid"),
        all(rate == 1.0 for rate in reference_valid_rates),
        "generation_realism_eval.all_reference_tokens_valid",
    )
    _consistent_bool(
        report.get("all_metrics_finite"),
        all(finite_metric_flags),
        "generation_realism_eval.all_metrics_finite",
    )
    _consistent_bool(
        report.get("bounded_sanity_gate_passed"),
        all(bounded_gate_flags),
        "generation_realism_eval.bounded_sanity_gate_passed",
    )
    declared_paper_faithful = _bool(
        report.get("paper_faithful_statistical_distinguisher"),
        "generation_realism_eval.paper_faithful_statistical_distinguisher",
    )
    if declared_paper_faithful and not _full_statistical_distinguisher_suite_passed(report):
        raise DPHoneyPaperEvidenceError(
            "generation_realism_eval.paper_faithful_statistical_distinguisher requires a passed "
            "statistical_distinguisher_suite."
        )
    return report


def _validate_statistical_distinguisher_eval(report: Mapping[str, object]) -> Mapping[str, object]:
    if report.get("schema_version") != STATISTICAL_DISTINGUISHER_EVAL_SCHEMA_VERSION:
        raise DPHoneyPaperEvidenceError(
            "statistical_distinguisher_eval.schema_version must be "
            f"{STATISTICAL_DISTINGUISHER_EVAL_SCHEMA_VERSION}."
        )
    if report.get("status") != STATISTICAL_DISTINGUISHER_EVAL_STATUS:
        raise DPHoneyPaperEvidenceError(
            f"statistical_distinguisher_eval.status must be {STATISTICAL_DISTINGUISHER_EVAL_STATUS}."
        )
    _positive_int(report.get("train_count_per_format"), "statistical_distinguisher_eval.train_count_per_format")
    _positive_int(report.get("test_count_per_format"), "statistical_distinguisher_eval.test_count_per_format")
    format_count = _positive_int(report.get("format_count"), "statistical_distinguisher_eval.format_count")
    scannable_count = _nonnegative_int(
        report.get("scannable_format_count"), "statistical_distinguisher_eval.scannable_format_count"
    )
    if scannable_count > format_count:
        raise DPHoneyPaperEvidenceError(
            "statistical_distinguisher_eval.scannable_format_count must be <= format_count."
        )
    _finite_number(report.get("alpha"), "statistical_distinguisher_eval.alpha")
    if report.get("raw_values_serialized") is not False:
        raise DPHoneyPaperEvidenceError("statistical_distinguisher_eval.raw_values_serialized must be false.")
    audit_safety = _mapping(report.get("audit_safety"), "statistical_distinguisher_eval.audit_safety")
    if audit_safety.get("raw_secret_values_in_report") is not False:
        raise DPHoneyPaperEvidenceError(
            "statistical_distinguisher_eval.audit_safety.raw_secret_values_in_report must be false."
        )
    if audit_safety.get("finding_payload_redacted") is not True:
        raise DPHoneyPaperEvidenceError(
            "statistical_distinguisher_eval.audit_safety.finding_payload_redacted must be true."
        )
    _validate_generator_parameters(
        report.get("generator_parameters"), "statistical_distinguisher_eval.generator_parameters"
    )
    required_tests = frozenset(
        _string_list(report.get("required_tests"), "statistical_distinguisher_eval.required_tests")
    )
    missing_required_tests = frozenset(REQUIRED_STATISTICAL_DISTINGUISHER_TESTS) - required_tests
    if len(missing_required_tests) > 0:
        missing = ", ".join(sorted(missing_required_tests))
        raise DPHoneyPaperEvidenceError(f"statistical_distinguisher_eval.required_tests missing: {missing}.")
    suite = _mapping(
        report.get("statistical_distinguisher_suite"),
        "statistical_distinguisher_eval.statistical_distinguisher_suite",
    )
    test_statuses: list[bool] = []
    for test_name in REQUIRED_STATISTICAL_DISTINGUISHER_TESTS:
        test_result = _mapping(suite.get(test_name), f"statistical_distinguisher_eval.{test_name}")
        status = test_result.get("status")
        if status not in {"passed", "failed"}:
            raise DPHoneyPaperEvidenceError(f"statistical_distinguisher_eval.{test_name}.status is invalid.")
        _mapping(test_result.get("aggregate"), f"statistical_distinguisher_eval.{test_name}.aggregate")
        _required_string(
            test_result.get("pass_criterion"),
            f"statistical_distinguisher_eval.{test_name}.pass_criterion",
        )
        if test_name != "discriminator_mlp":
            format_metrics = _mapping_list(
                test_result.get("format_metrics"),
                f"statistical_distinguisher_eval.{test_name}.format_metrics",
            )
            if len(format_metrics) != format_count:
                raise DPHoneyPaperEvidenceError(
                    f"statistical_distinguisher_eval.{test_name}.format_metrics length must equal format_count."
                )
        test_statuses.append(status == "passed")
    all_tests_passed = all(test_statuses)
    _consistent_bool(
        report.get("all_required_tests_passed"),
        all_tests_passed,
        "statistical_distinguisher_eval.all_required_tests_passed",
    )
    declared_paper_faithful = _bool(
        report.get("paper_faithful_statistical_distinguisher"),
        "statistical_distinguisher_eval.paper_faithful_statistical_distinguisher",
    )
    reference_source = _required_string(
        report.get("reference_source"),
        "statistical_distinguisher_eval.reference_source",
    )
    if declared_paper_faithful and not all_tests_passed:
        raise DPHoneyPaperEvidenceError(
            "statistical_distinguisher_eval.paper_faithful_statistical_distinguisher requires all tests to pass."
        )
    if declared_paper_faithful and not _reference_source_is_paper_sufficient(reference_source):
        raise DPHoneyPaperEvidenceError(
            "statistical_distinguisher_eval.paper_faithful_statistical_distinguisher requires a provider-like "
            "or real-credential-distribution reference source."
        )
    expected_paper_faithful = all_tests_passed and _reference_source_is_paper_sufficient(reference_source)
    if declared_paper_faithful is not expected_paper_faithful:
        raise DPHoneyPaperEvidenceError(
            "statistical_distinguisher_eval.paper_faithful_statistical_distinguisher must match suite status "
            "and reference source sufficiency."
        )
    if _reference_source_is_paper_sufficient(reference_source):
        _validate_reference_feature_corpus_metadata(
            report.get("reference_feature_corpus"),
            reference_source,
            report.get("train_count_per_format"),
            report.get("test_count_per_format"),
        )
    elif report.get("reference_feature_corpus") not in {None, False}:
        raise DPHoneyPaperEvidenceError(
            "statistical_distinguisher_eval.reference_feature_corpus is only valid for provider-like or "
            "real-credential-distribution reference sources."
        )
    return report


def _validate_reference_feature_corpus_metadata(
    value: object,
    reference_source: str,
    train_count_per_format: object,
    test_count_per_format: object,
) -> None:
    metadata = _mapping(value, "statistical_distinguisher_eval.reference_feature_corpus")
    if metadata.get("schema_version") != REFERENCE_FEATURE_CORPUS_SCHEMA_VERSION:
        raise DPHoneyPaperEvidenceError(
            "statistical_distinguisher_eval.reference_feature_corpus.schema_version must be "
            f"{REFERENCE_FEATURE_CORPUS_SCHEMA_VERSION}."
        )
    if metadata.get("source") != reference_source:
        raise DPHoneyPaperEvidenceError(
            "statistical_distinguisher_eval.reference_feature_corpus.source must match reference_source."
        )
    _required_string(metadata.get("source_description"), "reference_feature_corpus.source_description")
    _required_string(metadata.get("source_generation_method"), "reference_feature_corpus.source_generation_method")
    _required_sha256(metadata.get("sha256"), "reference_feature_corpus.sha256")
    if metadata.get("raw_values_serialized") is not False:
        raise DPHoneyPaperEvidenceError("reference_feature_corpus.raw_values_serialized must be false.")
    feature_names = _string_list(metadata.get("feature_names"), "reference_feature_corpus.feature_names")
    if feature_names != REFERENCE_FEATURE_NAMES:
        raise DPHoneyPaperEvidenceError("reference_feature_corpus.feature_names must match DP-HONEY evaluator.")
    _positive_int(metadata.get("format_count"), "reference_feature_corpus.format_count")
    _consistent_int(
        metadata.get("train_count_per_format"),
        train_count_per_format,
        "reference_feature_corpus.train_count_per_format",
    )
    _consistent_int(
        metadata.get("test_count_per_format"),
        test_count_per_format,
        "reference_feature_corpus.test_count_per_format",
    )


def _validate_generator_parameters(value: object, field_name: str) -> Mapping[str, object]:
    parameters = _mapping(value, field_name)
    for key in GENERATOR_PARAMETER_FIELDS:
        if key in {"epsilon", "clip"}:
            _finite_number(parameters.get(key), f"{field_name}.{key}")
        elif key == "corpus_size":
            _positive_int(parameters.get(key), f"{field_name}.{key}")
        else:
            _nonnegative_int(parameters.get(key), f"{field_name}.{key}")
    return parameters


def _validate_generator_parameter_binding(
    generator_metadata: Mapping[str, object] | None,
    generation_realism_eval: Mapping[str, object],
    statistical_distinguisher_eval: Mapping[str, object] | None,
) -> None:
    if generator_metadata is None:
        return
    reference = _validate_generator_parameters(generator_metadata, "audit.dp_honey")
    _require_same_generator_parameters(
        reference=reference,
        candidate=_validate_generator_parameters(
            generation_realism_eval.get("generator_parameters"),
            "generation_realism_eval.generator_parameters",
        ),
        field_name="generation_realism_eval.generator_parameters",
    )
    if statistical_distinguisher_eval is not None:
        _require_same_generator_parameters(
            reference=reference,
            candidate=_validate_generator_parameters(
                statistical_distinguisher_eval.get("generator_parameters"),
                "statistical_distinguisher_eval.generator_parameters",
            ),
            field_name="statistical_distinguisher_eval.generator_parameters",
        )


def _require_same_generator_parameters(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    field_name: str,
) -> None:
    mismatches = tuple(
        key
        for key in GENERATOR_PARAMETER_FIELDS
        if _finite_number(reference.get(key), f"audit.dp_honey.{key}")
        != _finite_number(candidate.get(key), f"{field_name}.{key}")
    )
    if len(mismatches) > 0:
        fields = ", ".join(mismatches)
        raise DPHoneyPaperEvidenceError(f"{field_name} must match runtime audit DP-HONEY metadata: {fields}.")


def _read_jsonl_mappings(path: Path) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            decoded = json.loads(line)
            records.append(_mapping(decoded, f"{path}:{line_number}"))
    if len(records) == 0:
        raise DPHoneyPaperEvidenceError(f"No audit records found in {path}.")
    return tuple(records)


def _first_dp_honey_metadata(records: tuple[Mapping[str, object], ...]) -> Mapping[str, object] | None:
    for record in records:
        for metadata in _dp_honey_metadata_values(record):
            return metadata
    return None


def _dp_honey_metadata_values(value: object) -> Iterable[Mapping[str, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "dp_honey" and isinstance(child, Mapping):
                yield cast(Mapping[str, object], child)
            else:
                yield from _dp_honey_metadata_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _dp_honey_metadata_values(child)


def _audit_has_sensitive_span_source(records: tuple[Mapping[str, object], ...], source: str) -> bool:
    for record in records:
        turn = _mapping(record.get("normalized_turn"), "audit.normalized_turn")
        spans = _mapping_list(turn.get("sensitive_spans"), "audit.normalized_turn.sensitive_spans")
        if any(span.get("source") == source for span in spans):
            return True
    return False


def _first_match_field(mapping: Mapping[str, object], field_name: str) -> JsonValue:
    matches = _mapping_list(mapping.get("matches"), f"{field_name}.matches")
    if len(matches) == 0:
        return None
    return _json_value_or_none(matches[0].get(field_name))


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DPHoneyPaperEvidenceError(f"{field_name} must be an object.")
    return cast(Mapping[str, object], value)


def _mapping_list(value: object, field_name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise DPHoneyPaperEvidenceError(f"{field_name} must be a list.")
    mappings: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        mappings.append(_mapping(item, f"{field_name}[{index}]"))
    return tuple(mappings)


def _string_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DPHoneyPaperEvidenceError(f"{field_name} must be a list.")
    strings: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise DPHoneyPaperEvidenceError(f"{field_name}[{index}] must be a string.")
        strings.append(item)
    return tuple(strings)


def _json_mapping_or_empty(value: object) -> dict[str, JsonValue]:
    if value is None:
        return {}
    mapping = _mapping(value, "json mapping")
    return {str(key): _json_value_or_none(child) for key, child in mapping.items()}


def _json_value_or_none(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value_or_none(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value_or_none(child) for key, child in value.items()}
    return str(value)


def _full_statistical_distinguisher_suite_passed(generation_realism_eval: Mapping[str, object]) -> bool:
    if generation_realism_eval.get("paper_faithful_statistical_distinguisher") is not True:
        return False
    suite = generation_realism_eval.get("statistical_distinguisher_suite")
    if not isinstance(suite, Mapping):
        return False
    required_tests = (
        "character_entropy_tests",
        "bigram_likelihood_tests",
        "numeric_substring_tests",
        "discriminator_mlp",
    )
    for test_name in required_tests:
        test_result = suite.get(test_name)
        if not isinstance(test_result, Mapping) or test_result.get("status") != "passed":
            return False
    return True


def _statistical_distinguisher_suite_passed(statistical_distinguisher_eval: Mapping[str, object] | None) -> bool:
    if statistical_distinguisher_eval is None:
        return False
    if statistical_distinguisher_eval.get("all_required_tests_passed") is not True:
        return False
    suite = statistical_distinguisher_eval.get("statistical_distinguisher_suite")
    if not isinstance(suite, Mapping):
        return False
    return all(
        isinstance(suite.get(test_name), Mapping) and suite[test_name].get("status") == "passed"
        for test_name in REQUIRED_STATISTICAL_DISTINGUISHER_TESTS
    )


def _statistical_distinguisher_reference_is_paper_sufficient(
    statistical_distinguisher_eval: Mapping[str, object] | None,
) -> bool:
    if statistical_distinguisher_eval is None:
        return False
    reference_source = statistical_distinguisher_eval.get("reference_source")
    if not isinstance(reference_source, str):
        return False
    return _reference_source_is_paper_sufficient(reference_source)


def _reference_source_is_paper_sufficient(reference_source: str) -> bool:
    return reference_source in {"provider_like_sealed_holdout", "real_credential_distribution_redacted_features"}


def _statistical_distinguisher_test_statuses(
    statistical_distinguisher_eval: Mapping[str, object] | None,
) -> dict[str, JsonValue]:
    if statistical_distinguisher_eval is None:
        return {}
    suite = _mapping(
        statistical_distinguisher_eval.get("statistical_distinguisher_suite"),
        "statistical_distinguisher_eval.statistical_distinguisher_suite",
    )
    statuses: dict[str, JsonValue] = {}
    for test_name in REQUIRED_STATISTICAL_DISTINGUISHER_TESTS:
        test_result = _mapping(suite.get(test_name), f"statistical_distinguisher_eval.{test_name}")
        statuses[test_name] = _json_value_or_none(test_result.get("status"))
    return statuses


def _consistent_bool(value: object, expected: bool, field_name: str) -> None:
    actual = _bool(value, field_name)
    if actual is not expected:
        raise DPHoneyPaperEvidenceError(f"{field_name} must be consistent with per-format metrics.")


def _consistent_int(value: object, expected: object, field_name: str) -> None:
    actual_integer = _positive_int(value, field_name)
    expected_integer = _positive_int(expected, f"{field_name}.expected")
    if actual_integer != expected_integer:
        raise DPHoneyPaperEvidenceError(f"{field_name} must match statistical distinguisher split counts.")


def _bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise DPHoneyPaperEvidenceError(f"{field_name} must be a boolean.")
    return value


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise DPHoneyPaperEvidenceError(f"{field_name} must be a non-empty string.")
    return value


def _required_sha256(value: object, field_name: str) -> str:
    text = _required_string(value, field_name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise DPHoneyPaperEvidenceError(f"{field_name} must be a lowercase SHA-256 hex digest.")
    return text


def _int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DPHoneyPaperEvidenceError(f"{field_name} must be an integer.")
    return value


def _int_or_none(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _int(value, field_name)


def _positive_int(value: object, field_name: str) -> int:
    integer = _int(value, field_name)
    if integer < 1:
        raise DPHoneyPaperEvidenceError(f"{field_name} must be positive.")
    return integer


def _nonnegative_int(value: object, field_name: str) -> int:
    integer = _int(value, field_name)
    if integer < 0:
        raise DPHoneyPaperEvidenceError(f"{field_name} must be non-negative.")
    return integer


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DPHoneyPaperEvidenceError(f"{field_name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise DPHoneyPaperEvidenceError(f"{field_name} must be finite.")
    return numeric


def _number_or_none(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, field_name)


def _rate_number(value: object, field_name: str) -> float:
    numeric = _finite_number(value, field_name)
    if numeric < 0.0 or numeric > 1.0:
        raise DPHoneyPaperEvidenceError(f"{field_name} must be in [0.0, 1.0].")
    return numeric


def _positive_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return value > 0.0


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
