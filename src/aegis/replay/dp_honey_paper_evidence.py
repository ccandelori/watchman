from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

DP_HONEY_PAPER_EVIDENCE_SCHEMA_VERSION = "aegis.dp_honey_paper_evidence/v1"
_FORBIDDEN_AUDIT_MARKERS = ("{{CREDENTIAL:", "ghp_", "github_pat_", "sk_live_", "AKIA")


class DPHoneyPaperEvidenceError(ValueError):
    """Raised when DP-HONEY paper evidence inputs are missing or malformed."""


@dataclass(frozen=True)
class DPHoneyPaperEvidenceConfig:
    scanner_eval_path: Path
    smoke_path: Path
    audit_jsonl_path: Path


def build_dp_honey_paper_evidence_report(config: DPHoneyPaperEvidenceConfig) -> dict[str, JsonValue]:
    scanner_eval = _read_json_mapping(config.scanner_eval_path)
    smoke = _read_json_mapping(config.smoke_path)
    audit_records = _read_jsonl_mappings(config.audit_jsonl_path)
    audit_text = config.audit_jsonl_path.read_text(encoding="utf-8")
    generator_metadata = _first_dp_honey_metadata(audit_records)
    checks = _mapping(smoke.get("checks"), "smoke.checks")

    checklist = (
        _dp_noised_bigram_check(generator_metadata),
        _format_fidelity_check(scanner_eval),
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
    return {
        "schema_version": DP_HONEY_PAPER_EVIDENCE_SCHEMA_VERSION,
        "component": "dp_honey",
        "paper_source": "Research/2606.04141v1.pdf sections 4.3, 5.3, and 6",
        "promotion_status": "paper_aligned_operational_beta",
        "paper_faithful_plus": False,
        "promotion_eligible": False,
        "summary": (
            "DP-HONEY has DP-noised bigram provenance, split-conformal scanner calibration, held-out scanner "
            "FP/FN evidence, gateway substitution, canary leak detection, provider-egress blocking, and redacted "
            "audit evidence. It is still not paper-faithful+ because statistical distinguisher realism and "
            "tool-argument canary/leakage accounting remain incomplete."
        ),
        "artifact_hashes": {
            "scanner_eval_sha256": _sha256_file(config.scanner_eval_path),
            "smoke_sha256": _sha256_file(config.smoke_path),
            "audit_jsonl_sha256": _sha256_file(config.audit_jsonl_path),
        },
        "scanner_metrics": _scanner_metrics(scanner_eval),
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


def parse_args(argv: Sequence[str]) -> tuple[DPHoneyPaperEvidenceConfig, Path | None]:
    parser = argparse.ArgumentParser(description="Build a DP-HONEY paper-faithfulness evidence report.")
    parser.add_argument("--scanner-eval", required=True, type=Path, help="Path to dp_honey_scanner_eval_v1.json.")
    parser.add_argument("--smoke", required=True, type=Path, help="Path to default mock-provider smoke JSON.")
    parser.add_argument("--audit-jsonl", required=True, type=Path, help="Path to matching smoke audit JSONL.")
    parser.add_argument("--output", required=False, type=Path, help="Optional JSON output path.")
    args = parser.parse_args(argv)
    return (
        DPHoneyPaperEvidenceConfig(
            scanner_eval_path=args.scanner_eval,
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


def _format_fidelity_check(scanner_eval: Mapping[str, object]) -> dict[str, JsonValue]:
    format_metrics = _mapping_list(scanner_eval.get("format_metrics"), "scanner_eval.format_metrics")
    all_formats_detected = len(format_metrics) > 0 and all(
        _int(metric.get("false_negative"), "false_negative") == 0 for metric in format_metrics
    )
    gaps = (
        "Paper evaluates statistical distinguishers such as entropy, bigram likelihood, and discriminator models; "
        "this report only proves structural format fidelity and scanner recall."
    )
    return _checklist_item(
        requirement_id="format_fidelity",
        paper_requirement="Evaluate generated honeytokens for format fidelity and statistical distinguishability.",
        status="partial" if all_formats_detected else "missing",
        evidence={
            "scannable_format_count": _json_value_or_none(scanner_eval.get("scannable_format_count")),
            "positive_example_count": _json_value_or_none(scanner_eval.get("positive_example_count")),
            "all_scannable_positive_examples_detected": all_formats_detected,
        },
        gaps=(gaps,),
    )


def _conformal_check(scanner_eval: Mapping[str, object]) -> dict[str, JsonValue]:
    calibration = _mapping(scanner_eval.get("conformal_calibration"), "scanner_eval.conformal_calibration")
    implemented = calibration.get("implemented") is True
    status = "met" if implemented and calibration.get("status") == "split_conformal_confidence_threshold" else "missing"
    return _checklist_item(
        requirement_id="split_conformal_calibration",
        paper_requirement="Choose detector thresholds from held-out benign calibration scores at target alpha.",
        status=status,
        evidence={
            "target_alpha": _json_value_or_none(calibration.get("target_alpha")),
            "calibration_benign_count": _json_value_or_none(calibration.get("calibration_benign_count")),
            "threshold": _json_value_or_none(calibration.get("threshold")),
            "recommended_min_confidence": _json_value_or_none(calibration.get("recommended_min_confidence")),
            "empirical_calibration_false_positive_rate": _json_value_or_none(
                calibration.get("empirical_calibration_false_positive_rate")
            ),
        },
        gaps=() if status == "met" else ("split conformal calibration evidence missing or malformed",),
    )


def _scanner_fn_fp_check(scanner_eval: Mapping[str, object]) -> dict[str, JsonValue]:
    counts = _mapping(scanner_eval.get("counts"), "scanner_eval.counts")
    has_counts = all(key in counts for key in ("true_positive", "true_negative", "false_positive", "false_negative"))
    has_rates = all(
        key in scanner_eval for key in ("false_positive_rate", "false_negative_rate", "precision", "recall")
    )
    status = "met" if has_counts and has_rates else "missing"
    return _checklist_item(
        requirement_id="scanner_fn_fp",
        paper_requirement="Report canary detector false positives and false negatives separately.",
        status=status,
        evidence={
            "counts": _json_mapping_or_empty(counts),
            "precision": _json_value_or_none(scanner_eval.get("precision")),
            "recall": _json_value_or_none(scanner_eval.get("recall")),
            "false_positive_rate": _json_value_or_none(scanner_eval.get("false_positive_rate")),
            "false_negative_rate": _json_value_or_none(scanner_eval.get("false_negative_rate")),
        },
        gaps=() if status == "met" else ("scanner FP/FN evidence is incomplete",),
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
    blocks_raw_tool_secret = guard.get("provider_status") == "skipped" and guard.get("final_action") == "block"
    return _checklist_item(
        requirement_id="tool_argument_leakage",
        paper_requirement=(
            "Apply canary and leakage-accounting logic to serialized tool-call arguments before dispatch."
        ),
        status="partial" if blocks_raw_tool_secret else "missing",
        evidence={
            "raw_tool_payload_blocked_before_provider": blocks_raw_tool_secret,
            "tool_call_name": _first_match_field(guard, "tool_call_name"),
            "argument_path": _first_match_field(guard, "argument_path"),
        },
        gaps=(
            "Provider egress guard blocks raw credential-shaped tool payloads, but DP-HONEY canary detection and "
            "NIMBUS-style leakage accounting are not yet applied to outbound serialized tool-call arguments before "
            "tool dispatch.",
        ),
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
        "positive_example_count": _json_value_or_none(scanner_eval.get("positive_example_count")),
        "negative_example_count": _json_value_or_none(scanner_eval.get("negative_example_count")),
        "scannable_format_count": _json_value_or_none(scanner_eval.get("scannable_format_count")),
        "precision": _json_value_or_none(scanner_eval.get("precision")),
        "recall": _json_value_or_none(scanner_eval.get("recall")),
        "false_positive_rate": _json_value_or_none(scanner_eval.get("false_positive_rate")),
        "false_negative_rate": _json_value_or_none(scanner_eval.get("false_negative_rate")),
    }


def _gateway_metrics(
    checks: Mapping[str, object],
    audit_records: tuple[Mapping[str, object], ...],
) -> dict[str, JsonValue]:
    benign_chat = _mapping(checks.get("benign_chat"), "checks.benign_chat")
    slot_leak = _mapping(checks.get("metadata_slot_canary_leak"), "checks.metadata_slot_canary_leak")
    encoded_leak = _mapping(checks.get("encoded_canary_leak"), "checks.encoded_canary_leak")
    guard = _mapping(checks.get("provider_egress_guard_block"), "checks.provider_egress_guard_block")
    return {
        "audit_event_count": len(audit_records),
        "benign_credential_slot_status": _json_value_or_none(benign_chat.get("credential_slot_status")),
        "benign_final_action": _json_value_or_none(benign_chat.get("final_action")),
        "registered_canary_leak_final_action": _json_value_or_none(slot_leak.get("final_action")),
        "encoded_canary_leak_final_action": _json_value_or_none(encoded_leak.get("final_action")),
        "tool_payload_block_final_action": _json_value_or_none(guard.get("final_action")),
        "tool_payload_provider_status": _json_value_or_none(guard.get("provider_status")),
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


def _int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DPHoneyPaperEvidenceError(f"{field_name} must be an integer.")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
