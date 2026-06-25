from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import NoReturn, cast
from urllib.parse import urlparse

from aegis.core.contracts import JsonValue

PROVIDER_SMOKE_EVIDENCE_SCHEMA_VERSION = "aegis.provider_smoke_evidence/v1"
_PREFLIGHT_SCHEMA_VERSION = "aegis.provider_preflight/v1"
_LOOPBACK_PROVIDER_REQUEST_SCHEMA_VERSION = "aegis.loopback_openai_provider_request/v1"
_AUDIT_RUNTIME_EVIDENCE_SCHEMA_VERSION = "aegis.audit_runtime_evidence/v1"
_PLACEHOLDER_MARKER = "{{CREDENTIAL:"
_SMOKE_EGRESS_TRACE_ID = "smoke-egress-guard-trace"
_SMOKE_BENIGN_TRACE_ID = "smoke-benign-trace"


class ProviderSmokeEvidenceError(ValueError):
    """Raised when provider smoke evidence cannot be parsed or verified."""


@dataclass(frozen=True)
class ProviderSmokeEvidenceConfig:
    preflight_path: Path
    smoke_path: Path
    provider_request_log_path: Path
    audit_jsonl_path: Path
    output_path: Path | None
    forbidden_markers: tuple[str, ...]


def parse_args(argv: Sequence[str]) -> ProviderSmokeEvidenceConfig:
    parser = argparse.ArgumentParser(
        description="Verify saved local loopback real-provider smoke evidence for Aegis gateway hardening."
    )
    parser.add_argument("--preflight", required=True, type=Path, help="Provider preflight JSON report path.")
    parser.add_argument("--smoke", required=True, type=Path, help="Gateway smoke JSON report path.")
    parser.add_argument(
        "--provider-request-log",
        required=True,
        type=Path,
        help="Loopback OpenAI-compatible provider request-log JSONL path.",
    )
    parser.add_argument("--audit-jsonl", required=True, type=Path, help="Durable gateway audit JSONL path.")
    parser.add_argument("--output", required=False, type=Path, help="Optional verification JSON report path.")
    parser.add_argument(
        "--forbidden-marker",
        action="append",
        required=True,
        help="Sensitive marker that must not appear in saved smoke, request, or audit evidence.",
    )
    args = parser.parse_args(argv)
    forbidden_markers = tuple(str(marker) for marker in args.forbidden_marker)
    _validate_forbidden_markers(forbidden_markers)
    return ProviderSmokeEvidenceConfig(
        preflight_path=args.preflight,
        smoke_path=args.smoke,
        provider_request_log_path=args.provider_request_log,
        audit_jsonl_path=args.audit_jsonl,
        output_path=args.output,
        forbidden_markers=forbidden_markers,
    )


def verify_provider_smoke_evidence(config: ProviderSmokeEvidenceConfig) -> dict[str, JsonValue]:
    _validate_forbidden_markers(config.forbidden_markers)
    preflight = _read_json_object(config.preflight_path, "preflight")
    smoke = _read_json_object(config.smoke_path, "smoke")
    provider_records = _read_jsonl_objects(config.provider_request_log_path, "provider request log")
    audit_records = _read_jsonl_objects(config.audit_jsonl_path, "audit JSONL")
    raw_artifact_text = _raw_artifact_text(
        paths=(config.preflight_path, config.smoke_path, config.provider_request_log_path, config.audit_jsonl_path)
    )

    checks = (
        _check_preflight(preflight),
        _check_smoke_report(smoke),
        _check_readiness(smoke),
        _check_capabilities(smoke),
        _check_benign_provider_completion(smoke),
        _check_egress_guard_pre_provider_block(smoke),
        _check_mock_only_probes_skipped(smoke),
        _check_provider_request_log(provider_records, config.forbidden_markers),
        _check_audit_trace_receipts(audit_records),
        _check_no_forbidden_markers(raw_artifact_text, config.forbidden_markers),
    )
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {
        "schema_version": PROVIDER_SMOKE_EVIDENCE_SCHEMA_VERSION,
        "status": status,
        "evidence_scope": "local_loopback_openai_compatible_adapter",
        "external_provider_evidence": False,
        "production_claim": "not_external_provider_production_evidence",
        "input_artifacts": _input_artifacts(config),
        "checks": [cast(JsonValue, check) for check in checks],
    }


def render_provider_smoke_evidence_json(report: dict[str, JsonValue]) -> str:
    return json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"


def main() -> None:
    try:
        config = parse_args(tuple(sys.argv[1:]))
        report = verify_provider_smoke_evidence(config)
        rendered = render_provider_smoke_evidence_json(report)
        if config.output_path is not None:
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            config.output_path.write_text(rendered, encoding="utf-8")
        sys.stdout.write(rendered)
    except ProviderSmokeEvidenceError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
    if report["status"] != "pass":
        raise SystemExit(1)


def _check_preflight(preflight: dict[str, JsonValue]) -> dict[str, JsonValue]:
    passed = (
        preflight.get("schema_version") == _PREFLIGHT_SCHEMA_VERSION
        and preflight.get("ready") is True
        and preflight.get("require_real_provider") is True
        and preflight.get("provider_kind") == "openai_compatible"
        and preflight.get("provider_name") == "openai_compatible"
        and preflight.get("mock_controls_enabled") is False
        and preflight.get("network_access") == "not_attempted"
    )
    observed = {
        "schema_version": preflight.get("schema_version"),
        "ready": preflight.get("ready"),
        "provider_kind": preflight.get("provider_kind"),
        "provider_name": preflight.get("provider_name"),
        "mock_controls_enabled": preflight.get("mock_controls_enabled"),
        "network_access": preflight.get("network_access"),
    }
    return _check(
        name="provider_preflight",
        passed=passed,
        detail="Preflight must prove openai_compatible config with mock controls disabled and no network request.",
        observed=observed,
    )


def _check_smoke_report(smoke: dict[str, JsonValue]) -> dict[str, JsonValue]:
    base_url = smoke.get("base_url")
    passed = (
        smoke.get("status") == "ok"
        and smoke.get("provider_mode") == "real-provider"
        and isinstance(base_url, str)
        and _is_loopback_http_url(base_url)
    )
    observed = {
        "status": smoke.get("status"),
        "provider_mode": smoke.get("provider_mode"),
        "base_url": base_url,
    }
    return _check(
        name="gateway_smoke_report",
        passed=passed,
        detail="Smoke report must be a successful local real-provider-mode gateway run.",
        observed=observed,
    )


def _check_readiness(smoke: dict[str, JsonValue]) -> dict[str, JsonValue]:
    readiness = _smoke_check(smoke, "gateway_readiness")
    passed = (
        readiness.get("status") == "ready"
        and readiness.get("provider_name") == "openai_compatible"
        and readiness.get("provider_mock_controls_enabled") is False
        and readiness.get("dp_honey_status") == "ready"
    )
    observed = {
        "status": readiness.get("status"),
        "provider_name": readiness.get("provider_name"),
        "provider_mock_controls_enabled": readiness.get("provider_mock_controls_enabled"),
        "dp_honey_status": readiness.get("dp_honey_status"),
        "cift_capability_mode": readiness.get("cift_capability_mode"),
    }
    return _check(
        name="gateway_readiness",
        passed=passed,
        detail="/ready evidence must show the OpenAI-compatible provider path with mock controls disabled.",
        observed=observed,
    )


def _check_capabilities(smoke: dict[str, JsonValue]) -> dict[str, JsonValue]:
    capabilities = _smoke_check(smoke, "capabilities")
    mock_response_modes = capabilities.get("mock_response_modes")
    passed = (
        capabilities.get("provider_mode") == "real-provider"
        and isinstance(mock_response_modes, list)
        and len(mock_response_modes) == 0
    )
    observed = {
        "provider_mode": capabilities.get("provider_mode"),
        "mock_response_modes": mock_response_modes,
    }
    return _check(
        name="gateway_capabilities",
        passed=passed,
        detail="Capabilities must advertise real-provider smoke mode with no mock response controls.",
        observed=observed,
    )


def _check_benign_provider_completion(smoke: dict[str, JsonValue]) -> dict[str, JsonValue]:
    benign = _smoke_check(smoke, "benign_chat")
    passed = (
        benign.get("final_action") == "allow"
        and benign.get("dp_honey_status") == "active"
        and benign.get("credential_slot_status") == "honeytoken_substituted"
        and benign.get("provider_status") == "completed"
        and _stage_status(benign, "provider") == "completed"
    )
    observed = {
        "final_action": benign.get("final_action"),
        "dp_honey_status": benign.get("dp_honey_status"),
        "credential_slot_status": benign.get("credential_slot_status"),
        "provider_status": benign.get("provider_status"),
    }
    return _check(
        name="benign_provider_completion",
        passed=passed,
        detail="Benign protected workflow must substitute a honeytoken and complete the provider call.",
        observed=observed,
    )


def _check_egress_guard_pre_provider_block(smoke: dict[str, JsonValue]) -> dict[str, JsonValue]:
    guard = _smoke_check(smoke, "provider_egress_guard_block")
    passed = (
        guard.get("final_action") == "block"
        and guard.get("guard_action") == "block"
        and guard.get("guard_reason") == "blocked_sensitive_value_before_provider_egress"
        and guard.get("provider_status") == "skipped"
        and guard.get("provider_reason") == "pre_generation_policy_block"
        and _stage_status(guard, "provider") == "skipped"
    )
    observed = {
        "final_action": guard.get("final_action"),
        "guard_action": guard.get("guard_action"),
        "provider_status": guard.get("provider_status"),
        "provider_reason": guard.get("provider_reason"),
        "guard_reason": guard.get("guard_reason"),
    }
    return _check(
        name="egress_guard_pre_provider_block",
        passed=passed,
        detail="Raw credential-shaped tool payload must block before provider generation.",
        observed=observed,
    )


def _check_mock_only_probes_skipped(smoke: dict[str, JsonValue]) -> dict[str, JsonValue]:
    encoded = _smoke_check(smoke, "encoded_canary_leak")
    metadata = _smoke_check(smoke, "metadata_slot_canary_leak")
    nimbus = _smoke_check(smoke, "nimbus_partial_leak")
    passed = (
        encoded.get("status") == "skipped" and metadata.get("status") == "skipped" and nimbus.get("status") == "skipped"
    )
    observed = {
        "encoded_canary_leak": encoded.get("status"),
        "metadata_slot_canary_leak": metadata.get("status"),
        "nimbus_partial_leak": nimbus.get("status"),
    }
    return _check(
        name="mock_only_probe_skip",
        passed=passed,
        detail="Real-provider smoke must skip mock-only leak controls instead of exercising mock fixtures.",
        observed=observed,
    )


def _check_provider_request_log(
    provider_records: tuple[dict[str, JsonValue], ...],
    forbidden_markers: tuple[str, ...],
) -> dict[str, JsonValue]:
    expected_count = len(forbidden_markers)
    record = provider_records[0] if len(provider_records) == 1 else {}
    passed = (
        len(provider_records) == 1
        and record.get("schema_version") == _LOOPBACK_PROVIDER_REQUEST_SCHEMA_VERSION
        and record.get("method") == "POST"
        and record.get("path") == "/v1/chat/completions"
        and record.get("authorization_status") == "matched_expected"
        and record.get("forbidden_substring_count") == expected_count
        and record.get("forbidden_substring_present") is False
        and _positive_int(record.get("message_count"))
        and _positive_int(record.get("request_body_bytes"))
        and _sha256_string(record.get("request_body_sha256"))
    )
    observed = {
        "record_count": len(provider_records),
        "schema_version": record.get("schema_version"),
        "method": record.get("method"),
        "path": record.get("path"),
        "authorization_status": record.get("authorization_status"),
        "forbidden_substring_count": record.get("forbidden_substring_count"),
        "forbidden_substring_present": record.get("forbidden_substring_present"),
        "message_count": record.get("message_count"),
    }
    return _check(
        name="provider_request_receipts",
        passed=passed,
        detail="Exactly one benign provider call should be recorded; blocked attack traffic must not reach provider.",
        observed=observed,
    )


def _check_audit_trace_receipts(audit_records: tuple[dict[str, JsonValue], ...]) -> dict[str, JsonValue]:
    records_by_trace = _records_by_trace_id(audit_records)
    benign = records_by_trace.get(_SMOKE_BENIGN_TRACE_ID)
    egress = records_by_trace.get(_SMOKE_EGRESS_TRACE_ID)
    egress_detector = _detector_result(egress, "provider_egress_guard")
    benign_runtime = _runtime_evidence(benign)
    egress_runtime = _runtime_evidence(egress)
    passed = (
        len(audit_records) >= 2
        and benign is not None
        and egress is not None
        and _policy_action(benign) == "allow"
        and _model_response_value(benign, "provider") == "openai_compatible"
        and _policy_action(egress) == "block"
        and _model_response_value(egress, "provider") == "skipped"
        and _model_response_value(egress, "reason") == "pre_generation_policy_block"
        and egress_detector.get("recommended_action") == "block"
        and _detector_evidence_value(egress_detector, "reason") == "blocked_sensitive_value_before_provider_egress"
        and benign_runtime.get("schema_version") == _AUDIT_RUNTIME_EVIDENCE_SCHEMA_VERSION
        and benign_runtime.get("final_action") == "allow"
        and benign_runtime.get("credential_slot_status") == "honeytoken_substituted"
        and _runtime_provider_state_value(benign_runtime, "provider") == "openai_compatible"
        and _runtime_provider_state_value(benign_runtime, "status") == "completed"
        and _runtime_detector_version(benign_runtime, "provider_egress_guard") == "provider-egress-guard-v1"
        and _runtime_detector_latency_present(benign_runtime, "provider_egress_guard")
        and egress_runtime.get("schema_version") == _AUDIT_RUNTIME_EVIDENCE_SCHEMA_VERSION
        and egress_runtime.get("final_action") == "block"
        and egress_runtime.get("credential_slot_status") == "real_secret_present"
        and _runtime_provider_state_value(egress_runtime, "provider") == "skipped"
        and _runtime_provider_state_value(egress_runtime, "status") == "skipped"
        and _runtime_provider_state_value(egress_runtime, "reason") == "pre_generation_policy_block"
        and _runtime_detector_version(egress_runtime, "provider_egress_guard") == "provider-egress-guard-v1"
        and _runtime_detector_latency_present(egress_runtime, "provider_egress_guard")
        and _runtime_fail_closed_event_present(egress_runtime, "pre_generation_policy_block")
    )
    observed = {
        "audit_record_count": len(audit_records),
        "has_benign_trace": benign is not None,
        "has_egress_trace": egress is not None,
        "benign_policy_action": _policy_action(benign),
        "egress_policy_action": _policy_action(egress),
        "egress_provider": _model_response_value(egress, "provider"),
        "egress_provider_reason": _model_response_value(egress, "reason"),
        "benign_runtime_evidence_schema_version": benign_runtime.get("schema_version"),
        "benign_runtime_provider_status": _runtime_provider_state_value(benign_runtime, "status"),
        "egress_runtime_evidence_schema_version": egress_runtime.get("schema_version"),
        "egress_runtime_provider_status": _runtime_provider_state_value(egress_runtime, "status"),
        "egress_runtime_fail_closed_pre_generation_block": _runtime_fail_closed_event_present(
            egress_runtime, "pre_generation_policy_block"
        ),
    }
    return _check(
        name="audit_trace_receipts",
        passed=passed,
        detail=(
            "Durable audit must prove benign allow and egress block before provider completion, "
            "including audit runtime-evidence receipts."
        ),
        observed=observed,
    )


def _check_no_forbidden_markers(raw_artifact_text: str, forbidden_markers: tuple[str, ...]) -> dict[str, JsonValue]:
    markers = (*forbidden_markers, _PLACEHOLDER_MARKER)
    present = tuple(marker for marker in markers if marker in raw_artifact_text)
    passed = len(present) == 0
    observed: dict[str, JsonValue] = {
        "forbidden_marker_count": len(markers),
        "present_forbidden_marker_count": len(present),
    }
    return _check(
        name="artifact_redaction",
        passed=passed,
        detail="Saved provider, smoke, preflight, and audit evidence must not contain raw forbidden markers.",
        observed=observed,
    )


def _input_artifacts(config: ProviderSmokeEvidenceConfig) -> dict[str, JsonValue]:
    return {
        "preflight": _artifact_receipt(config.preflight_path),
        "smoke": _artifact_receipt(config.smoke_path),
        "provider_request_log": _artifact_receipt(config.provider_request_log_path),
        "audit_jsonl": _artifact_receipt(config.audit_jsonl_path),
    }


def _artifact_receipt(path: Path) -> dict[str, JsonValue]:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _check(
    name: str,
    passed: bool,
    detail: str,
    observed: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "detail": detail,
        "observed": observed,
    }


def _smoke_check(smoke: dict[str, JsonValue], name: str) -> dict[str, JsonValue]:
    checks = smoke.get("checks")
    if not isinstance(checks, dict):
        return {}
    check = checks.get(name)
    if not isinstance(check, dict):
        return {}
    return check


def _stage_status(summary: dict[str, JsonValue], stage_name: str) -> JsonValue:
    stage = _stage(summary, stage_name)
    if stage is None:
        return None
    return stage.get("status")


def _stage(summary: dict[str, JsonValue], stage_name: str) -> dict[str, JsonValue] | None:
    stages = summary.get("stage_evidence")
    if not isinstance(stages, list):
        return None
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        if stage.get("stage") == stage_name:
            return stage
    return None


def _records_by_trace_id(records: tuple[dict[str, JsonValue], ...]) -> dict[str, dict[str, JsonValue]]:
    records_by_trace: dict[str, dict[str, JsonValue]] = {}
    for record in records:
        trace_id = record.get("trace_id")
        if isinstance(trace_id, str):
            records_by_trace[trace_id] = record
    return records_by_trace


def _policy_action(record: dict[str, JsonValue] | None) -> JsonValue:
    if record is None:
        return None
    policy_decision = record.get("policy_decision")
    if not isinstance(policy_decision, dict):
        return None
    return policy_decision.get("final_action")


def _model_response_value(record: dict[str, JsonValue] | None, key: str) -> JsonValue:
    if record is None:
        return None
    metadata = record.get("model_response_metadata")
    if not isinstance(metadata, dict):
        return None
    return metadata.get(key)


def _detector_result(record: dict[str, JsonValue] | None, detector_name: str) -> dict[str, JsonValue]:
    if record is None:
        return {}
    detector_results = record.get("detector_results")
    if not isinstance(detector_results, list):
        return {}
    for result in detector_results:
        if not isinstance(result, dict):
            continue
        if result.get("detector_name") == detector_name:
            return result
    return {}


def _detector_evidence_value(result: dict[str, JsonValue], key: str) -> JsonValue:
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        return None
    return evidence.get(key)


def _runtime_evidence(record: dict[str, JsonValue] | None) -> dict[str, JsonValue]:
    if record is None:
        return {}
    runtime_evidence = record.get("runtime_evidence")
    if not isinstance(runtime_evidence, dict):
        return {}
    return runtime_evidence


def _runtime_provider_state_value(runtime_evidence: dict[str, JsonValue], key: str) -> JsonValue:
    provider_state = runtime_evidence.get("provider_state")
    if not isinstance(provider_state, dict):
        return None
    return provider_state.get(key)


def _runtime_detector_version(runtime_evidence: dict[str, JsonValue], detector_name: str) -> JsonValue:
    detector_versions = runtime_evidence.get("detector_versions")
    if not isinstance(detector_versions, dict):
        return None
    return detector_versions.get(detector_name)


def _runtime_detector_latency_present(runtime_evidence: dict[str, JsonValue], detector_name: str) -> bool:
    detector_latencies = runtime_evidence.get("detector_latency_ms")
    if not isinstance(detector_latencies, dict):
        return False
    latency = detector_latencies.get(detector_name)
    return isinstance(latency, int | float) and not isinstance(latency, bool) and latency >= 0.0


def _runtime_fail_closed_event_present(runtime_evidence: dict[str, JsonValue], kind: str) -> bool:
    events = runtime_evidence.get("fail_closed_events")
    if not isinstance(events, list):
        return False
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("kind") == kind:
            return True
    return False


def _positive_int(value: JsonValue) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _sha256_string(value: JsonValue) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _is_loopback_http_url(value: str) -> bool:
    parsed = urlparse(value)
    if parsed.scheme != "http":
        return False
    host = parsed.hostname
    if host is None:
        return False
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _read_json_object(path: Path, context: str) -> dict[str, JsonValue]:
    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_from_pairs,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ProviderSmokeEvidenceError) as exc:
        raise ProviderSmokeEvidenceError(f"Could not read {context} JSON at {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ProviderSmokeEvidenceError(f"{context} JSON at {path} must be an object.")
    return cast(dict[str, JsonValue], decoded)


def _read_jsonl_objects(path: Path, context: str) -> tuple[dict[str, JsonValue], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ProviderSmokeEvidenceError(f"Could not read {context} JSONL at {path}: {exc}") from exc
    records: list[dict[str, JsonValue]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if raw_line.strip() == "":
            continue
        try:
            decoded = json.loads(
                raw_line,
                object_pairs_hook=_json_object_from_pairs,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ProviderSmokeEvidenceError) as exc:
            raise ProviderSmokeEvidenceError(f"{path}:{line_number}: invalid {context} record: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ProviderSmokeEvidenceError(f"{path}:{line_number}: {context} record must be an object.")
        records.append(cast(dict[str, JsonValue], decoded))
    return tuple(records)


def _raw_artifact_text(paths: tuple[Path, ...]) -> str:
    raw_text = []
    for path in paths:
        try:
            raw_text.append(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ProviderSmokeEvidenceError(f"Could not read artifact text at {path}: {exc}") from exc
    return "\n".join(raw_text)


def _json_object_from_pairs(pairs: Sequence[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ProviderSmokeEvidenceError(f"duplicate JSON key '{key}'.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ProviderSmokeEvidenceError(f"invalid JSON constant {value}.")


def _validate_forbidden_markers(forbidden_markers: tuple[str, ...]) -> None:
    if len(forbidden_markers) == 0:
        raise ProviderSmokeEvidenceError("at least one forbidden marker is required.")
    for marker in forbidden_markers:
        if marker == "":
            raise ProviderSmokeEvidenceError("forbidden markers must not be empty.")
