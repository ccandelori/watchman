from __future__ import annotations

from collections.abc import Mapping

from aegis.core.contracts import JsonValue

_EXPLAIN_SCHEMA_VERSION = "aegis.audit_explain/v1"
_RUNTIME_EVIDENCE_SCHEMA_VERSION = "aegis.audit_runtime_evidence/v1"
_SAFE_EVIDENCE_KEYS = frozenset(
    (
        "reason",
        "capability_reason",
        "predicted_label",
        "positive_label",
        "feature_source",
        "cift_window_family",
        "cift_window_selection_reason",
        "cift_window_coverage",
        "certification_mode",
        "certification_id",
        "runtime_model_sha256",
        "release_gate_report_sha256",
        "runtime_model_bundle_id",
        "critic_version",
        "turn_estimated_leakage_bits",
        "cumulative_estimated_leakage_bits",
        "budget_fraction",
        "warn_threshold",
        "sanitize_threshold",
        "block_threshold",
        "match_count",
        "checked_span_count",
        "allowed_honeytoken_span_count",
        "registered_canary_count",
    )
)


def explain_audit_record(record: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    normalized_turn = _mapping(record.get("normalized_turn"), "normalized_turn")
    detector_results = _detector_results(record)
    policy_decision = _mapping(record.get("policy_decision"), "policy_decision")
    model_response_metadata = _optional_mapping(record.get("model_response_metadata"))
    return {
        "schema_version": _EXPLAIN_SCHEMA_VERSION,
        "trace_id": _string(record, "trace_id"),
        "session_id": _string(record, "session_id"),
        "turn_index": _int(record, "turn_index"),
        "created_at": _string(record, "created_at"),
        "latency_ms": _number(record, "latency_ms"),
        "policy_mode": "severity",
        "stage_timeline": _stage_timeline(
            normalized_turn=normalized_turn,
            detector_results=detector_results,
            policy_decision=policy_decision,
            model_response_metadata=model_response_metadata,
        ),
        "detectors": [_detector_summary(result) for result in detector_results],
        "artifacts": _artifact_summary(detector_results),
        "runtime_evidence": _runtime_evidence(
            record=record,
            normalized_turn=normalized_turn,
            detector_results=detector_results,
            policy_decision=policy_decision,
            model_response_metadata=model_response_metadata,
        ),
        "policy_decision": _policy_summary(policy_decision),
    }


def _stage_timeline(
    normalized_turn: Mapping[str, JsonValue],
    detector_results: tuple[Mapping[str, JsonValue], ...],
    policy_decision: Mapping[str, JsonValue],
    model_response_metadata: Mapping[str, JsonValue],
) -> list[JsonValue]:
    return [
        {"stage": "normalize", "status": "ok"},
        _dp_honey_stage(normalized_turn),
        _detector_stage(stage="cift", component="cift", detector_results=detector_results),
        _detector_stage(stage="provider_egress_guard", component="tool_scanner", detector_results=detector_results),
        _provider_stage(normalized_turn=normalized_turn, model_response_metadata=model_response_metadata),
        _detector_stage(stage="canary", component="text_canary", detector_results=detector_results),
        _detector_stage(stage="nimbus", component="nimbus", detector_results=detector_results),
        {
            "stage": "policy",
            "status": "decided",
            "final_action": _string(policy_decision, "final_action"),
            "reason": _string(policy_decision, "reason"),
        },
        {"stage": "audit", "status": "written"},
    ]


def _dp_honey_stage(normalized_turn: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    metadata = _optional_mapping(normalized_turn.get("metadata"))
    detection = _optional_mapping(metadata.get("aegis_credential_slot_detection"))
    canary_count = _optional_int(metadata.get("dp_honey_canary_count"))
    if canary_count is None:
        canary_count = _optional_int(detection.get("honeytoken_substituted_count")) or 0
    status = "active" if canary_count > 0 else "not_configured"
    stage: dict[str, JsonValue] = {
        "stage": "dp_honey",
        "status": status,
        "canary_count": canary_count,
    }
    _copy_optional_string(detection, stage, "status", "credential_slot_status")
    _copy_optional_int(detection, stage, "credential_needed_count")
    _copy_optional_int(detection, stage, "honeytoken_substituted_count")
    _copy_optional_int(detection, stage, "real_secret_present_count")
    return stage


def _detector_stage(
    stage: str,
    component: str,
    detector_results: tuple[Mapping[str, JsonValue], ...],
) -> dict[str, JsonValue]:
    matches = tuple(result for result in detector_results if result.get("component") == component)
    if len(matches) == 0:
        return {"stage": stage, "status": "not_configured", "detectors": []}
    statuses = tuple(_optional_string(result.get("capability_status")) for result in matches)
    if "active" in statuses:
        status = "active"
    elif "degraded" in statuses:
        status = "degraded"
    else:
        status = "unavailable"
    return {
        "stage": stage,
        "status": status,
        "detectors": [_string(result, "detector_name") for result in matches],
        "actions": [_string(result, "recommended_action") for result in matches],
    }


def _provider_stage(
    normalized_turn: Mapping[str, JsonValue],
    model_response_metadata: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    model = _optional_mapping(normalized_turn.get("model"))
    provider = _optional_string(model_response_metadata.get("provider"))
    if provider is None:
        provider = _optional_string(model.get("provider")) or "unknown"
    status = "skipped" if provider == "skipped" else "completed"
    stage: dict[str, JsonValue] = {
        "stage": "provider",
        "status": status,
        "provider": provider,
        "model_id": _optional_string(model_response_metadata.get("model_id"))
        or _optional_string(model.get("model_id")),
    }
    _copy_optional_string(model_response_metadata, stage, "reason", "reason")
    return stage


def _detector_summary(result: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    evidence = _optional_mapping(result.get("evidence"))
    return {
        "detector_name": _string(result, "detector_name"),
        "component": _string(result, "component"),
        "recommended_action": _string(result, "recommended_action"),
        "capability_status": _string(result, "capability_status"),
        "score": _number(result, "score"),
        "confidence": _number(result, "confidence"),
        "latency_ms": _number(result, "latency_ms"),
        "evidence": _safe_evidence_summary(evidence),
    }


def _safe_evidence_summary(evidence: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    summary: dict[str, JsonValue] = {}
    for key in sorted(_SAFE_EVIDENCE_KEYS):
        value = evidence.get(key)
        if value is not None:
            summary[key] = value
    critic_evidence = _optional_mapping(evidence.get("critic_evidence"))
    if len(critic_evidence) > 0:
        summary["critic_evidence"] = _safe_evidence_summary(critic_evidence)
    return summary


def _artifact_summary(detector_results: tuple[Mapping[str, JsonValue], ...]) -> dict[str, JsonValue]:
    artifacts: dict[str, JsonValue] = {}
    for result in detector_results:
        evidence = _optional_mapping(result.get("evidence"))
        for key in (
            "certification_id",
            "certification_mode",
            "runtime_model_sha256",
            "release_gate_report_sha256",
            "runtime_model_bundle_id",
            "feature_source",
            "critic_version",
        ):
            value = evidence.get(key)
            if value is not None:
                artifacts[key] = value
    return artifacts


def _runtime_evidence(
    record: Mapping[str, JsonValue],
    normalized_turn: Mapping[str, JsonValue],
    detector_results: tuple[Mapping[str, JsonValue], ...],
    policy_decision: Mapping[str, JsonValue],
    model_response_metadata: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    stored = _optional_mapping(record.get("runtime_evidence"))
    if len(stored) > 0:
        return dict(stored)
    provider_stage = _provider_stage(
        normalized_turn=normalized_turn,
        model_response_metadata=model_response_metadata,
    )
    return {
        "schema_version": _RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "policy_mode": "severity",
        "final_action": _string(policy_decision, "final_action"),
        "provider_state": {
            "status": provider_stage.get("status"),
            "provider": provider_stage.get("provider"),
            "model_id": provider_stage.get("model_id"),
            "reason": provider_stage.get("reason"),
        },
        "credential_slot_status": _runtime_credential_slot_status(normalized_turn),
        "detector_versions": _runtime_detector_versions(detector_results),
        "artifact_hashes": _runtime_artifact_hashes(detector_results),
        "cift": _runtime_cift_summary(detector_results),
        "fail_closed_events": _runtime_fail_closed_events(
            policy_decision=policy_decision,
            provider_stage=provider_stage,
        ),
        "latency_ms": _number(record, "latency_ms"),
    }


def _runtime_credential_slot_status(normalized_turn: Mapping[str, JsonValue]) -> str:
    metadata = _optional_mapping(normalized_turn.get("metadata"))
    detection = _optional_mapping(metadata.get("aegis_credential_slot_detection"))
    status = _optional_string(detection.get("status"))
    if status is None:
        return "unknown"
    return status


def _runtime_detector_versions(detector_results: tuple[Mapping[str, JsonValue], ...]) -> dict[str, JsonValue]:
    versions: dict[str, JsonValue] = {}
    for result in detector_results:
        evidence = _optional_mapping(result.get("evidence"))
        detector_name = _string(result, "detector_name")
        version = _optional_string(evidence.get("detector_version"))
        if version is None:
            version = _optional_string(evidence.get("critic_version"))
        if version is None:
            version = _optional_string(evidence.get("runtime_model_bundle_id"))
        if version is None:
            version = f"{_string(result, 'component')}-unknown"
        versions[detector_name] = version
    return versions


def _runtime_artifact_hashes(detector_results: tuple[Mapping[str, JsonValue], ...]) -> dict[str, JsonValue]:
    hashes: dict[str, JsonValue] = {}
    for result in detector_results:
        evidence = _optional_mapping(result.get("evidence"))
        for key in (
            "runtime_model_sha256",
            "release_gate_report_sha256",
            "feature_vector_sha256",
            "rendered_prompt_sha256",
            "certification_manifest_sha256",
            "certification_report_sha256",
        ):
            value = evidence.get(key)
            if isinstance(value, str) and value != "":
                hashes[key] = value
    return hashes


def _runtime_cift_summary(detector_results: tuple[Mapping[str, JsonValue], ...]) -> dict[str, JsonValue]:
    summary: dict[str, JsonValue] = {}
    for result in detector_results:
        if result.get("component") != "cift":
            continue
        evidence = _optional_mapping(result.get("evidence"))
        for key in (
            "certification_id",
            "certification_mode",
            "runtime_model_sha256",
            "release_gate_report_sha256",
            "runtime_model_bundle_id",
            "feature_source",
        ):
            value = evidence.get(key)
            if isinstance(value, str) and value != "":
                summary[key] = value
    return summary


def _runtime_fail_closed_events(
    policy_decision: Mapping[str, JsonValue],
    provider_stage: Mapping[str, JsonValue],
) -> list[JsonValue]:
    if provider_stage.get("status") != "skipped" or provider_stage.get("reason") != "pre_generation_policy_block":
        return []
    return [
        {
            "kind": "pre_generation_policy_block",
            "provider_status": "skipped",
            "final_action": _string(policy_decision, "final_action"),
            "triggered_detectors": _string_sequence(policy_decision.get("triggered_detectors")),
        }
    ]


def _policy_summary(policy_decision: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "final_action": _string(policy_decision, "final_action"),
        "reason": _string(policy_decision, "reason"),
        "risk_score": _number(policy_decision, "risk_score"),
        "triggered_detectors": _string_sequence(policy_decision.get("triggered_detectors")),
    }


def _detector_results(record: Mapping[str, JsonValue]) -> tuple[Mapping[str, JsonValue], ...]:
    value = record.get("detector_results")
    if not isinstance(value, list):
        raise ValueError("audit record detector_results must be a list.")
    results: list[Mapping[str, JsonValue]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"audit record detector_results[{index}] must be an object.")
        results.append(item)
    return tuple(results)


def _mapping(value: JsonValue, context: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"audit record {context} must be an object.")
    return value


def _optional_mapping(value: JsonValue) -> Mapping[str, JsonValue]:
    if isinstance(value, dict):
        return value
    return {}


def _string(record: Mapping[str, JsonValue], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"audit record {key} must be a non-empty string.")
    return value


def _optional_string(value: JsonValue) -> str | None:
    if isinstance(value, str) and value != "":
        return value
    return None


def _int(record: Mapping[str, JsonValue], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"audit record {key} must be an integer.")
    return value


def _optional_int(value: JsonValue) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _number(record: Mapping[str, JsonValue], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"audit record {key} must be numeric.")
    return float(value)


def _string_sequence(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _copy_optional_string(
    source: Mapping[str, JsonValue],
    target: dict[str, JsonValue],
    source_key: str,
    target_key: str,
) -> None:
    value = _optional_string(source.get(source_key))
    if value is not None:
        target[target_key] = value


def _copy_optional_int(source: Mapping[str, JsonValue], target: dict[str, JsonValue], key: str) -> None:
    value = _optional_int(source.get(key))
    if value is not None:
        target[key] = value
