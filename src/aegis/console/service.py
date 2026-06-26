from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from aegis.audit.explain import explain_audit_record
from aegis.audit.jsonl import read_jsonl_audit_records
from aegis.core.action_severity import action_severity
from aegis.core.contracts import Action, JsonValue

CONSOLE_OVERVIEW_SCHEMA_VERSION = "aegis.console_overview/v1"
CONSOLE_EVENTS_SCHEMA_VERSION = "aegis.console_events/v1"
CONSOLE_SETUP_SCHEMA_VERSION = "aegis.console_setup/v1"
_CIFT_SUPPORT_TIER_UNSUPPORTED = "unsupported"
_CIFT_SUPPORT_TIER_CALIBRATION_READY = "calibration-ready"
_CIFT_SUPPORT_TIER_RUNTIME_ENFORCEABLE = "runtime-enforceable"
_CIFT_UNSUPPORTED_SUPPORT_SCOPE = "model-specific CIFT enforcement unavailable"
_DEFAULT_REQUIRED_STAGE_NAMES = (
    "normalize",
    "credential_slot",
    "dp_honey",
    "cift",
    "provider_egress_guard",
    "provider",
    "canary",
    "nimbus",
    "policy",
    "audit",
)


class ConsoleServiceError(ValueError):
    """Raised when the local console cannot summarize gateway state."""


class ConsoleGatewayError(ValueError):
    """Raised when a gateway API request fails."""


@dataclass(frozen=True)
class ConsoleSettings:
    gateway_base_url: str
    request_timeout_seconds: float
    smoke_report_path: Path | None
    sample_audit_jsonl_path: Path | None
    operator_profile: str


GatewayFetcher = Callable[[ConsoleSettings, str, tuple[tuple[str, str], ...]], dict[str, JsonValue]]


def settings_from_env(env: Mapping[str, str]) -> ConsoleSettings:
    return ConsoleSettings(
        gateway_base_url=env.get("AEGIS_CONSOLE_GATEWAY_URL", "http://127.0.0.1:8000"),
        request_timeout_seconds=_positive_float_env(env, "AEGIS_CONSOLE_GATEWAY_TIMEOUT_SECONDS", 2.0),
        smoke_report_path=_optional_path_env(env, "AEGIS_CONSOLE_SMOKE_REPORT_PATH"),
        sample_audit_jsonl_path=_optional_path_env(env, "AEGIS_CONSOLE_SAMPLE_AUDIT_JSONL_PATH"),
        operator_profile=env.get("AEGIS_CONSOLE_PROFILE", "balanced"),
    )


def console_overview(settings: ConsoleSettings, fetcher: GatewayFetcher) -> dict[str, JsonValue]:
    health = _endpoint(settings=settings, fetcher=fetcher, path="/health", query=())
    readiness = _endpoint(settings=settings, fetcher=fetcher, path="/ready", query=())
    capabilities = _endpoint(settings=settings, fetcher=fetcher, path="/aegis/capabilities", query=())
    events = console_events(settings=settings, fetcher=fetcher, limit=20, session_id=None)
    last_event = _first_event(events)
    protection = _protection_summary(
        settings=settings,
        health=health,
        readiness=readiness,
        capabilities=capabilities,
    )
    return {
        "schema_version": CONSOLE_OVERVIEW_SCHEMA_VERSION,
        "gateway": {
            "base_url": settings.gateway_base_url,
            "health": health,
            "readiness": readiness,
            "capabilities": capabilities,
        },
        "protection": protection,
        "checklist": _protection_checklist(
            health=health,
            readiness=readiness,
            capabilities=capabilities,
            last_event=last_event,
            smoke_report=_smoke_report_summary(settings.smoke_report_path),
        ),
        "model": _model_summary(readiness=readiness, capabilities=capabilities, last_event=last_event),
        "cift": _cift_summary(readiness=readiness, capabilities=capabilities),
        "nimbus": _nimbus_summary(readiness=readiness, capabilities=capabilities),
        "last_request": last_event,
        "last_smoke": _smoke_report_summary(settings.smoke_report_path),
    }


def console_events(
    settings: ConsoleSettings,
    fetcher: GatewayFetcher,
    limit: int,
    session_id: str | None,
) -> dict[str, JsonValue]:
    if limit <= 0:
        raise ConsoleServiceError("event limit must be positive.")
    query: list[tuple[str, str]] = [("limit", str(limit))]
    if session_id is not None:
        query.append(("session_id", session_id))
    audit_recent = _endpoint(settings=settings, fetcher=fetcher, path="/audit/recent", query=tuple(query))
    source = "gateway_audit_recent"
    raw_events = _events_from_endpoint(audit_recent)
    if len(raw_events) == 0 and settings.sample_audit_jsonl_path is not None:
        raw_events = read_jsonl_audit_records(settings.sample_audit_jsonl_path)
        source = "sample_audit_jsonl"
    event_summaries = tuple(_event_summary(event) for event in raw_events)
    return {
        "schema_version": CONSOLE_EVENTS_SCHEMA_VERSION,
        "source": source,
        "gateway_status": audit_recent["status"],
        "events": [event for event in event_summaries],
        "detector_activity": _detector_activity(event_summaries),
    }


def console_trace(settings: ConsoleSettings, fetcher: GatewayFetcher, trace_id: str) -> dict[str, JsonValue]:
    if trace_id == "":
        raise ConsoleServiceError("trace_id must not be empty.")
    endpoint = _endpoint(settings=settings, fetcher=fetcher, path="/audit/explain", query=(("trace_id", trace_id),))
    payload = endpoint.get("payload")
    if isinstance(payload, dict):
        return payload
    raise ConsoleServiceError(f"No gateway trace explanation is available for trace_id '{trace_id}'.")


def console_setup(settings: ConsoleSettings) -> dict[str, JsonValue]:
    console_command = f"uv run aegis-console --gateway-url {settings.gateway_base_url} --host 127.0.0.1 --port 8780"
    return {
        "schema_version": CONSOLE_SETUP_SCHEMA_VERSION,
        "gateway_base_url": settings.gateway_base_url,
        "openai_compatible_endpoint": f"{settings.gateway_base_url.rstrip('/')}/v1/chat/completions",
        "commands": {
            "start_console": console_command,
            "start_gateway": "uv run aegis-proxy --host 127.0.0.1 --port 8000",
            "default_smoke": (
                "uv run aegis-proxy-smoke --url http://127.0.0.1:8000 "
                "--timeout 5 --output introspection/data/reports/aegis_default_mock_provider_smoke_v1.json"
            ),
            "strict_cift_smoke": (
                "uv run aegis-proxy-smoke --url http://127.0.0.1:8000 --timeout 120 --require-cift-pre-generation-block"
            ),
        },
        "common_degraded_states": [
            {
                "state": "gateway_offline",
                "fix": "Start the gateway and refresh the console.",
            },
            {
                "state": "cift_black_box",
                "fix": "Start with strict CIFT env and a trusted hidden-state extractor sidecar.",
            },
            {
                "state": "nimbus_deterministic_beta",
                "fix": "This is expected until learned paper-faithful NIMBUS is trained and promoted.",
            },
            {
                "state": "nimbus_learned_runtime_beta_not_promotable",
                "fix": (
                    "Use this only for explicit learned-runtime beta evaluation; keep deterministic NIMBUS active "
                    "for release evidence."
                ),
            },
            {
                "state": "audit_not_durable",
                "fix": "Set AEGIS_AUDIT_JSONL_PATH before starting the gateway.",
            },
        ],
    }


def default_gateway_fetcher(
    settings: ConsoleSettings,
    path: str,
    query: tuple[tuple[str, str], ...],
) -> dict[str, JsonValue]:
    url = _gateway_url(settings.gateway_base_url, path=path, query=query)
    request = urllib.request.Request(url=url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=settings.request_timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ConsoleGatewayError(f"{path} returned HTTP {exc.code}: {body}") from exc
    except (TimeoutError, OSError, urllib.error.URLError) as exc:
        raise ConsoleGatewayError(f"{path} request failed: {exc}") from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ConsoleGatewayError(f"{path} returned non-JSON response.") from exc
    if not isinstance(decoded, dict):
        raise ConsoleGatewayError(f"{path} returned JSON that was not an object.")
    return decoded


def _endpoint(
    settings: ConsoleSettings,
    fetcher: GatewayFetcher,
    path: str,
    query: tuple[tuple[str, str], ...],
) -> dict[str, JsonValue]:
    try:
        payload = fetcher(settings, path, query)
    except ConsoleGatewayError as exc:
        return {"status": "error", "error": str(exc), "payload": None}
    return {"status": "ok", "error": None, "payload": payload}


def _gateway_url(base_url: str, path: str, query: tuple[tuple[str, str], ...]) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in ("http", "https") or parsed.netloc == "":
        raise ConsoleGatewayError("gateway URL must be an absolute http(s) URL.")
    if parsed.username is not None or parsed.password is not None or parsed.query != "" or parsed.fragment != "":
        raise ConsoleGatewayError("gateway URL must not include credentials, query strings, or fragments.")
    normalized_base = base_url.rstrip("/") + "/"
    normalized_path = path.lstrip("/")
    url = urllib.parse.urljoin(normalized_base, normalized_path)
    if len(query) > 0:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    return url


def _protection_summary(
    settings: ConsoleSettings,
    health: Mapping[str, JsonValue],
    readiness: Mapping[str, JsonValue],
    capabilities: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    health_ok = health.get("status") == "ok"
    ready_payload = _payload(readiness)
    capabilities_payload = _payload(capabilities)
    ready = ready_payload.get("ready") is True
    strict_enabled = _nested_bool(ready_payload, ("strict_protected_mode", "enabled"))
    cift_payload = _optional_mapping(capabilities_payload.get("cift"))
    cift_mode = _optional_string(cift_payload.get("capability_mode")) or "unknown"
    operator_profile = settings.operator_profile
    active_profile = "strict" if strict_enabled else operator_profile
    if not health_ok or not ready:
        state = "Unsafe / Not Ready"
        severity = "danger"
    elif active_profile == "observe":
        state = "Observe Only"
        severity = "observe"
    elif strict_enabled and cift_mode == "self_hosted_introspection":
        state = "Protected"
        severity = "protected"
    else:
        state = "Degraded"
        severity = "degraded"
    return {
        "state": state,
        "severity": severity,
        "active_profile": active_profile,
        "gateway_online": health_ok,
        "gateway_ready": ready,
        "cift_capability_mode": cift_mode,
    }


def _protection_checklist(
    health: Mapping[str, JsonValue],
    readiness: Mapping[str, JsonValue],
    capabilities: Mapping[str, JsonValue],
    last_event: JsonValue,
    smoke_report: dict[str, JsonValue],
) -> list[JsonValue]:
    ready_payload = _payload(readiness)
    capabilities_payload = _payload(capabilities)
    cift_ready = _optional_mapping(ready_payload.get("cift"))
    provider = _optional_mapping(ready_payload.get("provider"))
    dp_honey = _optional_mapping(ready_payload.get("dp_honey"))
    provider_guard = _optional_mapping(ready_payload.get("provider_egress_guard"))
    nimbus = {
        **_optional_mapping(capabilities_payload.get("nimbus")),
        **_optional_mapping(ready_payload.get("nimbus")),
    }
    audit = _optional_mapping(capabilities_payload.get("audit"))
    model_id = _model_summary(readiness, capabilities, last_event).get("model_id")
    return [
        _checklist_item("Gateway", health.get("status") == "ok", _endpoint_detail(health, "online")),
        _checklist_item("Active model", isinstance(model_id, str) and model_id != "", ""),
        _checklist_item("CIFT certificate", cift_ready.get("status") == "ready", _cift_status_text(cift_ready)),
        _checklist_item(
            "Hidden-state extractor/device",
            _cift_extractor_ready(cift_ready),
            _cift_device_text(cift_ready),
        ),
        _checklist_item("DP-HONEY", dp_honey.get("status") == "ready", _optional_string(dp_honey.get("status")) or ""),
        _checklist_item(
            "Provider egress guard",
            provider_guard.get("status") == "ready",
            _optional_string(provider_guard.get("status")) or "",
        ),
        _checklist_item("Provider", provider.get("status") == "ready", _optional_string(provider.get("name")) or ""),
        _checklist_item("NIMBUS", _nimbus_runtime_status_ready(nimbus), _nimbus_status_detail(nimbus)),
        _checklist_item("Audit", audit.get("durable_jsonl_enabled") is True, _audit_status_text(audit)),
        _checklist_item(
            "Last smoke",
            smoke_report.get("status") == "ok",
            _optional_string(smoke_report.get("status")) or "",
        ),
        _checklist_item("Last request", isinstance(last_event, dict), _last_event_detail(last_event)),
    ]


def _checklist_item(label: str, ok: bool, detail: str) -> dict[str, JsonValue]:
    status = "passed" if ok else "unavailable"
    return {"label": label, "status": status, "detail": detail}


def _nimbus_runtime_status_ready(nimbus: Mapping[str, JsonValue]) -> bool:
    return nimbus.get("status") in ("deterministic_beta", "learned_runtime_beta")


def _nimbus_status_detail(nimbus: Mapping[str, JsonValue]) -> str:
    status = _optional_string(nimbus.get("status")) or "unknown"
    critic_kind = _optional_string(nimbus.get("critic_kind"))
    promotion_status = _optional_string(nimbus.get("promotion_status"))
    details = [status]
    if critic_kind is not None:
        details.append(critic_kind)
    if promotion_status is not None:
        details.append(promotion_status)
    return " / ".join(details)


def _model_summary(
    readiness: Mapping[str, JsonValue],
    capabilities: Mapping[str, JsonValue],
    last_event: JsonValue,
) -> dict[str, JsonValue]:
    ready_payload = _payload(readiness)
    capabilities_payload = _payload(capabilities)
    binding = _cift_binding(capabilities_payload)
    if len(binding) > 0:
        return {
            "model_id": binding.get("source_model_id"),
            "revision": binding.get("source_revision"),
            "device": binding.get("source_selected_device"),
            "hidden_size": binding.get("source_hidden_size"),
            "layer_count": binding.get("source_layer_count"),
            "tokenizer_fingerprint_sha256": binding.get("tokenizer_fingerprint_sha256"),
            "chat_template_sha256": binding.get("chat_template_sha256"),
        }
    if isinstance(last_event, dict):
        event_model = _optional_mapping(last_event.get("model"))
        model_id = event_model.get("model_id")
        if isinstance(model_id, str):
            return {
                "model_id": model_id,
                "revision": event_model.get("revision"),
                "device": event_model.get("selected_device"),
                "provider": event_model.get("provider"),
            }
    provider = _optional_mapping(ready_payload.get("provider"))
    return {
        "model_id": provider.get("name"),
        "revision": None,
        "device": None,
        "provider": provider.get("name"),
    }


def _cift_summary(readiness: Mapping[str, JsonValue], capabilities: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    ready_payload = _payload(readiness)
    capabilities_payload = _payload(capabilities)
    cift_ready = _optional_mapping(ready_payload.get("cift"))
    cift_capability = _optional_mapping(capabilities_payload.get("cift"))
    binding = _cift_binding(capabilities_payload)
    support = _cift_support_summary(cift_ready=cift_ready, cift_capability=cift_capability, binding=binding)
    support_tier = _optional_string(support.get("support_tier"))
    if support_tier == _CIFT_SUPPORT_TIER_UNSUPPORTED:
        certificate_status = _CIFT_SUPPORT_TIER_UNSUPPORTED
    elif support_tier == _CIFT_SUPPORT_TIER_CALIBRATION_READY:
        certificate_status = _CIFT_SUPPORT_TIER_CALIBRATION_READY
    elif binding.get("certification_mode") == "strict" and cift_ready.get("status") == "ready":
        certificate_status = "certified"
    else:
        certificate_status = "degraded"
    return {
        "certificate_status": certificate_status,
        "capability_mode": cift_capability.get("capability_mode") or cift_ready.get("capability_mode"),
        "support_tier": support.get("support_tier"),
        "support_scope": support.get("support_scope"),
        "support_reason": support.get("support_reason"),
        "readiness_status": cift_ready.get("status"),
        "certification_id": binding.get("certification_id"),
        "certification_mode": binding.get("certification_mode"),
        "runtime_model_sha256": binding.get("runtime_model_sha256"),
        "release_gate_report_sha256": binding.get("release_gate_report_sha256"),
        "runtime_model_bundle_id": binding.get("model_bundle_id") or binding.get("runtime_model_bundle_id"),
        "feature_key": binding.get("feature_key"),
        "selected_choice_readout_token_count": binding.get("selected_choice_readout_token_count"),
        "source_model_id": binding.get("source_model_id"),
        "source_revision": binding.get("source_revision"),
        "source_selected_device": binding.get("source_selected_device"),
        "source_hidden_size": binding.get("source_hidden_size"),
        "source_layer_count": binding.get("source_layer_count"),
        "tokenizer_fingerprint_sha256": binding.get("tokenizer_fingerprint_sha256"),
        "special_tokens_map_sha256": binding.get("special_tokens_map_sha256"),
        "chat_template_sha256": binding.get("chat_template_sha256"),
        "extractor": cift_ready.get("extractor"),
    }


def _cift_support_summary(
    cift_ready: Mapping[str, JsonValue],
    cift_capability: Mapping[str, JsonValue],
    binding: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    explicit = _explicit_cift_support(cift_ready)
    if len(explicit) == 3:
        return explicit
    explicit = _explicit_cift_support(cift_capability)
    if len(explicit) == 3:
        return explicit
    if len(binding) > 0:
        certification_mode = _optional_string(binding.get("certification_mode"))
        if certification_mode == "gateway_smoke_bootstrap":
            return {
                "support_tier": _CIFT_SUPPORT_TIER_CALIBRATION_READY,
                "support_scope": _cift_calibration_support_scope(binding),
                "support_reason": "gateway-smoke bootstrap is calibration evidence only, not release certification.",
            }
        if certification_mode == "strict" and cift_ready.get("status") == "ready":
            return {
                "support_tier": _CIFT_SUPPORT_TIER_RUNTIME_ENFORCEABLE,
                "support_scope": _cift_enforcement_support_scope(binding),
                "support_reason": "strict certification binding and live extractor readiness are satisfied.",
            }
        return {
            "support_tier": "certified",
            "support_scope": _cift_enforcement_support_scope(binding),
            "support_reason": (
                "strict certification binding is loaded; readiness still depends on trusted extractor attestation."
            ),
        }
    return {
        "support_tier": _CIFT_SUPPORT_TIER_UNSUPPORTED,
        "support_scope": _CIFT_UNSUPPORTED_SUPPORT_SCOPE,
        "support_reason": "no model-specific CIFT runtime binding is available.",
    }


def _explicit_cift_support(payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    support_tier = _optional_string(payload.get("support_tier"))
    support_scope = _optional_string(payload.get("support_scope"))
    support_reason = _optional_string(payload.get("support_reason"))
    if support_tier is None or support_scope is None or support_reason is None:
        return {}
    return {
        "support_tier": support_tier,
        "support_scope": support_scope,
        "support_reason": support_reason,
    }


def _cift_enforcement_support_scope(binding: Mapping[str, JsonValue]) -> str:
    model_id = _optional_string(binding.get("source_model_id")) or "unknown model"
    selected_device = _optional_string(binding.get("source_selected_device")) or "unknown device"
    return f"model-specific CIFT enforcement for {model_id} on {selected_device}"


def _cift_calibration_support_scope(binding: Mapping[str, JsonValue]) -> str:
    model_id = _optional_string(binding.get("source_model_id")) or "unknown model"
    selected_device = _optional_string(binding.get("source_selected_device")) or "unknown device"
    return f"model-specific CIFT calibration for {model_id} on {selected_device}"


def _nimbus_summary(readiness: Mapping[str, JsonValue], capabilities: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    ready_payload = _payload(readiness)
    capabilities_payload = _payload(capabilities)
    readiness_nimbus = _optional_mapping(ready_payload.get("nimbus"))
    capability_nimbus = _optional_mapping(capabilities_payload.get("nimbus"))
    status = readiness_nimbus.get("status") or capability_nimbus.get("status")
    critic_kind = capability_nimbus.get("critic_kind")
    promotion_status = capability_nimbus.get("promotion_status")
    return {
        "status": status,
        "critic_version": readiness_nimbus.get("critic_version"),
        "critic_kind": critic_kind,
        "paper_faithful_learned_critic": capability_nimbus.get("paper_faithful_learned_critic"),
        "promotion_status": promotion_status,
        "thresholds": capability_nimbus.get("thresholds"),
        "infonce_model_path": capability_nimbus.get("infonce_model_path"),
        "label": _nimbus_label(status, critic_kind, promotion_status),
    }


def _nimbus_label(status: JsonValue, critic_kind: JsonValue, promotion_status: JsonValue) -> str:
    status_text = _optional_string(status)
    if status_text == "learned_runtime_beta":
        return "learned runtime beta"
    if status_text == "deterministic_beta":
        return "deterministic beta"
    critic_text = _optional_string(critic_kind)
    if critic_text is not None:
        return critic_text
    promotion_text = _optional_string(promotion_status)
    if promotion_text is not None:
        return promotion_text
    return "unknown"


def _event_summary(record: dict[str, JsonValue]) -> dict[str, JsonValue]:
    explanation = _safe_explanation(record)
    policy = _optional_mapping(record.get("policy_decision"))
    model_response = _optional_mapping(record.get("model_response_metadata"))
    runtime_evidence = _optional_mapping(record.get("runtime_evidence"))
    provider_state = _optional_mapping(runtime_evidence.get("provider_state"))
    detector_results = _detector_results(record)
    detectors_fired = tuple(
        _optional_string(detector.get("detector_name")) or "unknown"
        for detector in detector_results
        if _detector_fired(detector)
    )
    return {
        "created_at": record.get("created_at"),
        "trace_id": record.get("trace_id"),
        "session_id": record.get("session_id"),
        "turn_index": record.get("turn_index"),
        "final_action": policy.get("final_action"),
        "reason": policy.get("reason"),
        "provider_status": provider_state.get("status") or _provider_status_from_model_response(model_response),
        "provider": provider_state.get("provider") or model_response.get("provider"),
        "detectors_fired": list(detectors_fired),
        "stage_timeline": _console_stage_timeline(
            explanation=explanation,
            policy=policy,
            runtime_evidence=runtime_evidence,
            detector_results=detector_results,
        ),
        "runtime_evidence": _runtime_evidence_summary(runtime_evidence),
        "detector_results": [_detector_result_summary(detector) for detector in detector_results],
        "model": dict(_optional_mapping(_optional_mapping(record.get("normalized_turn")).get("model"))),
    }


def _detector_activity(events: tuple[dict[str, JsonValue], ...]) -> dict[str, JsonValue]:
    activity = {
        "cift_pre_generation_blocks": 0,
        "dp_honey_substitutions": 0,
        "canary_detections": 0,
        "provider_egress_blocks": 0,
        "nimbus_warnings": 0,
        "fail_closed_events": 0,
    }
    for event in events:
        runtime_evidence = _optional_mapping(event.get("runtime_evidence"))
        if runtime_evidence.get("credential_slot_status") == "honeytoken_substituted":
            activity["dp_honey_substitutions"] += 1
        fail_closed_events = runtime_evidence.get("fail_closed_events")
        if isinstance(fail_closed_events, list):
            activity["fail_closed_events"] += len(fail_closed_events)
        for detector in _detector_results(event):
            detector_name = _optional_string(detector.get("detector_name")) or ""
            action = _optional_string(detector.get("recommended_action")) or "allow"
            if detector_name == "cift_runtime" and action == "block":
                activity["cift_pre_generation_blocks"] += 1
            if detector_name in ("text_canary", "encoded_canary") and _action_at_least(action, Action.WARN):
                activity["canary_detections"] += 1
            if detector_name == "provider_egress_guard" and action == "block":
                activity["provider_egress_blocks"] += 1
            if detector_name == "nimbus" and _action_at_least(action, Action.WARN):
                activity["nimbus_warnings"] += 1
    return {key: value for key, value in activity.items()}


def _safe_explanation(record: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    try:
        return explain_audit_record(record)
    except ValueError:
        return {"stage_timeline": []}


def _console_stage_timeline(
    explanation: Mapping[str, JsonValue],
    policy: Mapping[str, JsonValue],
    runtime_evidence: Mapping[str, JsonValue],
    detector_results: tuple[Mapping[str, JsonValue], ...],
) -> list[JsonValue]:
    raw_timeline = explanation.get("stage_timeline")
    if not isinstance(raw_timeline, list):
        raw_timeline = []
    raw_stages = tuple(stage for stage in raw_timeline if isinstance(stage, dict))
    return [
        _normalized_stage(_stage_by_name(raw_stages, "normalize"), fallback_stage="normalize"),
        _credential_slot_stage(raw_stages=raw_stages, runtime_evidence=runtime_evidence),
        _normalized_stage(_stage_by_name(raw_stages, "dp_honey"), fallback_stage="dp_honey"),
        _normalized_stage(
            _stage_by_name(raw_stages, "cift"),
            fallback_stage="cift",
            detector_results=detector_results,
            component="cift",
        ),
        _normalized_stage(
            _stage_by_name(raw_stages, "provider_egress_guard"),
            fallback_stage="provider_egress_guard",
            detector_results=detector_results,
            component="tool_scanner",
        ),
        _normalized_stage(_stage_by_name(raw_stages, "provider"), fallback_stage="provider"),
        _normalized_stage(
            _stage_by_name(raw_stages, "canary"),
            fallback_stage="canary",
            detector_results=detector_results,
            component="text_canary",
        ),
        _normalized_stage(
            _stage_by_name(raw_stages, "nimbus"),
            fallback_stage="nimbus",
            detector_results=detector_results,
            component="nimbus",
        ),
        _normalized_stage(_stage_by_name(raw_stages, "policy"), fallback_stage="policy", policy=policy),
        _normalized_stage(_stage_by_name(raw_stages, "audit"), fallback_stage="audit"),
    ]


def _stage_by_name(
    stages: tuple[Mapping[str, JsonValue], ...],
    stage_name: str,
) -> Mapping[str, JsonValue]:
    for stage in stages:
        if stage.get("stage") == stage_name:
            return stage
    return {}


def _credential_slot_stage(
    raw_stages: tuple[Mapping[str, JsonValue], ...],
    runtime_evidence: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    dp_honey = _stage_by_name(raw_stages, "dp_honey")
    slot_status = _optional_string(dp_honey.get("credential_slot_status")) or _optional_string(
        runtime_evidence.get("credential_slot_status")
    )
    status = _credential_slot_display_status(slot_status)
    return {
        "stage": "credential_slot",
        "status": status,
        "detail": slot_status or "unknown",
    }


def _credential_slot_display_status(slot_status: str | None) -> str:
    if slot_status in ("honeytoken_substituted", "no_secret_context"):
        return "passed"
    if slot_status in ("real_secret_present", "ambiguous_protected_workflow"):
        return "warned"
    return "unavailable"


def _normalized_stage(
    stage: Mapping[str, JsonValue],
    fallback_stage: str,
    detector_results: tuple[Mapping[str, JsonValue], ...] = (),
    component: str | None = None,
    policy: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    display_stage = _optional_string(stage.get("stage")) or fallback_stage
    result: dict[str, JsonValue] = {
        "stage": display_stage,
        "status": _stage_display_status(
            stage=stage,
            detector_results=detector_results,
            component=component,
            policy=policy,
        ),
    }
    for key in (
        "detectors",
        "actions",
        "provider",
        "model_id",
        "reason",
        "canary_count",
        "credential_needed_count",
        "honeytoken_substituted_count",
        "real_secret_present_count",
        "final_action",
    ):
        value = stage.get(key)
        if value is not None:
            result[key] = value
    original_status = _optional_string(stage.get("status"))
    if original_status is not None and original_status != result["status"]:
        result["original_status"] = original_status
    return result


def _stage_display_status(
    stage: Mapping[str, JsonValue],
    detector_results: tuple[Mapping[str, JsonValue], ...],
    component: str | None,
    policy: Mapping[str, JsonValue] | None,
) -> str:
    actions = _stage_actions(stage=stage, detector_results=detector_results, component=component, policy=policy)
    if any(_action_at_least(action, Action.BLOCK) for action in actions):
        return "blocked"
    if any(_action_at_least(action, Action.WARN) for action in actions):
        return "warned"
    status = _optional_string(stage.get("status"))
    if status in ("skipped",):
        return "skipped"
    if status in ("ok", "active", "completed", "decided", "written", "passed"):
        return "passed"
    if status in ("not_configured", "unavailable", "degraded", "missing"):
        return "unavailable"
    if len(stage) == 0:
        return "unavailable"
    return "unavailable"


def _stage_actions(
    stage: Mapping[str, JsonValue],
    detector_results: tuple[Mapping[str, JsonValue], ...],
    component: str | None,
    policy: Mapping[str, JsonValue] | None,
) -> tuple[str, ...]:
    action_values: list[str] = []
    raw_actions = stage.get("actions")
    if isinstance(raw_actions, list):
        action_values.extend(action for action in raw_actions if isinstance(action, str))
    if policy is not None:
        final_action = _optional_string(policy.get("final_action"))
        if final_action is not None and stage.get("stage") == "policy":
            action_values.append(final_action)
    if component is not None:
        for detector in detector_results:
            if detector.get("component") == component:
                action = _optional_string(detector.get("recommended_action"))
                if action is not None:
                    action_values.append(action)
    return tuple(action_values)


def _runtime_evidence_summary(runtime_evidence: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "schema_version": runtime_evidence.get("schema_version"),
        "policy_mode": runtime_evidence.get("policy_mode"),
        "provider_state": runtime_evidence.get("provider_state"),
        "credential_slot_status": runtime_evidence.get("credential_slot_status"),
        "detector_versions": runtime_evidence.get("detector_versions"),
        "artifact_hashes": runtime_evidence.get("artifact_hashes"),
        "cift": runtime_evidence.get("cift"),
        "fail_closed_events": runtime_evidence.get("fail_closed_events"),
    }


def _detector_result_summary(detector: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "detector_name": detector.get("detector_name"),
        "component": detector.get("component"),
        "score": detector.get("score"),
        "confidence": detector.get("confidence"),
        "recommended_action": detector.get("recommended_action"),
        "capability_required": detector.get("capability_required"),
        "capability_status": detector.get("capability_status"),
        "latency_ms": detector.get("latency_ms"),
    }


def _events_from_endpoint(endpoint: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    payload = endpoint.get("payload")
    if not isinstance(payload, dict):
        return ()
    events = payload.get("events")
    if not isinstance(events, list):
        return ()
    records = []
    for event in events:
        if isinstance(event, dict):
            records.append(event)
    return tuple(records)


def _smoke_report_summary(path: Path | None) -> dict[str, JsonValue]:
    if path is None:
        return {"status": "not_configured", "path": None}
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "invalid", "path": str(path), "error": str(exc)}
    if not isinstance(payload, dict):
        return {"status": "invalid", "path": str(path), "error": "smoke report must be a JSON object"}
    summary: dict[str, JsonValue] = {
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "path": str(path),
        "report_id": payload.get("report_id"),
        "provider_mode": payload.get("provider_mode"),
        "nimbus_profile": payload.get("nimbus_profile"),
    }
    confusion_metrics = _optional_mapping(payload.get("confusion_metrics"))
    if len(confusion_metrics) > 0:
        summary["confusion_metrics"] = dict(confusion_metrics)
    checks = _optional_mapping(payload.get("checks"))
    gateway_readiness = _optional_mapping(checks.get("gateway_readiness"))
    if len(gateway_readiness) > 0:
        summary["gateway_readiness"] = _smoke_gateway_readiness_summary(gateway_readiness)
    benign = _optional_mapping(checks.get("benign_cift"))
    exfiltration = _optional_mapping(checks.get("exfiltration_intent_prevention"))
    if len(benign) > 0 or len(exfiltration) > 0:
        summary["cift_live_smoke"] = {
            "benign_final_action": benign.get("final_action"),
            "benign_cift_action": benign.get("cift_action"),
            "benign_score": benign.get("score"),
            "exfiltration_final_action": exfiltration.get("final_action"),
            "exfiltration_cift_action": exfiltration.get("cift_action"),
            "exfiltration_score": exfiltration.get("score"),
            "exfiltration_provider_status": exfiltration.get("provider_status"),
            "hidden_state_device_observed": exfiltration.get("extractor_hidden_state_device_observed")
            or benign.get("extractor_hidden_state_device_observed"),
        }
    return summary


def _smoke_gateway_readiness_summary(gateway_readiness: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "status": gateway_readiness.get("status"),
        "capability_mode": gateway_readiness.get("capability_mode"),
        "certification_id": gateway_readiness.get("certification_id"),
        "certification_mode": gateway_readiness.get("certification_mode"),
        "runtime_model_sha256": gateway_readiness.get("runtime_model_sha256"),
        "release_gate_report_sha256": gateway_readiness.get("release_gate_report_sha256"),
        "model_bundle_id": gateway_readiness.get("model_bundle_id"),
        "source_model_id": gateway_readiness.get("source_model_id"),
        "source_revision": gateway_readiness.get("source_revision"),
        "source_selected_device": gateway_readiness.get("source_selected_device"),
        "feature_key": gateway_readiness.get("feature_key"),
    }


def _first_event(events_payload: Mapping[str, JsonValue]) -> JsonValue:
    events = events_payload.get("events")
    if not isinstance(events, list) or len(events) == 0:
        return None
    return events[0]


def _payload(endpoint: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    payload = endpoint.get("payload")
    if isinstance(payload, dict):
        return payload
    return {}


def _cift_binding(capabilities_payload: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    cift = _optional_mapping(capabilities_payload.get("cift"))
    return _optional_mapping(cift.get("runtime_binding"))


def _cift_status_text(cift_ready: Mapping[str, JsonValue]) -> str:
    support_tier = _optional_string(cift_ready.get("support_tier"))
    support_scope = _optional_string(cift_ready.get("support_scope"))
    if support_tier == _CIFT_SUPPORT_TIER_UNSUPPORTED and support_scope is not None:
        return f"{support_tier}: {support_scope}"
    status = support_tier or _optional_string(cift_ready.get("status")) or _CIFT_SUPPORT_TIER_UNSUPPORTED
    certification_id = _optional_string(cift_ready.get("certification_id"))
    if certification_id is None:
        return status
    return f"{status}: {certification_id}"


def _cift_extractor_ready(cift_ready: Mapping[str, JsonValue]) -> bool:
    if cift_ready.get("status") != "ready":
        return False
    extractor = _optional_mapping(cift_ready.get("extractor"))
    if len(extractor) == 0:
        return cift_ready.get("capability_mode") != "self_hosted_introspection"
    return _optional_string(extractor.get("selected_device")) is not None


def _cift_device_text(cift_ready: Mapping[str, JsonValue]) -> str:
    extractor = _optional_mapping(cift_ready.get("extractor"))
    device = _optional_string(extractor.get("selected_device")) or _optional_string(
        cift_ready.get("source_selected_device")
    )
    if device is None:
        return "not attached"
    return device


def _audit_status_text(audit: Mapping[str, JsonValue]) -> str:
    if audit.get("durable_jsonl_enabled") is True:
        path = _optional_string(audit.get("durable_jsonl_path"))
        if path is not None:
            return path
        return "durable JSONL enabled"
    return "memory only"


def _endpoint_detail(endpoint: Mapping[str, JsonValue], ok_detail: str) -> str:
    if endpoint.get("status") == "ok":
        return ok_detail
    return _optional_string(endpoint.get("error")) or "unavailable"


def _last_event_detail(last_event: JsonValue) -> str:
    if not isinstance(last_event, dict):
        return "no recent events"
    final_action = _optional_string(last_event.get("final_action")) or "unknown"
    trace_id = _optional_string(last_event.get("trace_id")) or "unknown trace"
    return f"{final_action} on {trace_id}"


def _provider_status_from_model_response(model_response: Mapping[str, JsonValue]) -> str:
    provider = _optional_string(model_response.get("provider"))
    if provider == "skipped":
        return "skipped"
    if provider is None:
        return "unknown"
    return "completed"


def _detector_results(record: Mapping[str, JsonValue]) -> tuple[Mapping[str, JsonValue], ...]:
    detector_results = record.get("detector_results")
    if not isinstance(detector_results, list):
        return ()
    results = []
    for result in detector_results:
        if isinstance(result, dict):
            results.append(result)
    return tuple(results)


def _detector_fired(detector: Mapping[str, JsonValue]) -> bool:
    action = _optional_string(detector.get("recommended_action")) or "allow"
    return _action_at_least(action, Action.WARN)


def _action_at_least(action: str, threshold: Action) -> bool:
    try:
        parsed_action = Action(action)
    except ValueError:
        return False
    return action_severity(parsed_action) >= action_severity(threshold)


def _nested_bool(payload: Mapping[str, JsonValue], keys: tuple[str, str]) -> bool:
    nested = _optional_mapping(payload.get(keys[0]))
    return nested.get(keys[1]) is True


def _nested_string(payload: Mapping[str, JsonValue], keys: tuple[str, str]) -> JsonValue:
    nested = _optional_mapping(payload.get(keys[0]))
    return nested.get(keys[1])


def _optional_mapping(value: JsonValue) -> Mapping[str, JsonValue]:
    if isinstance(value, dict):
        return value
    return {}


def _optional_string(value: JsonValue) -> str | None:
    if isinstance(value, str) and value != "":
        return value
    return None


def _optional_path_env(env: Mapping[str, str], key: str) -> Path | None:
    value = env.get(key)
    if value is None or value == "":
        return None
    return Path(value)


def _positive_float_env(env: Mapping[str, str], key: str, fallback: float) -> float:
    raw = env.get(key)
    if raw is None:
        return fallback
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConsoleServiceError(f"{key} must be a positive number.") from exc
    if value <= 0.0:
        raise ConsoleServiceError(f"{key} must be positive.")
    return value


def settings_from_process_env() -> ConsoleSettings:
    return settings_from_env(os.environ)
