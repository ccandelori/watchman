from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from aegis.console.app import create_app
from aegis.console.service import ConsoleSettings, console_events, console_overview
from aegis.core.contracts import JsonValue


def test_console_overview_summarizes_protected_cift_state() -> None:
    overview = console_overview(settings=_settings(), fetcher=_gateway_fetcher(events=(_audit_record(),)))

    assert overview["protection"] == {
        "state": "Protected",
        "severity": "protected",
        "active_profile": "strict",
        "gateway_online": True,
        "gateway_ready": True,
        "cift_capability_mode": "self_hosted_introspection",
    }
    assert overview["model"]["model_id"] == "Qwen/Qwen3-4B"
    assert overview["model"]["device"] == "mps"
    assert overview["cift"]["certificate_status"] == "certified"
    assert overview["cift"]["support_tier"] == "runtime-enforceable"
    assert overview["cift"]["support_scope"] == "model-specific CIFT enforcement for Qwen/Qwen3-4B on mps"
    assert (
        overview["cift"]["support_reason"]
        == "strict certification binding and live extractor readiness are satisfied."
    )
    assert overview["nimbus"]["label"] == "deterministic beta"
    assert overview["last_request"]["final_action"] == "block"


def test_console_overview_summarizes_learned_nimbus_runtime_beta() -> None:
    ready_payload = _ready_payload()
    ready_payload["nimbus"] = {
        "status": "learned_runtime_beta",
        "critic_version": "nimbus-infonce-lexical-v0",
        "promotion_status": "learned_runtime_beta_not_promotable",
    }
    capabilities_payload = _capabilities_payload()
    capabilities_payload["nimbus"] = {
        "status": "learned_runtime_beta",
        "critic_kind": "learned_infonce_beta",
        "critic_version": "nimbus-infonce-lexical-v0",
        "paper_faithful_learned_critic": False,
        "promotion_status": "learned_runtime_beta_not_promotable",
        "infonce_model_path": "introspection/data/reports/aegis_nimbus_infonce_model_v0.json",
    }

    overview = console_overview(
        settings=_settings(),
        fetcher=_gateway_fetcher_with_payloads(
            events=(_audit_record(),),
            ready_payload=ready_payload,
            capabilities_payload=capabilities_payload,
        ),
    )

    assert overview["nimbus"]["label"] == "learned runtime beta"
    assert overview["nimbus"]["critic_kind"] == "learned_infonce_beta"
    assert overview["nimbus"]["promotion_status"] == "learned_runtime_beta_not_promotable"
    assert overview["nimbus"]["infonce_model_path"] == "introspection/data/reports/aegis_nimbus_infonce_model_v0.json"
    checklist = {item["label"]: item for item in overview["checklist"]}
    assert checklist["NIMBUS"] == {
        "label": "NIMBUS",
        "status": "passed",
        "detail": "learned_runtime_beta / learned_infonce_beta / learned_runtime_beta_not_promotable",
    }


def test_console_overview_surfaces_cift_smoke_metrics(tmp_path: Path) -> None:
    smoke_path = tmp_path / "cift-smoke.json"
    smoke_path.write_text(json.dumps(_cift_smoke_report()), encoding="utf-8")

    overview = console_overview(
        settings=_settings(smoke_report_path=smoke_path),
        fetcher=_gateway_fetcher(events=(_audit_record(),)),
    )

    assert overview["cift"]["runtime_model_bundle_id"] == "selected-choice-linear"
    assert overview["last_smoke"]["schema_version"] == "aegis.proxy.cift_gateway_smoke/v1"
    assert overview["last_smoke"]["confusion_metrics"] == {
        "false_negative_count": 0,
        "false_negative_rate": 0.0,
        "false_positive_count": 0,
        "false_positive_rate": 0.0,
    }
    assert overview["last_smoke"]["cift_live_smoke"] == {
        "benign_final_action": "allow",
        "benign_cift_action": "allow",
        "benign_score": 0.2,
        "exfiltration_final_action": "block",
        "exfiltration_cift_action": "block",
        "exfiltration_score": 0.99,
        "exfiltration_provider_status": "skipped",
        "hidden_state_device_observed": "mps:0",
    }


def test_console_overview_marks_black_box_cift_as_unsupported() -> None:
    ready_payload = {
        **_ready_payload(),
        "strict_protected_mode": {"enabled": False},
        "cift": {
            "ready": True,
            "status": "not_required",
            "capability_mode": "black_box",
            "support_tier": "unsupported",
            "support_scope": "model-specific CIFT enforcement unavailable",
            "support_reason": (
                "black-box provider mode has no certified hidden-state extractor binding; "
                "DP-HONEY, NIMBUS, and provider egress remain available."
            ),
        },
    }
    capabilities_payload = {
        **_capabilities_payload(),
        "cift": {
            "capability_mode": "black_box",
            "detectors": ["activation_unavailable"],
            "support_tier": "unsupported",
            "support_scope": "model-specific CIFT enforcement unavailable",
            "support_reason": (
                "black-box provider mode has no certified hidden-state extractor binding; "
                "DP-HONEY, NIMBUS, and provider egress remain available."
            ),
            "turn_annotator_count": 0,
        },
    }

    overview = console_overview(
        settings=_settings(),
        fetcher=_gateway_fetcher_with_payloads(
            events=(),
            ready_payload=ready_payload,
            capabilities_payload=capabilities_payload,
        ),
    )

    assert overview["protection"]["state"] == "Degraded"
    assert overview["cift"]["certificate_status"] == "unsupported"
    assert overview["cift"]["support_tier"] == "unsupported"
    assert overview["cift"]["support_scope"] == "model-specific CIFT enforcement unavailable"
    checklist = {item["label"]: item for item in overview["checklist"]}
    assert checklist["CIFT certificate"]["detail"] == "unsupported: model-specific CIFT enforcement unavailable"


def test_console_events_use_sample_audit_when_live_audit_is_empty(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(json.dumps(_audit_record()) + "\n", encoding="utf-8")
    settings = _settings(sample_audit_jsonl_path=audit_path)

    events = console_events(settings=settings, fetcher=_gateway_fetcher(events=()), limit=10, session_id=None)

    assert events["source"] == "sample_audit_jsonl"
    assert events["detector_activity"]["provider_egress_blocks"] == 1
    timeline = events["events"][0]["stage_timeline"]
    assert [stage["stage"] for stage in timeline] == [
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
    ]
    assert {stage["status"] for stage in timeline} <= {"passed", "blocked", "warned", "skipped", "unavailable"}
    assert timeline[3]["status"] == "blocked"
    assert "matches" not in json.dumps(events, sort_keys=True)
    assert "synthetic-sensitive-marker" not in json.dumps(events, sort_keys=True)


def test_console_app_serves_static_shell_and_api() -> None:
    client = TestClient(create_app(settings=_settings(), gateway_fetcher=_gateway_fetcher(events=(_audit_record(),))))

    page = client.get("/")
    overview = client.get("/api/overview")
    setup = client.get("/api/setup")

    assert page.status_code == 200
    assert "Aegis Watchman Console" in page.text
    assert overview.status_code == 200
    assert overview.json()["protection"]["state"] == "Protected"
    assert setup.status_code == 200
    setup_payload = setup.json()
    assert setup_payload["agent_gateway_base_url"] == "http://127.0.0.1:8000/v1"
    assert setup_payload["openai_compatible_endpoint"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert setup_payload["agent_settings"] == {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "aegis-local-dev-key",
        "model": "match the local provider model configured by AEGIS_OPENAI_MODEL",
    }
    assert setup_payload["agent_settings_text"] == (
        "Base URL: http://127.0.0.1:8000/v1\n"
        "API key: aegis-local-dev-key\n"
        "Model: match AEGIS_OPENAI_MODEL"
    )
    assert setup_payload["cift_sidecar"] == {
        "default_base_url": "http://127.0.0.1:9000",
        "status_source": "/ready.cift",
        "certification_rule": "sidecar attestation must match the promoted model-specific CIFT artifact",
    }
    assert {item["component"] for item in setup_payload["architecture"]} == {
        "agent_app",
        "aegis_gateway",
        "local_model_provider",
        "cift_sidecar",
    }
    assert {item["name"] for item in setup_payload["provider_examples"]} == {
        "Generic OpenAI-compatible local server",
        "Ollama OpenAI-compatible endpoint",
        "LM Studio or llama.cpp OpenAI-compatible endpoint",
    }
    assert any("strict CIFT blocks exfiltration intent" in item for item in setup_payload["smoke_expectations"])
    degraded_states = {item["state"] for item in setup_payload["common_degraded_states"]}
    assert "nimbus_learned_runtime_beta_not_promotable" in degraded_states
    assert "provider_unreachable" in degraded_states


def _settings(
    smoke_report_path: Path | None = None,
    sample_audit_jsonl_path: Path | None = None,
) -> ConsoleSettings:
    return ConsoleSettings(
        gateway_base_url="http://127.0.0.1:8000",
        request_timeout_seconds=0.1,
        smoke_report_path=smoke_report_path,
        sample_audit_jsonl_path=sample_audit_jsonl_path,
        operator_profile="balanced",
    )


def _gateway_fetcher(
    events: tuple[dict[str, JsonValue], ...],
) -> Callable[[ConsoleSettings, str, tuple[tuple[str, str], ...]], dict[str, JsonValue]]:
    return _gateway_fetcher_with_payloads(
        events=events,
        ready_payload=_ready_payload(),
        capabilities_payload=_capabilities_payload(),
    )


def _gateway_fetcher_with_payloads(
    events: tuple[dict[str, JsonValue], ...],
    ready_payload: dict[str, JsonValue],
    capabilities_payload: dict[str, JsonValue],
) -> Callable[[ConsoleSettings, str, tuple[tuple[str, str], ...]], dict[str, JsonValue]]:
    payloads: dict[str, dict[str, JsonValue]] = {
        "/health": {"status": "ok"},
        "/ready": ready_payload,
        "/aegis/capabilities": capabilities_payload,
        "/audit/recent": {"events": [event for event in events]},
    }

    def fetcher(
        settings: ConsoleSettings,
        path: str,
        query: tuple[tuple[str, str], ...],
    ) -> dict[str, JsonValue]:
        _ = settings
        _ = query
        return payloads[path]

    return fetcher


def _ready_payload() -> dict[str, JsonValue]:
    return {
        "ready": True,
        "status": "ready",
        "strict_protected_mode": {"enabled": True},
        "provider": {"status": "ready", "name": "openai_compatible", "target_url": "http://127.0.0.1:8776/v1"},
        "dp_honey": {"status": "ready"},
        "provider_egress_guard": {"status": "ready"},
        "nimbus": {"status": "deterministic_beta", "critic_version": "canary-v0"},
        "cift": {
            "status": "ready",
            "capability_mode": "self_hosted_introspection",
            "support_tier": "runtime-enforceable",
            "support_scope": "model-specific CIFT enforcement for Qwen/Qwen3-4B on mps",
            "support_reason": "strict certification binding and live extractor readiness are satisfied.",
            "certification_id": "cert-qwen3-4b",
            "certification_mode": "strict",
            "runtime_model_sha256": "a" * 64,
            "release_gate_report_sha256": "b" * 64,
            "source_selected_device": "mps",
            "extractor": {"selected_device": "mps"},
        },
    }


def _capabilities_payload() -> dict[str, JsonValue]:
    return {
        "audit": {
            "durable_jsonl_enabled": True,
            "durable_jsonl_path": "/tmp/aegis-audit.jsonl",
            "explain_route": "/audit/explain",
        },
        "nimbus": {
            "status": "deterministic_beta",
            "critic_kind": "deterministic_canary",
            "paper_faithful_learned_critic": False,
        },
        "provider": {
            "name": "openai_compatible",
            "target_url": "http://127.0.0.1:8776/v1",
            "mock_controls_enabled": False,
        },
        "cift": {
            "capability_mode": "self_hosted_introspection",
            "support_tier": "runtime-enforceable",
            "support_scope": "model-specific CIFT enforcement for Qwen/Qwen3-4B on mps",
            "support_reason": (
                "strict certification binding is loaded; readiness still depends on trusted extractor attestation."
            ),
            "runtime_binding": {
                "certification_id": "cert-qwen3-4b",
                "certification_mode": "strict",
                "runtime_model_sha256": "a" * 64,
                "release_gate_report_sha256": "b" * 64,
                "model_bundle_id": "selected-choice-linear",
                "source_model_id": "Qwen/Qwen3-4B",
                "source_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                "source_selected_device": "mps",
                "source_hidden_size": 2560,
                "source_layer_count": 36,
                "feature_key": "selected_choice_window_layer_21",
                "feature_count": 2560,
                "selected_choice_readout_token_count": 4,
                "tokenizer_fingerprint_sha256": "c" * 64,
                "special_tokens_map_sha256": "d" * 64,
                "chat_template_sha256": "e" * 64,
            },
        },
    }


def _cift_smoke_report() -> dict[str, JsonValue]:
    return {
        "schema_version": "aegis.proxy.cift_gateway_smoke/v1",
        "status": "ok",
        "report_id": "console-test-cift-smoke",
        "confusion_metrics": {
            "false_negative_count": 0,
            "false_negative_rate": 0.0,
            "false_positive_count": 0,
            "false_positive_rate": 0.0,
        },
        "checks": {
            "gateway_readiness": {
                "status": "ready",
                "capability_mode": "self_hosted_introspection",
                "certification_id": "cert-qwen3-4b",
                "certification_mode": "strict",
                "runtime_model_sha256": "a" * 64,
                "release_gate_report_sha256": "b" * 64,
                "model_bundle_id": "selected-choice-linear",
                "source_model_id": "Qwen/Qwen3-4B",
                "source_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                "source_selected_device": "mps",
                "feature_key": "selected_choice_window_layer_21",
            },
            "benign_cift": {
                "final_action": "allow",
                "cift_action": "allow",
                "score": 0.2,
                "extractor_hidden_state_device_observed": "mps:0",
            },
            "exfiltration_intent_prevention": {
                "final_action": "block",
                "cift_action": "block",
                "score": 0.99,
                "provider_status": "skipped",
                "extractor_hidden_state_device_observed": "mps:0",
            },
        },
    }


def _audit_record() -> dict[str, JsonValue]:
    return {
        "trace_id": "trace-cift-block",
        "session_id": "session-cift-block",
        "turn_index": 0,
        "created_at": "2026-06-25T00:00:00+00:00",
        "latency_ms": 1.0,
        "normalized_turn": {
            "trace_id": "trace-cift-block",
            "session_id": "session-cift-block",
            "turn_index": 0,
            "capability_mode": "self_hosted_introspection",
            "model": {
                "provider": "mock",
                "model_id": "Qwen/Qwen3-4B",
                "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                "selected_device": "mps",
            },
            "messages": [{"role": "user", "content": "[REDACTED_SENSITIVE]"}],
            "tool_calls": [],
            "sensitive_spans": [],
            "metadata": {
                "aegis_credential_slot_detection": {
                    "status": "honeytoken_substituted",
                    "credential_needed_count": 1,
                    "honeytoken_substituted_count": 1,
                    "real_secret_present_count": 0,
                },
                "dp_honey_canary_count": 1,
            },
        },
        "detector_results": [
            {
                "detector_name": "cift_runtime",
                "component": "cift",
                "score": 0.97,
                "confidence": 0.99,
                "recommended_action": "block",
                "capability_required": "self_hosted_introspection",
                "capability_status": "active",
                "evidence": {"feature_source": "self_hosted_cift"},
                "latency_ms": 4.0,
            },
            {
                "detector_name": "provider_egress_guard",
                "component": "tool_scanner",
                "score": 1.0,
                "confidence": 1.0,
                "recommended_action": "block",
                "capability_required": None,
                "capability_status": "active",
                "evidence": {
                    "reason": "blocked_sensitive_value_before_provider_egress",
                    "matches": [{"marker": "synthetic-sensitive-marker"}],
                },
                "latency_ms": 0.0,
            },
        ],
        "policy_decision": {
            "final_action": "block",
            "reason": "cift_pre_generation_policy_block",
            "triggered_detectors": ["cift_runtime"],
            "risk_score": 1.0,
            "sanitized_output": "",
        },
        "model_response_metadata": {
            "provider": "skipped",
            "reason": "pre_generation_policy_block",
            "model_id": "Qwen/Qwen3-4B",
        },
        "runtime_evidence": {
            "schema_version": "aegis.audit_runtime_evidence/v1",
            "policy_mode": "severity",
            "provider_state": {"status": "skipped", "reason": "pre_generation_policy_block"},
            "detector_versions": {"cift_runtime": "linear-v1"},
            "fail_closed_events": [
                {
                    "kind": "pre_generation_policy_block",
                    "provider_status": "skipped",
                    "final_action": "block",
                    "triggered_detectors": ["cift_runtime"],
                }
            ],
        },
    }
