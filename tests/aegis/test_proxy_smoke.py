from __future__ import annotations

from pathlib import Path

import pytest

from aegis.core.contracts import JsonValue
from aegis.proxy.smoke import (
    GatewaySmokeConfig,
    GatewaySmokeError,
    GatewaySmokeProviderMode,
    HttpJsonResponse,
    NimbusSmokeProfile,
    parse_args,
    run_gateway_smoke,
)


class FakeSmokeClient:
    def __init__(self, responses: dict[tuple[str, str], tuple[HttpJsonResponse, ...]]) -> None:
        self._responses = responses
        self.requests: list[tuple[str, str, dict[str, JsonValue] | None]] = []

    def get_json(self, url: str, timeout_seconds: float) -> HttpJsonResponse:
        self.requests.append(("GET", url, None))
        return self._response(method="GET", url=url)

    def post_json(self, url: str, payload: dict[str, JsonValue], timeout_seconds: float) -> HttpJsonResponse:
        self.requests.append(("POST", url, payload))
        return self._response(method="POST", url=url)

    def _response(self, method: str, url: str) -> HttpJsonResponse:
        responses = self._responses.get((method, url))
        if responses is None or len(responses) == 0:
            raise GatewaySmokeError(f"unexpected request {method} {url}")
        response = responses[0]
        self._responses[(method, url)] = responses[1:]
        return response


def _reset_responses() -> tuple[HttpJsonResponse, ...]:
    return (
        HttpJsonResponse(status_code=200, payload={"status": "reset"}),
        HttpJsonResponse(status_code=200, payload={"status": "reset"}),
        HttpJsonResponse(status_code=200, payload={"status": "reset"}),
        HttpJsonResponse(status_code=200, payload={"status": "reset"}),
        HttpJsonResponse(status_code=200, payload={"status": "reset"}),
    )


def test_gateway_smoke_parses_explicit_url_and_timeout() -> None:
    config = parse_args(("--url", "http://127.0.0.1:8000/", "--timeout", "2.5"))

    assert config == GatewaySmokeConfig(
        base_url="http://127.0.0.1:8000",
        timeout_seconds=2.5,
        nimbus_profile=NimbusSmokeProfile.DEFAULT,
        require_cift_pre_generation_block=False,
        provider_mode=GatewaySmokeProviderMode.MOCK,
        output_path=None,
    )


def test_gateway_smoke_parses_strict_nimbus_profile() -> None:
    config = parse_args(
        (
            "--url",
            "http://127.0.0.1:8000",
            "--timeout",
            "2.5",
            "--nimbus-profile",
            "strict-partial-block",
        )
    )

    assert config == GatewaySmokeConfig(
        base_url="http://127.0.0.1:8000",
        timeout_seconds=2.5,
        nimbus_profile=NimbusSmokeProfile.STRICT_PARTIAL_BLOCK,
        require_cift_pre_generation_block=False,
        provider_mode=GatewaySmokeProviderMode.MOCK,
        output_path=None,
    )


def test_gateway_smoke_parses_required_cift_pre_generation_block() -> None:
    config = parse_args(
        (
            "--url",
            "http://127.0.0.1:8000",
            "--timeout",
            "2.5",
            "--require-cift-pre-generation-block",
        )
    )

    assert config == GatewaySmokeConfig(
        base_url="http://127.0.0.1:8000",
        timeout_seconds=2.5,
        nimbus_profile=NimbusSmokeProfile.DEFAULT,
        require_cift_pre_generation_block=True,
        provider_mode=GatewaySmokeProviderMode.MOCK,
        output_path=None,
    )


def test_gateway_smoke_parses_real_provider_mode() -> None:
    config = parse_args(
        (
            "--url",
            "http://127.0.0.1:8000",
            "--timeout",
            "2.5",
            "--provider-mode",
            "real-provider",
        )
    )

    assert config == GatewaySmokeConfig(
        base_url="http://127.0.0.1:8000",
        timeout_seconds=2.5,
        nimbus_profile=NimbusSmokeProfile.DEFAULT,
        require_cift_pre_generation_block=False,
        provider_mode=GatewaySmokeProviderMode.REAL_PROVIDER,
        output_path=None,
    )


def test_gateway_smoke_parses_output_path() -> None:
    config = parse_args(
        (
            "--url",
            "http://127.0.0.1:8000",
            "--timeout",
            "2.5",
            "--output",
            "introspection/data/reports/aegis-smoke.json",
        )
    )

    assert config.output_path == Path("introspection/data/reports/aegis-smoke.json")


def test_gateway_smoke_rejects_non_positive_timeout() -> None:
    with pytest.raises(GatewaySmokeError, match="timeout"):
        parse_args(("--url", "http://127.0.0.1:8000", "--timeout", "0"))


def test_gateway_smoke_accepts_healthy_gateway_contract() -> None:
    base_url = "http://gateway.test"
    client = FakeSmokeClient(
        {
            ("GET", f"{base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{base_url}/ready"): (HttpJsonResponse(status_code=200, payload=_ready_response()),),
            ("GET", f"{base_url}/aegis/capabilities"): (
                HttpJsonResponse(status_code=200, payload=_capabilities_response(include_seed_route=True)),
            ),
            ("POST", f"{base_url}/test/reset"): _reset_responses(),
            ("POST", f"{base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        detector_names=("activation_unavailable",),
                        dp_honey_status="active",
                        canary_count=1,
                    ),
                ),
                HttpJsonResponse(status_code=400, payload=_ambiguous_protected_workflow_error_response()),
                HttpJsonResponse(status_code=200, payload=_egress_guard_chat_response()),
                HttpJsonResponse(status_code=200, payload=_tool_argument_canary_chat_response()),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="escalate",
                        detector_names=("encoded_canary", "nimbus", "provider_egress_guard"),
                        dp_honey_status="active",
                        canary_count=1,
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="escalate",
                        detector_names=("text_canary", "nimbus", "provider_egress_guard"),
                        dp_honey_status="active",
                        canary_count=1,
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_partial_nimbus_chat_response(
                        final_action="warn",
                        nimbus_action="warn",
                        budget_fraction=0.4,
                        block_threshold=0.9,
                    ),
                ),
            ),
            ("GET", f"{base_url}/audit/recent"): (
                HttpJsonResponse(
                    status_code=200,
                    payload={"events": [{"trace_id": "smoke-slot-leak-trace"}]},
                ),
            ),
            ("GET", f"{base_url}/audit/explain?trace_id=smoke-egress-guard-trace"): (
                HttpJsonResponse(status_code=200, payload=_audit_explain_response()),
            ),
        }
    )

    report = run_gateway_smoke(
        GatewaySmokeConfig(
            base_url=base_url,
            timeout_seconds=1.0,
            nimbus_profile=NimbusSmokeProfile.DEFAULT,
            require_cift_pre_generation_block=False,
            provider_mode=GatewaySmokeProviderMode.MOCK,
            output_path=None,
        ),
        client,
    )

    assert report["status"] == "ok"
    assert report["nimbus_profile"] == "default"
    readiness = report["checks"]["gateway_readiness"]
    assert isinstance(readiness, dict)
    assert readiness["provider_name"] == "mock"
    assert readiness["provider_mock_controls_enabled"] is True
    assert client.requests[0] == ("GET", f"{base_url}/health", None)
    assert client.requests[1] == ("GET", f"{base_url}/ready", None)
    assert client.requests[2] == ("GET", f"{base_url}/aegis/capabilities", None)
    assert client.requests[3][0:2] == ("POST", f"{base_url}/test/reset")
    assert client.requests[4][0:2] == ("POST", f"{base_url}/test/reset")
    assert client.requests[5][0:2] == ("POST", f"{base_url}/test/reset")
    assert client.requests[6][0:2] == ("POST", f"{base_url}/test/reset")
    assert client.requests[7][0:2] == ("POST", f"{base_url}/test/reset")
    benign_request = client.requests[8][2]
    ambiguous_request = client.requests[9][2]
    egress_request = client.requests[10][2]
    tool_argument_request = client.requests[11][2]
    leak_request = client.requests[12][2]
    slot_leak_request = client.requests[13][2]
    partial_leak_request = client.requests[14][2]
    assert isinstance(benign_request, dict)
    assert isinstance(ambiguous_request, dict)
    assert isinstance(egress_request, dict)
    assert isinstance(tool_argument_request, dict)
    assert isinstance(leak_request, dict)
    assert isinstance(slot_leak_request, dict)
    assert isinstance(partial_leak_request, dict)
    assert benign_request["metadata"] != leak_request["metadata"]
    assert benign_request["metadata"]["session_id"] == "smoke-session"
    assert benign_request["metadata"]["protected_workflow"] is True
    assert benign_request["metadata"]["credential_slots"] == [
        {"slot_name": "repo_pat", "credential_type": "github_pat"}
    ]
    assert ambiguous_request["metadata"]["session_id"] == "smoke-ambiguous-protected-session"
    assert ambiguous_request["metadata"]["protected_workflow"] is True
    assert "credential_slots" not in ambiguous_request["metadata"]
    ambiguous_summary = report["checks"]["ambiguous_protected_workflow"]
    assert isinstance(ambiguous_summary, dict)
    assert ambiguous_summary["status"] == "fail_closed"
    assert ambiguous_summary["credential_slot_status"] == "ambiguous_protected_workflow"
    assert egress_request["metadata"]["session_id"] == "smoke-session"
    assert egress_request["tool_calls"] == [
        {
            "name": "external_ticket",
            "arguments": {"body": "credential=ghp_realLookingToolSecret1234567890"},
        }
    ]
    assert tool_argument_request["metadata"]["session_id"] == "smoke-tool-session"
    assert tool_argument_request["tool_calls"] == [
        {
            "name": "external_ticket",
            "arguments": {"body": "credential={{CREDENTIAL:repo_pat:github_pat}}"},
        }
    ]
    assert slot_leak_request["metadata"]["session_id"] == "smoke-slot-session"
    assert slot_leak_request["metadata"]["credential_slots"] == [
        {"slot_name": "repo_pat", "credential_type": "github_pat"}
    ]
    assert partial_leak_request["metadata"]["session_id"] == "smoke-partial-session"


def test_gateway_smoke_accepts_required_self_hosted_cift_pre_generation_block() -> None:
    base_url = "http://gateway.test"
    client = FakeSmokeClient(
        {
            ("GET", f"{base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{base_url}/ready"): (HttpJsonResponse(status_code=200, payload=_self_hosted_ready_response()),),
            ("GET", f"{base_url}/aegis/capabilities"): (
                HttpJsonResponse(status_code=200, payload=_self_hosted_capabilities_response()),
            ),
            ("POST", f"{base_url}/test/reset"): _reset_responses(),
            ("POST", f"{base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        detector_names=("cift_runtime",),
                        dp_honey_status="active",
                        canary_count=1,
                    ),
                ),
                HttpJsonResponse(status_code=400, payload=_ambiguous_protected_workflow_error_response()),
                HttpJsonResponse(status_code=200, payload=_cift_block_chat_response()),
                HttpJsonResponse(status_code=200, payload=_egress_guard_chat_response()),
                HttpJsonResponse(status_code=200, payload=_tool_argument_canary_chat_response()),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="escalate",
                        detector_names=("cift_runtime", "encoded_canary", "nimbus", "provider_egress_guard"),
                        dp_honey_status="active",
                        canary_count=1,
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="escalate",
                        detector_names=("cift_runtime", "text_canary", "nimbus", "provider_egress_guard"),
                        dp_honey_status="active",
                        canary_count=1,
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_partial_nimbus_chat_response(
                        final_action="warn",
                        nimbus_action="warn",
                        budget_fraction=0.4,
                        block_threshold=0.9,
                    ),
                ),
            ),
            ("GET", f"{base_url}/audit/recent"): (
                HttpJsonResponse(status_code=200, payload={"events": [{"trace_id": "smoke-cift-block-trace"}]}),
            ),
            ("GET", f"{base_url}/audit/explain?trace_id=smoke-egress-guard-trace"): (
                HttpJsonResponse(status_code=200, payload=_audit_explain_response()),
            ),
        }
    )

    report = run_gateway_smoke(
        GatewaySmokeConfig(
            base_url=base_url,
            timeout_seconds=1.0,
            nimbus_profile=NimbusSmokeProfile.DEFAULT,
            require_cift_pre_generation_block=True,
            provider_mode=GatewaySmokeProviderMode.MOCK,
            output_path=None,
        ),
        client,
    )

    cift_summary = report["checks"]["cift_pre_generation_block"]
    assert isinstance(cift_summary, dict)
    assert cift_summary["status"] == "blocked"
    assert cift_summary["final_action"] == "block"
    assert cift_summary["cift_action"] == "block"
    assert cift_summary["provider_status"] == "skipped"
    assert cift_summary["provider_reason"] == "pre_generation_policy_block"
    cift_request = client.requests[10][2]
    assert isinstance(cift_request, dict)
    assert cift_request["metadata"]["session_id"] == "smoke-cift-session"
    assert cift_request["metadata"]["protected_workflow"] is True
    egress_summary = report["checks"]["provider_egress_guard_block"]
    assert isinstance(egress_summary, dict)
    assert egress_summary["final_action"] == "block"
    assert egress_summary["guard_action"] == "block"
    assert egress_summary["provider_status"] == "skipped"


def test_gateway_smoke_accepts_strict_partial_block_profile() -> None:
    base_url = "http://gateway.test"
    client = _healthy_smoke_client(
        base_url=base_url,
        partial_final_action="block",
        partial_nimbus_action="block",
        partial_block_threshold=0.36,
    )

    report = run_gateway_smoke(
        GatewaySmokeConfig(
            base_url=base_url,
            timeout_seconds=1.0,
            nimbus_profile=NimbusSmokeProfile.STRICT_PARTIAL_BLOCK,
            require_cift_pre_generation_block=False,
            provider_mode=GatewaySmokeProviderMode.MOCK,
            output_path=None,
        ),
        client,
    )

    partial_summary = report["checks"]["nimbus_partial_leak"]
    assert isinstance(partial_summary, dict)
    assert partial_summary["final_action"] == "block"
    assert partial_summary["nimbus_action"] == "block"


def test_gateway_smoke_accepts_real_provider_mode_without_mock_leak_controls() -> None:
    base_url = "http://gateway.test"
    client = FakeSmokeClient(
        {
            ("GET", f"{base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{base_url}/ready"): (HttpJsonResponse(status_code=200, payload=_real_provider_ready_response()),),
            ("GET", f"{base_url}/aegis/capabilities"): (
                HttpJsonResponse(status_code=200, payload=_real_provider_capabilities_response()),
            ),
            ("POST", f"{base_url}/test/reset"): _reset_responses(),
            ("POST", f"{base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        detector_names=("activation_unavailable",),
                        dp_honey_status="active",
                        canary_count=1,
                    ),
                ),
                HttpJsonResponse(status_code=400, payload=_ambiguous_protected_workflow_error_response()),
                HttpJsonResponse(status_code=200, payload=_egress_guard_chat_response()),
            ),
            ("GET", f"{base_url}/audit/recent"): (
                HttpJsonResponse(status_code=200, payload={"events": [{"trace_id": "smoke-egress-guard-trace"}]}),
            ),
            ("GET", f"{base_url}/audit/explain?trace_id=smoke-egress-guard-trace"): (
                HttpJsonResponse(status_code=200, payload=_audit_explain_response()),
            ),
        }
    )

    report = run_gateway_smoke(
        GatewaySmokeConfig(
            base_url=base_url,
            timeout_seconds=1.0,
            nimbus_profile=NimbusSmokeProfile.DEFAULT,
            require_cift_pre_generation_block=False,
            provider_mode=GatewaySmokeProviderMode.REAL_PROVIDER,
            output_path=None,
        ),
        client,
    )

    assert report["provider_mode"] == "real-provider"
    readiness = report["checks"]["gateway_readiness"]
    assert isinstance(readiness, dict)
    assert readiness["provider_name"] == "openai_compatible"
    assert readiness["provider_mock_controls_enabled"] is False
    assert report["checks"]["tool_argument_canary_leak"]["status"] == "skipped"
    assert report["checks"]["encoded_canary_leak"]["status"] == "skipped"
    assert report["checks"]["metadata_slot_canary_leak"]["status"] == "skipped"
    assert report["checks"]["nimbus_partial_leak"]["status"] == "skipped"
    chat_requests = [
        request for request in client.requests if request[0] == "POST" and request[1].endswith("/v1/chat/completions")
    ]
    assert len(chat_requests) == 3


def test_gateway_smoke_rejects_real_provider_mode_when_ready_reports_mock_provider() -> None:
    base_url = "http://gateway.test"
    client = FakeSmokeClient(
        {
            ("GET", f"{base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{base_url}/ready"): (HttpJsonResponse(status_code=200, payload=_ready_response()),),
        }
    )

    with pytest.raises(GatewaySmokeError, match=r"provider\.name must be openai_compatible"):
        run_gateway_smoke(
            GatewaySmokeConfig(
                base_url=base_url,
                timeout_seconds=1.0,
                nimbus_profile=NimbusSmokeProfile.DEFAULT,
                require_cift_pre_generation_block=False,
                provider_mode=GatewaySmokeProviderMode.REAL_PROVIDER,
                output_path=None,
            ),
            client,
        )


def test_gateway_smoke_rejects_default_partial_behavior_under_strict_profile() -> None:
    base_url = "http://gateway.test"
    client = _healthy_smoke_client(
        base_url=base_url,
        partial_final_action="warn",
        partial_nimbus_action="warn",
        partial_block_threshold=0.9,
    )

    with pytest.raises(GatewaySmokeError, match="strict NIMBUS profile expected"):
        run_gateway_smoke(
            GatewaySmokeConfig(
                base_url=base_url,
                timeout_seconds=1.0,
                nimbus_profile=NimbusSmokeProfile.STRICT_PARTIAL_BLOCK,
                require_cift_pre_generation_block=False,
                provider_mode=GatewaySmokeProviderMode.MOCK,
                output_path=None,
            ),
            client,
        )


def test_gateway_smoke_rejects_unhealthy_health_payload() -> None:
    base_url = "http://gateway.test"
    client = FakeSmokeClient(
        {
            ("GET", f"{base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "not-ok"}),),
        }
    )

    with pytest.raises(GatewaySmokeError, match="status=ok"):
        run_gateway_smoke(
            GatewaySmokeConfig(
                base_url=base_url,
                timeout_seconds=1.0,
                nimbus_profile=NimbusSmokeProfile.DEFAULT,
                require_cift_pre_generation_block=False,
                provider_mode=GatewaySmokeProviderMode.MOCK,
                output_path=None,
            ),
            client,
        )


def test_gateway_smoke_rejects_missing_encoded_canary_result() -> None:
    base_url = "http://gateway.test"
    client = FakeSmokeClient(
        {
            ("GET", f"{base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{base_url}/ready"): (HttpJsonResponse(status_code=200, payload=_ready_response()),),
            ("GET", f"{base_url}/aegis/capabilities"): (
                HttpJsonResponse(status_code=200, payload=_capabilities_response(include_seed_route=True)),
            ),
            ("POST", f"{base_url}/test/reset"): _reset_responses(),
            ("POST", f"{base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        detector_names=("activation_unavailable",),
                        dp_honey_status="active",
                        canary_count=1,
                    ),
                ),
                HttpJsonResponse(status_code=400, payload=_ambiguous_protected_workflow_error_response()),
                HttpJsonResponse(status_code=200, payload=_egress_guard_chat_response()),
                HttpJsonResponse(status_code=200, payload=_tool_argument_canary_chat_response()),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        detector_names=("activation_unavailable",),
                        dp_honey_status="active",
                        canary_count=1,
                    ),
                ),
            ),
        }
    )

    with pytest.raises(GatewaySmokeError, match="expected block-or-stronger"):
        run_gateway_smoke(
            GatewaySmokeConfig(
                base_url=base_url,
                timeout_seconds=1.0,
                nimbus_profile=NimbusSmokeProfile.DEFAULT,
                require_cift_pre_generation_block=False,
                provider_mode=GatewaySmokeProviderMode.MOCK,
                output_path=None,
            ),
            client,
        )


def _healthy_smoke_client(
    base_url: str,
    partial_final_action: str,
    partial_nimbus_action: str,
    partial_block_threshold: float,
) -> FakeSmokeClient:
    return FakeSmokeClient(
        {
            ("GET", f"{base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{base_url}/ready"): (HttpJsonResponse(status_code=200, payload=_ready_response()),),
            ("GET", f"{base_url}/aegis/capabilities"): (
                HttpJsonResponse(status_code=200, payload=_capabilities_response(include_seed_route=True)),
            ),
            ("POST", f"{base_url}/test/reset"): _reset_responses(),
            ("POST", f"{base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        detector_names=("activation_unavailable",),
                        dp_honey_status="active",
                        canary_count=1,
                    ),
                ),
                HttpJsonResponse(status_code=400, payload=_ambiguous_protected_workflow_error_response()),
                HttpJsonResponse(status_code=200, payload=_egress_guard_chat_response()),
                HttpJsonResponse(status_code=200, payload=_tool_argument_canary_chat_response()),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="escalate",
                        detector_names=("encoded_canary", "nimbus", "provider_egress_guard"),
                        dp_honey_status="active",
                        canary_count=1,
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="escalate",
                        detector_names=("text_canary", "nimbus", "provider_egress_guard"),
                        dp_honey_status="active",
                        canary_count=1,
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_partial_nimbus_chat_response(
                        final_action=partial_final_action,
                        nimbus_action=partial_nimbus_action,
                        budget_fraction=0.4,
                        block_threshold=partial_block_threshold,
                    ),
                ),
            ),
            ("GET", f"{base_url}/audit/recent"): (
                HttpJsonResponse(
                    status_code=200,
                    payload={"events": [{"trace_id": "smoke-slot-leak-trace"}]},
                ),
            ),
            ("GET", f"{base_url}/audit/explain?trace_id=smoke-egress-guard-trace"): (
                HttpJsonResponse(status_code=200, payload=_audit_explain_response()),
            ),
        }
    )


def _capabilities_response(include_seed_route: bool) -> dict[str, JsonValue]:
    routes: list[JsonValue] = [
        {"method": "GET", "path": "/health"},
        {"method": "GET", "path": "/ready"},
        {"method": "GET", "path": "/aegis/capabilities"},
        {"method": "POST", "path": "/v1/chat/completions"},
        {"method": "GET", "path": "/audit/recent"},
        {"method": "GET", "path": "/audit/explain"},
        {"method": "POST", "path": "/test/reset"},
    ]
    test_controls: dict[str, JsonValue] = {}
    if include_seed_route:
        routes.append({"method": "POST", "path": "/test/seed-canary"})
        test_controls = {
            "seed_canary": {
                "enabled": True,
                "route": "/test/seed-canary",
                "schema_version": "aegis.test_seed_canary/v1",
                "supported_credential_types": ["github_pat"],
            }
        }
    return {
        "schema_version": "aegis.proxy_capabilities/v1",
        "provider": {"name": "mock", "mock_controls_enabled": True},
        "nimbus": {
            "critic_version": "canary-v0",
            "budget_bits": 1.0,
            "max_turns": 20,
            "thresholds": {"warn": 0.3, "sanitize": 0.6, "block": 0.9},
            "critic": {
                "exact_match_leakage_bits": 1.0,
                "encoded_match_leakage_bits": 1.0,
                "partial_match_leakage_bits": 0.8,
                "partial_match_threshold": 0.4,
                "confidence": 0.8,
            },
        },
        "cift": {
            "capability_mode": "black_box",
            "detectors": ["activation_unavailable"],
            "turn_annotator_count": 0,
        },
        "routes": routes,
        "mock_response_modes": ["default", "leak_first_honeytoken", "base64_first_honeytoken"],
        "test_controls": test_controls,
    }


def _real_provider_capabilities_response() -> dict[str, JsonValue]:
    response = _capabilities_response(include_seed_route=False)
    response["provider"] = {"name": "openai_compatible", "mock_controls_enabled": False}
    response["mock_response_modes"] = []
    return response


def _self_hosted_capabilities_response() -> dict[str, JsonValue]:
    response = _capabilities_response(include_seed_route=True)
    response["cift"] = {
        "capability_mode": "self_hosted_introspection",
        "detectors": ["cift_runtime"],
        "turn_annotator_count": 1,
    }
    return response


def _ready_response() -> dict[str, JsonValue]:
    return {
        "schema_version": "aegis.proxy_readiness/v1",
        "ready": True,
        "status": "ready",
        "cift": {
            "status": "not_required",
            "capability_mode": "black_box",
        },
        "provider": {
            "status": "ready",
            "name": "mock",
            "mock_controls_enabled": True,
        },
        "dp_honey": {"status": "ready"},
        "nimbus": {"status": "deterministic_beta"},
    }


def _real_provider_ready_response() -> dict[str, JsonValue]:
    response = _ready_response()
    response["provider"] = {
        "status": "ready",
        "name": "openai_compatible",
        "mock_controls_enabled": False,
    }
    return response


def _self_hosted_ready_response() -> dict[str, JsonValue]:
    response = _ready_response()
    response["cift"] = {
        "status": "ready",
        "capability_mode": "self_hosted_introspection",
    }
    return response


def _seed_canary_response() -> dict[str, JsonValue]:
    return {
        "schema_version": "aegis.test_seed_canary/v1",
        "status": "seeded",
        "created": True,
        "session_id": "smoke-seeded-session",
        "canary": {
            "canary_id": "hny_smoke-seeded-session_repo_pat",
            "slot_name": "repo_pat",
            "credential_type": "github_pat",
            "sha256": "abc123",
            "source": "test_seed_canary",
            "metadata": {"slot_name": "repo_pat"},
        },
        "mock_response_modes": ["leak_first_honeytoken"],
    }


def _ambiguous_protected_workflow_error_response() -> dict[str, JsonValue]:
    return {
        "error": {
            "schema_version": "aegis.proxy_error/v1",
            "code": "invalid_request",
            "message": (
                "protected_workflow=true requires at least one credential slot declaration "
                "or deterministic credential reference."
            ),
            "details": {
                "credential_slot_status": "ambiguous_protected_workflow",
                "protected_workflow": True,
                "fail_closed": True,
                "credential_needed_count": 0,
                "honeytoken_substituted_count": 0,
                "real_secret_present_count": 0,
                "accepted_detection_sources": [
                    "metadata.credential_slots",
                    "message_credential_placeholder",
                    "message_env_field",
                    "tool_call_credential_placeholder",
                    "tool_schema",
                    "tool_call_secret_like_field",
                ],
            },
        }
    }


def _chat_response(
    final_action: str,
    detector_names: tuple[str, ...],
    dp_honey_status: str,
    canary_count: int,
) -> dict[str, JsonValue]:
    return {
        "id": "chatcmpl-smoke",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "aegis": {
            "trace_id": "smoke-trace",
            "runtime_trace": _runtime_trace(dp_honey_status=dp_honey_status, canary_count=canary_count),
            "policy_decision": {
                "final_action": final_action,
                "reason": "test",
                "triggered_detectors": [],
                "risk_score": 0.0,
                "sanitized_output": None,
            },
            "detector_results": [
                {
                    "detector_name": detector_name,
                    "component": "text_canary",
                    "score": 0.0,
                    "confidence": 1.0,
                    "recommended_action": "allow",
                    "capability_required": None,
                    "capability_status": "active",
                    "evidence": {},
                    "latency_ms": 0.0,
                }
                for detector_name in detector_names
            ],
        },
    }


def _partial_nimbus_chat_response(
    final_action: str,
    nimbus_action: str,
    budget_fraction: float,
    block_threshold: float,
) -> dict[str, JsonValue]:
    return {
        "id": "chatcmpl-smoke-partial",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "partial"}, "finish_reason": "stop"}],
        "aegis": {
            "trace_id": "smoke-partial-trace",
            "runtime_trace": _runtime_trace(dp_honey_status="active", canary_count=1),
            "policy_decision": {
                "final_action": final_action,
                "reason": "test",
                "triggered_detectors": ["nimbus"],
                "risk_score": budget_fraction,
                "sanitized_output": None,
            },
            "detector_results": [
                {
                    "detector_name": "nimbus",
                    "component": "nimbus",
                    "score": budget_fraction,
                    "confidence": 0.8,
                    "recommended_action": nimbus_action,
                    "capability_required": None,
                    "capability_status": "active",
                    "evidence": {
                        "budget_fraction": budget_fraction,
                        "block_threshold": block_threshold,
                    },
                    "latency_ms": 0.0,
                }
            ],
        },
    }


def _cift_block_chat_response() -> dict[str, JsonValue]:
    return {
        "id": "chatcmpl-smoke-cift-block",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        "aegis": {
            "trace_id": "smoke-cift-block-trace",
            "runtime_trace": _cift_block_runtime_trace(),
            "policy_decision": {
                "final_action": "block",
                "reason": "test",
                "triggered_detectors": ["cift_runtime"],
                "risk_score": 1.0,
                "sanitized_output": None,
            },
            "detector_results": [
                {
                    "detector_name": "cift_runtime",
                    "component": "cift",
                    "score": 0.99,
                    "confidence": 0.99,
                    "recommended_action": "block",
                    "capability_required": "self_hosted_introspection",
                    "capability_status": "active",
                    "evidence": {
                        "predicted_label": "exfiltration_intent",
                        "positive_label": "exfiltration_intent",
                    },
                    "latency_ms": 0.0,
                },
                {
                    "detector_name": "provider_egress_guard",
                    "component": "tool_scanner",
                    "score": 0.0,
                    "confidence": 1.0,
                    "recommended_action": "allow",
                    "capability_required": None,
                    "capability_status": "active",
                    "evidence": {},
                    "latency_ms": 0.0,
                },
            ],
        },
    }


def _egress_guard_chat_response() -> dict[str, JsonValue]:
    return {
        "id": "chatcmpl-smoke-egress-guard",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        "aegis": {
            "trace_id": "smoke-egress-guard-trace",
            "runtime_trace": _egress_guard_runtime_trace(),
            "policy_decision": {
                "final_action": "block",
                "reason": "test",
                "triggered_detectors": ["provider_egress_guard"],
                "risk_score": 1.0,
                "sanitized_output": None,
            },
            "detector_results": [
                {
                    "detector_name": "activation_unavailable",
                    "component": "cift",
                    "score": 0.0,
                    "confidence": 0.0,
                    "recommended_action": "allow",
                    "capability_required": None,
                    "capability_status": "unavailable",
                    "evidence": {},
                    "latency_ms": 0.0,
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
                        "matches": [
                            {
                                "kind": "raw_credential",
                                "tool_call_name": "external_ticket",
                                "argument_path": "arguments.body",
                            }
                        ],
                    },
                    "latency_ms": 0.0,
                },
            ],
        },
    }


def _tool_argument_canary_chat_response() -> dict[str, JsonValue]:
    return {
        "id": "chatcmpl-smoke-tool-canary",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        "aegis": {
            "trace_id": "smoke-tool-canary-trace",
            "runtime_trace": _tool_argument_canary_runtime_trace(),
            "policy_decision": {
                "final_action": "escalate",
                "reason": "test",
                "triggered_detectors": ["tool_call_canary"],
                "risk_score": 1.0,
                "sanitized_output": "",
            },
            "detector_results": [
                {
                    "detector_name": "activation_unavailable",
                    "component": "cift",
                    "score": 0.0,
                    "confidence": 0.0,
                    "recommended_action": "allow",
                    "capability_required": None,
                    "capability_status": "unavailable",
                    "evidence": {},
                    "latency_ms": 0.0,
                },
                {
                    "detector_name": "tool_call_canary",
                    "component": "tool_scanner",
                    "score": 1.0,
                    "confidence": 1.0,
                    "recommended_action": "escalate",
                    "capability_required": None,
                    "capability_status": "active",
                    "evidence": {
                        "reason": "registered_canary_tool_egress_detected",
                        "match_count": 1,
                        "matches": [
                            {
                                "tool_name": "external_ticket",
                                "argument_path": "$.body",
                                "credential_type": "github_pat",
                                "sha256": "a" * 64,
                            }
                        ],
                    },
                    "latency_ms": 0.0,
                },
                {
                    "detector_name": "nimbus_tool_egress",
                    "component": "nimbus",
                    "score": 1.0,
                    "confidence": 0.8,
                    "recommended_action": "block",
                    "capability_required": None,
                    "capability_status": "active",
                    "evidence": {
                        "reason": "nimbus_tool_argument_leakage_pre_dispatch_block",
                        "turn_estimated_leakage_bits": 1.0,
                        "cumulative_estimated_leakage_bits": 1.0,
                        "budget_fraction": 1.0,
                    },
                    "latency_ms": 0.0,
                },
                {
                    "detector_name": "provider_egress_guard",
                    "component": "tool_scanner",
                    "score": 0.0,
                    "confidence": 1.0,
                    "recommended_action": "allow",
                    "capability_required": None,
                    "capability_status": "active",
                    "evidence": {"reason": "no_blocked_sensitive_egress_detected"},
                    "latency_ms": 0.0,
                },
            ],
        },
    }


def _audit_explain_response() -> dict[str, JsonValue]:
    return {
        "schema_version": "aegis.audit_explain/v1",
        "trace_id": "smoke-egress-guard-trace",
        "session_id": "smoke-session",
        "turn_index": 2,
        "created_at": "2026-06-25T00:00:00Z",
        "latency_ms": 1.0,
        "policy_mode": "severity",
        "stage_timeline": _egress_guard_runtime_trace()["stages"],
        "detectors": [],
        "artifacts": {},
        "policy_decision": {
            "final_action": "block",
            "reason": "test",
            "risk_score": 1.0,
            "triggered_detectors": ["provider_egress_guard"],
        },
    }


def _runtime_trace(dp_honey_status: str, canary_count: int) -> dict[str, JsonValue]:
    dp_honey_stage: dict[str, JsonValue] = {
        "stage": "dp_honey",
        "status": dp_honey_status,
        "canary_count": canary_count,
    }
    if dp_honey_status == "active":
        dp_honey_stage["credential_slot_status"] = "honeytoken_substituted"
    return {
        "schema_version": "aegis.runtime_trace/v1",
        "stages": [
            {"stage": "normalize", "status": "ok"},
            dp_honey_stage,
            {"stage": "cift", "status": "unavailable", "detectors": ["activation_unavailable"]},
            {"stage": "provider_egress_guard", "status": "active", "detectors": ["provider_egress_guard"]},
            {"stage": "provider", "status": "completed", "provider": "mock", "model_id": "mock-model"},
            {"stage": "canary", "status": "active", "detectors": ["encoded_canary"]},
            {"stage": "nimbus", "status": "active", "detectors": ["nimbus"]},
            {"stage": "policy", "status": "decided", "final_action": "allow"},
            {"stage": "audit", "status": "written"},
        ],
    }


def _egress_guard_runtime_trace() -> dict[str, JsonValue]:
    return {
        "schema_version": "aegis.runtime_trace/v1",
        "stages": [
            {"stage": "normalize", "status": "ok"},
            {"stage": "dp_honey", "status": "not_configured", "canary_count": 0},
            {"stage": "cift", "status": "unavailable", "detectors": ["activation_unavailable"]},
            {"stage": "provider_egress_guard", "status": "active", "detectors": ["provider_egress_guard"]},
            {
                "stage": "provider",
                "status": "skipped",
                "provider": "skipped",
                "model_id": "mock-model",
                "reason": "pre_generation_policy_block",
            },
            {"stage": "canary", "status": "not_configured", "detectors": []},
            {"stage": "nimbus", "status": "not_configured", "detectors": []},
            {"stage": "policy", "status": "decided", "final_action": "block"},
            {"stage": "audit", "status": "written"},
        ],
    }


def _tool_argument_canary_runtime_trace() -> dict[str, JsonValue]:
    return {
        "schema_version": "aegis.runtime_trace/v1",
        "stages": [
            {"stage": "normalize", "status": "ok"},
            {
                "stage": "dp_honey",
                "status": "active",
                "canary_count": 1,
                "credential_slot_status": "honeytoken_substituted",
            },
            {"stage": "cift", "status": "unavailable", "detectors": ["activation_unavailable"]},
            {
                "stage": "provider_egress_guard",
                "status": "active",
                "detectors": ["tool_call_canary", "provider_egress_guard"],
            },
            {
                "stage": "provider",
                "status": "skipped",
                "provider": "skipped",
                "model_id": "mock-model",
                "reason": "pre_generation_policy_block",
            },
            {"stage": "canary", "status": "not_configured", "detectors": []},
            {"stage": "nimbus", "status": "active", "detectors": ["nimbus_tool_egress"]},
            {"stage": "policy", "status": "decided", "final_action": "escalate"},
            {"stage": "audit", "status": "written"},
        ],
    }


def _cift_block_runtime_trace() -> dict[str, JsonValue]:
    return {
        "schema_version": "aegis.runtime_trace/v1",
        "stages": [
            {"stage": "normalize", "status": "ok"},
            {
                "stage": "dp_honey",
                "status": "active",
                "canary_count": 1,
                "credential_slot_status": "honeytoken_substituted",
            },
            {"stage": "cift", "status": "active", "detectors": ["cift_runtime"]},
            {"stage": "provider_egress_guard", "status": "active", "detectors": ["provider_egress_guard"]},
            {
                "stage": "provider",
                "status": "skipped",
                "provider": "skipped",
                "model_id": "mock-model",
                "reason": "pre_generation_policy_block",
            },
            {"stage": "canary", "status": "not_configured", "detectors": []},
            {"stage": "nimbus", "status": "not_configured", "detectors": []},
            {"stage": "policy", "status": "decided", "final_action": "block"},
            {"stage": "audit", "status": "written"},
        ],
    }
