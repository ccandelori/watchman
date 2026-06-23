from __future__ import annotations

import pytest

from aegis.core.contracts import JsonValue
from aegis.proxy.smoke import (
    GatewaySmokeConfig,
    GatewaySmokeError,
    HttpJsonResponse,
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


def test_gateway_smoke_parses_explicit_url_and_timeout() -> None:
    config = parse_args(("--url", "http://127.0.0.1:8000/", "--timeout", "2.5"))

    assert config == GatewaySmokeConfig(base_url="http://127.0.0.1:8000", timeout_seconds=2.5)


def test_gateway_smoke_rejects_non_positive_timeout() -> None:
    with pytest.raises(GatewaySmokeError, match="timeout"):
        parse_args(("--url", "http://127.0.0.1:8000", "--timeout", "0"))


def test_gateway_smoke_accepts_healthy_gateway_contract() -> None:
    base_url = "http://gateway.test"
    client = FakeSmokeClient(
        {
            ("GET", f"{base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{base_url}/aegis/capabilities"): (
                HttpJsonResponse(status_code=200, payload=_capabilities_response(include_seed_route=True)),
            ),
            ("POST", f"{base_url}/test/reset"): (
                HttpJsonResponse(status_code=200, payload={"status": "reset"}),
                HttpJsonResponse(status_code=200, payload={"status": "reset"}),
            ),
            ("POST", f"{base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        detector_names=("activation_unavailable",),
                        dp_honey_status="not_configured",
                        canary_count=0,
                    ),
                ),
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
                        dp_honey_status="not_configured",
                        canary_count=0,
                    ),
                ),
            ),
            ("POST", f"{base_url}/test/seed-canary"): (
                HttpJsonResponse(status_code=200, payload=_seed_canary_response()),
            ),
            ("GET", f"{base_url}/audit/recent"): (
                HttpJsonResponse(
                    status_code=200,
                    payload={"events": [{"trace_id": "smoke-seeded-leak-trace"}]},
                ),
            ),
        }
    )

    report = run_gateway_smoke(GatewaySmokeConfig(base_url=base_url, timeout_seconds=1.0), client)

    assert report["status"] == "ok"
    assert client.requests[0] == ("GET", f"{base_url}/health", None)
    assert client.requests[1] == ("GET", f"{base_url}/aegis/capabilities", None)
    assert client.requests[2][0:2] == ("POST", f"{base_url}/test/reset")
    assert client.requests[3][0:2] == ("POST", f"{base_url}/test/reset")
    benign_request = client.requests[4][2]
    leak_request = client.requests[5][2]
    seed_request = client.requests[6][2]
    seeded_leak_request = client.requests[7][2]
    assert isinstance(benign_request, dict)
    assert isinstance(leak_request, dict)
    assert isinstance(seed_request, dict)
    assert isinstance(seeded_leak_request, dict)
    assert benign_request["metadata"] != leak_request["metadata"]
    assert seed_request["session_id"] == seeded_leak_request["metadata"]["session_id"]


def test_gateway_smoke_rejects_unhealthy_health_payload() -> None:
    base_url = "http://gateway.test"
    client = FakeSmokeClient(
        {
            ("GET", f"{base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "not-ok"}),),
        }
    )

    with pytest.raises(GatewaySmokeError, match="status=ok"):
        run_gateway_smoke(GatewaySmokeConfig(base_url=base_url, timeout_seconds=1.0), client)


def test_gateway_smoke_rejects_missing_encoded_canary_result() -> None:
    base_url = "http://gateway.test"
    client = FakeSmokeClient(
        {
            ("GET", f"{base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{base_url}/aegis/capabilities"): (
                HttpJsonResponse(status_code=200, payload=_capabilities_response(include_seed_route=True)),
            ),
            ("POST", f"{base_url}/test/reset"): (
                HttpJsonResponse(status_code=200, payload={"status": "reset"}),
                HttpJsonResponse(status_code=200, payload={"status": "reset"}),
            ),
            ("POST", f"{base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        detector_names=("activation_unavailable",),
                        dp_honey_status="not_configured",
                        canary_count=0,
                    ),
                ),
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
        run_gateway_smoke(GatewaySmokeConfig(base_url=base_url, timeout_seconds=1.0), client)


def _capabilities_response(include_seed_route: bool) -> dict[str, JsonValue]:
    routes: list[JsonValue] = [
        {"method": "GET", "path": "/health"},
        {"method": "GET", "path": "/aegis/capabilities"},
        {"method": "POST", "path": "/v1/chat/completions"},
        {"method": "GET", "path": "/audit/recent"},
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
        "routes": routes,
        "mock_response_modes": ["default", "leak_first_honeytoken", "base64_first_honeytoken"],
        "test_controls": test_controls,
    }


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


def _runtime_trace(dp_honey_status: str, canary_count: int) -> dict[str, JsonValue]:
    return {
        "schema_version": "aegis.runtime_trace/v1",
        "stages": [
            {"stage": "normalize", "status": "ok"},
            {"stage": "dp_honey", "status": dp_honey_status, "canary_count": canary_count},
            {"stage": "cift", "status": "unavailable", "detectors": ["activation_unavailable"]},
            {"stage": "provider_egress_guard", "status": "active", "detectors": ["provider_egress_guard"]},
            {"stage": "provider", "status": "completed", "provider": "mock", "model_id": "mock-model"},
            {"stage": "canary", "status": "active", "detectors": ["encoded_canary"]},
            {"stage": "nimbus", "status": "active", "detectors": ["nimbus"]},
            {"stage": "policy", "status": "decided", "final_action": "allow"},
            {"stage": "audit", "status": "written"},
        ],
    }
