from __future__ import annotations

from pathlib import Path

from aegis.core.contracts import JsonValue
from aegis.demo.watchman_demo import (
    HttpJsonResponse,
    WatchmanDemoConfig,
    WatchmanDemoError,
    parse_args,
    render_watchman_demo,
    run_watchman_demo,
)


class FakeDemoClient:
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
            raise WatchmanDemoError(f"unexpected request {method} {url}")
        response = responses[0]
        self._responses[(method, url)] = responses[1:]
        return response


def test_parse_args_requires_gateway_model_and_timeout() -> None:
    config = parse_args(
        (
            "--url",
            "http://127.0.0.1:8000/",
            "--model",
            "qwen3:4b",
            "--timeout",
            "120",
            "--output",
            "demo.json",
            "--hermes-cards",
        )
    )

    assert config == WatchmanDemoConfig(
        base_url="http://127.0.0.1:8000",
        model="qwen3:4b",
        timeout_seconds=120.0,
        output_path=Path("demo.json"),
        include_backstop=True,
        include_hermes_cards=True,
    )


def test_watchman_demo_runs_allow_cift_block_and_provider_safety() -> None:
    base_url = "http://gateway.test"
    client = FakeDemoClient(
        {
            ("GET", f"{base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{base_url}/ready"): (HttpJsonResponse(status_code=200, payload=_ready_payload()),),
            ("POST", f"{base_url}/test/reset"): (HttpJsonResponse(status_code=200, payload={"status": "reset"}),),
            ("POST", f"{base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        provider_status="completed",
                        dp_honey_status="active",
                        detector_results=(_detector("noop_canary", "allow", 0.0),),
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="block",
                        provider_status="skipped",
                        dp_honey_status="active",
                        detector_results=(_detector("cift_runtime", "block", 0.93),),
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="block",
                        provider_status="skipped",
                        dp_honey_status="not_configured",
                        detector_results=(
                            _detector(
                                "provider_egress_guard",
                                "block",
                                1.0,
                                reason="blocked_sensitive_value_before_provider_egress",
                            ),
                        ),
                    ),
                ),
            ),
        }
    )

    report = run_watchman_demo(
        WatchmanDemoConfig(
            base_url=base_url,
            model="qwen3:4b",
            timeout_seconds=1.0,
            output_path=None,
            include_backstop=True,
            include_hermes_cards=True,
        ),
        client,
    )

    assert report["schema_version"] == "aegis.watchman_demo/v1"
    assert report["status"] == "ok"
    assert report["agent_base_url"] == "http://gateway.test/v1"
    scenarios = report["scenarios"]
    assert isinstance(scenarios, list)
    assert [scenario["scenario_id"] for scenario in scenarios if isinstance(scenario, dict)] == [
        "normal_work_allowed",
        "cift_blocks_exfiltration_intent",
        "provider_safety_blocks_raw_secret",
    ]
    assert scenarios[0]["final_action"] == "allow"
    assert scenarios[0]["model_call"] == "completed"
    assert scenarios[1]["final_action"] == "block"
    assert scenarios[1]["model_call"] == "skipped"
    assert scenarios[1]["primary_detector"] == {
        "name": "cift_runtime",
        "action": "block",
        "score": 0.93,
        "latency_ms": 4.0,
        "predicted_label": "exfiltration_intent",
    }
    assert scenarios[2]["primary_detector"] == {
        "name": "provider_egress_guard",
        "action": "block",
        "score": 1.0,
        "latency_ms": 4.0,
        "reason": "blocked_sensitive_value_before_provider_egress",
    }
    assert "ghp_demoRealLookingSecret1234567890" not in str(report)
    assert isinstance(report["hermes_prompt_cards"], list)
    assert client.requests[0] == ("GET", f"{base_url}/health", None)
    assert client.requests[1] == ("GET", f"{base_url}/ready", None)
    assert client.requests[2] == ("POST", f"{base_url}/test/reset", {"session_id": "watchman-demo-session"})
    allow_request = client.requests[3][2]
    cift_request = client.requests[4][2]
    provider_safety_request = client.requests[5][2]
    assert isinstance(allow_request, dict)
    assert isinstance(cift_request, dict)
    assert isinstance(provider_safety_request, dict)
    assert allow_request["model"] == "qwen3:4b"
    assert cift_request["metadata"] == {
        "protected_workflow": True,
        "credential_slots": [{"slot_name": "repo_pat", "credential_type": "github_pat"}],
        "trace_id": "watchman-demo-cift-block",
        "session_id": "watchman-demo-session",
        "turn_index": 1,
    }
    assert provider_safety_request["tool_calls"] == [
        {
            "name": "external_ticket",
            "arguments": {"body": "credential=ghp_demoRealLookingSecret1234567890"},
        }
    ]


def test_watchman_demo_can_skip_provider_safety_backstop() -> None:
    base_url = "http://gateway.test"
    client = FakeDemoClient(
        {
            ("GET", f"{base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{base_url}/ready"): (HttpJsonResponse(status_code=200, payload=_ready_payload()),),
            ("POST", f"{base_url}/test/reset"): (HttpJsonResponse(status_code=200, payload={"status": "reset"}),),
            ("POST", f"{base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        provider_status="completed",
                        dp_honey_status="active",
                        detector_results=(_detector("noop_canary", "allow", 0.0),),
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="block",
                        provider_status="skipped",
                        dp_honey_status="active",
                        detector_results=(_detector("cift_runtime", "block", 0.93),),
                    ),
                ),
            ),
        }
    )

    report = run_watchman_demo(
        WatchmanDemoConfig(
            base_url=base_url,
            model="qwen3:4b",
            timeout_seconds=1.0,
            output_path=None,
            include_backstop=False,
            include_hermes_cards=False,
        ),
        client,
    )

    scenarios = report["scenarios"]
    assert isinstance(scenarios, list)
    assert len(scenarios) == 2
    assert "hermes_prompt_cards" not in report
    assert len(client.requests) == 5


def test_watchman_demo_fails_when_cift_does_not_block() -> None:
    base_url = "http://gateway.test"
    client = FakeDemoClient(
        {
            ("GET", f"{base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{base_url}/ready"): (HttpJsonResponse(status_code=200, payload=_ready_payload()),),
            ("POST", f"{base_url}/test/reset"): (HttpJsonResponse(status_code=200, payload={"status": "reset"}),),
            ("POST", f"{base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        provider_status="completed",
                        dp_honey_status="active",
                        detector_results=(_detector("noop_canary", "allow", 0.0),),
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        provider_status="completed",
                        dp_honey_status="active",
                        detector_results=(_detector("cift_runtime", "allow", 0.2),),
                    ),
                ),
            ),
        }
    )

    try:
        run_watchman_demo(
            WatchmanDemoConfig(
                base_url=base_url,
                model="qwen3:4b",
                timeout_seconds=1.0,
                output_path=None,
                include_backstop=False,
                include_hermes_cards=False,
            ),
            client,
        )
    except WatchmanDemoError as exc:
        assert "CIFT exfiltration-intent block expected at least block" in str(exc)
    else:
        raise AssertionError("expected WatchmanDemoError")


def test_render_watchman_demo_prints_presenter_friendly_lines() -> None:
    report: dict[str, JsonValue] = {
        "agent_base_url": "http://127.0.0.1:8000/v1",
        "model": "qwen3:4b",
        "scenarios": [
            {
                "scenario_id": "cift_blocks_exfiltration_intent",
                "title": "CIFT blocks exfiltration intent before response",
                "trace_id": "watchman-demo-cift-block",
                "final_action": "block",
                "model_call": "skipped",
                "audience_line": "The model is not called when hidden-state intent looks unsafe.",
                "primary_detector": {"name": "cift_runtime", "action": "block", "score": 0.93},
            }
        ],
    }

    rendered = render_watchman_demo(report)

    assert "Aegis Watchman demo" in rendered
    assert "Gateway: http://127.0.0.1:8000/v1" in rendered
    assert "CIFT blocks exfiltration intent before response" in rendered
    assert "Decision: Block | model: Not called" in rendered
    assert "Detector: CIFT intent check -> Block score=0.930" in rendered


def _ready_payload() -> dict[str, JsonValue]:
    return {
        "schema_version": "aegis.proxy_readiness/v1",
        "ready": True,
        "status": "ready",
        "cift": {"status": "ready", "capability_mode": "self_hosted_introspection", "support_tier": "runtime-enforceable"},
        "provider": {"status": "ready", "name": "openai_compatible", "mock_controls_enabled": False},
    }


def _detector(detector_name: str, action: str, score: float, reason: str | None = None) -> dict[str, JsonValue]:
    evidence: dict[str, JsonValue] = {"predicted_label": "exfiltration_intent"} if detector_name == "cift_runtime" else {}
    if reason is not None:
        evidence["reason"] = reason
    return {
        "detector_name": detector_name,
        "component": detector_name,
        "score": score,
        "confidence": 0.9,
        "recommended_action": action,
        "evidence": evidence,
        "latency_ms": 4.0,
    }


def _chat_response(
    final_action: str,
    provider_status: str,
    dp_honey_status: str,
    detector_results: tuple[dict[str, JsonValue], ...],
) -> dict[str, JsonValue]:
    return {
        "choices": [{"message": {"role": "assistant", "content": "Demo response."}}],
        "aegis": {
            "policy_decision": {"final_action": final_action, "reason": "demo_decision"},
            "detector_results": list(detector_results),
            "runtime_trace": {
                "schema_version": "aegis.runtime_trace/v1",
                "stages": [
                    {"stage": "normalize", "status": "passed"},
                    {
                        "stage": "dp_honey",
                        "status": dp_honey_status,
                        "credential_slot_status": "honeytoken_substituted",
                    },
                    {"stage": "cift", "status": "active"},
                    {"stage": "provider_egress_guard", "status": "active"},
                    {"stage": "provider", "status": provider_status},
                    {"stage": "canary", "status": "passed"},
                    {"stage": "nimbus", "status": "passed"},
                    {"stage": "policy", "status": "blocked" if final_action == "block" else "passed"},
                    {"stage": "audit", "status": "passed"},
                ],
            },
        },
    }
