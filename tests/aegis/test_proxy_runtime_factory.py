from __future__ import annotations

from aegis.audit.memory import InMemoryAuditSink
from aegis.core.contracts import Action, CapabilityMode, Message, ModelInfo
from aegis.core.orchestrator import RuntimeRequest
from aegis.detectors.canary import CanaryRecord, canary_sha256
from aegis.detectors.nimbus import BaselineNimbusCritic, InMemoryNimbusStateStore, NimbusConfig, NimbusDetector
from aegis.providers.mock import MockModelProvider
from aegis.proxy.runtime_factory import ProxyRuntimeFactory, black_box_cift_capability


def test_proxy_runtime_factory_builds_canary_aware_runtime() -> None:
    canary = CanaryRecord(
        canary_id="canary-test-1",
        credential_type="github_pat",
        value="canary_test_value_123",
        sha256=canary_sha256("canary_test_value_123"),
        source="test",
        metadata={"slot_name": "repo_pat", "turn_planted": 0},
    )
    runtime = ProxyRuntimeFactory(
        audit_sink=InMemoryAuditSink(),
        nimbus_detector=_nimbus_detector(),
        cift_capability=black_box_cift_capability(),
        model_provider=MockModelProvider(default_content=canary.value),
    ).build(canary_records=(canary,))

    response = runtime.evaluate_turn(_runtime_request())
    detector_names = tuple(result.detector_name for result in response.detector_results)

    assert detector_names == (
        "activation_unavailable",
        "provider_egress_guard",
        "text_canary",
        "encoded_canary",
        "nimbus",
    )
    assert response.policy_decision.final_action == Action.ESCALATE


def _nimbus_detector() -> NimbusDetector:
    return NimbusDetector(
        config=NimbusConfig(
            budget_bits=1.0,
            warn_threshold=0.3,
            sanitize_threshold=0.6,
            block_threshold=0.9,
            max_turns=20,
            critic_version="baseline-test",
        ),
        critic=BaselineNimbusCritic(fixed_estimated_leakage_bits=0.0, fixed_confidence=0.5),
        state_store=InMemoryNimbusStateStore(max_turns=20),
    )


def _runtime_request() -> RuntimeRequest:
    return RuntimeRequest(
        trace_id="trace-runtime-factory",
        session_id="session-runtime-factory",
        turn_index=0,
        capability_mode=CapabilityMode.BLACK_BOX,
        model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
        messages=(Message(role="user", content="hello"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={},
    )
