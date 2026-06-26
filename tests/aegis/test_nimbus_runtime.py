"""Runtime integration tests for NIMBUS."""

from aegis.core.contracts import (
    Action,
    CapabilityMode,
    Message,
    ModelInfo,
    NormalizedTurn,
)
from aegis.core.orchestrator import ModelResponse
from aegis.detectors.nimbus import (
    BaselineNimbusCritic,
    InMemoryNimbusStateStore,
    NimbusConfig,
    NimbusDetector,
)
from aegis.proxy.mock_app import create_default_proxy


def test_default_proxy_emits_nimbus_result():
    proxy = create_default_proxy()
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
        "metadata": {
            "credential_slots": [{"slot_name": "repo_pat", "credential_type": "github_pat"}],
            "protected_workflow": True,
        },
    }
    status, response = proxy.handle("POST", "/v1/chat/completions", body)
    assert status == 200

    detector_results = response["aegis"]["detector_results"]
    nimbus_results = [r for r in detector_results if r["detector_name"] == "nimbus"]
    assert len(nimbus_results) >= 1
    assert any(r["capability_status"] == "active" for r in nimbus_results)
    active_nimbus = next(r for r in nimbus_results if r["capability_status"] == "active")
    assert active_nimbus["evidence"]["paper_faithful_learned_critic"] is False
    assert active_nimbus["evidence"]["promotion_status"] == "deterministic_canary_beta"


def test_nimbus_unavailable_without_secret_handle():
    proxy = create_default_proxy()
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
    }
    status, response = proxy.handle("POST", "/v1/chat/completions", body)
    assert status == 200

    detector_results = response["aegis"]["detector_results"]
    nimbus_results = [r for r in detector_results if r["detector_name"] == "nimbus"]
    assert any(r["capability_status"] == "unavailable" for r in nimbus_results)


def test_nimbus_rejects_client_supplied_secret_context_handle():
    proxy = create_default_proxy()
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
        "metadata": {"secret_context_handle": "secret-xyz"},
    }
    status, response = proxy.handle("POST", "/v1/chat/completions", body)
    assert status == 400
    assert response["error"]["code"] == "invalid_request"
    assert "secret_context_handle" in response["error"]["message"]


def test_nimbus_active_with_runtime_owned_canary_context():
    proxy = create_default_proxy()
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
        "metadata": {
            "credential_slots": [{"slot_name": "repo_pat", "credential_type": "github_pat"}],
            "protected_workflow": True,
        },
    }
    status, response = proxy.handle("POST", "/v1/chat/completions", body)
    assert status == 200

    # Response-level check
    detector_results = response["aegis"]["detector_results"]
    nimbus_results = [r for r in detector_results if r["detector_name"] == "nimbus"]
    assert any(r["capability_status"] == "active" for r in nimbus_results)

    # Audit-level check (nested detector_results)
    audit_status, audit_payload = proxy.handle("GET", "/audit/recent", {})
    assert audit_status == 200

    audit_events = audit_payload["events"]
    assert len(audit_events) >= 1

    # Find an audit event that contains NIMBUS results
    nimbus_audit_results = []
    for event in audit_events:
        detector_results = event.get("detector_results")
        if isinstance(detector_results, list):
            for result in detector_results:
                if isinstance(result, dict) and result.get("detector_name") == "nimbus":
                    nimbus_audit_results.append(result)

    assert len(nimbus_audit_results) >= 1
    assert any(r.get("capability_status") == "active" for r in nimbus_audit_results)

    # Ensure the actual secret handle is not leaked in evidence
    for result in nimbus_audit_results:
        assert "hny_" not in str(result.get("evidence", {}))


def test_nimbus_multi_turn_escalation():
    """Test that NIMBUS correctly accumulates leakage across multiple turns
    and escalates advisory actions (warn -> sanitize -> block)."""
    # Custom small-budget configuration for focused test
    config = NimbusConfig(
        budget_bits=1.0,
        warn_threshold=0.3,
        sanitize_threshold=0.5,
        block_threshold=0.7,
        max_turns=20,
        critic_version="test-escalation-v0",
    )

    # Higher fixed score so we reach block in a small number of turns
    critic = BaselineNimbusCritic(fixed_estimated_leakage_bits=0.3, fixed_confidence=0.7)
    store = InMemoryNimbusStateStore(max_turns=20)
    detector = NimbusDetector(config, critic, store)

    session_id = "multi-turn-session"
    secret_handle = "secret-multi-turn"

    actions = []
    cumulatives = []

    for turn_idx in range(6):  # Enough turns to hit block
        model = ModelInfo(provider="mock", model_id="test", revision="v0", selected_device=None)

        turn = NormalizedTurn(
            trace_id=f"t{turn_idx}",
            session_id=session_id,
            turn_index=turn_idx,
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            model=model,
            messages=(Message(role="user", content="test"),),
            tool_calls=(),
            sensitive_spans=(),
            metadata={"secret_context_handle": secret_handle},
        )

        response = ModelResponse(output_text="response", metadata={})

        result = detector.evaluate(turn, response)

        actions.append(result.recommended_action)
        cumulatives.append(result.evidence["cumulative_estimated_leakage_bits"])

        # Secret handle must never appear in evidence
        assert secret_handle not in str(result.evidence)
        assert result.evidence["paper_faithful_learned_critic"] is False
        assert result.evidence["promotion_status"] == "demo_only_baseline"

    # Cumulative should be strictly increasing
    assert all(cumulatives[i] < cumulatives[i + 1] for i in range(len(cumulatives) - 1))

    # Action escalation
    assert Action.WARN in actions
    assert Action.SANITIZE in actions
    assert Action.BLOCK in actions

    # Final action should be BLOCK
    assert actions[-1] == Action.BLOCK
