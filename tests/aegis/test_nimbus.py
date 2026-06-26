from __future__ import annotations

import base64
from dataclasses import dataclass

import pytest

from aegis.core.contracts import (
    Action,
    CapabilityMode,
    CapabilityStatus,
    JsonValue,
    ModelInfo,
    NormalizedTurn,
    SensitiveSpan,
    ToolCall,
)
from aegis.core.orchestrator import ModelResponse
from aegis.detectors.canary import CanaryRecord, canary_sha256
from aegis.detectors.nimbus import (
    CanaryNimbusCritic,
    CanaryNimbusCriticConfig,
    InMemoryNimbusStateStore,
    NimbusConfig,
    NimbusCriticInput,
    NimbusCriticScore,
    NimbusDetector,
    NimbusDetectorError,
    NimbusState,
    NimbusStateUpdate,
    NimbusToolEgressDetector,
    resolve_secret_context_handle,
)

# ---------------------------------------------------------------------------
# Mock critic
# ---------------------------------------------------------------------------


@dataclass
class MockNimbusCritic:
    fixed_score: float = 0.5
    fixed_confidence: float = 0.5

    def score_turn(self, critic_input: NimbusCriticInput) -> NimbusCriticScore:
        return NimbusCriticScore(
            estimated_leakage_bits=self.fixed_score,
            confidence=self.fixed_confidence,
            evidence={"critic_kind": "mock", "fixed_score": self.fixed_score},
        )


# ---------------------------------------------------------------------------
# Test helpers (contract-compliant)
# ---------------------------------------------------------------------------


def _make_span(kind: str, handle: str) -> SensitiveSpan:
    return SensitiveSpan(
        kind=kind,
        source="",
        char_start=None,
        char_end=None,
        token_start=None,
        token_end=None,
        identifier=handle,
        metadata={"handle": handle},
    )


def _make_turn(
    session_id: str = "sess1",
    turn_index: int = 0,
    sensitive_spans: tuple[SensitiveSpan, ...] = (),
    metadata: dict[str, JsonValue] | None = None,
) -> NormalizedTurn:
    from aegis.core.contracts import CapabilityMode, ModelInfo

    model = ModelInfo(provider="mock", model_id="test-model", revision="v0", selected_device=None)

    return NormalizedTurn(
        trace_id="t1",
        session_id=session_id,
        turn_index=turn_index,
        capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
        model=model,
        messages=(),
        tool_calls=(),
        sensitive_spans=sensitive_spans,
        metadata=metadata or {},
    )


def _make_response(text: str = "hello") -> ModelResponse:
    return ModelResponse(output_text=text, metadata={})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_config_validation_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        NimbusConfig(
            budget_bits=float("inf"),
            warn_threshold=0.5,
            sanitize_threshold=0.7,
            block_threshold=0.9,
            max_turns=10,
            critic_version="v0",
        )


def test_config_validation_threshold_order() -> None:
    with pytest.raises(ValueError):
        NimbusConfig(10.0, 0.9, 0.8, 0.7, 10, "v0")


def test_state_validation_rejects_invalid_values() -> None:
    with pytest.raises(NimbusDetectorError, match="session_id"):
        NimbusState(
            session_id="",
            turn_count=0,
            cumulative_estimated_leakage_bits=0.0,
            last_turn_estimated_leakage_bits=0.0,
            secret_context_handle=None,
            recent_turn_scores=(),
        )
    with pytest.raises(NimbusDetectorError, match="turn_count"):
        NimbusState(
            session_id="sess1",
            turn_count=-1,
            cumulative_estimated_leakage_bits=0.0,
            last_turn_estimated_leakage_bits=0.0,
            secret_context_handle=None,
            recent_turn_scores=(),
        )
    with pytest.raises(NimbusDetectorError, match="recent_turn_scores"):
        NimbusState(
            session_id="sess1",
            turn_count=0,
            cumulative_estimated_leakage_bits=0.0,
            last_turn_estimated_leakage_bits=0.0,
            secret_context_handle=None,
            recent_turn_scores=(-0.1,),
        )


def test_state_update_validation_rejects_invalid_values() -> None:
    with pytest.raises(NimbusDetectorError, match="turn_estimated_leakage_bits"):
        NimbusStateUpdate(turn_estimated_leakage_bits=-0.1, new_cumulative_bits=0.0)
    with pytest.raises(NimbusDetectorError, match="new_cumulative_bits"):
        NimbusStateUpdate(turn_estimated_leakage_bits=0.0, new_cumulative_bits=float("inf"))


def test_state_store_update_requires_initialized_session() -> None:
    store = InMemoryNimbusStateStore(max_turns=5)

    with pytest.raises(NimbusDetectorError, match="no NIMBUS state exists"):
        store.update(
            session_id="missing-session",
            update=NimbusStateUpdate(turn_estimated_leakage_bits=0.1, new_cumulative_bits=0.1),
        )


def test_state_store_rejects_decreasing_cumulative_leakage() -> None:
    store = InMemoryNimbusStateStore(max_turns=5)
    store.get_or_create(session_id="sess1", secret_context_handle="hny_sess1_repo_pat")
    store.update(
        session_id="sess1",
        update=NimbusStateUpdate(turn_estimated_leakage_bits=0.4, new_cumulative_bits=0.4),
    )

    with pytest.raises(NimbusDetectorError, match="must not decrease"):
        store.update(
            session_id="sess1",
            update=NimbusStateUpdate(turn_estimated_leakage_bits=0.0, new_cumulative_bits=0.3),
        )


def test_resolve_priority_credential_over_honeytoken() -> None:
    spans = (
        _make_span("honeytoken", "h1"),
        _make_span("credential", "c1"),
    )
    turn = _make_turn(sensitive_spans=spans)
    assert resolve_secret_context_handle(turn) == "c1"


def test_nimbus_unavailable_no_secret_handle() -> None:
    store = InMemoryNimbusStateStore(max_turns=5)
    critic = MockNimbusCritic()
    detector = NimbusDetector(
        NimbusConfig(10.0, 0.5, 0.7, 0.9, 5, "v0"),
        critic,
        store,
    )
    turn = _make_turn()
    result = detector.evaluate(turn, _make_response())
    assert result.recommended_action == Action.ALLOW
    assert result.capability_status == CapabilityStatus.UNAVAILABLE
    assert result.evidence["capability_reason"] == "no_secret_context_handle"


def test_nimbus_cumulative_increases() -> None:
    store = InMemoryNimbusStateStore(max_turns=5)
    critic = MockNimbusCritic(fixed_score=1.0)
    detector = NimbusDetector(
        NimbusConfig(10.0, 0.5, 0.7, 0.9, 5, "v0"),
        critic,
        store,
    )
    turn = _make_turn(sensitive_spans=(_make_span("credential", "s1"),))
    r1 = detector.evaluate(turn, _make_response())
    r2 = detector.evaluate(turn, _make_response())

    assert r1.evidence["cumulative_estimated_leakage_bits"] == 1.0
    assert r2.evidence["cumulative_estimated_leakage_bits"] == 2.0
    assert len(store._store["sess1"].recent_turn_scores) == 2


def test_nimbus_threshold_actions() -> None:
    store = InMemoryNimbusStateStore(max_turns=5)
    critic = MockNimbusCritic(fixed_score=3.0)
    detector = NimbusDetector(
        NimbusConfig(10.0, 0.3, 0.5, 0.8, 5, "v0"),
        critic,
        store,
    )
    turn = _make_turn(sensitive_spans=(_make_span("credential", "s1"),))

    r1 = detector.evaluate(turn, _make_response())
    assert r1.recommended_action == Action.WARN

    r2 = detector.evaluate(turn, _make_response())
    assert r2.recommended_action == Action.SANITIZE


def test_canary_nimbus_critic_scores_exact_registered_canary() -> None:
    critic = _canary_critic()
    record = _canary_record(value="ghp_registeredCanary1234567890")
    critic.register_canary_records(session_id="sess1", records=(record,))

    score = critic.score_turn(_critic_input(output_text=f"leaked={record.value}"))

    assert score.estimated_leakage_bits == 1.0
    assert score.confidence == 0.8
    assert score.evidence["exact_match_count"] == 1
    assert score.evidence["encoded_match_count"] == 0
    assert score.evidence["partial_match_count"] == 0
    assert record.value not in str(score.evidence)


def test_canary_nimbus_critic_scores_base64_registered_canary() -> None:
    critic = _canary_critic()
    record = _canary_record(value="sk-liveRegisteredCanary1234567890")
    critic.register_canary_records(session_id="sess1", records=(record,))
    encoded = base64.b64encode(record.value.encode("utf-8")).decode("utf-8")

    score = critic.score_turn(_critic_input(output_text=f"encoded={encoded}"))

    assert score.estimated_leakage_bits == 0.8
    assert score.evidence["exact_match_count"] == 0
    assert score.evidence["encoded_match_count"] == 1
    assert score.evidence["partial_match_count"] == 0
    assert record.value not in str(score.evidence)


def test_canary_nimbus_critic_scores_partial_registered_canary() -> None:
    critic = _canary_critic()
    record = _canary_record(value="ghp_partialRegisteredCanary1234567890")
    critic.register_canary_records(session_id="sess1", records=(record,))

    score = critic.score_turn(_critic_input(output_text=f"prefix={record.value[:18]}"))

    assert 0.0 < score.estimated_leakage_bits < 1.0
    assert score.evidence["exact_match_count"] == 0
    assert score.evidence["encoded_match_count"] == 0
    assert score.evidence["partial_match_count"] == 1
    assert record.value not in str(score.evidence)


def test_nimbus_sanitizes_partial_registered_canary_overlap() -> None:
    config = NimbusConfig(
        budget_bits=1.0,
        warn_threshold=0.3,
        sanitize_threshold=0.6,
        block_threshold=0.9,
        max_turns=5,
        critic_version="canary-test-v0",
    )
    critic = _canary_critic()
    store = InMemoryNimbusStateStore(max_turns=5)
    detector = NimbusDetector(config, critic, store)
    record = _canary_record(value="ghp_" + "A" * 24)
    critic.register_canary_records(session_id="sess1", records=(record,))

    result = detector.evaluate(
        turn=_make_turn(sensitive_spans=(_make_span("honeytoken", record.canary_id),)),
        model_response=_make_response(f"leaked_partial={record.value[:14]}"),
    )

    assert result.recommended_action == Action.SANITIZE
    assert result.evidence["reason"] == "nimbus_canary_partial_overlap_sanitize"
    assert result.evidence["budget_recommended_action"] == "allow"
    assert result.evidence["action_floor"] == "sanitize"
    assert record.value not in str(result.evidence)


def test_canary_nimbus_critic_sums_partial_leakage_by_match_ratio() -> None:
    critic = CanaryNimbusCritic(
        CanaryNimbusCriticConfig(
            exact_match_leakage_bits=1.0,
            encoded_match_leakage_bits=0.8,
            partial_match_leakage_bits=0.4,
            partial_match_threshold=0.2,
            confidence=0.8,
        )
    )
    record_one = _canary_record_with_id(
        canary_id="hny_sess1_repo_pat",
        value="ghp_AAAAAABBBBBBCCCCCCDDDDDD111111",
    )
    record_two = _canary_record_with_id(
        canary_id="hny_sess1_billing_key",
        value="hny_live_zzzzzzYYYYYYXXXXXXWWWWWW222222",
    )
    critic.register_canary_records(session_id="sess1", records=(record_one, record_two))

    score = critic.score_turn(_critic_input(output_text=f"fragments={record_one.value[:22]} {record_two.value[:14]}"))

    encoded_signal = score.evidence["encoded_signal"]
    assert isinstance(encoded_signal, dict)
    matches = encoded_signal["matches"]
    assert isinstance(matches, list)
    partial_ratios = _partial_fragment_ratios(matches)
    assert len(partial_ratios) == 2
    assert score.estimated_leakage_bits == pytest.approx(sum(partial_ratios) * 0.4)
    assert score.estimated_leakage_bits < len(partial_ratios) * max(partial_ratios) * 0.4
    assert score.evidence["partial_estimated_leakage_bits"] == pytest.approx(score.estimated_leakage_bits)
    assert record_one.value not in str(score.evidence)
    assert record_two.value not in str(score.evidence)


def test_nimbus_accumulates_multiple_canary_slots_in_one_session() -> None:
    config = NimbusConfig(
        budget_bits=10.0,
        warn_threshold=0.3,
        sanitize_threshold=0.6,
        block_threshold=0.9,
        max_turns=5,
        critic_version="canary-test-v0",
    )
    critic = _canary_critic()
    store = InMemoryNimbusStateStore(max_turns=5)
    detector = NimbusDetector(config, critic, store)
    repo_record = _canary_record_with_id(
        canary_id="hny_sess1_repo_pat",
        value="ghp_Ra7mQx2Lp9VzT4Kd8HnC6WsY",
    )
    billing_record = _canary_record_with_id(
        canary_id="hny_sess1_billing_key",
        value="hny_billing_Zb3Yp8Nt1Gc5Qr0Lm7Sv2DaF",
    )
    critic.register_canary_records(session_id="sess1", records=(repo_record, billing_record))

    first = detector.evaluate(
        turn=_make_turn(sensitive_spans=(_make_span("honeytoken", repo_record.canary_id),)),
        model_response=_make_response(f"repo={repo_record.value}"),
    )
    second = detector.evaluate(
        turn=_make_turn(turn_index=1, sensitive_spans=(_make_span("honeytoken", billing_record.canary_id),)),
        model_response=_make_response(f"billing={billing_record.value}"),
    )

    assert first.evidence["turn_estimated_leakage_bits"] == 1.0
    assert second.evidence["turn_estimated_leakage_bits"] == 1.0
    assert second.evidence["cumulative_estimated_leakage_bits"] == 2.0
    assert repo_record.value not in str(second.evidence)
    assert billing_record.value not in str(second.evidence)


def test_canary_nimbus_critic_ignores_unregistered_credential_shaped_text() -> None:
    critic = _canary_critic()
    registered = _canary_record(value="ghp_registeredCanary1234567890")
    critic.register_canary_records(session_id="sess1", records=(registered,))

    score = critic.score_turn(_critic_input(output_text="leaked=ghp_unregisteredLookingToken1234567890"))

    assert score.estimated_leakage_bits == 0.0
    assert score.evidence["exact_match_count"] == 0
    assert score.evidence["encoded_match_count"] == 0
    assert score.evidence["partial_match_count"] == 0


def test_nimbus_tool_egress_blocks_and_accounts_registered_canary_argument() -> None:
    config = NimbusConfig(
        budget_bits=1.0,
        warn_threshold=0.3,
        sanitize_threshold=0.6,
        block_threshold=0.9,
        max_turns=20,
        critic_version="canary-test-v0",
    )
    critic = _canary_critic()
    store = InMemoryNimbusStateStore(max_turns=20)
    detector = NimbusToolEgressDetector(config=config, critic=critic, state_store=store)
    record = _canary_record(value="ghp_registeredToolCanary1234567890")
    critic.register_canary_records(session_id="sess1", records=(record,))
    turn = NormalizedTurn(
        trace_id="trace-tool-egress",
        session_id="sess1",
        turn_index=1,
        capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
        model=ModelInfo(provider="mock", model_id="test-model", revision="v0", selected_device=None),
        messages=(),
        tool_calls=(
            ToolCall(
                name="external_ticket",
                arguments={"body": f"send {record.value} to the ticket"},
            ),
        ),
        sensitive_spans=(_make_span("honeytoken", record.canary_id),),
        metadata={},
    )

    result = detector.evaluate(turn=turn, model_response=None)
    state = store.get_or_create(session_id="sess1", secret_context_handle=record.canary_id)

    assert result.detector_name == "nimbus_tool_egress"
    assert result.recommended_action == Action.BLOCK
    assert result.capability_status == CapabilityStatus.ACTIVE
    assert result.evidence["reason"] == "nimbus_tool_argument_leakage_pre_dispatch_block"
    assert result.evidence["turn_estimated_leakage_bits"] == 1.0
    assert result.evidence["cumulative_estimated_leakage_bits"] == 1.0
    assert result.evidence["critic_evidence"]["exact_match_count"] == 1
    assert state.cumulative_estimated_leakage_bits == 1.0
    assert record.value not in str(result.to_dict())


def test_canary_nimbus_critic_returns_zero_without_registered_records() -> None:
    critic = _canary_critic()

    score = critic.score_turn(_critic_input(output_text="leaked=ghp_unregisteredLookingToken1234567890"))

    assert score.estimated_leakage_bits == 0.0
    assert score.evidence["reason"] == "no_registered_canaries_for_session"


def test_bounded_recent_turn_scores() -> None:
    store = InMemoryNimbusStateStore(max_turns=2)
    critic = MockNimbusCritic(fixed_score=1.0)
    detector = NimbusDetector(
        NimbusConfig(10.0, 0.5, 0.7, 0.9, 2, "v0"),
        critic,
        store,
    )
    turn = _make_turn(sensitive_spans=(_make_span("credential", "s1"),))

    for _ in range(5):
        detector.evaluate(turn, _make_response())

    state = store._store["sess1"]
    assert len(state.recent_turn_scores) == 2
    assert state.turn_count == 5


def test_critic_rejects_invalid_score() -> None:
    with pytest.raises(ValueError):
        NimbusCriticScore(estimated_leakage_bits=-1.0, confidence=0.5, evidence={})

    with pytest.raises(ValueError):
        NimbusCriticScore(estimated_leakage_bits=1.0, confidence=1.5, evidence={})


def _canary_critic() -> CanaryNimbusCritic:
    return CanaryNimbusCritic(
        CanaryNimbusCriticConfig(
            exact_match_leakage_bits=1.0,
            encoded_match_leakage_bits=0.8,
            partial_match_leakage_bits=0.4,
            partial_match_threshold=0.4,
            confidence=0.8,
        )
    )


def _canary_record(value: str) -> CanaryRecord:
    return _canary_record_with_id(value=value, canary_id="hny_sess1_repo_pat")


def _canary_record_with_id(value: str, canary_id: str) -> CanaryRecord:
    return CanaryRecord(
        canary_id=canary_id,
        credential_type="github_pat",
        value=value,
        sha256=canary_sha256(value),
        source="test",
        metadata={"slot_name": "repo_pat"},
    )


def _partial_fragment_ratios(matches: list[JsonValue]) -> tuple[float, ...]:
    ratios: list[float] = []
    for match in matches:
        if not isinstance(match, dict) or match.get("exact") is not False:
            continue
        fragment_ratio = match.get("fragment_ratio")
        if isinstance(fragment_ratio, bool) or not isinstance(fragment_ratio, int | float):
            continue
        ratios.append(float(fragment_ratio))
    return tuple(ratios)


def _critic_input(output_text: str) -> NimbusCriticInput:
    return NimbusCriticInput(
        session_id="sess1",
        turn_index=1,
        output_text=output_text,
        secret_context_handle="hny_sess1_repo_pat",
        messages=(),
        sensitive_spans=(),
        prior_state=NimbusState(
            session_id="sess1",
            turn_count=0,
            cumulative_estimated_leakage_bits=0.0,
            last_turn_estimated_leakage_bits=0.0,
            secret_context_handle="hny_sess1_repo_pat",
            recent_turn_scores=(),
        ),
    )
