from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from aegis.core.contracts import (
    Action,
    CapabilityStatus,
    NormalizedTurn,
    SensitiveSpan,
)
from aegis.core.orchestrator import ModelResponse
from aegis.detectors.nimbus import (
    NimbusConfig,
    NimbusCriticInput,
    NimbusCriticScore,
    NimbusDetector,
    NimbusState,
    NimbusStateUpdate,
    resolve_secret_context_handle,
)

# ---------------------------------------------------------------------------
# In-memory implementation (for tests)
# ---------------------------------------------------------------------------

@dataclass
class InMemoryNimbusStateStore:
    max_turns: int
    _store: dict[str, NimbusState] = field(default_factory=dict)

    def get_or_create(
        self, session_id: str, secret_context_handle: str | None
    ) -> NimbusState:
        if session_id not in self._store:
            self._store[session_id] = NimbusState(
                session_id=session_id,
                turn_count=0,
                cumulative_estimated_leakage_bits=0.0,
                last_turn_estimated_leakage_bits=0.0,
                secret_context_handle=secret_context_handle,
                recent_turn_scores=(),
            )
        return self._store[session_id]

    def update(self, session_id: str, update: NimbusStateUpdate) -> NimbusState:
        state = self._store[session_id]
        new_scores = (*state.recent_turn_scores, update.turn_estimated_leakage_bits)
        if len(new_scores) > self.max_turns:
            new_scores = new_scores[-self.max_turns :]
        new_state = NimbusState(
            session_id=state.session_id,
            turn_count=state.turn_count + 1,
            cumulative_estimated_leakage_bits=update.new_cumulative_bits,
            last_turn_estimated_leakage_bits=update.turn_estimated_leakage_bits,
            secret_context_handle=state.secret_context_handle,
            recent_turn_scores=new_scores,
        )
        self._store[session_id] = new_state
        return new_state

    def destroy(self, session_id: str) -> None:
        self._store.pop(session_id, None)


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
    metadata: dict | None = None,
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

def test_config_validation_rejects_non_finite():
    with pytest.raises(ValueError):
        NimbusConfig(
            budget_bits=float("inf"),
            warn_threshold=0.5,
            sanitize_threshold=0.7,
            block_threshold=0.9,
            max_turns=10,
            critic_version="v0",
        )


def test_config_validation_threshold_order():
    with pytest.raises(ValueError):
        NimbusConfig(10.0, 0.9, 0.8, 0.7, 10, "v0")


def test_resolve_priority_credential_over_honeytoken():
    spans = (
        _make_span("honeytoken", "h1"),
        _make_span("credential", "c1"),
    )
    turn = _make_turn(sensitive_spans=spans)
    assert resolve_secret_context_handle(turn) == "c1"


def test_nimbus_unavailable_no_secret_handle():
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


def test_nimbus_cumulative_increases():
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


def test_nimbus_threshold_actions():
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


def test_bounded_recent_turn_scores():
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


def test_critic_rejects_invalid_score():
    with pytest.raises(ValueError):
        NimbusCriticScore(estimated_leakage_bits=-1.0, confidence=0.5)

    with pytest.raises(ValueError):
        NimbusCriticScore(estimated_leakage_bits=1.0, confidence=1.5)

