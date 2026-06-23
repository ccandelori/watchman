"""Runtime integration tests for NIMBUS."""

from typing import cast

import pytest

from aegis.core.contracts import (
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
    NimbusCritic,
    NimbusCriticInput,
    NimbusCriticScore,
    NimbusDetector,
    NimbusDetectorError,
    NimbusState,
)


def make_model_info() -> ModelInfo:
    model = ModelInfo(provider="mock", model_id="test", revision="v0", selected_device=None)
    return model


def make_critic_input(
    secret_handle: str,
    output_text: str,
    messages: tuple[Message, ...],
) -> NimbusCriticInput:
    prior_state = NimbusState(
        session_id="sess-contract",
        turn_count=0,
        cumulative_estimated_leakage_bits=0.0,
        last_turn_estimated_leakage_bits=0.0,
        secret_context_handle=secret_handle,
        recent_turn_scores=(),
    )

    return NimbusCriticInput(
        session_id="sess-contract",
        turn_index=0,
        output_text=output_text,
        secret_context_handle=secret_handle,
        messages=messages,
        sensitive_spans=(),
        prior_state=prior_state,
    )


@pytest.mark.parametrize("critic", [BaselineNimbusCritic()])
def test_critic_returns_valid_score(critic: NimbusCritic) -> None:
    critic_input = make_critic_input(
        secret_handle="secret-1",
        output_text="some output",
        messages=(Message(role="user", content="test"),),
    )
    score = critic.score_turn(critic_input)

    assert isinstance(score, NimbusCriticScore)
    assert score.estimated_leakage_bits >= 0
    assert 0.0 <= score.confidence <= 1.0


@pytest.mark.parametrize("critic", [BaselineNimbusCritic()])
def test_critic_does_not_leak_secret_handle(critic: NimbusCritic) -> None:
    secret = "super-secret-value-xyz"
    critic_input = make_critic_input(
        secret_handle=secret,
        output_text="some output",
        messages=(Message(role="user", content="test"),),
    )

    score = critic.score_turn(critic_input)

    assert secret not in str(score)


def test_critic_must_handle_empty_output() -> None:
    critic = BaselineNimbusCritic()
    critic_input = make_critic_input(
        secret_handle="secret-1",
        output_text="",
        messages=(Message(role="user", content="test"),),
    )

    score = critic.score_turn(critic_input)
    assert score.estimated_leakage_bits >= 0


def test_detector_rejects_non_score_critic_result() -> None:
    class BadCritic:
        def score_turn(self, critic_input: NimbusCriticInput) -> object:
            return object()

    config = NimbusConfig(
        budget_bits=10.0,
        warn_threshold=0.5,
        sanitize_threshold=0.7,
        block_threshold=0.9,
        max_turns=10,
        critic_version="bad",
    )
    store = InMemoryNimbusStateStore(max_turns=10)
    detector = NimbusDetector(config, cast(NimbusCritic, BadCritic()), store)

    model = make_model_info()
    turn = NormalizedTurn(
        trace_id="t1",
        session_id="sess-bad",
        turn_index=0,
        capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
        model=model,
        messages=(Message(role="user", content="test"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={"secret_context_handle": "s1"},
    )

    with pytest.raises(NimbusDetectorError, match=r"critic\.score_turn must return NimbusCriticScore"):
        detector.evaluate(turn, ModelResponse(output_text="response", metadata={}))
