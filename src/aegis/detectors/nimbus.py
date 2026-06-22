from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from aegis.core.contracts import (
    Action,
    CapabilityStatus,
    DetectorComponent,
    DetectorResult,
    JsonValue,
    Message,
    NormalizedTurn,
    SensitiveSpan,
)
from aegis.core.orchestrator import ModelResponse


@dataclass(frozen=True)
class NimbusConfig:
    budget_bits: float
    warn_threshold: float
    sanitize_threshold: float
    block_threshold: float
    max_turns: int
    critic_version: str

    def __post_init__(self) -> None:
        for name, value in [
            ("budget_bits", self.budget_bits),
            ("warn_threshold", self.warn_threshold),
            ("sanitize_threshold", self.sanitize_threshold),
            ("block_threshold", self.block_threshold),
        ]:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.budget_bits <= 0:
            raise ValueError("budget_bits must be > 0")
        if not (0 <= self.warn_threshold <= self.sanitize_threshold <= self.block_threshold <= 1):
            raise ValueError("thresholds must satisfy 0 <= warn <= sanitize <= block <= 1")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be > 0")


@dataclass(frozen=True)
class NimbusState:
    session_id: str
    turn_count: int
    cumulative_estimated_leakage_bits: float
    last_turn_estimated_leakage_bits: float
    secret_context_handle: str | None
    recent_turn_scores: tuple[float, ...]


@dataclass(frozen=True)
class NimbusStateUpdate:
    turn_estimated_leakage_bits: float
    new_cumulative_bits: float
    budget_fraction: float


@dataclass(frozen=True)
class NimbusCriticInput:
    session_id: str
    turn_index: int
    output_text: str
    secret_context_handle: str
    messages: tuple[Message, ...]
    sensitive_spans: tuple[SensitiveSpan, ...]
    prior_state: NimbusState


@dataclass(frozen=True)
class NimbusCriticScore:
    estimated_leakage_bits: float
    confidence: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.estimated_leakage_bits) or self.estimated_leakage_bits < 0:
            raise ValueError("estimated_leakage_bits must be finite and >= 0")
        if not (0.0 <= self.confidence <= 1.0) or not math.isfinite(self.confidence):
            raise ValueError("confidence must be finite and in [0, 1]")


class NimbusCritic(Protocol):
    def score_turn(self, critic_input: NimbusCriticInput) -> NimbusCriticScore:
        ...


class NimbusStateStore(Protocol):
    def get_or_create(
        self, session_id: str, secret_context_handle: str | None
    ) -> NimbusState:
        ...

    def update(self, session_id: str, update: NimbusStateUpdate) -> NimbusState:
        ...

    def destroy(self, session_id: str) -> None:
        ...


def resolve_secret_context_handle(turn: NormalizedTurn) -> str | None:
    """Priority: credential > honeytoken > secret, then metadata handle."""
    for kind in ("credential", "honeytoken", "secret"):
        for span in turn.sensitive_spans:
            if span.kind == kind:
                handle = span.metadata.get("handle")
                if isinstance(handle, str) and handle:
                    return handle
                if span.identifier:
                    return span.identifier
    handle = turn.metadata.get("secret_context_handle")
    if isinstance(handle, str) and handle:
        return handle
    return None


# ---------------------------------------------------------------------------
# Runtime-usable implementations
# ---------------------------------------------------------------------------

@dataclass
class InMemoryNimbusStateStore:
    """Runtime-safe in-memory store with bounded history."""

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
        new_scores = state.recent_turn_scores + (update.turn_estimated_leakage_bits,)
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


@dataclass
class BaselineNimbusCritic:
    """Deterministic baseline critic for runtime use (no ML)."""

    fixed_estimated_leakage_bits: float = 0.1
    fixed_confidence: float = 0.6

    def score_turn(self, critic_input: NimbusCriticInput) -> NimbusCriticScore:
        return NimbusCriticScore(
            estimated_leakage_bits=self.fixed_estimated_leakage_bits,
            confidence=self.fixed_confidence,
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@dataclass
class NimbusDetector:
    config: NimbusConfig
    critic: NimbusCritic
    state_store: NimbusStateStore

    def evaluate(
        self,
        turn: NormalizedTurn,
        model_response: ModelResponse | None,
    ) -> DetectorResult:
        if model_response is None:
            return self._unavailable_result(turn, "no_model_response")

        secret_handle = resolve_secret_context_handle(turn)
        if secret_handle is None:
            return self._unavailable_result(turn, "no_secret_context_handle")

        state = self.state_store.get_or_create(turn.session_id, secret_handle)

        critic_input = NimbusCriticInput(
            session_id=turn.session_id,
            turn_index=turn.turn_index,
            output_text=model_response.output_text,
            secret_context_handle=secret_handle,
            messages=turn.messages,
            sensitive_spans=turn.sensitive_spans,
            prior_state=state,
        )

        score = self.critic.score_turn(critic_input)

        new_cumulative = state.cumulative_estimated_leakage_bits + score.estimated_leakage_bits
        budget_fraction = min(new_cumulative / self.config.budget_bits, 1.0)

        update = NimbusStateUpdate(
            turn_estimated_leakage_bits=score.estimated_leakage_bits,
            new_cumulative_bits=new_cumulative,
            budget_fraction=budget_fraction,
        )
        new_state = self.state_store.update(turn.session_id, update)

        evidence: dict[str, JsonValue] = {
            "turn_estimated_leakage_bits": score.estimated_leakage_bits,
            "cumulative_estimated_leakage_bits": new_state.cumulative_estimated_leakage_bits,
            "budget_bits": self.config.budget_bits,
            "budget_fraction": budget_fraction,
            "session_id": turn.session_id,
            "state_turn_count": new_state.turn_count,
            "critic_version": self.config.critic_version,
            "capability_reason": "active",
        }

        recommended = self._recommend_action(budget_fraction)

        return DetectorResult(
            detector_name="nimbus",
            component=DetectorComponent.NIMBUS,
            recommended_action=recommended,
            score=budget_fraction,
            confidence=score.confidence,
            evidence=evidence,
            capability_required="secret_context_handle",
            capability_status=CapabilityStatus.ACTIVE,
            latency_ms=0.0,
        )

    def _recommend_action(self, budget_fraction: float) -> Action:
        if budget_fraction >= self.config.block_threshold:
            return Action.BLOCK
        if budget_fraction >= self.config.sanitize_threshold:
            return Action.SANITIZE
        if budget_fraction >= self.config.warn_threshold:
            return Action.WARN
        return Action.ALLOW

    def _unavailable_result(
        self, turn: NormalizedTurn, reason: str
    ) -> DetectorResult:
        return DetectorResult(
            detector_name="nimbus",
            component=DetectorComponent.NIMBUS,
            recommended_action=Action.ALLOW,
            score=0.0,
            confidence=0.0,
            evidence={
                "turn_estimated_leakage_bits": 0.0,
                "cumulative_estimated_leakage_bits": 0.0,
                "budget_bits": self.config.budget_bits,
                "budget_fraction": 0.0,
                "session_id": turn.session_id,
                "state_turn_count": 0,
                "critic_version": self.config.critic_version,
                "capability_reason": reason,
            },
            capability_required="secret_context_handle",
            capability_status=CapabilityStatus.UNAVAILABLE,
            latency_ms=0.0,
        )
