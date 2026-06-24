from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Protocol

from aegis.core.action_severity import highest_action
from aegis.core.contracts import Action, CapabilityStatus, DetectorComponent, DetectorResult, JsonValue, NormalizedTurn
from aegis.core.orchestrator import ModelResponse
from aegis.detectors.canary import EncodedCanaryDetector, InMemoryCanaryRegistry, TextCanaryDetector


class NimbusDetectorError(ValueError):
    """Raised when NIMBUS detector configuration is invalid."""


@dataclass(frozen=True)
class NimbusConfig:
    budget_bits: float
    warn_threshold: float
    sanitize_threshold: float
    block_threshold: float
    max_turns: int
    critic_version: str

    def __post_init__(self) -> None:
        _validate_positive_finite_bits(self.budget_bits, "budget_bits")
        _validate_probability(self.warn_threshold, "warn_threshold")
        _validate_probability(self.sanitize_threshold, "sanitize_threshold")
        _validate_probability(self.block_threshold, "block_threshold")
        if self.warn_threshold > self.sanitize_threshold or self.sanitize_threshold > self.block_threshold:
            raise NimbusDetectorError(
                "thresholds must be ordered: warn_threshold <= sanitize_threshold <= block_threshold."
            )
        if self.max_turns <= 0:
            raise NimbusDetectorError("max_turns must be positive.")
        if self.critic_version == "":
            raise NimbusDetectorError("critic_version must not be empty.")


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
    confidence: float


@dataclass(frozen=True)
class NimbusCriticInput:
    session_id: str
    turn_index: int
    secret_context_handle: str
    model_output_text: str
    state: NimbusState


@dataclass(frozen=True)
class NimbusCriticScore:
    estimated_leakage_bits: float
    confidence: float

    def __post_init__(self) -> None:
        _validate_nonnegative_finite_bits(self.estimated_leakage_bits, "estimated_leakage_bits")
        _validate_probability(self.confidence, "confidence")


class NimbusStateStore(Protocol):
    def get_or_create(self, session_id: str, secret_context_handle: str | None) -> NimbusState:
        """Return existing session state, or create state for a new secret context."""
        ...

    def update(self, session_id: str, update: NimbusStateUpdate) -> NimbusState:
        """Persist a cumulative NIMBUS state update."""
        ...

    def destroy(self, session_id: str) -> None:
        """Delete session state."""
        ...


class NimbusCritic(Protocol):
    def score_turn(self, critic_input: NimbusCriticInput) -> NimbusCriticScore:
        """Estimate how many bits of protected context leaked this turn."""
        ...


@dataclass
class InMemoryNimbusStateStore:
    max_turns: int
    _store: dict[str, NimbusState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise NimbusDetectorError("max_turns must be positive.")

    def get_or_create(self, session_id: str, secret_context_handle: str | None) -> NimbusState:
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
        recent_turn_scores = (*state.recent_turn_scores, update.turn_estimated_leakage_bits)
        if len(recent_turn_scores) > self.max_turns:
            recent_turn_scores = recent_turn_scores[-self.max_turns :]
        new_state = NimbusState(
            session_id=state.session_id,
            turn_count=state.turn_count + 1,
            cumulative_estimated_leakage_bits=update.new_cumulative_bits,
            last_turn_estimated_leakage_bits=update.turn_estimated_leakage_bits,
            secret_context_handle=state.secret_context_handle,
            recent_turn_scores=recent_turn_scores,
        )
        self._store[session_id] = new_state
        return new_state

    def destroy(self, session_id: str) -> None:
        self._store.pop(session_id, None)


@dataclass(frozen=True)
class BaselineNimbusCritic:
    fixed_estimated_leakage_bits: float
    fixed_confidence: float

    def __post_init__(self) -> None:
        _validate_nonnegative_finite_bits(self.fixed_estimated_leakage_bits, "fixed_estimated_leakage_bits")
        _validate_probability(self.fixed_confidence, "fixed_confidence")

    def score_turn(self, critic_input: NimbusCriticInput) -> NimbusCriticScore:
        return NimbusCriticScore(
            estimated_leakage_bits=self.fixed_estimated_leakage_bits,
            confidence=self.fixed_confidence,
        )


class NimbusDetector:
    def __init__(self, config: NimbusConfig, critic: NimbusCritic, state_store: NimbusStateStore) -> None:
        self.detector_name = "nimbus"
        self._config = config
        self._critic = critic
        self._state_store = state_store

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        started_at = time.perf_counter()
        secret_context_handle = resolve_secret_context_handle(turn)
        if secret_context_handle is None:
            return DetectorResult(
                detector_name=self.detector_name,
                component=DetectorComponent.NIMBUS,
                score=0.0,
                confidence=0.0,
                recommended_action=Action.ALLOW,
                capability_required="secret_context_handle",
                capability_status=CapabilityStatus.UNAVAILABLE,
                evidence={
                    "capability_reason": "no_secret_context_handle",
                    "session_id": turn.session_id,
                },
                latency_ms=_elapsed_ms(started_at),
            )
        if model_response is None:
            return DetectorResult(
                detector_name=self.detector_name,
                component=DetectorComponent.NIMBUS,
                score=0.0,
                confidence=0.0,
                recommended_action=Action.ALLOW,
                capability_required="model_response",
                capability_status=CapabilityStatus.DEGRADED,
                evidence={
                    "capability_reason": "model_response_required",
                    "session_id": turn.session_id,
                },
                latency_ms=_elapsed_ms(started_at),
            )

        state = self._state_store.get_or_create(
            session_id=turn.session_id,
            secret_context_handle=secret_context_handle,
        )
        critic_score = self._critic.score_turn(
            NimbusCriticInput(
                session_id=turn.session_id,
                turn_index=turn.turn_index,
                secret_context_handle=secret_context_handle,
                model_output_text=model_response.output_text,
                state=state,
            )
        )
        cumulative_bits = state.cumulative_estimated_leakage_bits + critic_score.estimated_leakage_bits
        updated_state = self._state_store.update(
            turn.session_id,
            NimbusStateUpdate(
                turn_estimated_leakage_bits=critic_score.estimated_leakage_bits,
                new_cumulative_bits=cumulative_bits,
                confidence=critic_score.confidence,
            ),
        )
        budget_fraction = cumulative_bits / self._config.budget_bits
        recommended_action = _budget_action(
            budget_fraction=budget_fraction,
            warn_threshold=self._config.warn_threshold,
            sanitize_threshold=self._config.sanitize_threshold,
            block_threshold=self._config.block_threshold,
        )
        return DetectorResult(
            detector_name=self.detector_name,
            component=DetectorComponent.NIMBUS,
            score=min(1.0, budget_fraction),
            confidence=critic_score.confidence,
            recommended_action=recommended_action,
            capability_required=None,
            capability_status=CapabilityStatus.ACTIVE,
            evidence={
                "reason": _budget_reason(recommended_action),
                "session_id": turn.session_id,
                "turn_count": updated_state.turn_count,
                "critic_version": self._config.critic_version,
                "secret_context_sha256": _sha256_text(secret_context_handle),
                "turn_estimated_leakage_bits": critic_score.estimated_leakage_bits,
                "last_turn_estimated_leakage_bits": updated_state.last_turn_estimated_leakage_bits,
                "cumulative_estimated_leakage_bits": updated_state.cumulative_estimated_leakage_bits,
                "budget_bits": self._config.budget_bits,
                "budget_fraction": budget_fraction,
                "warn_threshold": self._config.warn_threshold,
                "sanitize_threshold": self._config.sanitize_threshold,
                "block_threshold": self._config.block_threshold,
                "recent_turn_scores": list(updated_state.recent_turn_scores),
            },
            latency_ms=_elapsed_ms(started_at),
        )


def resolve_secret_context_handle(turn: NormalizedTurn) -> str | None:
    for preferred_kind in ("credential", "honeytoken"):
        for span in turn.sensitive_spans:
            if span.kind != preferred_kind:
                continue
            handle = _handle_from_span(span.metadata, span.identifier)
            if handle is not None:
                return handle
    value = turn.metadata.get("secret_context_handle")
    if isinstance(value, str) and value != "":
        return value
    return None


@dataclass(frozen=True)
class NimbusLeakageState:
    session_id: str
    score: float


class NimbusLeakageDetector:
    def __init__(
        self,
        detector_name: str,
        registry: InMemoryCanaryRegistry,
        partial_match_threshold: float,
        decay: float,
        warn_threshold: float,
        escalate_threshold: float,
        confidence: float,
    ) -> None:
        if detector_name == "":
            raise NimbusDetectorError("detector_name must not be empty.")
        _validate_probability(partial_match_threshold, "partial_match_threshold")
        _validate_probability(decay, "decay")
        _validate_probability(warn_threshold, "warn_threshold")
        _validate_probability(escalate_threshold, "escalate_threshold")
        _validate_probability(confidence, "confidence")
        if escalate_threshold < warn_threshold:
            raise NimbusDetectorError("escalate_threshold must be greater than or equal to warn_threshold.")
        self.detector_name = detector_name
        self._text_detector = TextCanaryDetector(detector_name=f"{detector_name}_exact_signal", registry=registry)
        self._encoded_detector = EncodedCanaryDetector(
            detector_name=f"{detector_name}_encoded_signal",
            registry=registry,
            partial_match_threshold=partial_match_threshold,
        )
        self._decay = decay
        self._warn_threshold = warn_threshold
        self._escalate_threshold = escalate_threshold
        self._confidence = confidence
        self._scores_by_session_id: dict[str, float] = {}

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        started_at = time.perf_counter()
        if model_response is None:
            return DetectorResult(
                detector_name=self.detector_name,
                component=DetectorComponent.NIMBUS,
                score=self._score_for_session(turn.session_id),
                confidence=self._confidence,
                recommended_action=Action.ALLOW,
                capability_required=None,
                capability_status=CapabilityStatus.DEGRADED,
                evidence={
                    "reason": "model_response_required",
                    "session_id": turn.session_id,
                    "current_leakage_score": self._score_for_session(turn.session_id),
                },
                latency_ms=_elapsed_ms(started_at),
            )

        exact_result = self._text_detector.evaluate(turn=turn, model_response=model_response)
        encoded_result = self._encoded_detector.evaluate(turn=turn, model_response=model_response)
        previous_score = self._score_for_session(turn.session_id)
        signal_score = max(exact_result.score, encoded_result.score)
        signal_action = highest_action((exact_result.recommended_action, encoded_result.recommended_action))
        updated_score = min(1.0, previous_score * self._decay + signal_score)
        self._scores_by_session_id[turn.session_id] = updated_score
        recommended_action = _recommended_action(
            signal_action=signal_action,
            updated_score=updated_score,
            warn_threshold=self._warn_threshold,
            escalate_threshold=self._escalate_threshold,
        )
        return DetectorResult(
            detector_name=self.detector_name,
            component=DetectorComponent.NIMBUS,
            score=updated_score,
            confidence=self._confidence,
            recommended_action=recommended_action,
            capability_required=None,
            capability_status=CapabilityStatus.ACTIVE,
            evidence={
                "reason": _reason(recommended_action),
                "session_id": turn.session_id,
                "previous_leakage_score": previous_score,
                "turn_signal_score": signal_score,
                "current_leakage_score": updated_score,
                "decay": self._decay,
                "warn_threshold": self._warn_threshold,
                "escalate_threshold": self._escalate_threshold,
                "exact_signal": _signal_summary(exact_result),
                "encoded_signal": _signal_summary(encoded_result),
            },
            latency_ms=_elapsed_ms(started_at),
        )

    def state(self, session_id: str) -> NimbusLeakageState:
        return NimbusLeakageState(session_id=session_id, score=self._score_for_session(session_id))

    def _score_for_session(self, session_id: str) -> float:
        return self._scores_by_session_id.get(session_id, 0.0)


def _handle_from_span(metadata: dict[str, JsonValue], identifier: str | None) -> str | None:
    handle = metadata.get("handle")
    if isinstance(handle, str) and handle != "":
        return handle
    if identifier is not None and identifier != "":
        return identifier
    return None


def _budget_action(
    budget_fraction: float,
    warn_threshold: float,
    sanitize_threshold: float,
    block_threshold: float,
) -> Action:
    if budget_fraction >= block_threshold:
        return Action.BLOCK
    if budget_fraction >= sanitize_threshold:
        return Action.SANITIZE
    if budget_fraction >= warn_threshold:
        return Action.WARN
    return Action.ALLOW


def _budget_reason(action: Action) -> str:
    if action == Action.BLOCK:
        return "nimbus_budget_block_threshold_reached"
    if action == Action.SANITIZE:
        return "nimbus_budget_sanitize_threshold_reached"
    if action == Action.WARN:
        return "nimbus_budget_warn_threshold_reached"
    return "nimbus_budget_within_limits"


def _recommended_action(
    signal_action: Action,
    updated_score: float,
    warn_threshold: float,
    escalate_threshold: float,
) -> Action:
    if signal_action == Action.ESCALATE or updated_score >= escalate_threshold:
        return Action.ESCALATE
    if updated_score >= warn_threshold or signal_action in (Action.SANITIZE, Action.BLOCK):
        return Action.WARN
    return Action.ALLOW


def _reason(action: Action) -> str:
    if action == Action.ESCALATE:
        return "cumulative_leakage_budget_exhausted"
    if action == Action.WARN:
        return "cumulative_leakage_budget_warning"
    return "no_cumulative_leakage_detected"


def _signal_summary(result: DetectorResult) -> dict[str, JsonValue]:
    summary: dict[str, JsonValue] = {
        "detector_name": result.detector_name,
        "score": result.score,
        "recommended_action": result.recommended_action.value,
        "reason": result.evidence.get("reason"),
    }
    match_count = result.evidence.get("match_count")
    if isinstance(match_count, int):
        summary["match_count"] = match_count
    matches = result.evidence.get("matches")
    if isinstance(matches, list):
        summary["matches"] = matches
    return summary


def _validate_probability(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise NimbusDetectorError(f"{field_name} must be a finite value in [0.0, 1.0].")


def _validate_positive_finite_bits(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise NimbusDetectorError(f"{field_name} must be a positive finite value.")


def _validate_nonnegative_finite_bits(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise NimbusDetectorError(f"{field_name} must be a non-negative finite value.")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0
