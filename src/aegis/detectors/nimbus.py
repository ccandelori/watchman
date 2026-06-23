from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol

from aegis.core.action_severity import highest_action
from aegis.core.contracts import (
    Action,
    CapabilityMode,
    CapabilityStatus,
    DetectorComponent,
    DetectorResult,
    JsonValue,
    Message,
    ModelInfo,
    NormalizedTurn,
    SensitiveSpan,
)
from aegis.core.orchestrator import ModelResponse
from aegis.detectors.canary import CanaryRecord, EncodedCanaryDetector, InMemoryCanaryRegistry, TextCanaryDetector


class NimbusDetectorError(ValueError):
    """Raised when NIMBUS-lite detector configuration is invalid."""


@dataclass(frozen=True)
class NimbusLeakageState:
    session_id: str
    score: float


@dataclass(frozen=True)
class NimbusConfig:
    budget_bits: float
    warn_threshold: float
    sanitize_threshold: float
    block_threshold: float
    max_turns: int
    critic_version: str

    def __post_init__(self) -> None:
        _validate_positive_finite(self.budget_bits, "budget_bits")
        _validate_probability(self.warn_threshold, "warn_threshold")
        _validate_probability(self.sanitize_threshold, "sanitize_threshold")
        _validate_probability(self.block_threshold, "block_threshold")
        if not self.warn_threshold <= self.sanitize_threshold <= self.block_threshold:
            raise NimbusDetectorError(
                "thresholds must satisfy warn_threshold <= sanitize_threshold <= block_threshold."
            )
        if self.max_turns < 1:
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

    def __post_init__(self) -> None:
        _validate_session_id(self.session_id)
        if self.turn_count < 0:
            raise NimbusDetectorError("turn_count must be non-negative.")
        _validate_non_negative_finite(
            self.cumulative_estimated_leakage_bits,
            "cumulative_estimated_leakage_bits",
        )
        _validate_non_negative_finite(
            self.last_turn_estimated_leakage_bits,
            "last_turn_estimated_leakage_bits",
        )
        if self.secret_context_handle == "":
            raise NimbusDetectorError("secret_context_handle must not be empty when provided.")
        for score in self.recent_turn_scores:
            _validate_non_negative_finite(score, "recent_turn_scores entry")


@dataclass(frozen=True)
class NimbusStateUpdate:
    turn_estimated_leakage_bits: float
    new_cumulative_bits: float

    def __post_init__(self) -> None:
        _validate_non_negative_finite(self.turn_estimated_leakage_bits, "turn_estimated_leakage_bits")
        _validate_non_negative_finite(self.new_cumulative_bits, "new_cumulative_bits")


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
    evidence: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _validate_non_negative_finite(self.estimated_leakage_bits, "estimated_leakage_bits")
        _validate_probability(self.confidence, "confidence")


class NimbusCritic(Protocol):
    def score_turn(self, critic_input: NimbusCriticInput) -> NimbusCriticScore:
        """Estimate leakage bits for one model response."""


class NimbusStateStore(Protocol):
    def get_or_create(self, session_id: str, secret_context_handle: str | None) -> NimbusState:
        """Load or initialize NIMBUS state for a session."""

    def update(self, session_id: str, update: NimbusStateUpdate) -> NimbusState:
        """Persist one NIMBUS state update."""

    def destroy(self, session_id: str) -> None:
        """Remove NIMBUS state for a completed session."""


@dataclass(frozen=True)
class BaselineNimbusCritic:
    fixed_estimated_leakage_bits: float = 0.0
    fixed_confidence: float = 0.5

    def __post_init__(self) -> None:
        _validate_non_negative_finite(self.fixed_estimated_leakage_bits, "fixed_estimated_leakage_bits")
        _validate_probability(self.fixed_confidence, "fixed_confidence")

    def score_turn(self, critic_input: NimbusCriticInput) -> NimbusCriticScore:
        return NimbusCriticScore(
            estimated_leakage_bits=self.fixed_estimated_leakage_bits,
            confidence=self.fixed_confidence,
            evidence={
                "critic_kind": "baseline",
                "critic_version": "fixed",
                "estimated_leakage_bits": self.fixed_estimated_leakage_bits,
            },
        )


@dataclass(frozen=True)
class CanaryNimbusCriticConfig:
    exact_match_leakage_bits: float
    encoded_match_leakage_bits: float
    partial_match_leakage_bits: float
    partial_match_threshold: float
    confidence: float

    def __post_init__(self) -> None:
        _validate_non_negative_finite(self.exact_match_leakage_bits, "exact_match_leakage_bits")
        _validate_non_negative_finite(self.encoded_match_leakage_bits, "encoded_match_leakage_bits")
        _validate_non_negative_finite(self.partial_match_leakage_bits, "partial_match_leakage_bits")
        _validate_probability(self.partial_match_threshold, "partial_match_threshold")
        _validate_probability(self.confidence, "confidence")


class CanaryNimbusCritic:
    def __init__(self, config: CanaryNimbusCriticConfig) -> None:
        self._config = config
        self._records_by_session_id: dict[str, tuple[CanaryRecord, ...]] = {}

    def register_canary_records(self, session_id: str, records: tuple[CanaryRecord, ...]) -> None:
        if session_id == "":
            raise NimbusDetectorError("session_id must not be empty.")
        if len(records) == 0:
            return
        records_by_id = {record.canary_id: record for record in self._records_by_session_id.get(session_id, ())}
        for record in records:
            records_by_id[record.canary_id] = record
        self._records_by_session_id[session_id] = tuple(records_by_id.values())

    def destroy_session(self, session_id: str) -> None:
        self._records_by_session_id.pop(session_id, None)

    def score_turn(self, critic_input: NimbusCriticInput) -> NimbusCriticScore:
        records = self._records_by_session_id.get(critic_input.session_id, ())
        if len(records) == 0:
            return NimbusCriticScore(
                estimated_leakage_bits=0.0,
                confidence=self._config.confidence,
                evidence={
                    "critic_kind": "canary",
                    "reason": "no_registered_canaries_for_session",
                    "registered_canary_count": 0,
                },
            )

        return _canary_nimbus_score(
            critic_input=critic_input,
            records=records,
            config=self._config,
        )


@dataclass
class InMemoryNimbusStateStore:
    max_turns: int
    _store: dict[str, NimbusState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise NimbusDetectorError("max_turns must be positive.")

    def get_or_create(self, session_id: str, secret_context_handle: str | None) -> NimbusState:
        _validate_session_id(session_id)
        state = self._store.get(session_id)
        if state is not None:
            return state
        state = NimbusState(
            session_id=session_id,
            turn_count=0,
            cumulative_estimated_leakage_bits=0.0,
            last_turn_estimated_leakage_bits=0.0,
            secret_context_handle=secret_context_handle,
            recent_turn_scores=(),
        )
        self._store[session_id] = state
        return state

    def update(self, session_id: str, update: NimbusStateUpdate) -> NimbusState:
        _validate_session_id(session_id)
        state = self._store.get(session_id)
        if state is None:
            raise NimbusDetectorError(f"no NIMBUS state exists for session_id '{session_id}'.")
        if update.new_cumulative_bits < state.cumulative_estimated_leakage_bits:
            raise NimbusDetectorError("new_cumulative_bits must not decrease existing cumulative leakage.")
        recent_turn_scores = (*state.recent_turn_scores, update.turn_estimated_leakage_bits)
        if len(recent_turn_scores) > self.max_turns:
            recent_turn_scores = recent_turn_scores[-self.max_turns :]
        updated_state = NimbusState(
            session_id=state.session_id,
            turn_count=state.turn_count + 1,
            cumulative_estimated_leakage_bits=update.new_cumulative_bits,
            last_turn_estimated_leakage_bits=update.turn_estimated_leakage_bits,
            secret_context_handle=state.secret_context_handle,
            recent_turn_scores=recent_turn_scores,
        )
        self._store[session_id] = updated_state
        return updated_state

    def destroy(self, session_id: str) -> None:
        _validate_session_id(session_id)
        self._store.pop(session_id, None)


class NimbusDetector:
    def __init__(self, config: NimbusConfig, critic: NimbusCritic, state_store: NimbusStateStore) -> None:
        self._config = config
        self._critic = critic
        self._state_store = state_store

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        started_at = time.perf_counter()
        secret_context_handle = resolve_secret_context_handle(turn)
        if secret_context_handle is None:
            return _nimbus_unavailable_result(
                turn=turn,
                reason="no_secret_context_handle",
                started_at=started_at,
            )
        if model_response is None:
            return DetectorResult(
                detector_name="nimbus",
                component=DetectorComponent.NIMBUS,
                score=0.0,
                confidence=0.0,
                recommended_action=Action.ALLOW,
                capability_required="model_response",
                capability_status=CapabilityStatus.DEGRADED,
                evidence={
                    "capability_reason": "model_response_required",
                    "turn_index": turn.turn_index,
                    "critic_version": self._config.critic_version,
                },
                latency_ms=_elapsed_ms(started_at),
            )

        prior_state = self._state_store.get_or_create(
            session_id=turn.session_id,
            secret_context_handle=secret_context_handle,
        )
        critic_score = self._critic.score_turn(
            NimbusCriticInput(
                session_id=turn.session_id,
                turn_index=turn.turn_index,
                output_text=model_response.output_text,
                secret_context_handle=secret_context_handle,
                messages=turn.messages,
                sensitive_spans=turn.sensitive_spans,
                prior_state=prior_state,
            )
        )
        if not isinstance(critic_score, NimbusCriticScore):
            raise NimbusDetectorError("critic.score_turn must return NimbusCriticScore.")
        new_cumulative_bits = prior_state.cumulative_estimated_leakage_bits + critic_score.estimated_leakage_bits
        updated_state = self._state_store.update(
            session_id=turn.session_id,
            update=NimbusStateUpdate(
                turn_estimated_leakage_bits=critic_score.estimated_leakage_bits,
                new_cumulative_bits=new_cumulative_bits,
            ),
        )
        budget_fraction = min(1.0, updated_state.cumulative_estimated_leakage_bits / self._config.budget_bits)
        recommended_action = _budget_action(
            budget_fraction=budget_fraction,
            warn_threshold=self._config.warn_threshold,
            sanitize_threshold=self._config.sanitize_threshold,
            block_threshold=self._config.block_threshold,
        )
        return DetectorResult(
            detector_name="nimbus",
            component=DetectorComponent.NIMBUS,
            score=budget_fraction,
            confidence=critic_score.confidence,
            recommended_action=recommended_action,
            capability_required=None,
            capability_status=CapabilityStatus.ACTIVE,
            evidence={
                "reason": _budget_reason(recommended_action),
                "turn_index": turn.turn_index,
                "turn_count": updated_state.turn_count,
                "turn_estimated_leakage_bits": critic_score.estimated_leakage_bits,
                "cumulative_estimated_leakage_bits": updated_state.cumulative_estimated_leakage_bits,
                "budget_bits": self._config.budget_bits,
                "budget_fraction": budget_fraction,
                "warn_threshold": self._config.warn_threshold,
                "sanitize_threshold": self._config.sanitize_threshold,
                "block_threshold": self._config.block_threshold,
                "critic_version": self._config.critic_version,
                "critic_evidence": critic_score.evidence,
            },
            latency_ms=_elapsed_ms(started_at),
        )

    def destroy_session(self, session_id: str) -> None:
        self._state_store.destroy(session_id)


class NimbusLeakageDetector:
    """Legacy canary-signal accumulator kept for demos and compatibility tests."""

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


def resolve_secret_context_handle(turn: NormalizedTurn) -> str | None:
    credential_handle = _handle_from_sensitive_spans(turn.sensitive_spans, "credential")
    if credential_handle is not None:
        return credential_handle
    honeytoken_handle = _handle_from_sensitive_spans(turn.sensitive_spans, "honeytoken")
    if honeytoken_handle is not None:
        return honeytoken_handle
    metadata_handle = turn.metadata.get("secret_context_handle")
    if isinstance(metadata_handle, str) and metadata_handle != "":
        return metadata_handle
    return None


def _canary_nimbus_score(
    critic_input: NimbusCriticInput,
    records: tuple[CanaryRecord, ...],
    config: CanaryNimbusCriticConfig,
) -> NimbusCriticScore:
    registry = InMemoryCanaryRegistry(records=records)
    turn = _turn_for_critic_input(critic_input)
    response = ModelResponse(output_text=critic_input.output_text, metadata={})
    exact_result = TextCanaryDetector(detector_name="nimbus_exact_signal", registry=registry).evaluate(
        turn=turn,
        model_response=response,
    )
    encoded_result = EncodedCanaryDetector(
        detector_name="nimbus_encoded_signal",
        registry=registry,
        partial_match_threshold=config.partial_match_threshold,
    ).evaluate(turn=turn, model_response=response)

    exact_matches = _matches_from_result(exact_result)
    encoded_matches = _matches_from_result(encoded_result)
    exact_match_count = len(exact_matches)
    encoded_exact_matches = tuple(match for match in encoded_matches if match.get("exact") is True)
    partial_matches = tuple(match for match in encoded_matches if match.get("exact") is False)
    encoded_match_count = len(encoded_exact_matches)
    partial_match_count = len(partial_matches)
    partial_fragment_ratios = _fragment_ratios(partial_matches)
    max_partial_fragment_ratio = _max_fragment_ratio(partial_fragment_ratios)
    partial_estimated_leakage_bits = sum(
        fragment_ratio * config.partial_match_leakage_bits for fragment_ratio in partial_fragment_ratios
    )
    matched_canary_ids = _matched_canary_ids(exact_matches + encoded_matches)
    estimated_leakage_bits = (
        exact_match_count * config.exact_match_leakage_bits
        + encoded_match_count * config.encoded_match_leakage_bits
        + partial_estimated_leakage_bits
    )

    evidence: dict[str, JsonValue] = {
        "critic_kind": "canary",
        "registered_canary_count": len(records),
        "exact_match_count": exact_match_count,
        "encoded_match_count": encoded_match_count,
        "partial_match_count": partial_match_count,
        "max_partial_fragment_ratio": max_partial_fragment_ratio,
        "partial_estimated_leakage_bits": partial_estimated_leakage_bits,
        "matched_canary_ids": list(matched_canary_ids),
        "estimated_leakage_bits": estimated_leakage_bits,
        "partial_match_threshold": config.partial_match_threshold,
        "exact_signal": _signal_summary(exact_result),
        "encoded_signal": _signal_summary(encoded_result),
    }
    return NimbusCriticScore(
        estimated_leakage_bits=estimated_leakage_bits,
        confidence=config.confidence,
        evidence=evidence,
    )


def _turn_for_critic_input(critic_input: NimbusCriticInput) -> NormalizedTurn:
    return NormalizedTurn(
        trace_id=f"nimbus-critic-{critic_input.session_id}-{critic_input.turn_index}",
        session_id=critic_input.session_id,
        turn_index=critic_input.turn_index,
        capability_mode=CapabilityMode.OFFLINE_EVAL,
        model=ModelInfo(provider="nimbus", model_id="canary-critic", revision=None, selected_device=None),
        messages=critic_input.messages,
        tool_calls=(),
        sensitive_spans=critic_input.sensitive_spans,
        metadata={},
    )


def _matches_from_result(result: DetectorResult) -> tuple[dict[str, JsonValue], ...]:
    matches = result.evidence.get("matches")
    if not isinstance(matches, list):
        return ()
    values: list[dict[str, JsonValue]] = []
    for match in matches:
        if isinstance(match, dict):
            values.append(match)
    return tuple(values)


def _fragment_ratios(matches: tuple[dict[str, JsonValue], ...]) -> tuple[float, ...]:
    ratios = tuple(match.get("fragment_ratio") for match in matches)
    return tuple(float(ratio) for ratio in ratios if isinstance(ratio, int | float))


def _max_fragment_ratio(ratios: tuple[float, ...]) -> float:
    if len(ratios) == 0:
        return 0.0
    return max(ratios)


def _matched_canary_ids(matches: tuple[dict[str, JsonValue], ...]) -> tuple[str, ...]:
    ids: list[str] = []
    for match in matches:
        canary_id = match.get("canary_id")
        if isinstance(canary_id, str) and canary_id not in ids:
            ids.append(canary_id)
    return tuple(ids)


def _handle_from_sensitive_spans(spans: tuple[SensitiveSpan, ...], kind: str) -> str | None:
    for span in spans:
        if span.kind != kind:
            continue
        metadata_handle = span.metadata.get("handle")
        if isinstance(metadata_handle, str) and metadata_handle != "":
            return metadata_handle
        if span.identifier is not None and span.identifier != "":
            return span.identifier
    return None


def _nimbus_unavailable_result(turn: NormalizedTurn, reason: str, started_at: float) -> DetectorResult:
    return DetectorResult(
        detector_name="nimbus",
        component=DetectorComponent.NIMBUS,
        score=0.0,
        confidence=0.0,
        recommended_action=Action.ALLOW,
        capability_required="secret_context_handle",
        capability_status=CapabilityStatus.UNAVAILABLE,
        evidence={
            "capability_reason": reason,
            "turn_index": turn.turn_index,
        },
        latency_ms=_elapsed_ms(started_at),
    )


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
        return "nimbus_leakage_budget_block"
    if action == Action.SANITIZE:
        return "nimbus_leakage_budget_sanitize"
    if action == Action.WARN:
        return "nimbus_leakage_budget_warning"
    return "nimbus_leakage_budget_available"


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
    if not isfinite(value) or value < 0.0 or value > 1.0:
        raise NimbusDetectorError(f"{field_name} must be in [0.0, 1.0].")


def _validate_positive_finite(value: float, field_name: str) -> None:
    if not isfinite(value) or value <= 0.0:
        raise NimbusDetectorError(f"{field_name} must be a finite positive value.")


def _validate_non_negative_finite(value: float, field_name: str) -> None:
    if not isfinite(value) or value < 0.0:
        raise NimbusDetectorError(f"{field_name} must be a finite non-negative value.")


def _validate_session_id(session_id: str) -> None:
    if session_id == "":
        raise NimbusDetectorError("session_id must not be empty.")


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0
