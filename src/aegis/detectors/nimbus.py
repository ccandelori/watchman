from __future__ import annotations

import json
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
    ToolCall,
)
from aegis.core.orchestrator import ModelResponse
from aegis.detectors.canary import (
    CanaryRecord,
    EncodedCanaryDetector,
    InMemoryCanaryRegistry,
    TextCanaryDetector,
    canary_sha256,
)
from aegis.replay.nimbus_infonce import (
    NIMBUS_INFONCE_DIAGNOSTIC_ONLY_FEATURE_NAMES,
    NimbusInfoNCEModel,
    score_nimbus_infonce_runtime_candidate,
)


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


@dataclass(frozen=True)
class NimbusRuntimeCandidateContext:
    context_id: str
    credential_type: str
    positive_context_text: str
    negative_context_texts: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        if self.context_id == "":
            raise NimbusDetectorError("context_id must not be empty.")
        if self.credential_type == "":
            raise NimbusDetectorError("credential_type must not be empty.")
        if self.positive_context_text == "":
            raise NimbusDetectorError("positive_context_text must not be empty.")
        if len(self.negative_context_texts) == 0:
            raise NimbusDetectorError("negative_context_texts must not be empty.")
        if any(context_text == "" for context_text in self.negative_context_texts):
            raise NimbusDetectorError("negative_context_texts entries must not be empty.")
        if self.source == "":
            raise NimbusDetectorError("source must not be empty.")


class NimbusCritic(Protocol):
    def score_turn(self, critic_input: NimbusCriticInput) -> NimbusCriticScore:
        """Estimate leakage bits for one model response."""


class RegisteredCanaryNimbusCritic(NimbusCritic, Protocol):
    def register_canary_records(self, session_id: str, records: tuple[CanaryRecord, ...]) -> None:
        """Register runtime canaries that define a session secret context."""

    def destroy_session(self, session_id: str) -> None:
        """Discard session-local canary context."""

    def clear(self) -> None:
        """Discard all canary context."""


class NimbusStateStore(Protocol):
    def get_or_create(self, session_id: str, secret_context_handle: str | None) -> NimbusState:
        """Load or initialize NIMBUS state for a session."""

    def update(self, session_id: str, update: NimbusStateUpdate) -> NimbusState:
        """Persist one NIMBUS state update."""

    def destroy(self, session_id: str) -> None:
        """Remove NIMBUS state for a completed session."""

    def clear(self) -> None:
        """Remove all NIMBUS session state."""


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
                "paper_faithful_learned_critic": False,
                "promotion_status": "demo_only_baseline",
                "deterministic_fallback": True,
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

    def clear(self) -> None:
        self._records_by_session_id.clear()

    def score_turn(self, critic_input: NimbusCriticInput) -> NimbusCriticScore:
        records = self._records_by_session_id.get(critic_input.session_id, ())
        if len(records) == 0:
            return NimbusCriticScore(
                estimated_leakage_bits=0.0,
                confidence=self._config.confidence,
                evidence={
                    "critic_kind": "canary",
                    "paper_faithful_learned_critic": False,
                    "promotion_status": "deterministic_canary_beta",
                    "deterministic_fallback": True,
                    "reason": "no_registered_canaries_for_session",
                    "registered_canary_count": 0,
                },
            )

        return _canary_nimbus_score(
            critic_input=critic_input,
            records=records,
            config=self._config,
        )


class LearnedInfoNCENimbusCritic:
    def __init__(self, model: NimbusInfoNCEModel, confidence: float) -> None:
        _validate_probability(confidence, "confidence")
        self._model = model
        self._confidence = confidence
        self._records_by_session_id: dict[str, tuple[CanaryRecord, ...]] = {}
        self._candidate_contexts_by_session_id: dict[str, tuple[NimbusRuntimeCandidateContext, ...]] = {}

    def register_canary_records(self, session_id: str, records: tuple[CanaryRecord, ...]) -> None:
        if session_id == "":
            raise NimbusDetectorError("session_id must not be empty.")
        if len(records) == 0:
            return
        records_by_id = {record.canary_id: record for record in self._records_by_session_id.get(session_id, ())}
        for record in records:
            records_by_id[record.canary_id] = record
        self._records_by_session_id[session_id] = tuple(records_by_id.values())

    def register_candidate_contexts(
        self,
        session_id: str,
        contexts: tuple[NimbusRuntimeCandidateContext, ...],
    ) -> None:
        if session_id == "":
            raise NimbusDetectorError("session_id must not be empty.")
        if len(contexts) == 0:
            return
        contexts_by_id = {
            context.context_id: context for context in self._candidate_contexts_by_session_id.get(session_id, ())
        }
        for context in contexts:
            if len(context.negative_context_texts) != self._model.negative_count:
                raise NimbusDetectorError(
                    f"candidate context '{context.context_id}' must include {self._model.negative_count} negatives."
                )
            contexts_by_id[context.context_id] = context
        self._candidate_contexts_by_session_id[session_id] = tuple(contexts_by_id.values())

    def destroy_session(self, session_id: str) -> None:
        self._records_by_session_id.pop(session_id, None)
        self._candidate_contexts_by_session_id.pop(session_id, None)

    def clear(self) -> None:
        self._records_by_session_id.clear()
        self._candidate_contexts_by_session_id.clear()

    def score_turn(self, critic_input: NimbusCriticInput) -> NimbusCriticScore:
        candidate_contexts = self._candidate_contexts_by_session_id.get(critic_input.session_id, ())
        if len(candidate_contexts) > 0:
            return _learned_infonce_nimbus_score(
                critic_input=critic_input,
                contexts=candidate_contexts,
                model=self._model,
                confidence=self._confidence,
                runtime_context_source="registered_candidate_contexts",
            )
        records = self._records_by_session_id.get(critic_input.session_id, ())
        if len(records) == 0:
            return NimbusCriticScore(
                estimated_leakage_bits=0.0,
                confidence=self._confidence,
                evidence={
                    "critic_kind": "learned_infonce_beta",
                    "critic_version": self._model.model_id,
                    "paper_faithful_learned_critic": False,
                    "promotion_status": "learned_runtime_beta_not_promotable",
                    "deterministic_fallback": False,
                    "runtime_context_source": "registered_canary_records",
                    "reason": "no_registered_canaries_for_session",
                    "registered_canary_count": 0,
                    "registered_candidate_context_count": 0,
                },
            )

        return _learned_infonce_nimbus_score(
            critic_input=critic_input,
            contexts=tuple(
                _candidate_context_from_canary_record(record, self._model.negative_count) for record in records
            ),
            model=self._model,
            confidence=self._confidence,
            runtime_context_source="registered_canary_records",
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

    def clear(self) -> None:
        self._store.clear()


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
                    "critic_kind": "unscored",
                    "paper_faithful_learned_critic": False,
                    "promotion_status": "unscored_runtime_precondition_missing",
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
        budget_action = _budget_action(
            budget_fraction=budget_fraction,
            warn_threshold=self._config.warn_threshold,
            sanitize_threshold=self._config.sanitize_threshold,
            block_threshold=self._config.block_threshold,
        )
        action_floor = _canary_critic_action_floor(critic_score.evidence)
        recommended_action = highest_action((budget_action, action_floor))
        return DetectorResult(
            detector_name="nimbus",
            component=DetectorComponent.NIMBUS,
            score=budget_fraction,
            confidence=critic_score.confidence,
            recommended_action=recommended_action,
            capability_required=None,
            capability_status=CapabilityStatus.ACTIVE,
            evidence={
                "reason": _nimbus_reason(
                    recommended_action=recommended_action,
                    budget_action=budget_action,
                    action_floor=action_floor,
                ),
                "turn_index": turn.turn_index,
                "turn_count": updated_state.turn_count,
                "turn_estimated_leakage_bits": critic_score.estimated_leakage_bits,
                "cumulative_estimated_leakage_bits": updated_state.cumulative_estimated_leakage_bits,
                "budget_bits": self._config.budget_bits,
                "budget_fraction": budget_fraction,
                "budget_recommended_action": budget_action.value,
                "action_floor": action_floor.value,
                "warn_threshold": self._config.warn_threshold,
                "sanitize_threshold": self._config.sanitize_threshold,
                "block_threshold": self._config.block_threshold,
                "critic_version": self._config.critic_version,
                "critic_kind": _optional_string(critic_score.evidence, "critic_kind", "unknown"),
                "paper_faithful_learned_critic": _optional_bool(
                    critic_score.evidence,
                    "paper_faithful_learned_critic",
                    False,
                ),
                "promotion_status": _optional_string(
                    critic_score.evidence,
                    "promotion_status",
                    "unknown",
                ),
                "critic_evidence": critic_score.evidence,
            },
            latency_ms=_elapsed_ms(started_at),
        )

    def destroy_session(self, session_id: str) -> None:
        self._state_store.destroy(session_id)

    def clear(self) -> None:
        self._state_store.clear()


class NimbusToolEgressDetector:
    def __init__(self, config: NimbusConfig, critic: NimbusCritic, state_store: NimbusStateStore) -> None:
        self._config = config
        self._critic = critic
        self._state_store = state_store

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        started_at = time.perf_counter()
        if len(turn.tool_calls) == 0:
            return DetectorResult(
                detector_name="nimbus_tool_egress",
                component=DetectorComponent.NIMBUS,
                score=0.0,
                confidence=1.0,
                recommended_action=Action.ALLOW,
                capability_required=None,
                capability_status=CapabilityStatus.ACTIVE,
                evidence={
                    "reason": "no_tool_arguments_to_score",
                    "turn_index": turn.turn_index,
                    "tool_call_count": 0,
                    "critic_version": self._config.critic_version,
                    "critic_kind": "canary",
                    "paper_faithful_learned_critic": False,
                    "promotion_status": "deterministic_canary_beta",
                },
                latency_ms=_elapsed_ms(started_at),
            )

        secret_context_handle = resolve_secret_context_handle(turn)
        if secret_context_handle is None:
            return DetectorResult(
                detector_name="nimbus_tool_egress",
                component=DetectorComponent.NIMBUS,
                score=0.0,
                confidence=0.0,
                recommended_action=Action.ALLOW,
                capability_required="secret_context_handle",
                capability_status=CapabilityStatus.UNAVAILABLE,
                evidence={
                    "capability_reason": "no_secret_context_handle",
                    "turn_index": turn.turn_index,
                    "tool_call_count": len(turn.tool_calls),
                    "critic_kind": "unavailable",
                    "paper_faithful_learned_critic": False,
                    "promotion_status": "unscored_runtime_precondition_missing",
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
                output_text=_serialized_tool_call_arguments(turn.tool_calls),
                secret_context_handle=secret_context_handle,
                messages=turn.messages,
                sensitive_spans=turn.sensitive_spans,
                prior_state=prior_state,
            )
        )
        if not isinstance(critic_score, NimbusCriticScore):
            raise NimbusDetectorError("critic.score_turn must return NimbusCriticScore.")
        if critic_score.estimated_leakage_bits == 0.0:
            return DetectorResult(
                detector_name="nimbus_tool_egress",
                component=DetectorComponent.NIMBUS,
                score=min(1.0, prior_state.cumulative_estimated_leakage_bits / self._config.budget_bits),
                confidence=critic_score.confidence,
                recommended_action=Action.ALLOW,
                capability_required=None,
                capability_status=CapabilityStatus.ACTIVE,
                evidence={
                    "reason": "no_tool_argument_leakage_detected",
                    "turn_index": turn.turn_index,
                    "tool_call_count": len(turn.tool_calls),
                    "turn_estimated_leakage_bits": 0.0,
                    "cumulative_estimated_leakage_bits": prior_state.cumulative_estimated_leakage_bits,
                    "budget_bits": self._config.budget_bits,
                    "budget_fraction": min(
                        1.0,
                        prior_state.cumulative_estimated_leakage_bits / self._config.budget_bits,
                    ),
                    "critic_version": self._config.critic_version,
                    "critic_kind": _optional_string(critic_score.evidence, "critic_kind", "unknown"),
                    "paper_faithful_learned_critic": _optional_bool(
                        critic_score.evidence,
                        "paper_faithful_learned_critic",
                        False,
                    ),
                    "promotion_status": _optional_string(
                        critic_score.evidence,
                        "promotion_status",
                        "unknown",
                    ),
                    "critic_evidence": critic_score.evidence,
                },
                latency_ms=_elapsed_ms(started_at),
            )

        new_cumulative_bits = prior_state.cumulative_estimated_leakage_bits + critic_score.estimated_leakage_bits
        updated_state = self._state_store.update(
            session_id=turn.session_id,
            update=NimbusStateUpdate(
                turn_estimated_leakage_bits=critic_score.estimated_leakage_bits,
                new_cumulative_bits=new_cumulative_bits,
            ),
        )
        budget_fraction = min(1.0, updated_state.cumulative_estimated_leakage_bits / self._config.budget_bits)
        budget_action = _budget_action(
            budget_fraction=budget_fraction,
            warn_threshold=self._config.warn_threshold,
            sanitize_threshold=self._config.sanitize_threshold,
            block_threshold=self._config.block_threshold,
        )
        recommended_action = highest_action((budget_action, Action.BLOCK))
        return DetectorResult(
            detector_name="nimbus_tool_egress",
            component=DetectorComponent.NIMBUS,
            score=budget_fraction,
            confidence=critic_score.confidence,
            recommended_action=recommended_action,
            capability_required=None,
            capability_status=CapabilityStatus.ACTIVE,
            evidence={
                "reason": "nimbus_tool_argument_leakage_pre_dispatch_block",
                "turn_index": turn.turn_index,
                "turn_count": updated_state.turn_count,
                "tool_call_count": len(turn.tool_calls),
                "turn_estimated_leakage_bits": critic_score.estimated_leakage_bits,
                "cumulative_estimated_leakage_bits": updated_state.cumulative_estimated_leakage_bits,
                "budget_bits": self._config.budget_bits,
                "budget_fraction": budget_fraction,
                "budget_recommended_action": budget_action.value,
                "pre_dispatch_action_floor": Action.BLOCK.value,
                "warn_threshold": self._config.warn_threshold,
                "sanitize_threshold": self._config.sanitize_threshold,
                "block_threshold": self._config.block_threshold,
                "critic_version": self._config.critic_version,
                "critic_kind": _optional_string(critic_score.evidence, "critic_kind", "unknown"),
                "paper_faithful_learned_critic": _optional_bool(
                    critic_score.evidence,
                    "paper_faithful_learned_critic",
                    False,
                ),
                "promotion_status": _optional_string(
                    critic_score.evidence,
                    "promotion_status",
                    "unknown",
                ),
                "critic_evidence": critic_score.evidence,
            },
            latency_ms=_elapsed_ms(started_at),
        )


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
        "paper_faithful_learned_critic": False,
        "promotion_status": "deterministic_canary_beta",
        "deterministic_fallback": True,
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


def _learned_infonce_nimbus_score(
    critic_input: NimbusCriticInput,
    contexts: tuple[NimbusRuntimeCandidateContext, ...],
    model: NimbusInfoNCEModel,
    confidence: float,
    runtime_context_source: str,
) -> NimbusCriticScore:
    state_text = " ".join(message.content for message in critic_input.messages)
    scored_contexts = tuple(
        (
            context,
            score_nimbus_infonce_runtime_candidate(
                model=model,
                output_text=critic_input.output_text,
                state_text=state_text,
                true_context_text=context.positive_context_text,
                negative_context_texts=context.negative_context_texts,
            ),
        )
        for context in contexts
    )
    if len(scored_contexts) == 0:
        raise NimbusDetectorError("learned InfoNCE runtime scoring requires at least one candidate context.")
    selected_context, selected_score = max(scored_contexts, key=lambda item: item[1].estimated_leakage_bits)
    evidence: dict[str, JsonValue] = {
        "critic_kind": "learned_infonce_beta",
        "critic_version": model.model_id,
        "paper_faithful_learned_critic": False,
        "promotion_status": "learned_runtime_beta_not_promotable",
        "deterministic_fallback": False,
        "runtime_context_source": runtime_context_source,
        "reason": "runtime_learned_infonce_beta_score",
        "registered_candidate_context_count": len(contexts),
        "selected_context_id": selected_context.context_id,
        "selected_context_sha256": canary_sha256(selected_context.positive_context_text),
        "selected_credential_type": selected_context.credential_type,
        "selected_context_source": selected_context.source,
        "positive_probability": selected_score.positive_probability,
        "nce_loss_bits": selected_score.nce_loss_bits,
        "positive_rank": selected_score.positive_rank,
        "candidate_count": selected_score.candidate_count,
        "estimated_leakage_bits": selected_score.estimated_leakage_bits,
        "feature_names": list(model.feature_names),
        "feature_values": list(selected_score.feature_values),
        "feature_weights": list(model.weights),
        "diagnostic_only_feature_names": list(NIMBUS_INFONCE_DIAGNOSTIC_ONLY_FEATURE_NAMES),
        "state_token_overlap_policy": "diagnostic_only_not_current_turn_leakage",
        "model_source_corpus_sha256": model.source_corpus_sha256,
        "model_training_record_count": model.training_record_count,
        "model_training_split_group_count": model.training_split_group_count,
    }
    return NimbusCriticScore(
        estimated_leakage_bits=selected_score.estimated_leakage_bits,
        confidence=confidence,
        evidence=evidence,
    )


def _candidate_context_from_canary_record(record: CanaryRecord, negative_count: int) -> NimbusRuntimeCandidateContext:
    slot_name = _safe_canary_slot_name(record)
    return NimbusRuntimeCandidateContext(
        context_id=record.canary_id,
        credential_type=record.credential_type,
        positive_context_text=f"{record.credential_type} credential context {slot_name} contains {record.value}",
        negative_context_texts=_runtime_negative_context_texts(record, negative_count),
        source=record.source,
    )


def _runtime_negative_context_texts(record: CanaryRecord, count: int) -> tuple[str, ...]:
    if count < 1:
        raise NimbusDetectorError("runtime negative context count must be positive.")
    slot_name = _safe_canary_slot_name(record)
    return tuple(
        f"{record.credential_type} decoy context {slot_name} contains safe-decoy-marker-{index:02d}-5813"
        for index in range(count)
    )


def _safe_canary_slot_name(record: CanaryRecord) -> str:
    slot_name = record.metadata.get("slot_name")
    if isinstance(slot_name, str) and slot_name != "":
        return slot_name
    return "runtime-slot"


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


def _serialized_tool_call_arguments(tool_calls: tuple[ToolCall, ...]) -> str:
    payload = [
        {
            "tool_call_index": index,
            "tool_name": tool_call.name,
            "arguments": tool_call.arguments,
        }
        for index, tool_call in enumerate(tool_calls)
    ]
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


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
            "critic_kind": "unavailable",
            "paper_faithful_learned_critic": False,
            "promotion_status": "unscored_runtime_precondition_missing",
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


def _canary_critic_action_floor(evidence: dict[str, JsonValue]) -> Action:
    if evidence.get("critic_kind") != "canary":
        return Action.ALLOW
    partial_match_count = evidence.get("partial_match_count")
    if isinstance(partial_match_count, int) and partial_match_count > 0:
        return Action.SANITIZE
    return Action.ALLOW


def _nimbus_reason(recommended_action: Action, budget_action: Action, action_floor: Action) -> str:
    if recommended_action != budget_action and action_floor == Action.SANITIZE:
        return "nimbus_canary_partial_overlap_sanitize"
    return _budget_reason(recommended_action)


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


def _optional_bool(evidence: dict[str, JsonValue], field_name: str, fallback: bool) -> bool:
    value = evidence.get(field_name)
    if isinstance(value, bool):
        return value
    return fallback


def _optional_string(evidence: dict[str, JsonValue], field_name: str, fallback: str) -> str:
    value = evidence.get(field_name)
    if isinstance(value, str) and value != "":
        return value
    return fallback


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
