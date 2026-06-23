from __future__ import annotations

import time
from dataclasses import dataclass

from aegis.core.action_severity import highest_action
from aegis.core.contracts import Action, CapabilityStatus, DetectorComponent, DetectorResult, JsonValue, NormalizedTurn
from aegis.core.orchestrator import ModelResponse
from aegis.detectors.canary import EncodedCanaryDetector, InMemoryCanaryRegistry, TextCanaryDetector


class NimbusDetectorError(ValueError):
    """Raised when NIMBUS-lite detector configuration is invalid."""


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
    if value < 0.0 or value > 1.0:
        raise NimbusDetectorError(f"{field_name} must be in [0.0, 1.0].")


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0
