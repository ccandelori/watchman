from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from aegis.core.contracts import (
    AuditEvent,
    CapabilityMode,
    DetectorResult,
    JsonValue,
    Message,
    ModelInfo,
    NormalizedTurn,
    PolicyDecision,
    SensitiveSpan,
    ToolCall,
)


@dataclass(frozen=True)
class RuntimeRequest:
    trace_id: str
    session_id: str
    turn_index: int
    capability_mode: CapabilityMode
    model: ModelInfo
    messages: tuple[Message, ...]
    tool_calls: tuple[ToolCall, ...]
    sensitive_spans: tuple[SensitiveSpan, ...]
    metadata: dict[str, JsonValue]


@dataclass(frozen=True)
class ModelResponse:
    output_text: str
    metadata: dict[str, JsonValue]


@dataclass(frozen=True)
class AegisRuntimeResponse:
    output_text: str
    detector_results: tuple[DetectorResult, ...]
    policy_decision: PolicyDecision
    audit_event: AuditEvent


class Detector(Protocol):
    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        """Evaluate a normalized turn and optional model response."""


class PolicyEngine(Protocol):
    def decide(self, detector_results: tuple[DetectorResult, ...]) -> PolicyDecision:
        """Combine detector outputs into one final policy decision."""


class AuditSink(Protocol):
    def write(self, event: AuditEvent) -> None:
        """Persist or publish an audit event."""


class ModelProvider(Protocol):
    def generate(self, turn: NormalizedTurn) -> ModelResponse:
        """Generate model output for a normalized turn."""


class TurnAnnotator(Protocol):
    def annotate(self, turn: NormalizedTurn) -> NormalizedTurn:
        """Attach derived runtime metadata before detector stages run."""


class AegisRuntime:
    def __init__(
        self,
        turn_annotators: tuple[TurnAnnotator, ...],
        pre_generation_detectors: tuple[Detector, ...],
        post_generation_detectors: tuple[Detector, ...],
        session_detectors: tuple[Detector, ...],
        policy_engine: PolicyEngine,
        audit_sink: AuditSink,
        model_provider: ModelProvider,
    ) -> None:
        self._turn_annotators = turn_annotators
        self._pre_generation_detectors = pre_generation_detectors
        self._post_generation_detectors = post_generation_detectors
        self._session_detectors = session_detectors
        self._policy_engine = policy_engine
        self._audit_sink = audit_sink
        self._model_provider = model_provider

    def evaluate_turn(self, request: RuntimeRequest) -> AegisRuntimeResponse:
        started_at = time.perf_counter()
        turn = NormalizedTurn(
            trace_id=request.trace_id,
            session_id=request.session_id,
            turn_index=request.turn_index,
            capability_mode=request.capability_mode,
            model=request.model,
            messages=request.messages,
            tool_calls=request.tool_calls,
            sensitive_spans=request.sensitive_spans,
            metadata=request.metadata,
        )
        for annotator in self._turn_annotators:
            turn = annotator.annotate(turn)

        pre_generation_results = tuple(detector.evaluate(turn, None) for detector in self._pre_generation_detectors)
        model_response = self._model_provider.generate(turn)
        post_generation_results = tuple(
            detector.evaluate(turn, model_response) for detector in self._post_generation_detectors
        )
        session_results = tuple(detector.evaluate(turn, model_response) for detector in self._session_detectors)
        detector_results = pre_generation_results + post_generation_results + session_results
        policy_decision = self._policy_engine.decide(detector_results)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        audit_event = AuditEvent(
            trace_id=turn.trace_id,
            session_id=turn.session_id,
            turn_index=turn.turn_index,
            normalized_turn=turn,
            detector_results=detector_results,
            policy_decision=policy_decision,
            latency_ms=latency_ms,
            created_at=datetime.now(UTC).isoformat(),
        )
        self._audit_sink.write(audit_event)

        output_text = policy_decision.sanitized_output
        if output_text is None:
            output_text = model_response.output_text

        return AegisRuntimeResponse(
            output_text=output_text,
            detector_results=detector_results,
            policy_decision=policy_decision,
            audit_event=audit_event,
        )
