from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class Action(StrEnum):
    ALLOW = "allow"
    WARN = "warn"
    SANITIZE = "sanitize"
    BLOCK = "block"
    ESCALATE = "escalate"


class CapabilityMode(StrEnum):
    SELF_HOSTED_INTROSPECTION = "self_hosted_introspection"
    BLACK_BOX = "black_box"
    SDK_EMBEDDED = "sdk_embedded"
    OFFLINE_EVAL = "offline_eval"


class CapabilityStatus(StrEnum):
    ACTIVE = "active"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


class DetectorComponent(StrEnum):
    CIFT = "cift"
    DP_HONEY = "dp_honey"
    TEXT_CANARY = "text_canary"
    TOOL_SCANNER = "tool_scanner"
    NIMBUS = "nimbus"
    CAPABILITY = "capability"


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, JsonValue]

    def to_dict(self) -> dict[str, JsonValue]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class SensitiveSpan:
    kind: str
    source: str
    char_start: int | None
    char_end: int | None
    token_start: int | None
    token_end: int | None
    identifier: str | None
    metadata: dict[str, JsonValue]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind,
            "source": self.source,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "identifier": self.identifier,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model_id: str
    revision: str | None
    selected_device: str | None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "revision": self.revision,
            "selected_device": self.selected_device,
        }


@dataclass(frozen=True)
class NormalizedTurn:
    trace_id: str
    session_id: str
    turn_index: int
    capability_mode: CapabilityMode
    model: ModelInfo
    messages: tuple[Message, ...]
    tool_calls: tuple[ToolCall, ...]
    sensitive_spans: tuple[SensitiveSpan, ...]
    metadata: dict[str, JsonValue]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "capability_mode": self.capability_mode.value,
            "model": self.model.to_dict(),
            "messages": [message.to_dict() for message in self.messages],
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
            "sensitive_spans": [span.to_dict() for span in self.sensitive_spans],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DetectorResult:
    detector_name: str
    component: DetectorComponent
    score: float
    confidence: float
    recommended_action: Action
    capability_required: str | None
    capability_status: CapabilityStatus
    evidence: dict[str, JsonValue]
    latency_ms: float

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "detector_name": self.detector_name,
            "component": self.component.value,
            "score": self.score,
            "confidence": self.confidence,
            "recommended_action": self.recommended_action.value,
            "capability_required": self.capability_required,
            "capability_status": self.capability_status.value,
            "evidence": self.evidence,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class PolicyDecision:
    final_action: Action
    reason: str
    triggered_detectors: tuple[str, ...]
    risk_score: float
    sanitized_output: str | None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "final_action": self.final_action.value,
            "reason": self.reason,
            "triggered_detectors": list(self.triggered_detectors),
            "risk_score": self.risk_score,
            "sanitized_output": self.sanitized_output,
        }


@dataclass(frozen=True)
class AuditEvent:
    trace_id: str
    session_id: str
    turn_index: int
    normalized_turn: NormalizedTurn
    detector_results: tuple[DetectorResult, ...]
    policy_decision: PolicyDecision
    latency_ms: float
    created_at: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "normalized_turn": self.normalized_turn.to_dict(),
            "detector_results": [result.to_dict() for result in self.detector_results],
            "policy_decision": self.policy_decision.to_dict(),
            "latency_ms": self.latency_ms,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class CapabilityReport:
    capability_mode: CapabilityMode
    active_detectors: tuple[str, ...]
    unavailable_detectors: dict[str, str]
    model: ModelInfo

    def to_dict(self) -> dict[str, JsonValue]:
        unavailable_detectors: dict[str, JsonValue] = {
            detector_name: reason for detector_name, reason in self.unavailable_detectors.items()
        }
        return {
            "capability_mode": self.capability_mode.value,
            "active_detectors": list(self.active_detectors),
            "unavailable_detectors": unavailable_detectors,
            "model": self.model.to_dict(),
        }
