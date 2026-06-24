from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
_AUDIT_ALLOWED_SENSITIVE_SPAN_KINDS = frozenset(("honeytoken",))
_AUDIT_REDACTION_MARKER = "[REDACTED_SENSITIVE]"


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
            "normalized_turn": _audit_safe_normalized_turn(self.normalized_turn),
            "detector_results": [result.to_dict() for result in self.detector_results],
            "policy_decision": self.policy_decision.to_dict(),
            "latency_ms": self.latency_ms,
            "created_at": self.created_at,
        }


def _audit_safe_sensitive_span(span: SensitiveSpan) -> dict[str, JsonValue]:
    return {
        "kind": span.kind,
        "source": span.source,
        "char_start": span.char_start,
        "char_end": span.char_end,
        "token_start": span.token_start,
        "token_end": span.token_end,
        "identifier": span.identifier,
        "metadata": _audit_safe_sensitive_span_metadata(span.metadata),
    }


def _audit_safe_sensitive_span_metadata(metadata: dict[str, JsonValue]) -> dict[str, JsonValue]:
    safe_keys = ("sha256", "honeytoken_sha256", "credential_type", "source", "slot_name", "turn_planted")
    return {key: metadata[key] for key in safe_keys if key in metadata}


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


def _audit_safe_normalized_turn(turn: NormalizedTurn) -> dict[str, JsonValue]:
    return {
        "trace_id": turn.trace_id,
        "session_id": turn.session_id,
        "turn_index": turn.turn_index,
        "capability_mode": turn.capability_mode.value,
        "model": turn.model.to_dict(),
        "message_count": len(turn.messages),
        "messages": _audit_safe_messages(turn),
        "tool_call_count": len(turn.tool_calls),
        "tool_calls": _audit_safe_tool_calls(turn),
        "sensitive_span_count": len(turn.sensitive_spans),
        "sensitive_spans": [_audit_safe_sensitive_span(span) for span in turn.sensitive_spans],
        "metadata_key_count": len(turn.metadata),
        "metadata": _audit_safe_turn_metadata(turn.metadata),
    }


def _audit_safe_messages(turn: NormalizedTurn) -> list[JsonValue]:
    messages: list[JsonValue] = []
    for index, message in enumerate(turn.messages):
        messages.append(
            {
                "role": message.role,
                "content": _audit_safe_message_content(
                    message=message,
                    message_index=index,
                    sensitive_spans=turn.sensitive_spans,
                ),
            }
        )
    return messages


def _audit_safe_turn_metadata(metadata: dict[str, JsonValue]) -> dict[str, JsonValue]:
    safe_keys = (
        "dp_honey_canary_count",
        "dp_honey_canary_ids",
        "test_seed_canary_count",
        "test_seed_canary_ids",
    )
    return {key: metadata[key] for key in safe_keys if key in metadata}


def _audit_safe_message_content(
    message: Message,
    message_index: int,
    sensitive_spans: tuple[SensitiveSpan, ...],
) -> str:
    blocked_spans = tuple(
        span
        for span in sensitive_spans
        if span.kind not in _AUDIT_ALLOWED_SENSITIVE_SPAN_KINDS or span.metadata.get("audit_redact") is True
    )
    if len(blocked_spans) == 0:
        return message.content
    content = message.content
    redaction_ranges: list[tuple[int, int]] = []
    for span in blocked_spans:
        span_message_index = span.metadata.get("message_index")
        if isinstance(span_message_index, int) and span_message_index != message_index:
            continue
        if span.char_start is None or span.char_end is None:
            return _AUDIT_REDACTION_MARKER
        if span.char_start < 0 or span.char_end <= span.char_start or span.char_end > len(content):
            return _AUDIT_REDACTION_MARKER
        redaction_ranges.append((span.char_start, span.char_end))
    return _redact_ranges(content=content, ranges=tuple(redaction_ranges))


def _audit_safe_tool_calls(turn: NormalizedTurn) -> list[JsonValue]:
    blocked_spans = tuple(span for span in turn.sensitive_spans if span.kind not in _AUDIT_ALLOWED_SENSITIVE_SPAN_KINDS)
    if len(blocked_spans) == 0:
        return [
            {
                "name": tool_call.name,
                "argument_key_count": len(tool_call.arguments),
            }
            for tool_call in turn.tool_calls
        ]
    return [
        {
            "name": tool_call.name,
            "argument_key_count": len(tool_call.arguments),
            "arguments": {
                "redacted": True,
                "reason": "non_honeytoken_sensitive_span_present",
            },
        }
        for tool_call in turn.tool_calls
    ]


def _redact_ranges(content: str, ranges: tuple[tuple[int, int], ...]) -> str:
    if len(ranges) == 0:
        return content
    redacted = content
    for start, end in sorted(ranges, reverse=True):
        redacted = redacted[:start] + _AUDIT_REDACTION_MARKER + redacted[end:]
    return redacted
