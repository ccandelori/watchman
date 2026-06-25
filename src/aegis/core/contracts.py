from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
_AUDIT_ALLOWED_SENSITIVE_SPAN_KINDS = frozenset(("honeytoken",))
_AUDIT_REDACTION_MARKER = "[REDACTED_SENSITIVE]"
_AUDIT_RUNTIME_EVIDENCE_SCHEMA_VERSION = "aegis.audit_runtime_evidence/v1"
_AUDIT_POLICY_MODE = "severity"
_AUDIT_ARTIFACT_HASH_KEYS = (
    "runtime_model_sha256",
    "release_gate_report_sha256",
    "feature_vector_sha256",
    "rendered_prompt_sha256",
    "certification_manifest_sha256",
    "certification_report_sha256",
)
_AUDIT_CIFT_KEYS = (
    "certification_id",
    "certification_mode",
    "runtime_model_sha256",
    "release_gate_report_sha256",
    "runtime_model_bundle_id",
    "feature_source",
)
_DETECTOR_VERSION_BY_NAME = {
    "activation_unavailable": "activation-unavailable-v1",
    "provider_egress_guard": "provider-egress-guard-v1",
    "text_canary": "text-canary-v1",
    "encoded_canary": "encoded-canary-v1",
    "noop_canary": "noop-canary-v1",
}


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
    model_response_metadata: dict[str, JsonValue] = field(default_factory=dict)

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
            "model_response_metadata": self.model_response_metadata,
            "runtime_evidence": _audit_runtime_evidence(
                turn=self.normalized_turn,
                detector_results=self.detector_results,
                policy_decision=self.policy_decision,
                latency_ms=self.latency_ms,
                model_response_metadata=self.model_response_metadata,
            ),
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


def _audit_safe_normalized_turn(turn: NormalizedTurn) -> dict[str, JsonValue]:
    return {
        "trace_id": turn.trace_id,
        "session_id": turn.session_id,
        "turn_index": turn.turn_index,
        "capability_mode": turn.capability_mode.value,
        "model": turn.model.to_dict(),
        "messages": _audit_safe_messages(turn),
        "tool_calls": _audit_safe_tool_calls(turn),
        "sensitive_spans": [span.to_dict() for span in turn.sensitive_spans],
        "metadata": turn.metadata,
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
        return [tool_call.to_dict() for tool_call in turn.tool_calls]
    return [
        {
            "name": tool_call.name,
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


def _audit_runtime_evidence(
    turn: NormalizedTurn,
    detector_results: tuple[DetectorResult, ...],
    policy_decision: PolicyDecision,
    latency_ms: float,
    model_response_metadata: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    provider_state = _audit_provider_state(turn=turn, model_response_metadata=model_response_metadata)
    return {
        "schema_version": _AUDIT_RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "policy_mode": _AUDIT_POLICY_MODE,
        "final_action": policy_decision.final_action.value,
        "provider_state": provider_state,
        "credential_slot_status": _audit_credential_slot_status(turn),
        "detector_versions": _audit_detector_versions(detector_results),
        "detector_latency_ms": _audit_detector_latencies(detector_results),
        "artifact_hashes": _audit_artifact_hashes(detector_results),
        "cift": _audit_cift_summary(detector_results),
        "fail_closed_events": _audit_fail_closed_events(
            turn=turn,
            policy_decision=policy_decision,
            provider_state=provider_state,
        ),
        "latency_ms": latency_ms,
    }


def _audit_provider_state(
    turn: NormalizedTurn,
    model_response_metadata: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    provider = model_response_metadata.get("provider")
    if not isinstance(provider, str) or provider == "":
        provider = turn.model.provider
    model_id = model_response_metadata.get("model_id")
    if not isinstance(model_id, str) or model_id == "":
        model_id = turn.model.model_id
    state: dict[str, JsonValue] = {
        "status": "skipped" if provider == "skipped" else "completed",
        "provider": provider,
        "model_id": model_id,
    }
    reason = model_response_metadata.get("reason")
    if isinstance(reason, str) and reason != "":
        state["reason"] = reason
    return state


def _audit_credential_slot_status(turn: NormalizedTurn) -> str:
    detection = _json_object_or_empty(turn.metadata.get("aegis_credential_slot_detection"))
    status = detection.get("status")
    if isinstance(status, str) and status != "":
        return status
    return "unknown"


def _audit_detector_versions(detector_results: tuple[DetectorResult, ...]) -> dict[str, JsonValue]:
    versions: dict[str, JsonValue] = {}
    for result in detector_results:
        versions[result.detector_name] = _audit_detector_version(result)
    return versions


def _audit_detector_version(result: DetectorResult) -> str:
    evidence = result.evidence
    version = evidence.get("detector_version")
    if isinstance(version, str) and version != "":
        return version
    critic_version = evidence.get("critic_version")
    if result.component == DetectorComponent.NIMBUS and isinstance(critic_version, str) and critic_version != "":
        return critic_version
    runtime_model_bundle_id = evidence.get("runtime_model_bundle_id")
    if (
        result.component == DetectorComponent.CIFT
        and isinstance(runtime_model_bundle_id, str)
        and runtime_model_bundle_id != ""
    ):
        return runtime_model_bundle_id
    mapped = _DETECTOR_VERSION_BY_NAME.get(result.detector_name)
    if mapped is not None:
        return mapped
    return f"{result.component.value}-unknown"


def _audit_detector_latencies(detector_results: tuple[DetectorResult, ...]) -> dict[str, JsonValue]:
    return {result.detector_name: result.latency_ms for result in detector_results}


def _audit_artifact_hashes(detector_results: tuple[DetectorResult, ...]) -> dict[str, JsonValue]:
    hashes: dict[str, JsonValue] = {}
    for result in detector_results:
        for key in _AUDIT_ARTIFACT_HASH_KEYS:
            value = result.evidence.get(key)
            if _is_audit_safe_hash(value):
                hashes[key] = value
    return hashes


def _audit_cift_summary(detector_results: tuple[DetectorResult, ...]) -> dict[str, JsonValue]:
    summary: dict[str, JsonValue] = {}
    for result in detector_results:
        if result.component != DetectorComponent.CIFT:
            continue
        for key in _AUDIT_CIFT_KEYS:
            value = result.evidence.get(key)
            if isinstance(value, str) and value != "":
                summary[key] = value
    return summary


def _audit_fail_closed_events(
    turn: NormalizedTurn,
    policy_decision: PolicyDecision,
    provider_state: dict[str, JsonValue],
) -> list[JsonValue]:
    events: list[JsonValue] = []
    detection = _json_object_or_empty(turn.metadata.get("aegis_credential_slot_detection"))
    if detection.get("status") == "ambiguous_protected_workflow" or detection.get("fail_closed") is True:
        events.append(
            {
                "kind": "ambiguous_protected_workflow",
                "credential_slot_status": detection.get("status"),
                "final_action": policy_decision.final_action.value,
            }
        )
    if provider_state.get("status") == "skipped" and provider_state.get("reason") == "pre_generation_policy_block":
        events.append(
            {
                "kind": "pre_generation_policy_block",
                "provider_status": "skipped",
                "final_action": policy_decision.final_action.value,
                "triggered_detectors": list(policy_decision.triggered_detectors),
            }
        )
    return events


def _json_object_or_empty(value: JsonValue) -> dict[str, JsonValue]:
    if isinstance(value, dict):
        return value
    return {}


def _is_audit_safe_hash(value: JsonValue) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) == 64:
        return all(character in "0123456789abcdef" for character in value)
    if value.startswith("sha256:") and len(value) == len("sha256:") + 64:
        return all(character in "0123456789abcdef" for character in value[len("sha256:") :])
    return False
