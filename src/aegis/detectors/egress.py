from __future__ import annotations

import hashlib
from dataclasses import dataclass

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

_ALLOWED_PROVIDER_EGRESS_KINDS = frozenset(("honeytoken",))


@dataclass(frozen=True)
class EgressGuardMatch:
    span_kind: str
    span_source: str
    identifier: str | None
    sha256: str | None
    message_role: str | None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "span_kind": self.span_kind,
            "span_source": self.span_source,
            "identifier": self.identifier,
            "sha256": self.sha256,
            "message_role": self.message_role,
        }


class ProviderEgressGuardDetector:
    detector_name = "provider_egress_guard"

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        matches = _blocked_egress_matches(turn)
        if len(matches) == 0:
            return DetectorResult(
                detector_name=self.detector_name,
                component=DetectorComponent.TOOL_SCANNER,
                score=0.0,
                confidence=1.0,
                recommended_action=Action.ALLOW,
                capability_required=None,
                capability_status=CapabilityStatus.ACTIVE,
                evidence={
                    "reason": "no_blocked_sensitive_egress_detected",
                    "checked_span_count": _blocked_span_count(turn.sensitive_spans),
                    "allowed_honeytoken_span_count": _allowed_span_count(turn.sensitive_spans),
                },
                latency_ms=0.0,
            )

        return DetectorResult(
            detector_name=self.detector_name,
            component=DetectorComponent.TOOL_SCANNER,
            score=1.0,
            confidence=1.0,
            recommended_action=Action.BLOCK,
            capability_required=None,
            capability_status=CapabilityStatus.ACTIVE,
            evidence={
                "reason": "blocked_sensitive_value_before_provider_egress",
                "match_count": len(matches),
                "matches": [match.to_dict() for match in matches],
            },
            latency_ms=0.0,
        )


def _blocked_egress_matches(turn: NormalizedTurn) -> tuple[EgressGuardMatch, ...]:
    matches: list[EgressGuardMatch] = []
    for span in turn.sensitive_spans:
        if span.kind in _ALLOWED_PROVIDER_EGRESS_KINDS:
            continue
        span_matches = _matches_for_span(span=span, messages=turn.messages)
        if len(span_matches) == 0:
            matches.append(
                EgressGuardMatch(
                    span_kind=span.kind,
                    span_source=span.source,
                    identifier=span.identifier,
                    sha256=_span_sha256(span),
                    message_role=None,
                )
            )
            continue
        matches.extend(span_matches)
    return tuple(matches)


def _matches_for_span(span: SensitiveSpan, messages: tuple[Message, ...]) -> tuple[EgressGuardMatch, ...]:
    if span.char_start is None or span.char_end is None:
        return ()
    if span.char_start < 0 or span.char_end <= span.char_start:
        return ()

    expected_sha256 = _span_sha256(span)
    matches: list[EgressGuardMatch] = []
    for message in messages:
        if span.char_end > len(message.content):
            continue
        candidate = message.content[span.char_start : span.char_end]
        if candidate == "":
            continue
        if expected_sha256 is not None and _sha256(candidate) != expected_sha256:
            continue
        matches.append(
            EgressGuardMatch(
                span_kind=span.kind,
                span_source=span.source,
                identifier=span.identifier,
                sha256=expected_sha256,
                message_role=message.role,
            )
        )
    return tuple(matches)


def _span_sha256(span: SensitiveSpan) -> str | None:
    value = span.metadata.get("sha256")
    if isinstance(value, str) and value != "":
        return value
    return None


def _blocked_span_count(spans: tuple[SensitiveSpan, ...]) -> int:
    return sum(1 for span in spans if span.kind not in _ALLOWED_PROVIDER_EGRESS_KINDS)


def _allowed_span_count(spans: tuple[SensitiveSpan, ...]) -> int:
    return sum(1 for span in spans if span.kind in _ALLOWED_PROVIDER_EGRESS_KINDS)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
