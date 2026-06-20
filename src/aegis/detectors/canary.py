from __future__ import annotations

import hashlib
from dataclasses import dataclass

from aegis.core.contracts import Action, CapabilityStatus, DetectorComponent, DetectorResult, JsonValue, NormalizedTurn
from aegis.core.orchestrator import ModelResponse


class CanaryDetectorError(ValueError):
    """Raised when canary detector configuration is invalid."""


@dataclass(frozen=True)
class CanaryRecord:
    canary_id: str
    credential_type: str
    value: str
    sha256: str
    source: str
    metadata: dict[str, JsonValue]


@dataclass(frozen=True)
class CanaryMatch:
    canary_id: str
    credential_type: str
    sha256: str
    source: str
    char_start: int
    char_end: int
    metadata: dict[str, JsonValue]


class InMemoryCanaryRegistry:
    def __init__(self, records: tuple[CanaryRecord, ...]) -> None:
        self._records = _validate_records(records)

    def records(self) -> tuple[CanaryRecord, ...]:
        return self._records


class TextCanaryDetector:
    def __init__(self, detector_name: str, registry: InMemoryCanaryRegistry) -> None:
        if detector_name == "":
            raise CanaryDetectorError("detector_name must not be empty.")
        self.detector_name = detector_name
        self._registry = registry

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        if model_response is None:
            return DetectorResult(
                detector_name=self.detector_name,
                component=DetectorComponent.TEXT_CANARY,
                score=0.0,
                confidence=1.0,
                recommended_action=Action.ALLOW,
                capability_required=None,
                capability_status=CapabilityStatus.DEGRADED,
                evidence={
                    "reason": "model_response_required",
                    "session_id": turn.session_id,
                },
                latency_ms=0.0,
            )

        matches = _scan_text_for_canaries(text=model_response.output_text, records=self._registry.records())
        if len(matches) == 0:
            return DetectorResult(
                detector_name=self.detector_name,
                component=DetectorComponent.TEXT_CANARY,
                score=0.0,
                confidence=1.0,
                recommended_action=Action.ALLOW,
                capability_required=None,
                capability_status=CapabilityStatus.ACTIVE,
                evidence={
                    "reason": "no_canary_leak_detected",
                    "session_id": turn.session_id,
                    "match_count": 0,
                    "matches": [],
                },
                latency_ms=0.0,
            )

        match_values: list[JsonValue] = [_match_to_dict(match) for match in matches]
        return DetectorResult(
            detector_name=self.detector_name,
            component=DetectorComponent.TEXT_CANARY,
            score=1.0,
            confidence=1.0,
            recommended_action=Action.ESCALATE,
            capability_required=None,
            capability_status=CapabilityStatus.ACTIVE,
            evidence={
                "reason": "registered_canary_leak_detected",
                "session_id": turn.session_id,
                "match_count": len(matches),
                "matches": match_values,
            },
            latency_ms=0.0,
        )


class NoopCanaryDetector:
    detector_name = "noop_canary"

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        return DetectorResult(
            detector_name=self.detector_name,
            component=DetectorComponent.DP_HONEY,
            score=0.0,
            confidence=1.0,
            recommended_action=Action.ALLOW,
            capability_required=None,
            capability_status=CapabilityStatus.DEGRADED,
            evidence={
                "reason": "canary_registry_not_configured",
                "session_id": turn.session_id,
            },
            latency_ms=0.0,
        )


def canary_sha256(value: str) -> str:
    if value == "":
        raise CanaryDetectorError("canary value must not be empty.")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_records(records: tuple[CanaryRecord, ...]) -> tuple[CanaryRecord, ...]:
    seen_ids: set[str] = set()
    for record in records:
        _validate_record(record)
        if record.canary_id in seen_ids:
            raise CanaryDetectorError(f"duplicate canary_id '{record.canary_id}'.")
        seen_ids.add(record.canary_id)
    return records


def _validate_record(record: CanaryRecord) -> None:
    for field_name, value in (
        ("canary_id", record.canary_id),
        ("credential_type", record.credential_type),
        ("value", record.value),
        ("sha256", record.sha256),
        ("source", record.source),
    ):
        if value == "":
            raise CanaryDetectorError(f"CanaryRecord field '{field_name}' must not be empty.")
    expected_hash = canary_sha256(record.value)
    if record.sha256 != expected_hash:
        raise CanaryDetectorError(f"CanaryRecord '{record.canary_id}' has sha256 that does not match its value.")


def _scan_text_for_canaries(text: str, records: tuple[CanaryRecord, ...]) -> tuple[CanaryMatch, ...]:
    matches: list[CanaryMatch] = []
    for record in records:
        start_index = text.find(record.value)
        while start_index != -1:
            end_index = start_index + len(record.value)
            matches.append(
                CanaryMatch(
                    canary_id=record.canary_id,
                    credential_type=record.credential_type,
                    sha256=record.sha256,
                    source=record.source,
                    char_start=start_index,
                    char_end=end_index,
                    metadata=record.metadata,
                )
            )
            start_index = text.find(record.value, end_index)
    return tuple(matches)


def _match_to_dict(match: CanaryMatch) -> dict[str, JsonValue]:
    return {
        "canary_id": match.canary_id,
        "credential_type": match.credential_type,
        "sha256": match.sha256,
        "source": match.source,
        "char_start": match.char_start,
        "char_end": match.char_end,
        "metadata": match.metadata,
    }
