from __future__ import annotations

import time

from aegis.core.contracts import Action, CapabilityStatus, DetectorComponent, DetectorResult, JsonValue, NormalizedTurn
from aegis.core.orchestrator import ModelResponse
from detect.dp_honey import scanner as dp_honey_scanner

_CONFIDENCE_VALUES: dict[str, float] = {
    "high": 0.95,
    "medium": 0.65,
    "low": 0.35,
}


class DPHoneyDetectorError(ValueError):
    """Raised when a DP-HONEY detector is configured incorrectly."""


class DPHoneyTextDetector:
    """Aegis detector adapter for DP-HONEY's registry-derived text scanner."""

    def __init__(self, detector_name: str = "dp_honey_text", include_low_confidence: bool = True) -> None:
        if detector_name == "":
            raise DPHoneyDetectorError("detector_name must not be empty.")
        self.detector_name = detector_name
        self._include_low_confidence = include_low_confidence

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        started_at = time.perf_counter()
        if model_response is None:
            return DetectorResult(
                detector_name=self.detector_name,
                component=DetectorComponent.DP_HONEY,
                score=0.0,
                confidence=1.0,
                recommended_action=Action.ALLOW,
                capability_required=None,
                capability_status=CapabilityStatus.DEGRADED,
                evidence={
                    "reason": "model_response_required",
                    "session_id": turn.session_id,
                },
                latency_ms=_elapsed_ms(started_at),
            )

        findings = tuple(_finding_to_evidence(finding) for finding in dp_honey_scanner.scan(model_response.output_text))
        if not self._include_low_confidence:
            findings = tuple(finding for finding in findings if finding["confidence"] != "low")

        if len(findings) == 0:
            return DetectorResult(
                detector_name=self.detector_name,
                component=DetectorComponent.DP_HONEY,
                score=0.0,
                confidence=1.0,
                recommended_action=Action.ALLOW,
                capability_required=None,
                capability_status=CapabilityStatus.ACTIVE,
                evidence={
                    "reason": "no_secret_shape_detected",
                    "session_id": turn.session_id,
                    "match_count": 0,
                    "matches": [],
                },
                latency_ms=_elapsed_ms(started_at),
            )

        max_confidence = max(_confidence_value(str(finding["confidence"])) for finding in findings)
        matches: list[JsonValue] = [dict(finding) for finding in findings]
        return DetectorResult(
            detector_name=self.detector_name,
            component=DetectorComponent.DP_HONEY,
            score=max_confidence,
            confidence=max_confidence,
            recommended_action=Action.SANITIZE,
            capability_required=None,
            capability_status=CapabilityStatus.ACTIVE,
            evidence={
                "reason": "secret_shape_detected",
                "session_id": turn.session_id,
                "match_count": len(findings),
                "matches": matches,
            },
            latency_ms=_elapsed_ms(started_at),
        )


def _finding_to_evidence(finding: dict[str, int | str]) -> dict[str, JsonValue]:
    return {
        "format": str(finding["format"]),
        "confidence": str(finding["confidence"]),
        "char_start": int(finding["start"]),
        "char_end": int(finding["end"]),
    }


def _confidence_value(confidence: str) -> float:
    return _CONFIDENCE_VALUES.get(confidence, 0.0)


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0
