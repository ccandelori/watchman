from __future__ import annotations

from aegis.core.contracts import Action, CapabilityStatus, DetectorComponent, DetectorResult, NormalizedTurn
from aegis.core.orchestrator import ModelResponse


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
