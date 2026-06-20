from __future__ import annotations

from aegis.core.contracts import (
    Action,
    CapabilityMode,
    CapabilityStatus,
    DetectorComponent,
    DetectorResult,
    NormalizedTurn,
)
from aegis.core.orchestrator import ModelResponse


class ActivationUnavailableDetector:
    detector_name = "activation_unavailable"

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        if turn.capability_mode == CapabilityMode.SELF_HOSTED_INTROSPECTION:
            status = CapabilityStatus.ACTIVE
            reason = "self_hosted_introspection_available"
        else:
            status = CapabilityStatus.UNAVAILABLE
            reason = "black_box_mode"

        return DetectorResult(
            detector_name=self.detector_name,
            component=DetectorComponent.CIFT,
            score=0.0,
            confidence=1.0,
            recommended_action=Action.ALLOW,
            capability_required=CapabilityMode.SELF_HOSTED_INTROSPECTION.value,
            capability_status=status,
            evidence={
                "reason": reason,
                "model_id": turn.model.model_id,
                "selected_device": turn.model.selected_device,
            },
            latency_ms=0.0,
        )
