from __future__ import annotations

from aegis.core.action_severity import action_severity, highest_action
from aegis.core.contracts import Action, DetectorResult, PolicyDecision


class SeverityPolicyEngine:
    def decide(self, detector_results: tuple[DetectorResult, ...]) -> PolicyDecision:
        if len(detector_results) == 0:
            return PolicyDecision(
                final_action=Action.ALLOW,
                reason="No detectors were configured.",
                triggered_detectors=(),
                risk_score=0.0,
                sanitized_output=None,
            )

        final_action = highest_action(tuple(result.recommended_action for result in detector_results))
        highest_severity = action_severity(final_action)
        triggering_results = tuple(
            result for result in detector_results if action_severity(result.recommended_action) == highest_severity
        )
        if final_action == Action.ALLOW:
            triggered_detectors: tuple[str, ...] = ()
            reason = "No detector requested intervention."
        else:
            triggered_detectors = tuple(result.detector_name for result in triggering_results)
            reason = f"Selected {final_action.value} from highest-severity detector recommendation."

        return PolicyDecision(
            final_action=final_action,
            reason=reason,
            triggered_detectors=triggered_detectors,
            risk_score=max(result.score for result in triggering_results),
            sanitized_output=None,
        )
