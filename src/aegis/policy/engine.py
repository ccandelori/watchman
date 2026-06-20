from __future__ import annotations

from aegis.core.contracts import Action, DetectorResult, PolicyDecision

_ACTION_SEVERITY: dict[Action, int] = {
    Action.ALLOW: 0,
    Action.WARN: 1,
    Action.SANITIZE: 2,
    Action.BLOCK: 3,
    Action.ESCALATE: 4,
}


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

        highest_severity = max(_ACTION_SEVERITY[result.recommended_action] for result in detector_results)
        final_action = next(action for action, severity in _ACTION_SEVERITY.items() if severity == highest_severity)
        triggering_results = tuple(
            result for result in detector_results if _ACTION_SEVERITY[result.recommended_action] == highest_severity
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
