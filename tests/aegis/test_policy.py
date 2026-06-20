import unittest

from aegis.core.contracts import Action, CapabilityStatus, DetectorComponent, DetectorResult
from aegis.policy.engine import SeverityPolicyEngine


def _result(name: str, action: Action, score: float) -> DetectorResult:
    return DetectorResult(
        detector_name=name,
        component=DetectorComponent.TOOL_SCANNER,
        score=score,
        confidence=1.0,
        recommended_action=action,
        capability_required=None,
        capability_status=CapabilityStatus.ACTIVE,
        evidence={"reason": name},
        latency_ms=0.5,
    )


class SeverityPolicyEngineTest(unittest.TestCase):
    def test_all_allow_results_produce_allow_decision(self) -> None:
        decision = SeverityPolicyEngine().decide((_result("safe", Action.ALLOW, 0.0),))

        self.assertEqual(Action.ALLOW, decision.final_action)
        self.assertEqual((), decision.triggered_detectors)
        self.assertEqual(0.0, decision.risk_score)

    def test_highest_severity_result_wins(self) -> None:
        decision = SeverityPolicyEngine().decide(
            (
                _result("warning", Action.WARN, 0.2),
                _result("blocker", Action.BLOCK, 0.9),
                _result("sanitizer", Action.SANITIZE, 0.5),
            )
        )

        self.assertEqual(Action.BLOCK, decision.final_action)
        self.assertEqual(("blocker",), decision.triggered_detectors)
        self.assertEqual(0.9, decision.risk_score)

    def test_tied_highest_severity_keeps_all_triggering_detectors(self) -> None:
        decision = SeverityPolicyEngine().decide(
            (
                _result("canary", Action.ESCALATE, 0.8),
                _result("ledger", Action.ESCALATE, 0.7),
                _result("warning", Action.WARN, 0.3),
            )
        )

        self.assertEqual(Action.ESCALATE, decision.final_action)
        self.assertEqual(("canary", "ledger"), decision.triggered_detectors)
        self.assertEqual(0.8, decision.risk_score)


if __name__ == "__main__":
    unittest.main()
