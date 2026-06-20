import unittest

from aegis.core.contracts import (
    Action,
    AuditEvent,
    CapabilityMode,
    CapabilityReport,
    CapabilityStatus,
    DetectorComponent,
    DetectorResult,
    Message,
    ModelInfo,
    NormalizedTurn,
    PolicyDecision,
    SensitiveSpan,
    ToolCall,
)


class RuntimeContractsTest(unittest.TestCase):
    def test_normalized_turn_serializes_to_json_safe_dict(self) -> None:
        turn = NormalizedTurn(
            trace_id="trace-1",
            session_id="session-1",
            turn_index=1,
            capability_mode=CapabilityMode.BLACK_BOX,
            model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
            messages=(Message(role="user", content="hello"),),
            tool_calls=(ToolCall(name="send_email", arguments={"to": "team@example.com", "body": "hello"}),),
            sensitive_spans=(
                SensitiveSpan(
                    kind="honeytoken",
                    source="dp_honey",
                    char_start=10,
                    char_end=20,
                    token_start=None,
                    token_end=None,
                    identifier="canary-1",
                    metadata={"credential_family": "api_key"},
                ),
            ),
            metadata={"scenario_id": "benign_email"},
        )

        encoded = turn.to_dict()

        self.assertEqual("black_box", encoded["capability_mode"])
        self.assertEqual("mock-model", encoded["model"]["model_id"])
        self.assertEqual("send_email", encoded["tool_calls"][0]["name"])
        self.assertEqual("honeytoken", encoded["sensitive_spans"][0]["kind"])

    def test_detector_result_policy_decision_and_audit_event_serialize(self) -> None:
        turn = NormalizedTurn(
            trace_id="trace-1",
            session_id="session-1",
            turn_index=1,
            capability_mode=CapabilityMode.BLACK_BOX,
            model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
            messages=(Message(role="user", content="hello"),),
            tool_calls=(),
            sensitive_spans=(),
            metadata={},
        )
        detector_result = DetectorResult(
            detector_name="activation_unavailable",
            component=DetectorComponent.CIFT,
            score=0.0,
            confidence=1.0,
            recommended_action=Action.ALLOW,
            capability_required="self_hosted_introspection",
            capability_status=CapabilityStatus.UNAVAILABLE,
            evidence={"reason": "black_box_mode"},
            latency_ms=1.2,
        )
        decision = PolicyDecision(
            final_action=Action.ALLOW,
            reason="No detector requested intervention.",
            triggered_detectors=(),
            risk_score=0.0,
            sanitized_output=None,
        )
        event = AuditEvent(
            trace_id=turn.trace_id,
            session_id=turn.session_id,
            turn_index=turn.turn_index,
            normalized_turn=turn,
            detector_results=(detector_result,),
            policy_decision=decision,
            latency_ms=2.4,
            created_at="2026-06-20T12:00:00Z",
        )

        encoded = event.to_dict()

        self.assertEqual("activation_unavailable", encoded["detector_results"][0]["detector_name"])
        self.assertEqual("unavailable", encoded["detector_results"][0]["capability_status"])
        self.assertEqual("allow", encoded["policy_decision"]["final_action"])

    def test_capability_report_records_active_and_unavailable_detectors(self) -> None:
        report = CapabilityReport(
            capability_mode=CapabilityMode.BLACK_BOX,
            active_detectors=("noop_canary",),
            unavailable_detectors={"activation_probe": "black_box_mode"},
            model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
        )

        encoded = report.to_dict()

        self.assertEqual("black_box", encoded["capability_mode"])
        self.assertEqual(["noop_canary"], encoded["active_detectors"])
        self.assertEqual("black_box_mode", encoded["unavailable_detectors"]["activation_probe"])


if __name__ == "__main__":
    unittest.main()
