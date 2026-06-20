import unittest

from aegis.audit.memory import InMemoryAuditSink
from aegis.core.contracts import (
    Action,
    CapabilityMode,
    CapabilityStatus,
    DetectorComponent,
    DetectorResult,
    Message,
    ModelInfo,
    NormalizedTurn,
)
from aegis.core.orchestrator import AegisRuntime, ModelResponse, RuntimeRequest
from aegis.detectors.activation import ActivationUnavailableDetector
from aegis.detectors.canary import NoopCanaryDetector
from aegis.policy.engine import SeverityPolicyEngine
from aegis.providers.mock import MockModelProvider


class OutputAwareDetector:
    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        if model_response is None:
            raise AssertionError("post-generation detector must receive model output.")
        return DetectorResult(
            detector_name="output_aware",
            component=DetectorComponent.TEXT_CANARY,
            score=0.4,
            confidence=1.0,
            recommended_action=Action.WARN,
            capability_required=None,
            capability_status=CapabilityStatus.ACTIVE,
            evidence={"output_text": model_response.output_text},
            latency_ms=0.1,
        )


class AegisRuntimeTest(unittest.TestCase):
    def test_mock_turn_produces_detector_results_policy_decision_and_audit_event(self) -> None:
        audit_sink = InMemoryAuditSink()
        runtime = AegisRuntime(
            pre_generation_detectors=(ActivationUnavailableDetector(),),
            post_generation_detectors=(NoopCanaryDetector(),),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=audit_sink,
            model_provider=MockModelProvider(default_content="hello from mock"),
        )
        request = RuntimeRequest(
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

        response = runtime.evaluate_turn(request)

        self.assertEqual("hello from mock", response.output_text)
        self.assertEqual(Action.ALLOW, response.policy_decision.final_action)
        self.assertEqual(2, len(response.detector_results))
        self.assertEqual(1, len(audit_sink.recent(limit=10)))
        self.assertEqual("trace-1", audit_sink.recent(limit=1)[0].trace_id)

    def test_black_box_runtime_emits_activation_unavailable_result(self) -> None:
        runtime = AegisRuntime(
            pre_generation_detectors=(ActivationUnavailableDetector(),),
            post_generation_detectors=(),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=InMemoryAuditSink(),
            model_provider=MockModelProvider(default_content="ok"),
        )
        request = RuntimeRequest(
            trace_id="trace-2",
            session_id="session-2",
            turn_index=1,
            capability_mode=CapabilityMode.BLACK_BOX,
            model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
            messages=(Message(role="user", content="hello"),),
            tool_calls=(),
            sensitive_spans=(),
            metadata={},
        )

        response = runtime.evaluate_turn(request)
        activation_result = response.detector_results[0]

        self.assertEqual(DetectorComponent.CIFT, activation_result.component)
        self.assertEqual(CapabilityStatus.UNAVAILABLE, activation_result.capability_status)
        self.assertEqual("black_box_mode", activation_result.evidence["reason"])

    def test_post_generation_detector_receives_model_output_before_policy(self) -> None:
        runtime = AegisRuntime(
            pre_generation_detectors=(),
            post_generation_detectors=(OutputAwareDetector(),),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=InMemoryAuditSink(),
            model_provider=MockModelProvider(default_content="generated text"),
        )
        request = RuntimeRequest(
            trace_id="trace-3",
            session_id="session-3",
            turn_index=1,
            capability_mode=CapabilityMode.BLACK_BOX,
            model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
            messages=(Message(role="user", content="hello"),),
            tool_calls=(),
            sensitive_spans=(),
            metadata={},
        )

        response = runtime.evaluate_turn(request)

        self.assertEqual(Action.WARN, response.policy_decision.final_action)
        self.assertEqual("generated text", response.detector_results[0].evidence["output_text"])


if __name__ == "__main__":
    unittest.main()
