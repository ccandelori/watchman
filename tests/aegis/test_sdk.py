import unittest

from aegis.audit.memory import InMemoryAuditSink
from aegis.core.contracts import Action, CapabilityMode, Message, ModelInfo
from aegis.core.orchestrator import AegisRuntime, RuntimeRequest
from aegis.detectors.activation import ActivationUnavailableDetector
from aegis.policy.engine import SeverityPolicyEngine
from aegis.providers.mock import MockModelProvider
from aegis.sdk.runtime import evaluate_turn


class SdkRuntimeTest(unittest.TestCase):
    def test_evaluate_turn_uses_shared_runtime_orchestrator(self) -> None:
        runtime = AegisRuntime(
            turn_annotators=(),
            pre_generation_detectors=(ActivationUnavailableDetector(),),
            post_generation_detectors=(),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=InMemoryAuditSink(),
            model_provider=MockModelProvider(default_content="sdk output"),
        )
        request = RuntimeRequest(
            trace_id="trace-sdk",
            session_id="session-sdk",
            turn_index=1,
            capability_mode=CapabilityMode.BLACK_BOX,
            model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
            messages=(Message(role="user", content="hello"),),
            tool_calls=(),
            sensitive_spans=(),
            metadata={},
        )

        response = evaluate_turn(runtime=runtime, request=request)

        self.assertEqual("sdk output", response.output_text)
        self.assertEqual(Action.ALLOW, response.policy_decision.final_action)


if __name__ == "__main__":
    unittest.main()
