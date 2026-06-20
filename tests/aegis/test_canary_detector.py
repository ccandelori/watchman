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
from aegis.detectors.canary import (
    CanaryRecord,
    InMemoryCanaryRegistry,
    TextCanaryDetector,
    canary_sha256,
)
from aegis.policy.engine import SeverityPolicyEngine


class CanaryAwareModelProvider:
    def __init__(self, output_text: str) -> None:
        self._output_text = output_text

    def generate(self, turn: NormalizedTurn) -> ModelResponse:
        return ModelResponse(output_text=self._output_text, metadata={"provider": "canary_aware_fixture"})


class CiftWarningDetector:
    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        return DetectorResult(
            detector_name="cift_fixture",
            component=DetectorComponent.CIFT,
            score=0.35,
            confidence=0.75,
            recommended_action=Action.WARN,
            capability_required="self_hosted_introspection",
            capability_status=CapabilityStatus.ACTIVE,
            evidence={"reason": "fixture_cift_review_band", "session_id": turn.session_id},
            latency_ms=0.0,
        )


def _record(value: str) -> CanaryRecord:
    return CanaryRecord(
        canary_id="hny_api_key_123",
        credential_type="api_key",
        value=value,
        sha256=canary_sha256(value),
        source="dp_honey_lite",
        metadata={"scenario": "unit_test"},
    )


def _request() -> RuntimeRequest:
    return RuntimeRequest(
        trace_id="trace-canary",
        session_id="session-canary",
        turn_index=1,
        capability_mode=CapabilityMode.BLACK_BOX,
        model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
        messages=(Message(role="user", content="summarize the incident"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={},
    )


class TextCanaryDetectorTest(unittest.TestCase):
    def test_exact_canary_leak_escalates_with_audit_safe_evidence(self) -> None:
        canary_value = "sk-hny-testCanaryValue123"
        registry = InMemoryCanaryRegistry(records=(_record(canary_value),))
        detector = TextCanaryDetector(detector_name="text_canary", registry=registry)
        output = f"The credential is {canary_value}. Rotate it immediately."

        result = detector.evaluate(turn=_runtime_turn(), model_response=ModelResponse(output_text=output, metadata={}))

        self.assertEqual("text_canary", result.detector_name)
        self.assertEqual(DetectorComponent.TEXT_CANARY, result.component)
        self.assertEqual(Action.ESCALATE, result.recommended_action)
        self.assertEqual(CapabilityStatus.ACTIVE, result.capability_status)
        self.assertEqual(1.0, result.score)
        self.assertEqual(1, result.evidence["match_count"])
        self.assertEqual("hny_api_key_123", result.evidence["matches"][0]["canary_id"])
        self.assertEqual(canary_sha256(canary_value), result.evidence["matches"][0]["sha256"])
        self.assertEqual("api_key", result.evidence["matches"][0]["credential_type"])
        self.assertEqual(output.index(canary_value), result.evidence["matches"][0]["char_start"])
        self.assertEqual(output.index(canary_value) + len(canary_value), result.evidence["matches"][0]["char_end"])
        self.assertNotIn(canary_value, str(result.to_dict()))

    def test_safe_output_allows_without_matches(self) -> None:
        registry = InMemoryCanaryRegistry(records=(_record("sk-hny-testCanaryValue123"),))
        detector = TextCanaryDetector(detector_name="text_canary", registry=registry)

        result = detector.evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text="No protected values are included.", metadata={}),
        )

        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual(0.0, result.score)
        self.assertEqual(0, result.evidence["match_count"])
        self.assertEqual([], result.evidence["matches"])

    def test_runtime_escalates_when_model_output_contains_registered_canary(self) -> None:
        canary_value = "sk-hny-testCanaryValue123"
        audit_sink = InMemoryAuditSink()
        runtime = AegisRuntime(
            pre_generation_detectors=(CiftWarningDetector(),),
            post_generation_detectors=(
                TextCanaryDetector(
                    detector_name="text_canary",
                    registry=InMemoryCanaryRegistry(records=(_record(canary_value),)),
                ),
            ),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=audit_sink,
            model_provider=CanaryAwareModelProvider(output_text=f"Leaked value: {canary_value}"),
        )

        response = runtime.evaluate_turn(_request())

        self.assertEqual(Action.ESCALATE, response.policy_decision.final_action)
        self.assertEqual(("text_canary",), response.policy_decision.triggered_detectors)
        self.assertEqual(2, len(response.detector_results))
        self.assertEqual(1, len(audit_sink.recent(limit=10)))
        self.assertNotIn(canary_value, str(response.audit_event.to_dict()))


def _runtime_turn() -> NormalizedTurn:
    request = _request()
    return NormalizedTurn(
        trace_id=request.trace_id,
        session_id=request.session_id,
        turn_index=request.turn_index,
        capability_mode=request.capability_mode,
        model=request.model,
        messages=request.messages,
        tool_calls=request.tool_calls,
        sensitive_spans=request.sensitive_spans,
        metadata=request.metadata,
    )


if __name__ == "__main__":
    unittest.main()
