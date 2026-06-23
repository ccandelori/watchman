import unittest

from aegis.audit.memory import InMemoryAuditSink
from aegis.core.contracts import (
    Action,
    CapabilityMode,
    CapabilityStatus,
    DetectorComponent,
    DetectorResult,
    JsonValue,
    Message,
    ModelInfo,
    NormalizedTurn,
    SensitiveSpan,
    ToolCall,
)
from aegis.core.orchestrator import AegisRuntime, ModelResponse, RuntimeRequest
from aegis.detectors.activation import ActivationUnavailableDetector
from aegis.detectors.canary import NoopCanaryDetector
from aegis.detectors.egress import ProviderEgressGuardDetector
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


class MetadataAnnotator:
    def __init__(self, key: str, value: JsonValue) -> None:
        self._key = key
        self._value = value

    def annotate(self, turn: NormalizedTurn) -> NormalizedTurn:
        metadata = dict(turn.metadata)
        metadata[self._key] = self._value
        return NormalizedTurn(
            trace_id=turn.trace_id,
            session_id=turn.session_id,
            turn_index=turn.turn_index,
            capability_mode=turn.capability_mode,
            model=turn.model,
            messages=turn.messages,
            tool_calls=turn.tool_calls,
            sensitive_spans=turn.sensitive_spans,
            metadata=metadata,
        )


class MetadataDetector:
    def __init__(self, key: str) -> None:
        self._key = key

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        return DetectorResult(
            detector_name="metadata_detector",
            component=DetectorComponent.CAPABILITY,
            score=1.0 if turn.metadata.get(self._key) == "attached" else 0.0,
            confidence=1.0,
            recommended_action=Action.ALLOW,
            capability_required=None,
            capability_status=CapabilityStatus.ACTIVE,
            evidence={"observed_value": turn.metadata.get(self._key)},
            latency_ms=0.1,
        )


class FailingModelProvider:
    def generate(self, turn: NormalizedTurn) -> ModelResponse:
        raise AssertionError("model provider must not be called.")


class AegisRuntimeTest(unittest.TestCase):
    def test_mock_turn_produces_detector_results_policy_decision_and_audit_event(self) -> None:
        audit_sink = InMemoryAuditSink()
        runtime = AegisRuntime(
            turn_annotators=(),
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
            turn_annotators=(),
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
            turn_annotators=(),
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

    def test_turn_annotators_run_before_pre_generation_detectors_and_audit(self) -> None:
        runtime = AegisRuntime(
            turn_annotators=(MetadataAnnotator(key="derived_feature", value="attached"),),
            pre_generation_detectors=(MetadataDetector(key="derived_feature"),),
            post_generation_detectors=(),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=InMemoryAuditSink(),
            model_provider=MockModelProvider(default_content="ok"),
        )
        request = RuntimeRequest(
            trace_id="trace-annotator",
            session_id="session-annotator",
            turn_index=1,
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device="cpu"),
            messages=(Message(role="user", content="hello"),),
            tool_calls=(),
            sensitive_spans=(),
            metadata={},
        )

        response = runtime.evaluate_turn(request)

        self.assertEqual(1.0, response.detector_results[0].score)
        self.assertEqual("attached", response.detector_results[0].evidence["observed_value"])
        self.assertEqual("attached", response.audit_event.normalized_turn.metadata["derived_feature"])

    def test_pre_generation_block_skips_model_provider_and_writes_audit(self) -> None:
        audit_sink = InMemoryAuditSink()
        runtime = AegisRuntime(
            turn_annotators=(),
            pre_generation_detectors=(ProviderEgressGuardDetector(),),
            post_generation_detectors=(OutputAwareDetector(),),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=audit_sink,
            model_provider=FailingModelProvider(),
        )
        secret = "sk_live_raw_secret"
        request = RuntimeRequest(
            trace_id="trace-egress-block",
            session_id="session-egress-block",
            turn_index=1,
            capability_mode=CapabilityMode.BLACK_BOX,
            model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
            messages=(Message(role="system", content=f"Credential: {secret}"),),
            tool_calls=(ToolCall(name="send_secret", arguments={"token": secret}),),
            sensitive_spans=(
                SensitiveSpan(
                    kind="credential",
                    source="test",
                    char_start=len("Credential: "),
                    char_end=len("Credential: ") + len(secret),
                    token_start=None,
                    token_end=None,
                    identifier="cred-1",
                    metadata={},
                ),
            ),
            metadata={},
        )

        response = runtime.evaluate_turn(request)

        self.assertEqual("", response.output_text)
        self.assertEqual(Action.BLOCK, response.policy_decision.final_action)
        self.assertEqual(("provider_egress_guard",), response.policy_decision.triggered_detectors)
        self.assertEqual(1, len(response.detector_results))
        self.assertEqual(
            "blocked_sensitive_value_before_provider_egress",
            response.detector_results[0].evidence["reason"],
        )
        self.assertNotIn(secret, str(response.detector_results[0].evidence))
        self.assertEqual(1, len(audit_sink.recent(limit=10)))
        self.assertNotIn(secret, str(audit_sink.recent(limit=1)[0].to_dict()))
        audit_message = audit_sink.recent(limit=1)[0].to_dict()["normalized_turn"]["messages"][0]["content"]
        audit_tool_call = audit_sink.recent(limit=1)[0].to_dict()["normalized_turn"]["tool_calls"][0]
        self.assertIn("[REDACTED_SENSITIVE]", audit_message)
        self.assertEqual(
            {"redacted": True, "reason": "non_honeytoken_sensitive_span_present"},
            audit_tool_call["arguments"],
        )

    def test_provider_egress_guard_allows_honeytoken_spans(self) -> None:
        runtime = AegisRuntime(
            turn_annotators=(),
            pre_generation_detectors=(ProviderEgressGuardDetector(),),
            post_generation_detectors=(),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=InMemoryAuditSink(),
            model_provider=MockModelProvider(default_content="provider called"),
        )
        honeytoken = "ghp_fakeHoneytoken1234567890"
        request = RuntimeRequest(
            trace_id="trace-egress-allow",
            session_id="session-egress-allow",
            turn_index=1,
            capability_mode=CapabilityMode.BLACK_BOX,
            model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
            messages=(Message(role="system", content=f"Credential: {honeytoken}"),),
            tool_calls=(),
            sensitive_spans=(
                SensitiveSpan(
                    kind="honeytoken",
                    source="dp_honey_dev_proxy",
                    char_start=len("Credential: "),
                    char_end=len("Credential: ") + len(honeytoken),
                    token_start=None,
                    token_end=None,
                    identifier="hny-1",
                    metadata={"sha256": "not-needed-for-allow"},
                ),
            ),
            metadata={},
        )

        response = runtime.evaluate_turn(request)

        self.assertEqual("provider called", response.output_text)
        self.assertEqual(Action.ALLOW, response.policy_decision.final_action)
        self.assertEqual("no_blocked_sensitive_egress_detected", response.detector_results[0].evidence["reason"])
        self.assertEqual(1, response.detector_results[0].evidence["allowed_honeytoken_span_count"])


if __name__ == "__main__":
    unittest.main()
