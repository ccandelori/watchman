import base64
import codecs
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
    CanaryDetectorError,
    CanaryRecord,
    EncodedCanaryDetector,
    InMemoryCanaryRegistry,
    NoopCanaryDetector,
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
    def test_empty_detector_name_is_rejected(self) -> None:
        registry = InMemoryCanaryRegistry(records=(_record("sk-hny-testCanaryValue123"),))

        with self.assertRaisesRegex(CanaryDetectorError, "detector_name"):
            TextCanaryDetector(detector_name="", registry=registry)

    def test_missing_model_response_degrades(self) -> None:
        registry = InMemoryCanaryRegistry(records=(_record("sk-hny-testCanaryValue123"),))
        detector = TextCanaryDetector(detector_name="text_canary", registry=registry)

        result = detector.evaluate(turn=_runtime_turn(), model_response=None)

        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual("model_response_required", result.evidence["reason"])

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


class EncodedCanaryDetectorTest(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self) -> None:
        registry = InMemoryCanaryRegistry(records=(_record("sk-hny-testCanaryValue123"),))

        with self.assertRaisesRegex(CanaryDetectorError, "detector_name"):
            EncodedCanaryDetector(detector_name="", registry=registry, partial_match_threshold=0.8)
        with self.assertRaisesRegex(CanaryDetectorError, "partial_match_threshold"):
            EncodedCanaryDetector(detector_name="encoded_canary", registry=registry, partial_match_threshold=1.1)

    def test_missing_model_response_degrades(self) -> None:
        registry = InMemoryCanaryRegistry(records=(_record("sk-hny-testCanaryValue123"),))
        detector = EncodedCanaryDetector(
            detector_name="encoded_canary",
            registry=registry,
            partial_match_threshold=0.8,
        )

        result = detector.evaluate(turn=_runtime_turn(), model_response=None)

        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual("model_response_required", result.evidence["reason"])

    def test_safe_output_allows_without_matches(self) -> None:
        registry = InMemoryCanaryRegistry(records=(_record("sk-hny-testCanaryValue123"),))
        detector = EncodedCanaryDetector(
            detector_name="encoded_canary",
            registry=registry,
            partial_match_threshold=0.8,
        )

        result = detector.evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text="No protected values are included.", metadata={}),
        )

        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual("no_encoded_canary_leak_detected", result.evidence["reason"])
        self.assertEqual(0, result.evidence["match_count"])

    def test_base64_canary_leak_escalates_when_exact_text_detector_is_dark(self) -> None:
        canary_value = "sk-hny-testCanaryValue123"
        encoded_value = base64.b64encode(canary_value.encode("utf-8")).decode("utf-8")
        registry = InMemoryCanaryRegistry(records=(_record(canary_value),))
        exact_detector = TextCanaryDetector(detector_name="text_canary", registry=registry)
        encoded_detector = EncodedCanaryDetector(
            detector_name="encoded_canary",
            registry=registry,
            partial_match_threshold=0.8,
        )
        output = f"The requested encoded value is {encoded_value}."

        exact_result = exact_detector.evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text=output, metadata={}),
        )
        encoded_result = encoded_detector.evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text=output, metadata={}),
        )

        self.assertEqual(Action.ALLOW, exact_result.recommended_action)
        self.assertEqual(Action.ESCALATE, encoded_result.recommended_action)
        self.assertEqual(DetectorComponent.TEXT_CANARY, encoded_result.component)
        self.assertEqual("encoded_canary_leak_detected", encoded_result.evidence["reason"])
        self.assertEqual(1, encoded_result.evidence["match_count"])
        self.assertEqual("base64", encoded_result.evidence["matches"][0]["encoding"])
        self.assertEqual(True, encoded_result.evidence["matches"][0]["exact"])
        self.assertEqual("hny_api_key_123", encoded_result.evidence["matches"][0]["canary_id"])
        self.assertEqual(canary_sha256(canary_value), encoded_result.evidence["matches"][0]["sha256"])
        self.assertNotIn(canary_value, str(encoded_result.to_dict()))

    def test_hex_rot13_and_leet_canary_leaks_are_detected(self) -> None:
        canary_value = "sk-hny-testCanaryValue123"
        registry = InMemoryCanaryRegistry(records=(_record(canary_value),))
        detector = EncodedCanaryDetector(
            detector_name="encoded_canary",
            registry=registry,
            partial_match_threshold=0.8,
        )
        encoded_values = {
            "hex": canary_value.encode("utf-8").hex(),
            "rot13": codecs.encode(canary_value, "rot_13"),
            "leet": "5k-hny-7357C4n4ryV4lu3123",
        }

        for encoding, encoded_value in encoded_values.items():
            with self.subTest(encoding=encoding):
                result = detector.evaluate(
                    turn=_runtime_turn(),
                    model_response=ModelResponse(output_text=f"Encoded: {encoded_value}", metadata={}),
                )

                self.assertEqual(Action.ESCALATE, result.recommended_action)
                self.assertEqual(encoding, result.evidence["matches"][0]["encoding"])

    def test_reverse_and_fragmented_canary_leaks_are_detected(self) -> None:
        canary_value = "sk-hny-testCanaryValue123"
        registry = InMemoryCanaryRegistry(records=(_record(canary_value),))
        detector = EncodedCanaryDetector(
            detector_name="encoded_canary",
            registry=registry,
            partial_match_threshold=0.8,
        )

        reverse_result = detector.evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text=f"Reversed: {canary_value[::-1]}", metadata={}),
        )
        fragmented_result = detector.evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text="Fragmented: sk-hny-test Canary Value123", metadata={}),
        )

        self.assertEqual(Action.ESCALATE, reverse_result.recommended_action)
        self.assertEqual("reverse", reverse_result.evidence["matches"][0]["encoding"])
        self.assertEqual(Action.ESCALATE, fragmented_result.recommended_action)
        self.assertEqual("fragmentation", fragmented_result.evidence["matches"][0]["encoding"])

    def test_larger_decoded_blob_containing_canary_is_detected(self) -> None:
        canary_value = "sk-hny-testCanaryValue123"
        registry = InMemoryCanaryRegistry(records=(_record(canary_value),))
        detector = EncodedCanaryDetector(
            detector_name="encoded_canary",
            registry=registry,
            partial_match_threshold=0.8,
        )
        encoded_blob = base64.b64encode(f"prefix::{canary_value}::suffix".encode()).decode("utf-8")

        result = detector.evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text=f"Blob: {encoded_blob}", metadata={}),
        )

        self.assertEqual(Action.ESCALATE, result.recommended_action)
        self.assertEqual("decoded_base64_blob", result.evidence["matches"][0]["encoding"])

    def test_partial_canary_overlap_sanitizes(self) -> None:
        canary_value = "sk-hny-testCanaryValue123"
        registry = InMemoryCanaryRegistry(records=(_record(canary_value),))
        detector = EncodedCanaryDetector(
            detector_name="encoded_canary",
            registry=registry,
            partial_match_threshold=0.5,
        )

        result = detector.evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text="Partial leak: sk-hny-testCan", metadata={}),
        )

        self.assertEqual(Action.SANITIZE, result.recommended_action)
        self.assertEqual("partial_canary_overlap_detected", result.evidence["reason"])
        self.assertEqual("partial", result.evidence["matches"][0]["encoding"])
        self.assertEqual(False, result.evidence["matches"][0]["exact"])

    def test_verbatim_canary_is_left_to_exact_text_detector(self) -> None:
        canary_value = "sk-hny-testCanaryValue123"
        registry = InMemoryCanaryRegistry(records=(_record(canary_value),))
        detector = EncodedCanaryDetector(
            detector_name="encoded_canary",
            registry=registry,
            partial_match_threshold=0.8,
        )

        result = detector.evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text=f"Raw value: {canary_value}", metadata={}),
        )

        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual(0, result.evidence["match_count"])

    def test_runtime_escalates_when_model_output_contains_encoded_canary(self) -> None:
        canary_value = "sk-hny-testCanaryValue123"
        encoded_value = base64.b64encode(canary_value.encode("utf-8")).decode("utf-8")
        registry = InMemoryCanaryRegistry(records=(_record(canary_value),))
        audit_sink = InMemoryAuditSink()
        runtime = AegisRuntime(
            pre_generation_detectors=(),
            post_generation_detectors=(
                TextCanaryDetector(detector_name="text_canary", registry=registry),
                EncodedCanaryDetector(
                    detector_name="encoded_canary",
                    registry=registry,
                    partial_match_threshold=0.8,
                ),
            ),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=audit_sink,
            model_provider=CanaryAwareModelProvider(output_text=f"Leaked value: {encoded_value}"),
        )

        response = runtime.evaluate_turn(_request())

        self.assertEqual(Action.ESCALATE, response.policy_decision.final_action)
        self.assertEqual(("encoded_canary",), response.policy_decision.triggered_detectors)
        self.assertEqual(2, len(response.detector_results))
        self.assertNotIn(canary_value, str(response.audit_event.to_dict()))


class NoopCanaryDetectorTest(unittest.TestCase):
    def test_noop_canary_detector_reports_unconfigured_boundary(self) -> None:
        result = NoopCanaryDetector().evaluate(turn=_runtime_turn(), model_response=None)

        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual(DetectorComponent.DP_HONEY, result.component)
        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual("canary_registry_not_configured", result.evidence["reason"])


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
