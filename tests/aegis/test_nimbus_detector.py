from __future__ import annotations

import unittest

from aegis.core.contracts import (
    Action,
    CapabilityMode,
    CapabilityStatus,
    DetectorComponent,
    Message,
    ModelInfo,
    NormalizedTurn,
)
from aegis.core.orchestrator import ModelResponse
from aegis.detectors.canary import CanaryRecord, InMemoryCanaryRegistry, canary_sha256
from aegis.detectors.nimbus import NimbusDetectorError, NimbusLeakageDetector


class NimbusLeakageDetectorTest(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self) -> None:
        registry = _registry("sk-hny-testCanaryValue123")

        with self.assertRaisesRegex(NimbusDetectorError, "detector_name"):
            NimbusLeakageDetector(
                detector_name="",
                registry=registry,
                partial_match_threshold=0.5,
                decay=0.5,
                warn_threshold=0.4,
                escalate_threshold=0.9,
                confidence=0.7,
            )
        with self.assertRaisesRegex(NimbusDetectorError, "escalate_threshold"):
            NimbusLeakageDetector(
                detector_name="nimbus",
                registry=registry,
                partial_match_threshold=0.5,
                decay=0.5,
                warn_threshold=0.9,
                escalate_threshold=0.4,
                confidence=0.7,
            )

    def test_missing_model_response_degrades_without_updating_state(self) -> None:
        detector = _detector()

        result = detector.evaluate(turn=_runtime_turn("session-a"), model_response=None)

        self.assertEqual(DetectorComponent.NIMBUS, result.component)
        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual("model_response_required", result.evidence["reason"])
        self.assertEqual(0.0, detector.state("session-a").score)

    def test_clean_output_allows_and_keeps_session_score_zero(self) -> None:
        detector = _detector()

        result = detector.evaluate(
            turn=_runtime_turn("session-a"),
            model_response=ModelResponse(output_text="No protected values are present.", metadata={}),
        )

        self.assertEqual(CapabilityStatus.ACTIVE, result.capability_status)
        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual("no_cumulative_leakage_detected", result.evidence["reason"])
        self.assertEqual(0.0, result.score)
        self.assertEqual(0, result.evidence["exact_signal"]["match_count"])
        self.assertEqual(0.0, detector.state("session-a").score)

    def test_exact_canary_leak_escalates_with_audit_safe_evidence(self) -> None:
        canary_value = "sk-hny-testCanaryValue123"
        detector = _detector()

        result = detector.evaluate(
            turn=_runtime_turn("session-a"),
            model_response=ModelResponse(output_text=f"Leaked value: {canary_value}", metadata={}),
        )

        self.assertEqual(Action.ESCALATE, result.recommended_action)
        self.assertEqual(1.0, result.score)
        self.assertEqual("cumulative_leakage_budget_exhausted", result.evidence["reason"])
        self.assertEqual(1, result.evidence["exact_signal"]["match_count"])
        self.assertEqual(canary_sha256(canary_value), result.evidence["exact_signal"]["matches"][0]["sha256"])
        self.assertNotIn(canary_value, str(result.to_dict()))

    def test_partial_leaks_accumulate_by_session(self) -> None:
        detector = _detector()
        first_result = detector.evaluate(
            turn=_runtime_turn("session-a"),
            model_response=ModelResponse(output_text="Partial leak: sk-hny-testCan", metadata={}),
        )
        second_result = detector.evaluate(
            turn=_runtime_turn("session-a"),
            model_response=ModelResponse(output_text="Partial leak: sk-hny-testCan", metadata={}),
        )

        self.assertEqual(Action.WARN, first_result.recommended_action)
        self.assertEqual(Action.ESCALATE, second_result.recommended_action)
        self.assertGreater(second_result.score, first_result.score)
        self.assertEqual("partial_canary_overlap_detected", second_result.evidence["encoded_signal"]["reason"])

    def test_sessions_are_isolated(self) -> None:
        detector = _detector()
        detector.evaluate(
            turn=_runtime_turn("session-a"),
            model_response=ModelResponse(output_text="Partial leak: sk-hny-testCan", metadata={}),
        )

        result = detector.evaluate(
            turn=_runtime_turn("session-b"),
            model_response=ModelResponse(output_text="No protected values are present.", metadata={}),
        )

        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertGreater(detector.state("session-a").score, 0.0)
        self.assertEqual(0.0, detector.state("session-b").score)


def _detector() -> NimbusLeakageDetector:
    return NimbusLeakageDetector(
        detector_name="nimbus",
        registry=_registry("sk-hny-testCanaryValue123"),
        partial_match_threshold=0.5,
        decay=0.5,
        warn_threshold=0.4,
        escalate_threshold=0.75,
        confidence=0.7,
    )


def _registry(value: str) -> InMemoryCanaryRegistry:
    return InMemoryCanaryRegistry(
        records=(
            CanaryRecord(
                canary_id="hny_api_key_123",
                credential_type="api_key",
                value=value,
                sha256=canary_sha256(value),
                source="dp_honey",
                metadata={"scenario": "nimbus_unit_test"},
            ),
        )
    )


def _runtime_turn(session_id: str) -> NormalizedTurn:
    return NormalizedTurn(
        trace_id="trace-nimbus",
        session_id=session_id,
        turn_index=1,
        capability_mode=CapabilityMode.BLACK_BOX,
        model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
        messages=(Message(role="user", content="summarize the incident"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={},
    )


if __name__ == "__main__":
    unittest.main()
