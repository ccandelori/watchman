from __future__ import annotations

import unittest

from aegis.canaries.dp_honey import DPHoneyCanaryGenerator, build_dp_honey_ledger
from aegis.canaries.ledger import HoneytokenLedger, HoneytokenLedgerError, inject_honeytokens
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
from aegis.core.orchestrator import ModelResponse
from aegis.detectors.canary import TextCanaryDetector
from aegis.detectors.dp_honey import DPHoneyDetectorError, DPHoneyTextDetector
from aegis.policy.engine import SeverityPolicyEngine
from aegis.proxy.dp_honey import apply_dp_honey_auto_decoy
from detect.dp_honey import generate_honeytokens, get_format


def _runtime_turn() -> NormalizedTurn:
    return NormalizedTurn(
        trace_id="trace-dp-honey",
        session_id="session-dp-honey",
        turn_index=1,
        capability_mode=CapabilityMode.BLACK_BOX,
        model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
        messages=(Message(role="user", content="summarize the request"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={},
    )


def _non_dp_honey_result() -> DetectorResult:
    return DetectorResult(
        detector_name="tool_scanner",
        component=DetectorComponent.TOOL_SCANNER,
        score=0.5,
        confidence=1.0,
        recommended_action=Action.SANITIZE,
        capability_required=None,
        capability_status=CapabilityStatus.ACTIVE,
        evidence={"reason": "fixture"},
        latency_ms=0.0,
    )


class DPHoneyTextDetectorTest(unittest.TestCase):
    def test_invalid_detector_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(DPHoneyDetectorError, "detector_name"):
            DPHoneyTextDetector(detector_name="")

    def test_missing_model_response_degrades(self) -> None:
        result = DPHoneyTextDetector().evaluate(turn=_runtime_turn(), model_response=None)

        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual(DetectorComponent.DP_HONEY, result.component)
        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual("model_response_required", result.evidence["reason"])

    def test_registered_secret_shape_recommends_sanitize_without_echoing_match(self) -> None:
        token = generate_honeytokens("github-ghp", count=1, sample_seed=1)[0]
        output = f"leaked={token}"

        result = DPHoneyTextDetector().evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text=output, metadata={}),
        )

        self.assertEqual(Action.SANITIZE, result.recommended_action)
        self.assertEqual(DetectorComponent.DP_HONEY, result.component)
        self.assertEqual("secret_shape_detected", result.evidence["reason"])
        self.assertEqual(1, result.evidence["match_count"])
        self.assertEqual("github-ghp", result.evidence["matches"][0]["format"])
        self.assertEqual("high", result.evidence["matches"][0]["confidence"])
        self.assertNotIn(token, str(result.to_dict()))

    def test_unknown_fallback_can_be_excluded(self) -> None:
        output = "custom=vendor_live_abC123XYZ999qweRTY456mno"

        included = DPHoneyTextDetector().evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text=output, metadata={}),
        )
        excluded = DPHoneyTextDetector(include_low_confidence=False).evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text=output, metadata={}),
        )

        self.assertEqual(Action.SANITIZE, included.recommended_action)
        self.assertEqual("unknown-token", included.evidence["matches"][0]["format"])
        self.assertEqual(Action.ALLOW, excluded.recommended_action)
        self.assertEqual(0, excluded.evidence["match_count"])


class DPHoneyCanaryGeneratorTest(unittest.TestCase):
    def test_generator_maps_aegis_credential_type_to_dp_honey_format(self) -> None:
        generator = DPHoneyCanaryGenerator(corpus_size=20)

        token = generator(slot_name="repo_pat", credential_type="github_pat")

        self.assertTrue(get_format("github-ghp").validate(token))

    def test_dp_honey_ledger_plants_registered_canary_records(self) -> None:
        ledger = build_dp_honey_ledger(
            session_id="session-ledger",
            generator=DPHoneyCanaryGenerator(corpus_size=20),
        )
        messages = (Message(role="system", content="Use {{CREDENTIAL:repo_pat:github_pat}}."),)

        injection = inject_honeytokens(messages=messages, ledger=ledger, turn_index=0)
        registry = injection.canary_registry()
        detector = TextCanaryDetector(detector_name="text_canary", registry=registry)
        leaked = f"model repeated {injection.canary_records[0].value}"
        result = detector.evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text=leaked, metadata={}),
        )

        self.assertEqual("dp_honey", injection.canary_records[0].source)
        self.assertEqual(Action.ESCALATE, result.recommended_action)
        self.assertNotIn("{{CREDENTIAL", str(injection.to_dict()))

    def test_default_dp_honey_ledger_scopes_canaries_by_session(self) -> None:
        first = build_dp_honey_ledger(session_id="session-a", generator=DPHoneyCanaryGenerator(corpus_size=20))
        second = build_dp_honey_ledger(session_id="session-b", generator=DPHoneyCanaryGenerator(corpus_size=20))

        first_token = first.plant(slot_name="repo_pat", credential_type="github_pat", turn_index=0)
        second_token = second.plant(slot_name="repo_pat", credential_type="github_pat", turn_index=0)

        self.assertNotEqual(first_token.value, second_token.value)
        self.assertEqual(
            first_token.value, first.plant(slot_name="repo_pat", credential_type="github_pat", turn_index=1).value
        )

    def test_ledger_source_can_still_be_overridden_for_existing_callers(self) -> None:
        ledger = HoneytokenLedger(
            session_id="session-ledger",
            generator=lambda slot_name, credential_type: f"hny_{slot_name}_{credential_type}",
            source="fixture_source",
        )

        token = ledger.plant(slot_name="slot", credential_type="kind", turn_index=0)

        self.assertEqual("fixture_source", token.source)
        with self.assertRaisesRegex(HoneytokenLedgerError, "source"):
            HoneytokenLedger(session_id="session-ledger", generator=lambda slot, kind: "value", source="")


class DPHoneyAutoDecoyRemediationTest(unittest.TestCase):
    def test_auto_decoy_runs_only_after_policy_selects_dp_honey_sanitize(self) -> None:
        token = generate_honeytokens("github-ghp", count=1, sample_seed=2)[0]
        output = f"token={token}"
        detector_result = DPHoneyTextDetector().evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text=output, metadata={}),
        )
        decision = SeverityPolicyEngine().decide((detector_result,))

        remediation = apply_dp_honey_auto_decoy(
            output_text=output,
            policy_decision=decision,
            detector_results=(detector_result,),
            seed=9,
        )

        self.assertTrue(remediation.applied)
        self.assertEqual("dp_honey_auto_decoy_applied", remediation.reason)
        self.assertNotIn(token, remediation.output_text)
        self.assertEqual("github-ghp", remediation.findings[0]["format"])
        self.assertNotIn(token, str(remediation.to_dict()))

    def test_auto_decoy_does_not_run_for_non_dp_honey_sanitize(self) -> None:
        output = "token=vendor_live_abC123XYZ999qweRTY456mno"
        other_result = _non_dp_honey_result()
        decision = SeverityPolicyEngine().decide((other_result,))

        remediation = apply_dp_honey_auto_decoy(
            output_text=output,
            policy_decision=decision,
            detector_results=(other_result,),
            seed=9,
        )

        self.assertFalse(remediation.applied)
        self.assertEqual("dp_honey_not_triggered", remediation.reason)
        self.assertEqual(output, remediation.output_text)

    def test_auto_decoy_does_not_run_when_policy_does_not_sanitize(self) -> None:
        allow_result = DPHoneyTextDetector().evaluate(
            turn=_runtime_turn(),
            model_response=ModelResponse(output_text="nothing secret-shaped here", metadata={}),
        )
        decision = SeverityPolicyEngine().decide((allow_result,))

        remediation = apply_dp_honey_auto_decoy(
            output_text="nothing secret-shaped here",
            policy_decision=decision,
            detector_results=(allow_result,),
        )

        self.assertFalse(remediation.applied)
        self.assertEqual("policy_action_not_sanitize", remediation.reason)


if __name__ == "__main__":
    unittest.main()
