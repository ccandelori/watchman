import tempfile
import unittest
from pathlib import Path

from aegis.audit.memory import InMemoryAuditSink
from aegis.core.contracts import Action, CapabilityMode, CapabilityStatus, DetectorComponent, Message, ModelInfo
from aegis.core.orchestrator import AegisRuntime, RuntimeRequest
from aegis.detectors.cift_candidate import (
    CIFT_SELECTOR_PROBE_V0,
    CiftCandidateMonitorError,
    PrecomputedCiftCandidateDetector,
)
from aegis.policy.engine import SeverityPolicyEngine
from aegis.providers.mock import MockModelProvider
from aegis.replay.offline import (
    OfflineReplayError,
    load_cift_candidate_scores_jsonl,
    load_runtime_requests_jsonl,
    replay_requests,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "cift_candidate_monitor_v0"


class CiftCandidateMonitorTest(unittest.TestCase):
    def test_replay_uses_candidate_profile_thresholds_and_writes_audit_events(self) -> None:
        requests = load_runtime_requests_jsonl(FIXTURE_ROOT / "runtime_turns.jsonl")
        scores_by_example_id = load_cift_candidate_scores_jsonl(FIXTURE_ROOT / "detector_results.jsonl")
        audit_sink = InMemoryAuditSink()
        runtime = AegisRuntime(
            turn_annotators=(),
            pre_generation_detectors=(
                PrecomputedCiftCandidateDetector(
                    profile=CIFT_SELECTOR_PROBE_V0,
                    scores_by_example_id=scores_by_example_id,
                ),
            ),
            post_generation_detectors=(),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=audit_sink,
            model_provider=MockModelProvider(default_content="mock response"),
        )

        responses = replay_requests(runtime=runtime, requests=requests)

        self.assertEqual(3, len(responses))
        self.assertEqual(Action.ALLOW, responses[0].policy_decision.final_action)
        self.assertEqual(Action.WARN, responses[1].policy_decision.final_action)
        self.assertEqual(Action.WARN, responses[2].policy_decision.final_action)
        self.assertEqual("allow", responses[0].detector_results[0].evidence["operating_band"])
        self.assertEqual("review", responses[1].detector_results[0].evidence["operating_band"])
        self.assertEqual("balanced", responses[2].detector_results[0].evidence["operating_band"])
        self.assertEqual(3, len(audit_sink.recent(limit=10)))
        self.assertEqual("trace-cift-balanced", audit_sink.recent(limit=1)[0].trace_id)

    def test_black_box_mode_emits_explicit_unavailable_cift_evidence(self) -> None:
        detector = PrecomputedCiftCandidateDetector(
            profile=CIFT_SELECTOR_PROBE_V0,
            scores_by_example_id={},
        )
        request = RuntimeRequest(
            trace_id="trace-black-box",
            session_id="session-black-box",
            turn_index=1,
            capability_mode=CapabilityMode.BLACK_BOX,
            model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
            messages=(),
            tool_calls=(),
            sensitive_spans=(),
            metadata={"example_id": "not-needed"},
        )
        runtime = AegisRuntime(
            turn_annotators=(),
            pre_generation_detectors=(detector,),
            post_generation_detectors=(),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=InMemoryAuditSink(),
            model_provider=MockModelProvider(default_content="mock response"),
        )

        response = runtime.evaluate_turn(request)
        result = response.detector_results[0]

        self.assertEqual(DetectorComponent.CIFT, result.component)
        self.assertEqual(CapabilityStatus.UNAVAILABLE, result.capability_status)
        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual("activation_access_unavailable", result.evidence["reason"])
        self.assertEqual("cift_selector_probe_v0", result.evidence["profile_id"])

    def test_self_hosted_mode_requires_matching_precomputed_score(self) -> None:
        detector = PrecomputedCiftCandidateDetector(
            profile=CIFT_SELECTOR_PROBE_V0,
            scores_by_example_id={},
        )
        request = RuntimeRequest(
            trace_id="trace-missing-score",
            session_id="session-missing-score",
            turn_index=1,
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device="cpu"),
            messages=(Message(role="user", content="hello"),),
            tool_calls=(),
            sensitive_spans=(),
            metadata={"example_id": "missing-score"},
        )
        runtime = AegisRuntime(
            turn_annotators=(),
            pre_generation_detectors=(detector,),
            post_generation_detectors=(),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=InMemoryAuditSink(),
            model_provider=MockModelProvider(default_content="mock response"),
        )

        with self.assertRaises(CiftCandidateMonitorError):
            runtime.evaluate_turn(request)

    def test_score_loader_rejects_duplicate_example_ids(self) -> None:
        first_line = (FIXTURE_ROOT / "detector_results.jsonl").read_text(encoding="utf-8").splitlines()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate_scores.jsonl"
            path.write_text(first_line + "\n" + first_line + "\n", encoding="utf-8")

            with self.assertRaises(OfflineReplayError):
                load_cift_candidate_scores_jsonl(path)


if __name__ == "__main__":
    unittest.main()
