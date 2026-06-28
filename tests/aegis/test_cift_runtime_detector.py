from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aegis.audit.memory import InMemoryAuditSink
from aegis.cift_contract import CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION
from aegis.core.contracts import Action, CapabilityMode, CapabilityStatus, Message, ModelInfo, NormalizedTurn
from aegis.core.orchestrator import AegisRuntime, RuntimeRequest
from aegis.detectors.cift_runtime import (
    CiftFeatureExtraction,
    CiftFeatureVectorAnnotator,
    CiftRuntimeDetector,
    CiftRuntimeDetectorError,
    CiftRuntimeLinearModel,
    CiftRuntimeWindowSelector,
    CiftRuntimeWindowSelectorConfig,
    build_cift_window_selector_runtime_components,
    cift_feature_vector_from_turn,
    cift_runtime_model_to_dict,
    load_cift_runtime_model,
    normalized_turn_with_cift_feature_vector,
    predict_cift_runtime_model,
    validate_cift_runtime_model,
)
from aegis.policy.engine import SeverityPolicyEngine
from aegis.providers.mock import MockModelProvider

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_MODEL_PATH = (
    REPOSITORY_ROOT
    / "introspection"
    / "data"
    / "models"
    / "cift_qwen3_0_6b_dp_honey_lite_v4_1_selector_window_layer_15_runtime_v1.json"
)


class CiftRuntimeDetectorTest(unittest.TestCase):
    def test_runtime_scores_feature_vector_attached_by_turn_annotator(self) -> None:
        extractor = RecordingFeatureExtractor(feature_vector=(3.0, 2.0))
        runtime = AegisRuntime(
            turn_annotators=(
                CiftFeatureVectorAnnotator(
                    feature_key="readout_window_layer_15",
                    extractor=extractor,
                    source="test_self_hosted_extractor",
                    selected_choice_window=False,
                ),
            ),
            pre_generation_detectors=(
                CiftRuntimeDetector(
                    detector_name="cift_runtime",
                    model=_runtime_model(positive_class_index=1),
                    activation_failure_action=Action.ALLOW,
                ),
            ),
            post_generation_detectors=(),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=InMemoryAuditSink(),
            model_provider=MockModelProvider(default_content="ok"),
        )

        response = runtime.evaluate_turn(_request(capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION))

        self.assertEqual([("trace-cift-runtime", "readout_window_layer_15")], extractor.calls)
        self.assertEqual(Action.WARN, response.policy_decision.final_action)
        self.assertEqual(CapabilityStatus.ACTIVE, response.detector_results[0].capability_status)
        self.assertEqual("test_self_hosted_extractor", _feature_source(response.audit_event.normalized_turn))
        self.assertNotIn("feature_vectors", response.detector_results[0].evidence)

    def test_self_hosted_turn_with_feature_vector_emits_active_detector_result(self) -> None:
        detector = CiftRuntimeDetector(
            detector_name="cift_runtime",
            model=_runtime_model(positive_class_index=1),
            activation_failure_action=Action.ALLOW,
        )
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={"cift": {"feature_vectors": {"readout_window_layer_15": [3.0, 2.0]}}},
        )

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.ACTIVE, result.capability_status)
        self.assertEqual(Action.WARN, result.recommended_action)
        self.assertAlmostEqual(_sigmoid(1.25), result.score)
        self.assertEqual("exfiltration_intent", result.evidence["predicted_label"])
        self.assertEqual("metadata.cift.feature_vectors", result.evidence["activation_source"])
        self.assertNotIn("feature_vectors", result.evidence)

    def test_black_box_turn_emits_explicit_unavailable_result(self) -> None:
        detector = CiftRuntimeDetector(
            detector_name="cift_runtime",
            model=_runtime_model(positive_class_index=1),
            activation_failure_action=Action.ALLOW,
        )
        turn = _turn(capability_mode=CapabilityMode.BLACK_BOX, metadata={})

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.UNAVAILABLE, result.capability_status)
        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual("activation_access_unavailable", result.evidence["reason"])
        self.assertEqual("black_box", result.evidence["actual_capability_mode"])

    def test_self_hosted_turn_without_feature_vector_emits_degraded_result(self) -> None:
        detector = CiftRuntimeDetector(
            detector_name="cift_runtime",
            model=_runtime_model(positive_class_index=1),
            activation_failure_action=Action.ALLOW,
        )
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={"cift": {"readout_token_indices": [1, 2, 3]}},
        )

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual("activation_feature_vector_missing", result.evidence["reason"])

    def test_runtime_candidate_without_feature_vector_blocks_generation(self) -> None:
        model = replace(
            _runtime_model(positive_class_index=1),
            candidate_status="runtime_candidate",
            positive_action=Action.BLOCK,
        )
        detector = CiftRuntimeDetector(
            detector_name="cift_runtime",
            model=model,
            activation_failure_action=Action.BLOCK,
        )
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={"cift": {"readout_token_indices": [1, 2, 3]}},
        )

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual(Action.BLOCK, result.recommended_action)
        self.assertEqual("activation_feature_vector_missing", result.evidence["reason"])
        self.assertEqual("runtime_candidate_missing_feature_fail_closed", result.evidence["fail_closed_reason"])

    def test_fail_closed_detector_blocks_malformed_cift_metadata(self) -> None:
        detector = CiftRuntimeDetector(
            detector_name="cift_runtime",
            model=replace(
                _runtime_model(positive_class_index=1),
                candidate_status="runtime_candidate",
                positive_action=Action.BLOCK,
            ),
            activation_failure_action=Action.BLOCK,
        )
        turn = _turn(capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION, metadata={"cift": "bad"})

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual(Action.BLOCK, result.recommended_action)
        self.assertEqual(1.0, result.score)
        self.assertEqual("activation_feature_vector_malformed", result.evidence["reason"])
        self.assertEqual("runtime_candidate_missing_feature_fail_closed", result.evidence["fail_closed_reason"])

    def test_fail_closed_detector_blocks_wrong_length_feature_vector(self) -> None:
        detector = CiftRuntimeDetector(
            detector_name="cift_runtime",
            model=replace(
                _runtime_model(positive_class_index=1),
                candidate_status="runtime_candidate",
                positive_action=Action.BLOCK,
            ),
            activation_failure_action=Action.BLOCK,
        )
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={"cift": {"feature_vectors": {"readout_window_layer_15": [1.0]}}},
        )

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual(Action.BLOCK, result.recommended_action)
        self.assertEqual(1.0, result.score)
        self.assertEqual("activation_feature_vector_malformed", result.evidence["reason"])
        self.assertEqual("runtime_candidate_missing_feature_fail_closed", result.evidence["fail_closed_reason"])

    def test_window_selector_uses_selected_choice_model_when_selected_choice_metadata_exists(self) -> None:
        selected_model = _selector_model(
            feature_key="selected_choice_window_layer_15",
            model_bundle_id="selected-choice-bundle",
            logistic_coefficients=(1.0, 1.0),
        )
        fallback_model = _selector_model(
            feature_key="readout_window_layer_15",
            model_bundle_id="fallback-bundle",
            logistic_coefficients=(-1.0, -1.0),
        )
        detector = CiftRuntimeWindowSelector(
            detector_name="cift_runtime",
            selected_choice_model=selected_model,
            fallback_model=fallback_model,
            activation_failure_action=Action.ALLOW,
        )
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={
                "cift": {
                    "selected_choice_readout_token_indices": [7, 8],
                    "feature_vectors": {
                        "selected_choice_window_layer_15": [2.0, 2.0],
                        "readout_window_layer_15": [2.0, 2.0],
                    },
                }
            },
        )

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.ACTIVE, result.capability_status)
        self.assertEqual(Action.WARN, result.recommended_action)
        self.assertEqual(0.7, result.confidence)
        self.assertEqual("selected_choice", result.evidence["cift_window_family"])
        self.assertEqual("selected_choice_metadata_present", result.evidence["cift_window_selection_reason"])
        self.assertEqual("primary", result.evidence["cift_window_coverage"])
        self.assertEqual("selected-choice-bundle", result.evidence["model_bundle_id"])

    def test_window_selector_uses_fallback_model_when_selected_choice_metadata_is_absent(self) -> None:
        selected_model = _selector_model(
            feature_key="selected_choice_window_layer_15",
            model_bundle_id="selected-choice-bundle",
            logistic_coefficients=(-1.0, -1.0),
        )
        fallback_model = _selector_model(
            feature_key="readout_window_layer_15",
            model_bundle_id="fallback-bundle",
            logistic_coefficients=(1.0, 1.0),
        )
        detector = CiftRuntimeWindowSelector(
            detector_name="cift_runtime",
            selected_choice_model=selected_model,
            fallback_model=fallback_model,
            activation_failure_action=Action.ALLOW,
        )
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={
                "cift": {
                    "readout_token_indices": [3, 4],
                    "feature_vectors": {
                        "selected_choice_window_layer_15": [2.0, 2.0],
                        "readout_window_layer_15": [2.0, 2.0],
                    },
                }
            },
        )

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual(Action.WARN, result.recommended_action)
        self.assertEqual(0.35, result.confidence)
        self.assertEqual("payload_query_fallback", result.evidence["cift_window_family"])
        self.assertEqual("selected_choice_metadata_absent", result.evidence["cift_window_selection_reason"])
        self.assertEqual("degraded_fallback", result.evidence["cift_window_coverage"])
        self.assertEqual("selected_choice_metadata_required_for_primary_cift", result.evidence["degradation_reason"])
        self.assertEqual("fallback-bundle", result.evidence["model_bundle_id"])

    def test_fail_closed_window_selector_uses_freeform_model_when_selected_choice_metadata_is_absent(self) -> None:
        selected_model = _selector_model(
            feature_key="selected_choice_window_layer_15",
            model_bundle_id="selected-choice-bundle",
            logistic_coefficients=(-1.0, -1.0),
        )
        fallback_model = replace(
            _selector_model(
                feature_key="query_tail_window_layer_21",
                model_bundle_id="freeform-bundle",
                logistic_coefficients=(-1.0, -1.0),
            ),
            candidate_status="runtime_candidate",
            positive_action=Action.BLOCK,
        )
        detector = CiftRuntimeWindowSelector(
            detector_name="cift_runtime",
            selected_choice_model=selected_model,
            fallback_model=fallback_model,
            activation_failure_action=Action.BLOCK,
        )
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={
                "cift": {
                    "readout_token_indices": [3, 4],
                    "feature_vectors": {
                        "selected_choice_window_layer_15": [0.0, 0.0],
                        "query_tail_window_layer_21": [2.0, 2.0],
                    },
                }
            },
        )

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.ACTIVE, result.capability_status)
        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual(
            "selected_choice_metadata_absent_freeform_route",
            result.evidence["cift_window_selection_reason"],
        )
        self.assertEqual("freeform_query_tail", result.evidence["cift_window_family"])
        self.assertEqual("certified_freeform", result.evidence["cift_window_coverage"])
        self.assertEqual("freeform-bundle", result.evidence["model_bundle_id"])
        self.assertEqual("freeform-bundle", result.evidence["freeform_model_bundle_id"])

    def test_fail_closed_window_selector_blocks_when_selected_choice_metadata_is_absent_without_freeform_model(
        self,
    ) -> None:
        selected_model = _selector_model(
            feature_key="selected_choice_window_layer_15",
            model_bundle_id="selected-choice-bundle",
            logistic_coefficients=(-1.0, -1.0),
        )
        detector = CiftRuntimeWindowSelector(
            detector_name="cift_runtime",
            selected_choice_model=selected_model,
            fallback_model=None,
            activation_failure_action=Action.BLOCK,
        )
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={
                "cift": {
                    "readout_token_indices": [3, 4],
                    "feature_vectors": {
                        "selected_choice_window_layer_15": [0.0, 0.0],
                    },
                }
            },
        )

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual(Action.BLOCK, result.recommended_action)
        self.assertEqual(1.0, result.score)
        self.assertEqual("selected_choice_metadata_absent", result.evidence["reason"])
        self.assertEqual("selected_choice_metadata_absent", result.evidence["cift_window_selection_reason"])
        self.assertEqual("selected_choice", result.evidence["cift_window_family"])
        self.assertEqual("unavailable", result.evidence["cift_window_coverage"])
        self.assertEqual("selected-choice-bundle", result.evidence["model_bundle_id"])
        self.assertEqual("runtime_candidate_missing_feature_fail_closed", result.evidence["fail_closed_reason"])

    def test_window_selector_degrades_when_selected_choice_metadata_exists_but_feature_is_missing(self) -> None:
        selected_model = _selector_model(
            feature_key="selected_choice_window_layer_15",
            model_bundle_id="selected-choice-bundle",
            logistic_coefficients=(1.0, 1.0),
        )
        fallback_model = _selector_model(
            feature_key="readout_window_layer_15",
            model_bundle_id="fallback-bundle",
            logistic_coefficients=(1.0, 1.0),
        )
        detector = CiftRuntimeWindowSelector(
            detector_name="cift_runtime",
            selected_choice_model=selected_model,
            fallback_model=fallback_model,
            activation_failure_action=Action.ALLOW,
        )
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={
                "cift": {
                    "selected_choice_readout_token_indices": [7, 8],
                    "feature_vectors": {"readout_window_layer_15": [2.0, 2.0]},
                }
            },
        )

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual("activation_feature_vector_missing", result.evidence["reason"])
        self.assertEqual("selected_choice", result.evidence["cift_window_family"])
        self.assertEqual("selected-choice-bundle", result.evidence["model_bundle_id"])

    def test_window_selector_rejects_malformed_selected_choice_indices(self) -> None:
        malformed_values: tuple[tuple[object, str], ...] = (
            ("not-a-list", "must be a list"),
            ([], "must not be empty"),
            ([True], "must be an integer"),
            (["bad"], "must be an integer"),
            ([-1], "must be non-negative"),
        )
        selected_model = _selector_model(
            feature_key="selected_choice_window_layer_15",
            model_bundle_id="selected-choice-bundle",
            logistic_coefficients=(1.0, 1.0),
        )
        fallback_model = _selector_model(
            feature_key="readout_window_layer_15",
            model_bundle_id="fallback-bundle",
            logistic_coefficients=(1.0, 1.0),
        )
        detector = CiftRuntimeWindowSelector(
            detector_name="cift_runtime",
            selected_choice_model=selected_model,
            fallback_model=fallback_model,
            activation_failure_action=Action.ALLOW,
        )

        for selected_choice_indices, message_fragment in malformed_values:
            with self.subTest(selected_choice_indices=selected_choice_indices):
                turn = _turn(
                    capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
                    metadata={
                        "cift": {
                            "selected_choice_readout_token_indices": selected_choice_indices,
                            "feature_vectors": {
                                "selected_choice_window_layer_15": [2.0, 2.0],
                                "readout_window_layer_15": [2.0, 2.0],
                            },
                        }
                    },
                )

                with self.assertRaisesRegex(CiftRuntimeDetectorError, message_fragment):
                    detector.evaluate(turn, None)

    def test_fail_closed_window_selector_blocks_malformed_selected_choice_metadata(self) -> None:
        selected_model = _selector_model(
            feature_key="selected_choice_window_layer_15",
            model_bundle_id="selected-choice-bundle",
            logistic_coefficients=(1.0, 1.0),
        )
        fallback_model = _selector_model(
            feature_key="readout_window_layer_15",
            model_bundle_id="fallback-bundle",
            logistic_coefficients=(1.0, 1.0),
        )
        detector = CiftRuntimeWindowSelector(
            detector_name="cift_runtime",
            selected_choice_model=selected_model,
            fallback_model=fallback_model,
            activation_failure_action=Action.BLOCK,
        )
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={
                "cift": {
                    "selected_choice_readout_token_indices": "not-a-list",
                    "feature_vectors": {
                        "selected_choice_window_layer_15": [2.0, 2.0],
                        "readout_window_layer_15": [2.0, 2.0],
                    },
                }
            },
        )

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual(Action.BLOCK, result.recommended_action)
        self.assertEqual("activation_feature_vector_malformed", result.evidence["reason"])
        self.assertEqual("runtime_candidate_missing_feature_fail_closed", result.evidence["fail_closed_reason"])

    def test_fail_closed_detector_blocks_non_finite_feature_vector(self) -> None:
        detector = CiftRuntimeDetector(
            detector_name="cift_runtime",
            model=replace(
                _runtime_model(positive_class_index=1),
                candidate_status="runtime_candidate",
                positive_action=Action.BLOCK,
            ),
            activation_failure_action=Action.BLOCK,
        )
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={"cift": {"feature_vectors": {"readout_window_layer_15": [float("nan"), 2.0]}}},
        )

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual(Action.BLOCK, result.recommended_action)
        self.assertEqual("activation_feature_vector_malformed", result.evidence["reason"])
        self.assertIn("finite", result.evidence["error"])
        self.assertEqual("runtime_candidate_missing_feature_fail_closed", result.evidence["fail_closed_reason"])

    def test_runtime_candidate_fails_closed_when_activation_failure_metadata_is_malformed(self) -> None:
        detector = CiftRuntimeDetector(
            detector_name="cift_runtime",
            model=replace(
                _runtime_model(positive_class_index=1),
                candidate_status="runtime_candidate",
                positive_action=Action.BLOCK,
            ),
            activation_failure_action=Action.BLOCK,
        )
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={"cift": {"activation_failures": []}},
        )

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual(Action.BLOCK, result.recommended_action)
        self.assertEqual("activation_feature_vector_malformed", result.evidence["reason"])
        self.assertIn("metadata.cift.activation_failures", result.evidence["error"])
        self.assertEqual("runtime_candidate_missing_feature_fail_closed", result.evidence["fail_closed_reason"])

    def test_feature_annotator_records_malformed_extractor_output_for_fail_closed_detector(self) -> None:
        extractor = MalformedFeatureExtractor(feature_vector=("not-a-number", 2.0))
        runtime = AegisRuntime(
            turn_annotators=(
                CiftFeatureVectorAnnotator(
                    feature_key="readout_window_layer_15",
                    extractor=extractor,
                    source="test_self_hosted_extractor",
                    selected_choice_window=False,
                ),
            ),
            pre_generation_detectors=(
                CiftRuntimeDetector(
                    detector_name="cift_runtime",
                    model=replace(
                        _runtime_model(positive_class_index=1),
                        candidate_status="runtime_candidate",
                        positive_action=Action.BLOCK,
                    ),
                    activation_failure_action=Action.BLOCK,
                ),
            ),
            post_generation_detectors=(),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=InMemoryAuditSink(),
            model_provider=MockModelProvider(default_content="unsafe provider output"),
        )

        response = runtime.evaluate_turn(_request(capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION))

        self.assertEqual(Action.BLOCK, response.policy_decision.final_action)
        self.assertEqual("", response.output_text)
        self.assertEqual(CapabilityStatus.DEGRADED, response.detector_results[0].capability_status)
        self.assertEqual(Action.BLOCK, response.detector_results[0].recommended_action)
        self.assertEqual("activation_feature_vector_malformed", response.detector_results[0].evidence["reason"])
        self.assertEqual(
            "runtime_candidate_missing_feature_fail_closed",
            response.detector_results[0].evidence["fail_closed_reason"],
        )

    def test_feature_annotator_records_non_finite_extractor_output_for_fail_closed_detector(self) -> None:
        extractor = MalformedFeatureExtractor(feature_vector=(float("nan"), 2.0))
        runtime = AegisRuntime(
            turn_annotators=(
                CiftFeatureVectorAnnotator(
                    feature_key="readout_window_layer_15",
                    extractor=extractor,
                    source="test_self_hosted_extractor",
                    selected_choice_window=False,
                ),
            ),
            pre_generation_detectors=(
                CiftRuntimeDetector(
                    detector_name="cift_runtime",
                    model=replace(
                        _runtime_model(positive_class_index=1),
                        candidate_status="runtime_candidate",
                        positive_action=Action.BLOCK,
                    ),
                    activation_failure_action=Action.BLOCK,
                ),
            ),
            post_generation_detectors=(),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=InMemoryAuditSink(),
            model_provider=MockModelProvider(default_content="unsafe provider output"),
        )

        response = runtime.evaluate_turn(_request(capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION))

        self.assertEqual(Action.BLOCK, response.policy_decision.final_action)
        self.assertEqual("", response.output_text)
        self.assertEqual(CapabilityStatus.DEGRADED, response.detector_results[0].capability_status)
        self.assertEqual(Action.BLOCK, response.detector_results[0].recommended_action)
        self.assertEqual("activation_feature_vector_malformed", response.detector_results[0].evidence["reason"])
        self.assertIn("finite", response.detector_results[0].evidence["error"])
        self.assertEqual(
            "runtime_candidate_missing_feature_fail_closed",
            response.detector_results[0].evidence["fail_closed_reason"],
        )

    def test_feature_annotator_records_extractor_exception_for_fail_closed_detector(self) -> None:
        extractor = RaisingFeatureExtractor(error=RuntimeError("extractor backend unavailable"))
        runtime = AegisRuntime(
            turn_annotators=(
                CiftFeatureVectorAnnotator(
                    feature_key="readout_window_layer_15",
                    extractor=extractor,
                    source="test_self_hosted_extractor",
                    selected_choice_window=False,
                ),
            ),
            pre_generation_detectors=(
                CiftRuntimeDetector(
                    detector_name="cift_runtime",
                    model=replace(
                        _runtime_model(positive_class_index=1),
                        candidate_status="runtime_candidate",
                        positive_action=Action.BLOCK,
                    ),
                    activation_failure_action=Action.BLOCK,
                ),
            ),
            post_generation_detectors=(),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=InMemoryAuditSink(),
            model_provider=MockModelProvider(default_content="unsafe provider output"),
        )

        response = runtime.evaluate_turn(_request(capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION))

        self.assertEqual(Action.BLOCK, response.policy_decision.final_action)
        self.assertEqual("", response.output_text)
        self.assertEqual(CapabilityStatus.DEGRADED, response.detector_results[0].capability_status)
        self.assertEqual(Action.BLOCK, response.detector_results[0].recommended_action)
        self.assertEqual("activation_feature_vector_malformed", response.detector_results[0].evidence["reason"])
        self.assertIn("RuntimeError", response.detector_results[0].evidence["error"])
        self.assertEqual(
            "runtime_candidate_missing_feature_fail_closed",
            response.detector_results[0].evidence["fail_closed_reason"],
        )

    def test_window_selector_components_load_artifacts_and_plug_into_runtime(self) -> None:
        extractor = FeatureMapExtractor(
            feature_vectors={
                ("trace-cift-runtime", "selected_choice_window_layer_15"): (2.0, 2.0),
                ("trace-cift-runtime", "readout_window_layer_15"): (-2.0, -2.0),
            }
        )
        selected_model = _selector_model(
            feature_key="selected_choice_window_layer_15",
            model_bundle_id="selected-choice-bundle",
            logistic_coefficients=(1.0, 1.0),
        )
        fallback_model = _selector_model(
            feature_key="readout_window_layer_15",
            model_bundle_id="fallback-bundle",
            logistic_coefficients=(1.0, 1.0),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_model_path = root / "selected.json"
            fallback_model_path = root / "fallback.json"
            selected_model_path.write_text(
                json.dumps(_runtime_candidate_selector_record(selected_model)),
                encoding="utf-8",
            )
            fallback_model_path.write_text(
                json.dumps(_runtime_candidate_selector_record(fallback_model)),
                encoding="utf-8",
            )
            components = build_cift_window_selector_runtime_components(
                CiftRuntimeWindowSelectorConfig(
                    detector_name="cift_runtime",
                    selected_choice_model_path=selected_model_path,
                    selected_choice_model_sha256=_sha256_file(selected_model_path),
                    fallback_model_path=fallback_model_path,
                    fallback_model_sha256=_sha256_file(fallback_model_path),
                    feature_extractor=extractor,
                    feature_source="test_feature_map",
                    activation_failure_action=Action.BLOCK,
                )
            )
            runtime = AegisRuntime(
                turn_annotators=components.turn_annotators,
                pre_generation_detectors=components.pre_generation_detectors,
                post_generation_detectors=(),
                session_detectors=(),
                policy_engine=SeverityPolicyEngine(),
                audit_sink=InMemoryAuditSink(),
                model_provider=MockModelProvider(default_content="ok"),
            )

            response = runtime.evaluate_turn(
                _request_with_metadata(
                    capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
                    metadata={"cift": {"selected_choice_readout_token_indices": [7, 8]}},
                )
            )

        self.assertEqual(
            [
                ("trace-cift-runtime", "selected_choice_window_layer_15"),
                ("trace-cift-runtime", "readout_window_layer_15"),
            ],
            extractor.calls,
        )
        self.assertEqual(2, len(components.turn_annotators))
        self.assertEqual(1, len(components.pre_generation_detectors))
        self.assertEqual(Action.BLOCK, response.policy_decision.final_action)
        self.assertEqual(CapabilityStatus.ACTIVE, response.detector_results[0].capability_status)
        self.assertEqual("selected_choice", response.detector_results[0].evidence["cift_window_family"])

    def test_window_selector_components_reject_offline_research_candidate_artifacts(self) -> None:
        extractor = RecordingFeatureExtractor(feature_vector=(1.0, 2.0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_model_path = root / "selected.json"
            fallback_model_path = root / "fallback.json"
            selected_model_path.write_text(
                json.dumps(
                    cift_runtime_model_to_dict(
                        _selector_model(
                            feature_key="selected_choice_window_layer_15",
                            model_bundle_id="selected-choice-bundle",
                            logistic_coefficients=(1.0, 1.0),
                        )
                    )
                ),
                encoding="utf-8",
            )
            fallback_model_path.write_text(
                json.dumps(
                    cift_runtime_model_to_dict(
                        _selector_model(
                            feature_key="readout_window_layer_15",
                            model_bundle_id="fallback-bundle",
                            logistic_coefficients=(1.0, 1.0),
                        )
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(CiftRuntimeDetectorError, "runtime_candidate"):
                build_cift_window_selector_runtime_components(
                    CiftRuntimeWindowSelectorConfig(
                        detector_name="cift_runtime",
                        selected_choice_model_path=selected_model_path,
                        selected_choice_model_sha256=_sha256_file(selected_model_path),
                        fallback_model_path=fallback_model_path,
                        fallback_model_sha256=_sha256_file(fallback_model_path),
                        feature_extractor=extractor,
                        feature_source="test_feature_map",
                        activation_failure_action=Action.BLOCK,
                    )
                )

    def test_window_selector_components_fail_closed_when_extractor_returns_no_feature_vectors(self) -> None:
        extractor = FeatureMapExtractor(feature_vectors={})
        selected_model = _selector_model(
            feature_key="selected_choice_window_layer_15",
            model_bundle_id="selected-choice-bundle",
            logistic_coefficients=(1.0, 1.0),
        )
        fallback_model = _selector_model(
            feature_key="readout_window_layer_15",
            model_bundle_id="fallback-bundle",
            logistic_coefficients=(1.0, 1.0),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_model_path = root / "selected.json"
            fallback_model_path = root / "fallback.json"
            selected_model_path.write_text(
                json.dumps(_runtime_candidate_selector_record(selected_model)),
                encoding="utf-8",
            )
            fallback_model_path.write_text(
                json.dumps(_runtime_candidate_selector_record(fallback_model)),
                encoding="utf-8",
            )
            components = build_cift_window_selector_runtime_components(
                CiftRuntimeWindowSelectorConfig(
                    detector_name="cift_runtime",
                    selected_choice_model_path=selected_model_path,
                    selected_choice_model_sha256=_sha256_file(selected_model_path),
                    fallback_model_path=fallback_model_path,
                    fallback_model_sha256=_sha256_file(fallback_model_path),
                    feature_extractor=extractor,
                    feature_source="test_feature_map",
                    activation_failure_action=Action.BLOCK,
                )
            )
            runtime = AegisRuntime(
                turn_annotators=components.turn_annotators,
                pre_generation_detectors=components.pre_generation_detectors,
                post_generation_detectors=(),
                session_detectors=(),
                policy_engine=SeverityPolicyEngine(),
                audit_sink=InMemoryAuditSink(),
                model_provider=MockModelProvider(default_content="ok"),
            )

            response = runtime.evaluate_turn(
                _request_with_metadata(
                    capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
                    metadata={"cift": {"selected_choice_readout_token_indices": [7, 8]}},
                )
            )

        self.assertEqual(Action.BLOCK, response.policy_decision.final_action)
        self.assertEqual("", response.output_text)
        self.assertEqual(CapabilityStatus.DEGRADED, response.detector_results[0].capability_status)
        self.assertEqual(Action.BLOCK, response.detector_results[0].recommended_action)
        self.assertEqual(
            "runtime_candidate_missing_feature_fail_closed",
            response.detector_results[0].evidence["fail_closed_reason"],
        )

    def test_window_selector_components_use_freeform_route_when_selected_choice_metadata_is_absent(self) -> None:
        extractor = FeatureMapExtractor(
            feature_vectors={
                ("trace-cift-runtime", "selected_choice_window_layer_15"): (0.0, 0.0),
                ("trace-cift-runtime", "query_tail_window_layer_21"): (2.0, 2.0),
            }
        )
        selected_model = _selector_model(
            feature_key="selected_choice_window_layer_15",
            model_bundle_id="selected-choice-bundle",
            logistic_coefficients=(-1.0, -1.0),
        )
        fallback_model = _selector_model(
            feature_key="query_tail_window_layer_21",
            model_bundle_id="freeform-bundle",
            logistic_coefficients=(-1.0, -1.0),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_model_path = root / "selected.json"
            fallback_model_path = root / "fallback.json"
            selected_model_path.write_text(
                json.dumps(_runtime_candidate_selector_record(selected_model)),
                encoding="utf-8",
            )
            fallback_model_path.write_text(
                json.dumps(_runtime_candidate_selector_record(fallback_model)),
                encoding="utf-8",
            )
            components = build_cift_window_selector_runtime_components(
                CiftRuntimeWindowSelectorConfig(
                    detector_name="cift_runtime",
                    selected_choice_model_path=selected_model_path,
                    selected_choice_model_sha256=_sha256_file(selected_model_path),
                    fallback_model_path=fallback_model_path,
                    fallback_model_sha256=_sha256_file(fallback_model_path),
                    feature_extractor=extractor,
                    feature_source="test_feature_map",
                    activation_failure_action=Action.BLOCK,
                )
            )
            runtime = AegisRuntime(
                turn_annotators=components.turn_annotators,
                pre_generation_detectors=components.pre_generation_detectors,
                post_generation_detectors=(),
                session_detectors=(),
                policy_engine=SeverityPolicyEngine(),
                audit_sink=InMemoryAuditSink(),
                model_provider=MockModelProvider(default_content="ok"),
            )

            response = runtime.evaluate_turn(
                _request_with_metadata(
                    capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
                    metadata={"cift": {"readout_token_indices": [3, 4]}},
                )
            )

        self.assertEqual(Action.ALLOW, response.policy_decision.final_action)
        self.assertEqual("ok", response.output_text)
        self.assertEqual(
            [
                ("trace-cift-runtime", "selected_choice_window_layer_15"),
                ("trace-cift-runtime", "query_tail_window_layer_21"),
            ],
            extractor.calls,
        )
        self.assertEqual(CapabilityStatus.ACTIVE, response.detector_results[0].capability_status)
        self.assertEqual(Action.ALLOW, response.detector_results[0].recommended_action)
        self.assertEqual("freeform_query_tail", response.detector_results[0].evidence["cift_window_family"])
        self.assertEqual(
            "selected_choice_metadata_absent_freeform_route",
            response.detector_results[0].evidence["cift_window_selection_reason"],
        )
        self.assertEqual("certified_freeform", response.detector_results[0].evidence["cift_window_coverage"])

    def test_window_selector_components_fail_closed_when_freeform_route_is_not_configured(self) -> None:
        extractor = FeatureMapExtractor(
            feature_vectors={
                ("trace-cift-runtime", "selected_choice_window_layer_15"): (0.0, 0.0),
            }
        )
        selected_model = _selector_model(
            feature_key="selected_choice_window_layer_15",
            model_bundle_id="selected-choice-bundle",
            logistic_coefficients=(-1.0, -1.0),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_model_path = root / "selected.json"
            selected_model_path.write_text(
                json.dumps(_runtime_candidate_selector_record(selected_model)),
                encoding="utf-8",
            )
            components = build_cift_window_selector_runtime_components(
                CiftRuntimeWindowSelectorConfig(
                    detector_name="cift_runtime",
                    selected_choice_model_path=selected_model_path,
                    selected_choice_model_sha256=_sha256_file(selected_model_path),
                    fallback_model_path=None,
                    fallback_model_sha256=None,
                    feature_extractor=extractor,
                    feature_source="test_feature_map",
                    activation_failure_action=Action.BLOCK,
                )
            )
            runtime = AegisRuntime(
                turn_annotators=components.turn_annotators,
                pre_generation_detectors=components.pre_generation_detectors,
                post_generation_detectors=(),
                session_detectors=(),
                policy_engine=SeverityPolicyEngine(),
                audit_sink=InMemoryAuditSink(),
                model_provider=MockModelProvider(default_content="unsafe provider output"),
            )

            response = runtime.evaluate_turn(
                _request_with_metadata(
                    capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
                    metadata={"cift": {"readout_token_indices": [3, 4]}},
                )
            )

        self.assertEqual(Action.BLOCK, response.policy_decision.final_action)
        self.assertEqual("", response.output_text)
        self.assertEqual([("trace-cift-runtime", "selected_choice_window_layer_15")], extractor.calls)
        self.assertEqual(CapabilityStatus.DEGRADED, response.detector_results[0].capability_status)
        self.assertEqual(Action.BLOCK, response.detector_results[0].recommended_action)
        self.assertEqual("selected_choice_metadata_absent", response.detector_results[0].evidence["reason"])
        self.assertEqual(
            "runtime_candidate_missing_feature_fail_closed",
            response.detector_results[0].evidence["fail_closed_reason"],
        )

    def test_window_selector_component_builder_rejects_missing_artifact_path(self) -> None:
        extractor = RecordingFeatureExtractor(feature_vector=(1.0, 2.0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_model_path = root / "selected.json"
            fallback_model_path = root / "missing.json"
            selected_model_path.write_text(
                json.dumps(
                    cift_runtime_model_to_dict(
                        _selector_model(
                            feature_key="selected_choice_window_layer_15",
                            model_bundle_id="selected-choice-bundle",
                            logistic_coefficients=(1.0, 1.0),
                        )
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(CiftRuntimeDetectorError, "CIFT runtime model path does not exist"):
                build_cift_window_selector_runtime_components(
                    CiftRuntimeWindowSelectorConfig(
                        detector_name="cift_runtime",
                        selected_choice_model_path=selected_model_path,
                        selected_choice_model_sha256=_sha256_file(selected_model_path),
                        fallback_model_path=fallback_model_path,
                        fallback_model_sha256=None,
                        feature_extractor=extractor,
                        feature_source="test_feature_map",
                        activation_failure_action=Action.BLOCK,
                    )
                )

    def test_window_selector_component_builder_rejects_selected_choice_sha256_mismatch(self) -> None:
        extractor = RecordingFeatureExtractor(feature_vector=(1.0, 2.0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_model_path = root / "selected.json"
            selected_model_path.write_text(
                json.dumps(
                    _runtime_candidate_selector_record(
                        _selector_model(
                            feature_key="selected_choice_window_layer_15",
                            model_bundle_id="selected-choice-bundle",
                            logistic_coefficients=(1.0, 1.0),
                        )
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(CiftRuntimeDetectorError, "sha256 mismatch"):
                build_cift_window_selector_runtime_components(
                    CiftRuntimeWindowSelectorConfig(
                        detector_name="cift_runtime",
                        selected_choice_model_path=selected_model_path,
                        selected_choice_model_sha256="f" * 64,
                        fallback_model_path=None,
                        fallback_model_sha256=None,
                        feature_extractor=extractor,
                        feature_source="test_feature_map",
                        activation_failure_action=Action.BLOCK,
                    )
                )

    def test_offline_eval_mode_can_score_feature_vector(self) -> None:
        detector = CiftRuntimeDetector(
            detector_name="cift_runtime",
            model=_runtime_model(positive_class_index=1),
            activation_failure_action=Action.ALLOW,
        )
        turn = _turn(
            capability_mode=CapabilityMode.OFFLINE_EVAL,
            metadata={"cift": {"feature_vectors": {"readout_window_layer_15": [0.0, 2.0]}}},
        )

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.ACTIVE, result.capability_status)
        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual("negative", result.evidence["operating_band"])

    def test_positive_class_zero_inverts_sklearn_binary_logistic_probability(self) -> None:
        model = _runtime_model(positive_class_index=0)

        prediction = predict_cift_runtime_model(model=model, feature_vector=(3.0, 2.0))

        self.assertAlmostEqual(1.0 - _sigmoid(1.25), prediction.score)
        self.assertEqual(Action.ALLOW, prediction.recommended_action)
        self.assertEqual("secret_present_safe", prediction.predicted_label)

    def test_model_round_trips_through_json_safe_dict(self) -> None:
        model = _runtime_model(positive_class_index=1)

        decoded = json.loads(json.dumps(cift_runtime_model_to_dict(model)))
        loaded = _model_from_temp_file(decoded)

        self.assertEqual(model.model_bundle_id, loaded.model_bundle_id)
        self.assertEqual(model.logistic_coefficients, loaded.logistic_coefficients)

    def test_mlp_runtime_model_scores_with_paper_forward_pass(self) -> None:
        model = _model_from_temp_file(_runtime_mlp_record())

        positive_prediction = predict_cift_runtime_model(model=model, feature_vector=(2.0, 2.0))
        negative_prediction = predict_cift_runtime_model(model=model, feature_vector=(0.0, 0.0))

        self.assertGreater(positive_prediction.score, 0.5)
        self.assertLess(negative_prediction.score, 0.5)
        self.assertEqual(Action.WARN, positive_prediction.recommended_action)
        self.assertEqual(Action.ALLOW, negative_prediction.recommended_action)
        self.assertEqual("exfiltration_intent", positive_prediction.predicted_label)

    def test_generated_v4_1_runtime_artifact_loads_without_research_imports(self) -> None:
        model = load_cift_runtime_model(RUNTIME_MODEL_PATH)

        self.assertEqual("aegis.cift_runtime_linear/v1", model.schema_version)
        self.assertEqual("readout_window_layer_15", model.feature_key)
        self.assertEqual(1024, model.feature_count)
        self.assertEqual("exfiltration_intent", model.positive_label)
        self.assertEqual(Action.WARN, model.positive_action)

    def test_missing_generated_artifact_would_break_the_runtime_integration_claim(self) -> None:
        self.assertTrue(RUNTIME_MODEL_PATH.exists())

    def test_feature_annotator_does_not_call_extractor_in_black_box_mode(self) -> None:
        extractor = RecordingFeatureExtractor(feature_vector=(3.0, 2.0))
        annotator = CiftFeatureVectorAnnotator(
            feature_key="readout_window_layer_15",
            extractor=extractor,
            source="test_self_hosted_extractor",
            selected_choice_window=False,
        )
        turn = _turn(capability_mode=CapabilityMode.BLACK_BOX, metadata={})

        annotated = annotator.annotate(turn)

        self.assertIs(turn, annotated)
        self.assertEqual([], extractor.calls)

    def test_feature_annotator_preserves_existing_cift_metadata_and_does_not_mutate_input(self) -> None:
        extractor = RecordingFeatureExtractor(feature_vector=(3.0, 2.0))
        annotator = CiftFeatureVectorAnnotator(
            feature_key="readout_window_layer_15",
            extractor=extractor,
            source="test_self_hosted_extractor",
            selected_choice_window=False,
        )
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={"cift": {"readout_token_indices": [1, 2, 3]}},
        )

        annotated = annotator.annotate(turn)

        self.assertNotEqual(id(turn.metadata), id(annotated.metadata))
        self.assertEqual({"readout_token_indices": [1, 2, 3]}, turn.metadata["cift"])
        self.assertEqual((3.0, 2.0), cift_feature_vector_from_turn(annotated, "readout_window_layer_15"))
        self.assertEqual([1, 2, 3], annotated.metadata["cift"]["readout_token_indices"])
        self.assertEqual("test_self_hosted_extractor", _feature_source(annotated))

    def test_feature_annotator_records_trusted_selected_choice_metadata(self) -> None:
        extractor = SelectedChoiceFeatureExtractor(
            feature_vector=(3.0, 2.0),
            selected_choice_readout_token_indices=(7, 8),
        )
        annotator = CiftFeatureVectorAnnotator(
            feature_key="selected_choice_window_layer_15",
            extractor=extractor,
            source="test_self_hosted_extractor",
            selected_choice_window=True,
        )
        turn = _turn(capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION, metadata={})

        annotated = annotator.annotate(turn)

        self.assertEqual(
            (3.0, 2.0),
            cift_feature_vector_from_turn(annotated, "selected_choice_window_layer_15"),
        )
        self.assertEqual(
            [7, 8],
            annotated.metadata["cift"]["selected_choice_readout_token_indices"],
        )
        self.assertEqual(
            {"source": "test_self_hosted_extractor", "token_count": 2},
            annotated.metadata["cift"]["selected_choice_readout_source"],
        )

    def test_feature_annotator_preserves_sidecar_provenance_in_detector_evidence(self) -> None:
        extractor = ProvenanceFeatureExtractor(
            feature_vector=(3.0, 2.0),
            selected_choice_readout_token_indices=(7, 8, 9, 10),
            provenance={
                "extractor_id": "trusted-activation-sidecar",
                "model_attestation_schema_version": "aegis.cift_model_attestation/v1",
                "model_id": "Qwen/Qwen3-4B",
                "revision": "main",
                "selected_device": "mps",
                "hidden_size": 4096,
                "layer_count": 36,
                "tokenizer_fingerprint_sha256": "b" * 64,
                "special_tokens_map_sha256": "c" * 64,
                "chat_template_sha256": "d" * 64,
                "prompt_renderer": "aegis.cift_prompt_renderer.trace_bridge/v1",
                "selected_choice_geometry": "aegis.cift_selected_choice_geometry.semantic_indirection/v1",
                "selected_choice_readout_token_count": 4,
                "extraction_receipt_schema_version": CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
                "feature_vector_length": 2,
                "feature_vector_sha256": "e" * 64,
                "rendered_prompt_sha256": "f" * 64,
                "readout_token_indices": [4, 5, 6, 7],
                "query_tail_readout_token_indices": [4, 5, 6, 7],
                "readout_window_source": "query_tail",
                "readout_source": {
                    "source": "trusted-activation-sidecar",
                    "readout_window": "query_tail",
                    "token_count": 4,
                },
                "selected_choice_readout_token_indices": [7, 8, 9, 10],
                "selected_choice_readout_token_indices_sha256": "a" * 64,
                "hidden_state_layer_count": 37,
                "hidden_state_device_observed": "mps:0",
                "input_device_observed": "mps:0",
            },
        )
        annotator = CiftFeatureVectorAnnotator(
            feature_key="selected_choice_window_layer_15",
            extractor=extractor,
            source="self_hosted_activation_extractor",
            selected_choice_window=True,
        )
        detector = CiftRuntimeWindowSelector(
            detector_name="cift_runtime",
            selected_choice_model=_selector_model(
                feature_key="selected_choice_window_layer_15",
                model_bundle_id="selected-choice-bundle",
                logistic_coefficients=(1.0, 1.0),
            ),
            fallback_model=None,
            activation_failure_action=Action.BLOCK,
        )
        annotated = annotator.annotate(_turn(capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION, metadata={}))

        result = detector.evaluate(turn=annotated, model_response=None)

        self.assertEqual([("trace-cift-runtime", "selected_choice_window_layer_15")], extractor.calls)
        self.assertEqual(CapabilityStatus.ACTIVE, result.capability_status)
        self.assertEqual("trusted-activation-sidecar", result.evidence["extractor_id"])
        self.assertEqual("Qwen/Qwen3-4B", result.evidence["extractor_model_id"])
        self.assertEqual("main", result.evidence["extractor_revision"])
        self.assertEqual("mps", result.evidence["extractor_selected_device"])
        self.assertEqual(4096, result.evidence["extractor_hidden_size"])
        self.assertEqual(36, result.evidence["extractor_layer_count"])
        self.assertEqual("b" * 64, result.evidence["extractor_tokenizer_fingerprint_sha256"])
        self.assertEqual("c" * 64, result.evidence["extractor_special_tokens_map_sha256"])
        self.assertEqual("d" * 64, result.evidence["extractor_chat_template_sha256"])
        self.assertEqual(
            "aegis.cift_prompt_renderer.trace_bridge/v1",
            result.evidence["extractor_prompt_renderer"],
        )
        self.assertEqual(
            "aegis.cift_selected_choice_geometry.semantic_indirection/v1",
            result.evidence["extractor_selected_choice_geometry"],
        )
        self.assertEqual(4, result.evidence["extractor_selected_choice_readout_token_count"])
        self.assertEqual(
            CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION, result.evidence["extractor_extraction_receipt_schema_version"]
        )
        self.assertEqual(2, result.evidence["extractor_feature_vector_length"])
        self.assertEqual("e" * 64, result.evidence["extractor_feature_vector_sha256"])
        self.assertEqual("f" * 64, result.evidence["extractor_rendered_prompt_sha256"])
        self.assertEqual([4, 5, 6, 7], result.evidence["extractor_readout_token_indices"])
        self.assertEqual([4, 5, 6, 7], result.evidence["extractor_query_tail_readout_token_indices"])
        self.assertEqual("query_tail", result.evidence["extractor_readout_window_source"])
        self.assertEqual(
            {"source": "trusted-activation-sidecar", "readout_window": "query_tail", "token_count": 4},
            result.evidence["extractor_readout_source"],
        )
        self.assertEqual([7, 8, 9, 10], result.evidence["extractor_selected_choice_readout_token_indices"])
        self.assertEqual("a" * 64, result.evidence["extractor_selected_choice_readout_token_indices_sha256"])
        self.assertEqual(37, result.evidence["extractor_hidden_state_layer_count"])
        self.assertEqual("mps:0", result.evidence["extractor_hidden_state_device_observed"])
        self.assertEqual("mps:0", result.evidence["extractor_input_device_observed"])

    def test_feature_annotator_rejects_sidecar_readout_count_mismatch(self) -> None:
        extractor = ProvenanceFeatureExtractor(
            feature_vector=(3.0, 2.0),
            selected_choice_readout_token_indices=(7, 8),
            provenance={
                "extractor_id": "trusted-activation-sidecar",
                "model_id": "Qwen/Qwen3-4B",
                "revision": "main",
                "selected_device": "mps",
                "selected_choice_readout_token_count": 4,
            },
        )
        annotator = CiftFeatureVectorAnnotator(
            feature_key="selected_choice_window_layer_15",
            extractor=extractor,
            source="self_hosted_activation_extractor",
            selected_choice_window=True,
        )

        annotated = annotator.annotate(_turn(capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION, metadata={}))

        self.assertNotIn("selected_choice_readout_token_indices", annotated.metadata["cift"])
        self.assertEqual(
            "extractor.selected_choice_readout_token_indices length must match selected_choice_readout_token_count",
            annotated.metadata["cift"]["activation_failures"]["selected_choice_window_layer_15"]["reason"],
        )

    def test_feature_annotator_requires_rich_extractor_to_return_selected_choice_indices_atomically(self) -> None:
        extractor = ProvenanceFeatureExtractor(
            feature_vector=(3.0, 2.0),
            selected_choice_readout_token_indices=None,
            provenance={
                "extractor_id": "trusted-activation-sidecar",
                "model_id": "Qwen/Qwen3-4B",
                "revision": "main",
                "selected_device": "mps",
                "selected_choice_readout_token_count": 4,
            },
        )
        annotator = CiftFeatureVectorAnnotator(
            feature_key="selected_choice_window_layer_15",
            extractor=extractor,
            source="self_hosted_activation_extractor",
            selected_choice_window=True,
        )
        detector = CiftRuntimeWindowSelector(
            detector_name="cift_runtime",
            selected_choice_model=_selector_model(
                feature_key="selected_choice_window_layer_15",
                model_bundle_id="selected-choice-bundle",
                logistic_coefficients=(1.0, 1.0),
            ),
            fallback_model=None,
            activation_failure_action=Action.BLOCK,
        )

        annotated = annotator.annotate(_turn(capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION, metadata={}))
        result = detector.evaluate(turn=annotated, model_response=None)

        self.assertEqual([("trace-cift-runtime", "selected_choice_window_layer_15")], extractor.calls)
        self.assertNotIn("selected_choice_readout_token_indices", annotated.metadata["cift"])
        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual(Action.BLOCK, result.recommended_action)
        self.assertEqual("activation_feature_vector_malformed", result.evidence["reason"])
        self.assertEqual(
            "selected_choice_feature_vector_activation_failure", result.evidence["cift_window_selection_reason"]
        )
        self.assertIn("required in the feature extraction response", result.evidence["error"])

    def test_feature_annotator_leaves_self_hosted_turn_degraded_when_extractor_returns_none(self) -> None:
        extractor = RecordingFeatureExtractor(feature_vector=None)
        annotator = CiftFeatureVectorAnnotator(
            feature_key="readout_window_layer_15",
            extractor=extractor,
            source="test_self_hosted_extractor",
            selected_choice_window=False,
        )
        turn = _turn(capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION, metadata={})

        annotated = annotator.annotate(turn)

        self.assertIs(turn, annotated)
        self.assertEqual([("trace-cift-runtime", "readout_window_layer_15")], extractor.calls)

    def test_feature_annotator_rejects_non_finite_extractor_values(self) -> None:
        extractor = RecordingFeatureExtractor(feature_vector=(1.0, math.inf))
        annotator = CiftFeatureVectorAnnotator(
            feature_key="readout_window_layer_15",
            extractor=extractor,
            source="test_self_hosted_extractor",
            selected_choice_window=False,
        )
        turn = _turn(capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION, metadata={})

        annotated = annotator.annotate(turn)

        self.assertEqual(
            "extractor.readout_window_layer_15[1] must be finite.",
            annotated.metadata["cift"]["activation_failures"]["readout_window_layer_15"]["reason"],
        )

    def test_normalized_turn_with_cift_feature_vector_rejects_bad_cift_metadata(self) -> None:
        turn = _turn(capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION, metadata={"cift": "bad"})

        with self.assertRaises(CiftRuntimeDetectorError):
            normalized_turn_with_cift_feature_vector(
                turn=turn,
                feature_key="readout_window_layer_15",
                feature_vector=(1.0, 2.0),
                source="test_self_hosted_extractor",
                provenance={},
            )

    def test_feature_vector_parser_rejects_malformed_cift_metadata(self) -> None:
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={"cift": "not-an-object"},
        )

        with self.assertRaises(CiftRuntimeDetectorError):
            cift_feature_vector_from_turn(turn=turn, feature_key="readout_window_layer_15")

    def test_feature_vector_parser_rejects_non_numeric_values(self) -> None:
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={"cift": {"feature_vectors": {"readout_window_layer_15": [1.0, "bad"]}}},
        )

        with self.assertRaises(CiftRuntimeDetectorError):
            cift_feature_vector_from_turn(turn=turn, feature_key="readout_window_layer_15")

    def test_feature_vector_parser_rejects_non_finite_values(self) -> None:
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={"cift": {"feature_vectors": {"readout_window_layer_15": [1.0, math.nan]}}},
        )

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "must be finite"):
            cift_feature_vector_from_turn(turn=turn, feature_key="readout_window_layer_15")

    def test_loader_rejects_invalid_schema(self) -> None:
        record = cift_runtime_model_to_dict(_runtime_model(positive_class_index=1))
        record["schema_version"] = "wrong"

        with self.assertRaises(CiftRuntimeDetectorError):
            _model_from_temp_file(record)

    def test_loader_rejects_non_finite_model_json_numbers(self) -> None:
        record = cift_runtime_model_to_dict(_runtime_model(positive_class_index=1))
        record["decision_threshold"] = float("nan")

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "non-finite JSON number"):
            _model_from_temp_file(record)

    def test_loader_rejects_feature_count_mismatch(self) -> None:
        record = cift_runtime_model_to_dict(_runtime_model(positive_class_index=1))
        record["scaler_scale"] = [1.0]

        with self.assertRaises(CiftRuntimeDetectorError):
            _model_from_temp_file(record)

    def test_loader_rejects_runtime_candidate_without_promotion_gates(self) -> None:
        record = cift_runtime_model_to_dict(_runtime_model(positive_class_index=1))
        record["candidate_status"] = "runtime_candidate"
        record["positive_action"] = "block"

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "promotion_gates"):
            _model_from_temp_file(record)

    def test_loader_rejects_runtime_candidate_with_failed_promotion_gates(self) -> None:
        record = _runtime_candidate_record_with_promotion_gates()
        promotion_gates = record["promotion_gates"]
        if not isinstance(promotion_gates, dict):
            raise AssertionError("promotion_gates must be an object.")
        runtime_candidate = promotion_gates["runtime_candidate"]
        if not isinstance(runtime_candidate, dict):
            raise AssertionError("promotion_gates.runtime_candidate must be an object.")
        runtime_candidate["eligible"] = False
        runtime_candidate["failed_requirements"] = ["metric_value must meet or exceed metric_threshold"]

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "failed_requirements"):
            _model_from_temp_file(record)

    def test_loader_accepts_runtime_candidate_with_promotion_gates(self) -> None:
        model = _model_from_temp_file(_runtime_candidate_record_with_promotion_gates())

        self.assertEqual("runtime_candidate", model.candidate_status)
        self.assertEqual("synthetic-cift-lab", model.training_dataset_id)

    def test_loader_rejects_linear_runtime_candidate_claiming_paper_mlp_promotion(self) -> None:
        record = _linear_runtime_candidate_record_with_paper_mlp_promotion_gates()

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "linear runtime model.*mlp_128_64_1"):
            _model_from_temp_file(record)

    def test_loader_accepts_mlp_runtime_candidate_claiming_paper_mlp_promotion(self) -> None:
        record = _runtime_mlp_record()
        record["candidate_status"] = "runtime_candidate"
        record["positive_action"] = "block"
        record["evaluation_report_ids"] = [
            "synthetic-sealed-holdout-report",
            "synthetic-metric-report",
            "synthetic-calibration-report",
            "synthetic-ablation-report",
            "synthetic-patching-report",
            "synthetic-failure-case-report",
            "synthetic-runtime-prevention-report",
            "synthetic-lineage-report",
        ]
        record["training_dataset_id"] = "synthetic-cift-lab"
        record["promotion_gates"] = _runtime_candidate_promotion_gates()

        model = _model_from_temp_file(record)

        self.assertEqual("runtime_candidate", model.candidate_status)
        self.assertEqual("synthetic-cift-lab", model.training_dataset_id)

    def test_loader_rejects_mlp_runtime_candidate_claiming_challenger_promotion(self) -> None:
        record = _runtime_candidate_record_with_promotion_gates()
        head_to_head_report_id = "synthetic-head-to-head-report"
        evaluation_report_ids = record["evaluation_report_ids"]
        if not isinstance(evaluation_report_ids, list):
            raise AssertionError("evaluation_report_ids must be a list.")
        evaluation_report_ids.append(head_to_head_report_id)
        required_report_ids = _promotion_gate_string_list(record=record, field_name="required_report_ids")
        required_report_ids.append(head_to_head_report_id)
        report_artifacts = _promotion_gate_report_artifacts(record)
        report_artifacts.append(
            {
                "report_id": head_to_head_report_id,
                "path": "introspection/data/reports/synthetic-head-to-head-report.json",
                "sha256": "f" * 64,
                "schema_version": "synthetic_report/v1",
            }
        )
        reports = _promotion_gate_mapping(record=record, field_name="reports")
        reports["head_to_head"] = head_to_head_report_id
        paper_method = _promotion_gate_mapping(record=record, field_name="paper_method")
        paper_method["probe_architecture"] = "linear_logistic"
        paper_method["training_loss"] = "bce"
        paper_method["head_to_head_report_id"] = head_to_head_report_id
        paper_method["paper_probe_metric_value"] = 0.9
        paper_method["candidate_probe_metric_value"] = 0.91

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "mlp_128_64_1 runtime model.*alternative"):
            _model_from_temp_file(record)

    def test_loader_rejects_offline_candidate_with_promotion_gates(self) -> None:
        record = cift_runtime_model_to_dict(_runtime_model(positive_class_index=1))
        record["promotion_gates"] = {"schema_version": "cift_promotion_gates/v1"}

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "only valid for runtime_candidate"):
            _model_from_temp_file(record)

    def test_loader_rejects_runtime_candidate_with_bad_promotion_schema(self) -> None:
        record = _runtime_candidate_record_with_promotion_gates()
        promotion_gates = _promotion_gates_from_record(record)
        promotion_gates["schema_version"] = "wrong"

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "promotion_gates.schema_version"):
            _model_from_temp_file(record)

    def test_loader_rejects_runtime_candidate_with_missing_report_ids(self) -> None:
        record = _runtime_candidate_record_with_promotion_gates()
        gate = _promotion_gate_from_record(record)
        gate["missing_report_ids"] = ["synthetic-missing-report"]

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "missing_report_ids"):
            _model_from_temp_file(record)

    def test_loader_rejects_runtime_candidate_with_missing_model_report(self) -> None:
        record = _runtime_candidate_record_with_promotion_gates()
        required_report_ids = _promotion_gate_string_list(record=record, field_name="required_report_ids")
        required_report_ids.append("synthetic-report-not-in-model")

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "missing from model evaluation_report_ids"):
            _model_from_temp_file(record)

    def test_loader_rejects_runtime_candidate_without_report_artifacts(self) -> None:
        record = _runtime_candidate_record_with_promotion_gates()
        gate = _promotion_gate_from_record(record)
        gate.pop("report_artifacts")

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "report_artifacts"):
            _model_from_temp_file(record)

    def test_loader_rejects_runtime_candidate_with_bad_report_artifact_digest(self) -> None:
        record = _runtime_candidate_record_with_promotion_gates()
        report_artifacts = _promotion_gate_report_artifacts(record)
        first_artifact = report_artifacts[0]
        if not isinstance(first_artifact, dict):
            raise AssertionError("report_artifacts entries must be objects.")
        first_artifact["sha256"] = "not-a-digest"

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "sha256"):
            _model_from_temp_file(record)

    def test_loader_rejects_runtime_candidate_with_non_preventive_positive_action(self) -> None:
        for positive_action in ("warn", "sanitize"):
            with self.subTest(positive_action=positive_action):
                record = _runtime_candidate_record_with_promotion_gates()
                record["positive_action"] = positive_action

                with self.assertRaisesRegex(CiftRuntimeDetectorError, "positive_action"):
                    _model_from_temp_file(record)

    def test_loader_rejects_runtime_candidate_with_metric_below_threshold(self) -> None:
        record = _runtime_candidate_record_with_promotion_gates()
        metric = _promotion_gate_mapping(record=record, field_name="metric")
        metric["value"] = 0.89

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "metric.value"):
            _model_from_temp_file(record)

    def test_loader_rejects_runtime_candidate_with_ablation_below_threshold(self) -> None:
        record = _runtime_candidate_record_with_promotion_gates()
        ablation = _promotion_gate_mapping(record=record, field_name="ablation")
        ablation["delta"] = 0.09

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "ablation.delta"):
            _model_from_temp_file(record)

    def test_loader_rejects_runtime_candidate_with_duplicate_split_ids(self) -> None:
        record = _runtime_candidate_record_with_promotion_gates()
        splits = _promotion_gate_mapping(record=record, field_name="splits")
        splits["sealed_holdout"] = splits["heldout"]

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "split ids must be distinct"):
            _model_from_temp_file(record)

    def test_loader_rejects_runtime_candidate_with_static_secret_token_paper_method(self) -> None:
        record = _runtime_candidate_record_with_promotion_gates()
        paper_method = _promotion_gate_mapping(record=record, field_name="paper_method")
        paper_method["uses_static_secret_token_positions"] = True

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "uses_static_secret_token_positions"):
            _model_from_temp_file(record)

    def test_loader_rejects_challenger_without_head_to_head_report(self) -> None:
        record = _linear_runtime_candidate_record_with_paper_mlp_promotion_gates()
        paper_method = _promotion_gate_mapping(record=record, field_name="paper_method")
        paper_method["probe_architecture"] = "linear_logistic"
        paper_method["training_loss"] = "bce"

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "head_to_head_report_id"):
            _model_from_temp_file(record)

    def test_loader_rejects_challenger_head_to_head_missing_from_model_reports(self) -> None:
        record = _linear_runtime_candidate_record_with_paper_mlp_promotion_gates()
        paper_method = _promotion_gate_mapping(record=record, field_name="paper_method")
        paper_method["probe_architecture"] = "linear_logistic"
        paper_method["training_loss"] = "bce"
        paper_method["head_to_head_report_id"] = "synthetic-head-to-head-report"
        paper_method["paper_probe_metric_value"] = 0.9
        paper_method["candidate_probe_metric_value"] = 0.91

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "head_to_head_report_id"):
            _model_from_temp_file(record)

    def test_loader_accepts_challenger_with_head_to_head_win(self) -> None:
        record = _linear_runtime_candidate_record_with_challenger_promotion_gates()

        model = _model_from_temp_file(record)

        self.assertEqual("runtime_candidate", model.candidate_status)

    def test_loader_accepts_raw_activation_challenger_promotion_exception(self) -> None:
        record = _raw_activation_challenger_record(paper_metric=0.9, candidate_metric=0.91)

        model = _model_from_temp_file(record)

        self.assertEqual("runtime_candidate", model.candidate_status)

    def test_loader_rejects_raw_activation_challenger_that_only_ties_paper(self) -> None:
        record = _raw_activation_challenger_record(paper_metric=0.91, candidate_metric=0.91)

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "candidate metric must exceed paper metric"):
            _model_from_temp_file(record)

    def test_loader_rejects_raw_activation_challenger_with_cci_covariance(self) -> None:
        record = _raw_activation_challenger_record(paper_metric=0.9, candidate_metric=0.91)
        paper_method = _promotion_gate_mapping(record=record, field_name="paper_method")
        paper_method["covariance_estimator"] = "diagonal_covariance"

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "covariance_estimator must be not_applicable"):
            _model_from_temp_file(record)

    def test_loader_rejects_raw_activation_challenger_with_layer_weighting(self) -> None:
        record = _raw_activation_challenger_record(paper_metric=0.9, candidate_metric=0.91)
        paper_method = _promotion_gate_mapping(record=record, field_name="paper_method")
        paper_method["layer_weighting"] = "softplus_nonnegative_cfs"

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "layer_weighting must be not_applicable"):
            _model_from_temp_file(record)

    def test_loader_rejects_raw_activation_challenger_with_nonzero_ridge(self) -> None:
        record = _raw_activation_challenger_record(paper_metric=0.9, candidate_metric=0.91)
        paper_method = _promotion_gate_mapping(record=record, field_name="paper_method")
        paper_method["ridge"] = 0.001

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "ridge must be 0.0"):
            _model_from_temp_file(record)

    def test_loader_rejects_raw_activation_challenger_without_exception_rationale(self) -> None:
        record = _raw_activation_challenger_record(paper_metric=0.9, candidate_metric=0.91)
        paper_method = _promotion_gate_mapping(record=record, field_name="paper_method")
        paper_method["paper_faithfulness_exception"] = ""

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "paper_faithfulness_exception"):
            _model_from_temp_file(record)

    def test_loader_rejects_non_finite_probability_fields(self) -> None:
        model = replace(_runtime_model(positive_class_index=1), decision_threshold=math.nan)

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "decision_threshold must be finite"):
            validate_cift_runtime_model(model)

    def test_loader_rejects_non_finite_vector_fields(self) -> None:
        model = replace(_runtime_model(positive_class_index=1), scaler_mean=(1.0, math.inf))

        with self.assertRaisesRegex(CiftRuntimeDetectorError, r"scaler_mean\[1\] must be finite"):
            validate_cift_runtime_model(model)

    def test_loader_rejects_non_finite_logistic_intercept(self) -> None:
        model = replace(_runtime_model(positive_class_index=1), logistic_intercept=math.inf)

        with self.assertRaisesRegex(CiftRuntimeDetectorError, "logistic_intercept must be finite"):
            validate_cift_runtime_model(model)

    def test_predict_rejects_wrong_feature_vector_length(self) -> None:
        model = _runtime_model(positive_class_index=1)

        with self.assertRaises(CiftRuntimeDetectorError):
            predict_cift_runtime_model(model=model, feature_vector=(1.0,))

    def test_predict_rejects_non_finite_feature_vector_values(self) -> None:
        model = _runtime_model(positive_class_index=1)

        with self.assertRaisesRegex(CiftRuntimeDetectorError, r"feature_vector\[1\] must be finite"):
            predict_cift_runtime_model(model=model, feature_vector=(1.0, math.nan))

    def test_model_validation_rejects_non_positive_scaler_scale(self) -> None:
        model = CiftRuntimeLinearModel(
            schema_version="aegis.cift_runtime_linear/v1",
            model_bundle_id="bundle",
            source_model_id="model",
            source_revision="main",
            source_selected_device="cpu",
            source_hidden_size=2,
            source_layer_count=1,
            tokenizer_fingerprint_sha256="b" * 64,
            special_tokens_map_sha256="c" * 64,
            chat_template_sha256="d" * 64,
            training_dataset_id="dataset",
            source_artifact_sha256="a" * 64,
            evaluation_report_ids=("report",),
            task_name="task",
            feature_key="readout_window_layer_15",
            feature_count=2,
            label_names=("secret_present_safe", "exfiltration_intent"),
            positive_label="exfiltration_intent",
            positive_class_index=1,
            class_indices=(0, 1),
            decision_threshold=0.5,
            score_semantics="probability",
            confidence=0.7,
            candidate_status="offline_research_candidate",
            scaler_mean=(0.0, 0.0),
            scaler_scale=(1.0, 0.0),
            logistic_coefficients=(1.0, 1.0),
            logistic_intercept=0.0,
            negative_action=Action.ALLOW,
            positive_action=Action.WARN,
        )

        with self.assertRaises(CiftRuntimeDetectorError):
            validate_cift_runtime_model(model)


def _runtime_model(positive_class_index: int) -> CiftRuntimeLinearModel:
    if positive_class_index == 1:
        label_names = ("secret_present_safe", "exfiltration_intent")
        positive_label = "exfiltration_intent"
    else:
        label_names = ("exfiltration_intent", "secret_present_safe")
        positive_label = "exfiltration_intent"
    return CiftRuntimeLinearModel(
        schema_version="aegis.cift_runtime_linear/v1",
        model_bundle_id="test_bundle",
        source_model_id="test-model",
        source_revision="main",
        source_selected_device="cpu",
        source_hidden_size=2,
        source_layer_count=1,
        tokenizer_fingerprint_sha256="b" * 64,
        special_tokens_map_sha256="c" * 64,
        chat_template_sha256="d" * 64,
        training_dataset_id="test-dataset",
        source_artifact_sha256="a" * 64,
        evaluation_report_ids=("test-report",),
        task_name="safe_secret_vs_exfiltration",
        feature_key="readout_window_layer_15",
        feature_count=2,
        label_names=label_names,
        positive_label=positive_label,
        positive_class_index=positive_class_index,
        class_indices=(0, 1),
        decision_threshold=0.5,
        score_semantics="test_probability",
        confidence=0.7,
        candidate_status="offline_research_candidate",
        scaler_mean=(1.0, 2.0),
        scaler_scale=(2.0, 4.0),
        logistic_coefficients=(1.0, -0.5),
        logistic_intercept=0.25,
        negative_action=Action.ALLOW,
        positive_action=Action.WARN,
    )


def _selector_model(
    feature_key: str,
    model_bundle_id: str,
    logistic_coefficients: tuple[float, float],
) -> CiftRuntimeLinearModel:
    return CiftRuntimeLinearModel(
        schema_version="aegis.cift_runtime_linear/v1",
        model_bundle_id=model_bundle_id,
        source_model_id="test-model",
        source_revision="main",
        source_selected_device="cpu",
        source_hidden_size=2,
        source_layer_count=1,
        tokenizer_fingerprint_sha256="b" * 64,
        special_tokens_map_sha256="c" * 64,
        chat_template_sha256="d" * 64,
        training_dataset_id="test-dataset",
        source_artifact_sha256="b" * 64,
        evaluation_report_ids=("test-report",),
        task_name="safe_secret_vs_exfiltration",
        feature_key=feature_key,
        feature_count=2,
        label_names=("secret_present_safe", "exfiltration_intent"),
        positive_label="exfiltration_intent",
        positive_class_index=1,
        class_indices=(0, 1),
        decision_threshold=0.5,
        score_semantics="test_probability",
        confidence=0.7,
        candidate_status="offline_research_candidate",
        scaler_mean=(0.0, 0.0),
        scaler_scale=(1.0, 1.0),
        logistic_coefficients=logistic_coefficients,
        logistic_intercept=0.0,
        negative_action=Action.ALLOW,
        positive_action=Action.WARN,
    )


def _runtime_candidate_selector_record(model: CiftRuntimeLinearModel) -> dict[str, object]:
    head_to_head_report_id = "synthetic-head-to-head-report"
    promoted_model = replace(
        model,
        training_dataset_id="synthetic-cift-lab",
        evaluation_report_ids=(
            "synthetic-sealed-holdout-report",
            "synthetic-metric-report",
            "synthetic-calibration-report",
            "synthetic-ablation-report",
            "synthetic-patching-report",
            "synthetic-failure-case-report",
            "synthetic-runtime-prevention-report",
            "synthetic-lineage-report",
            head_to_head_report_id,
        ),
        candidate_status="runtime_candidate",
        positive_action=Action.BLOCK,
    )
    record = cift_runtime_model_to_dict(promoted_model)
    record["promotion_gates"] = _runtime_candidate_promotion_gates()
    required_report_ids = _promotion_gate_string_list(record=record, field_name="required_report_ids")
    required_report_ids.append(head_to_head_report_id)
    report_artifacts = _promotion_gate_report_artifacts(record)
    report_artifacts.append(
        {
            "report_id": head_to_head_report_id,
            "path": "introspection/data/reports/synthetic-head-to-head-report.json",
            "sha256": "f" * 64,
            "schema_version": "synthetic_report/v1",
        }
    )
    reports = _promotion_gate_mapping(record=record, field_name="reports")
    reports["head_to_head"] = head_to_head_report_id
    paper_method = _promotion_gate_mapping(record=record, field_name="paper_method")
    paper_method["probe_architecture"] = "linear_logistic"
    paper_method["training_loss"] = "bce"
    paper_method["head_to_head_report_id"] = head_to_head_report_id
    paper_method["paper_probe_metric_value"] = 0.9
    paper_method["candidate_probe_metric_value"] = 0.91
    return record


def _runtime_mlp_record() -> dict[str, object]:
    first_weights = [[0.0 for _column in range(128)] for _row in range(2)]
    first_weights[0][0] = 1.0
    first_weights[1][1] = 1.0
    second_weights = [[0.0 for _column in range(64)] for _row in range(128)]
    second_weights[0][0] = 1.0
    second_weights[1][1] = 1.0
    output_weights = [0.0 for _index in range(64)]
    output_weights[0] = 1.0
    output_weights[1] = 1.0
    return {
        "schema_version": "aegis.cift_runtime_mlp/v1",
        "model_bundle_id": "paper-mlp-bundle",
        "source_model_id": "test-model",
        "source_revision": "main",
        "source_selected_device": "cpu",
        "source_hidden_size": 2,
        "source_layer_count": 1,
        "tokenizer_fingerprint_sha256": "b" * 64,
        "special_tokens_map_sha256": "c" * 64,
        "chat_template_sha256": "d" * 64,
        "training_dataset_id": "test-dataset",
        "source_artifact_sha256": "c" * 64,
        "evaluation_report_ids": ["test-report"],
        "task_name": "safe_secret_vs_exfiltration",
        "feature_key": "readout_window_layer_15",
        "feature_count": 2,
        "label_names": ["secret_present_safe", "exfiltration_intent"],
        "positive_label": "exfiltration_intent",
        "positive_class_index": 1,
        "class_indices": [0, 1],
        "decision_threshold": 0.5,
        "score_semantics": "test_probability",
        "confidence": 0.7,
        "candidate_status": "offline_research_candidate",
        "probe_architecture": "mlp_128_64_1",
        "raw_layer_weights": [0.0, 0.0],
        "first_weights": first_weights,
        "first_bias": [0.0 for _index in range(128)],
        "second_weights": second_weights,
        "second_bias": [0.0 for _index in range(64)],
        "output_weights": output_weights,
        "output_bias": -2.0,
        "negative_action": "allow",
        "positive_action": "warn",
    }


def _turn(capability_mode: CapabilityMode, metadata: dict[str, object]) -> NormalizedTurn:
    return NormalizedTurn(
        trace_id="trace-cift-runtime",
        session_id="session-cift-runtime",
        turn_index=1,
        capability_mode=capability_mode,
        model=ModelInfo(provider="mock", model_id="mock-qwen", revision=None, selected_device="cpu"),
        messages=(Message(role="user", content="hello"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata=metadata,
    )


def _request(capability_mode: CapabilityMode) -> RuntimeRequest:
    return _request_with_metadata(capability_mode=capability_mode, metadata={})


def _request_with_metadata(capability_mode: CapabilityMode, metadata: dict[str, object]) -> RuntimeRequest:
    return RuntimeRequest(
        trace_id="trace-cift-runtime",
        session_id="session-cift-runtime",
        turn_index=1,
        capability_mode=capability_mode,
        model=ModelInfo(provider="mock", model_id="mock-qwen", revision=None, selected_device="cpu"),
        messages=(Message(role="user", content="hello"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata=metadata,
    )


def _feature_source(turn: NormalizedTurn) -> object:
    cift_metadata = turn.metadata["cift"]
    if not isinstance(cift_metadata, dict):
        raise AssertionError("metadata.cift must be an object.")
    feature_sources = cift_metadata["feature_sources"]
    if not isinstance(feature_sources, dict):
        raise AssertionError("metadata.cift.feature_sources must be an object.")
    readout_source = feature_sources["readout_window_layer_15"]
    if not isinstance(readout_source, dict):
        raise AssertionError("feature source must be an object.")
    return readout_source["source"]


class RecordingFeatureExtractor:
    def __init__(self, feature_vector: tuple[float, ...] | None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._feature_vector = feature_vector

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        self.calls.append((turn.trace_id, feature_key))
        return self._feature_vector


class SelectedChoiceFeatureExtractor:
    def __init__(
        self,
        feature_vector: tuple[float, ...],
        selected_choice_readout_token_indices: tuple[int, ...],
    ) -> None:
        self._feature_vector = feature_vector
        self._selected_choice_readout_token_indices = selected_choice_readout_token_indices

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        return self._feature_vector

    def extract_selected_choice_readout_token_indices(
        self,
        turn: NormalizedTurn,
        feature_key: str,
    ) -> tuple[int, ...] | None:
        return self._selected_choice_readout_token_indices


class ProvenanceFeatureExtractor:
    def __init__(
        self,
        feature_vector: tuple[float, ...],
        selected_choice_readout_token_indices: tuple[int, ...] | None,
        provenance: dict[str, object],
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self._extraction = CiftFeatureExtraction(
            feature_vector=feature_vector,
            selected_choice_readout_token_indices=selected_choice_readout_token_indices,
            provenance=provenance,
        )

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        raise AssertionError("rich provenance extractor should use extract_feature_extraction.")

    def extract_feature_extraction(self, turn: NormalizedTurn, feature_key: str) -> CiftFeatureExtraction:
        self.calls.append((turn.trace_id, feature_key))
        return self._extraction


class MalformedFeatureExtractor:
    def __init__(self, feature_vector: tuple[object, ...]) -> None:
        self._feature_vector = feature_vector

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[object, ...]:
        return self._feature_vector


class RaisingFeatureExtractor:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        raise self._error


class FeatureMapExtractor:
    def __init__(self, feature_vectors: dict[tuple[str, str], tuple[float, ...]]) -> None:
        self.calls: list[tuple[str, str]] = []
        self._feature_vectors = feature_vectors

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        self.calls.append((turn.trace_id, feature_key))
        return self._feature_vectors.get((turn.trace_id, feature_key))


def _model_from_temp_file(record: dict[str, object]) -> CiftRuntimeLinearModel:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return load_cift_runtime_model(path)


def _runtime_candidate_record_with_promotion_gates() -> dict[str, object]:
    record = _runtime_mlp_record()
    record["candidate_status"] = "runtime_candidate"
    record["positive_action"] = "block"
    record["evaluation_report_ids"] = [
        "synthetic-sealed-holdout-report",
        "synthetic-metric-report",
        "synthetic-calibration-report",
        "synthetic-ablation-report",
        "synthetic-patching-report",
        "synthetic-failure-case-report",
        "synthetic-runtime-prevention-report",
        "synthetic-lineage-report",
    ]
    record["training_dataset_id"] = "synthetic-cift-lab"
    record["promotion_gates"] = _runtime_candidate_promotion_gates()
    return record


def _linear_runtime_candidate_record_with_paper_mlp_promotion_gates() -> dict[str, object]:
    record = cift_runtime_model_to_dict(_runtime_model(positive_class_index=1))
    record["candidate_status"] = "runtime_candidate"
    record["positive_action"] = "block"
    record["evaluation_report_ids"] = [
        "synthetic-sealed-holdout-report",
        "synthetic-metric-report",
        "synthetic-calibration-report",
        "synthetic-ablation-report",
        "synthetic-patching-report",
        "synthetic-failure-case-report",
        "synthetic-runtime-prevention-report",
        "synthetic-lineage-report",
    ]
    record["training_dataset_id"] = "synthetic-cift-lab"
    record["promotion_gates"] = _runtime_candidate_promotion_gates()
    return record


def _linear_runtime_candidate_record_with_challenger_promotion_gates() -> dict[str, object]:
    record = _linear_runtime_candidate_record_with_paper_mlp_promotion_gates()
    head_to_head_report_id = "synthetic-head-to-head-report"
    evaluation_report_ids = record["evaluation_report_ids"]
    if not isinstance(evaluation_report_ids, list):
        raise AssertionError("evaluation_report_ids must be a list.")
    evaluation_report_ids.append(head_to_head_report_id)
    required_report_ids = _promotion_gate_string_list(record=record, field_name="required_report_ids")
    required_report_ids.append(head_to_head_report_id)
    report_artifacts = _promotion_gate_report_artifacts(record)
    report_artifacts.append(
        {
            "report_id": head_to_head_report_id,
            "path": "introspection/data/reports/synthetic-head-to-head-report.json",
            "sha256": "f" * 64,
            "schema_version": "synthetic_report/v1",
        }
    )
    reports = _promotion_gate_mapping(record=record, field_name="reports")
    reports["head_to_head"] = head_to_head_report_id
    paper_method = _promotion_gate_mapping(record=record, field_name="paper_method")
    paper_method["probe_architecture"] = "linear_logistic"
    paper_method["training_loss"] = "bce"
    paper_method["head_to_head_report_id"] = head_to_head_report_id
    paper_method["paper_probe_metric_value"] = 0.9
    paper_method["candidate_probe_metric_value"] = 0.91
    return record


def _raw_activation_challenger_record(paper_metric: float, candidate_metric: float) -> dict[str, object]:
    record = _linear_runtime_candidate_record_with_challenger_promotion_gates()
    paper_method = _promotion_gate_mapping(record=record, field_name="paper_method")
    paper_method["feature_representation"] = "raw_activation"
    paper_method["covariance_estimator"] = "not_applicable"
    paper_method["ridge"] = 0.0
    paper_method["layer_weighting"] = "not_applicable"
    paper_method["paper_probe_metric_value"] = paper_metric
    paper_method["candidate_probe_metric_value"] = candidate_metric
    paper_method["paper_faithfulness_exception"] = (
        "Raw selected-choice activations beat the paper MLP in head-to-head evaluation."
    )
    return record


def _runtime_candidate_promotion_gates() -> dict[str, object]:
    return {
        "schema_version": "cift_promotion_gates/v1",
        "runtime_candidate": {
            "schema_version": "cift_promotion_gate_result/v1",
            "evidence_id": "synthetic-promotion-evidence",
            "candidate_status": "runtime_candidate",
            "eligible": True,
            "behavior_id": "secret-exfiltration-intent",
            "behavior_description": "User request attempts to move a protected secret into an external channel.",
            "training_dataset_id": "synthetic-cift-lab",
            "splits": {
                "train": "synthetic-cift-lab/train",
                "calibration": "synthetic-cift-lab/calibration",
                "heldout": "synthetic-cift-lab/heldout",
                "sealed_holdout": "synthetic-cift-lab/sealed-holdout",
            },
            "metric": {
                "report_id": "synthetic-metric-report",
                "name": "sealed_holdout_macro_f1",
                "value": 0.91,
                "threshold": 0.9,
            },
            "ablation": {
                "report_id": "synthetic-ablation-report",
                "delta": 0.18,
                "delta_threshold": 0.1,
            },
            "reports": {
                "sealed_holdout": "synthetic-sealed-holdout-report",
                "metric": "synthetic-metric-report",
                "calibration": "synthetic-calibration-report",
                "ablation": "synthetic-ablation-report",
                "patching": "synthetic-patching-report",
                "failure_cases": "synthetic-failure-case-report",
                "runtime_prevention": "synthetic-runtime-prevention-report",
                "lineage": "synthetic-lineage-report",
                "head_to_head": None,
            },
            "paper_method": {
                "readout_position_contract": "post_secret_post_query_causal_readout",
                "monitored_layer_policy": "last_quarter_transformer_layers",
                "feature_representation": "diagonal_mahalanobis_cci",
                "covariance_estimator": "diagonal_covariance",
                "ridge": 0.001,
                "layer_weighting": "softplus_nonnegative_cfs",
                "probe_architecture": "mlp_128_64_1",
                "training_loss": "bce_with_l1_softplus_weight_sparsity",
                "pre_output": True,
                "uses_static_secret_token_positions": False,
                "head_to_head_report_id": None,
                "paper_probe_metric_value": None,
                "candidate_probe_metric_value": None,
            },
            "required_report_ids": [
                "synthetic-sealed-holdout-report",
                "synthetic-metric-report",
                "synthetic-calibration-report",
                "synthetic-ablation-report",
                "synthetic-patching-report",
                "synthetic-failure-case-report",
                "synthetic-runtime-prevention-report",
                "synthetic-lineage-report",
            ],
            "report_artifacts": [
                {
                    "report_id": "synthetic-sealed-holdout-report",
                    "path": "introspection/data/reports/synthetic-sealed-holdout-report.json",
                    "sha256": "0".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-metric-report",
                    "path": "introspection/data/reports/synthetic-metric-report.json",
                    "sha256": "1".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-calibration-report",
                    "path": "introspection/data/reports/synthetic-calibration-report.json",
                    "sha256": "2".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-ablation-report",
                    "path": "introspection/data/reports/synthetic-ablation-report.json",
                    "sha256": "3".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-patching-report",
                    "path": "introspection/data/reports/synthetic-patching-report.json",
                    "sha256": "4".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-failure-case-report",
                    "path": "introspection/data/reports/synthetic-failure-case-report.json",
                    "sha256": "5".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-runtime-prevention-report",
                    "path": "introspection/data/reports/synthetic-runtime-prevention-report.json",
                    "sha256": "6".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
                {
                    "report_id": "synthetic-lineage-report",
                    "path": "introspection/data/reports/synthetic-lineage-report.json",
                    "sha256": "7".zfill(64),
                    "schema_version": "synthetic_report/v1",
                },
            ],
            "missing_report_ids": [],
            "failed_requirements": [],
            "created_at": "2026-06-23T00:00:00Z",
        },
    }


def _promotion_gates_from_record(record: dict[str, object]) -> dict[str, object]:
    promotion_gates = record["promotion_gates"]
    if not isinstance(promotion_gates, dict):
        raise AssertionError("promotion_gates must be an object.")
    return promotion_gates


def _promotion_gate_from_record(record: dict[str, object]) -> dict[str, object]:
    promotion_gates = _promotion_gates_from_record(record)
    runtime_candidate = promotion_gates["runtime_candidate"]
    if not isinstance(runtime_candidate, dict):
        raise AssertionError("promotion_gates.runtime_candidate must be an object.")
    return runtime_candidate


def _promotion_gate_mapping(record: dict[str, object], field_name: str) -> dict[str, object]:
    runtime_candidate = _promotion_gate_from_record(record)
    value = runtime_candidate[field_name]
    if not isinstance(value, dict):
        raise AssertionError(f"promotion_gates.runtime_candidate.{field_name} must be an object.")
    return value


def _promotion_gate_string_list(record: dict[str, object], field_name: str) -> list[str]:
    runtime_candidate = _promotion_gate_from_record(record)
    value = runtime_candidate[field_name]
    if not isinstance(value, list):
        raise AssertionError(f"promotion_gates.runtime_candidate.{field_name} must be a list.")
    for item in value:
        if not isinstance(item, str):
            raise AssertionError(f"promotion_gates.runtime_candidate.{field_name} must contain only strings.")
    return value


def _promotion_gate_report_artifacts(record: dict[str, object]) -> list[object]:
    runtime_candidate = _promotion_gate_from_record(record)
    value = runtime_candidate["report_artifacts"]
    if not isinstance(value, list):
        raise AssertionError("promotion_gates.runtime_candidate.report_artifacts must be a list.")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


if __name__ == "__main__":
    unittest.main()
