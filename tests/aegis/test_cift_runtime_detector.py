from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from aegis.audit.memory import InMemoryAuditSink
from aegis.core.contracts import Action, CapabilityMode, CapabilityStatus, Message, ModelInfo, NormalizedTurn
from aegis.core.orchestrator import AegisRuntime, RuntimeRequest
from aegis.detectors.cift_runtime import (
    CiftFeatureVectorAnnotator,
    CiftRuntimeDetector,
    CiftRuntimeDetectorError,
    CiftRuntimeLinearModel,
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
                ),
            ),
            pre_generation_detectors=(
                CiftRuntimeDetector(detector_name="cift_runtime", model=_runtime_model(positive_class_index=1)),
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
        detector = CiftRuntimeDetector(detector_name="cift_runtime", model=_runtime_model(positive_class_index=1))
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
        detector = CiftRuntimeDetector(detector_name="cift_runtime", model=_runtime_model(positive_class_index=1))
        turn = _turn(capability_mode=CapabilityMode.BLACK_BOX, metadata={})

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.UNAVAILABLE, result.capability_status)
        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual("activation_access_unavailable", result.evidence["reason"])
        self.assertEqual("black_box", result.evidence["actual_capability_mode"])

    def test_self_hosted_turn_without_feature_vector_emits_degraded_result(self) -> None:
        detector = CiftRuntimeDetector(detector_name="cift_runtime", model=_runtime_model(positive_class_index=1))
        turn = _turn(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            metadata={"cift": {"readout_token_indices": [1, 2, 3]}},
        )

        result = detector.evaluate(turn, None)

        self.assertEqual(CapabilityStatus.DEGRADED, result.capability_status)
        self.assertEqual(Action.ALLOW, result.recommended_action)
        self.assertEqual("activation_feature_vector_missing", result.evidence["reason"])

    def test_offline_eval_mode_can_score_feature_vector(self) -> None:
        detector = CiftRuntimeDetector(detector_name="cift_runtime", model=_runtime_model(positive_class_index=1))
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

    def test_feature_annotator_leaves_self_hosted_turn_degraded_when_extractor_returns_none(self) -> None:
        extractor = RecordingFeatureExtractor(feature_vector=None)
        annotator = CiftFeatureVectorAnnotator(
            feature_key="readout_window_layer_15",
            extractor=extractor,
            source="test_self_hosted_extractor",
        )
        turn = _turn(capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION, metadata={})

        annotated = annotator.annotate(turn)

        self.assertIs(turn, annotated)
        self.assertEqual([("trace-cift-runtime", "readout_window_layer_15")], extractor.calls)

    def test_normalized_turn_with_cift_feature_vector_rejects_bad_cift_metadata(self) -> None:
        turn = _turn(capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION, metadata={"cift": "bad"})

        with self.assertRaises(CiftRuntimeDetectorError):
            normalized_turn_with_cift_feature_vector(
                turn=turn,
                feature_key="readout_window_layer_15",
                feature_vector=(1.0, 2.0),
                source="test_self_hosted_extractor",
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

    def test_loader_rejects_invalid_schema(self) -> None:
        record = cift_runtime_model_to_dict(_runtime_model(positive_class_index=1))
        record["schema_version"] = "wrong"

        with self.assertRaises(CiftRuntimeDetectorError):
            _model_from_temp_file(record)

    def test_loader_rejects_feature_count_mismatch(self) -> None:
        record = cift_runtime_model_to_dict(_runtime_model(positive_class_index=1))
        record["scaler_scale"] = [1.0]

        with self.assertRaises(CiftRuntimeDetectorError):
            _model_from_temp_file(record)

    def test_predict_rejects_wrong_feature_vector_length(self) -> None:
        model = _runtime_model(positive_class_index=1)

        with self.assertRaises(CiftRuntimeDetectorError):
            predict_cift_runtime_model(model=model, feature_vector=(1.0,))

    def test_model_validation_rejects_non_positive_scaler_scale(self) -> None:
        model = CiftRuntimeLinearModel(
            schema_version="aegis.cift_runtime_linear/v1",
            model_bundle_id="bundle",
            source_model_id="model",
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
    return RuntimeRequest(
        trace_id="trace-cift-runtime",
        session_id="session-cift-runtime",
        turn_index=1,
        capability_mode=capability_mode,
        model=ModelInfo(provider="mock", model_id="mock-qwen", revision=None, selected_device="cpu"),
        messages=(Message(role="user", content="hello"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={},
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


def _model_from_temp_file(record: dict[str, object]) -> CiftRuntimeLinearModel:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return load_cift_runtime_model(path)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


if __name__ == "__main__":
    unittest.main()
