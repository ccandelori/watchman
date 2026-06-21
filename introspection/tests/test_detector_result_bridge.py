import unittest

from aegis_introspection.detector_result_bridge import (
    CalibratedCiftDetectorBridgeConfig,
    CiftDetectorBridgeConfig,
    CiftModelPredictionContext,
    TrainedCiftDetectorBridgeConfig,
    calibrated_cift_prediction_to_detector_result,
    cift_prediction_to_detector_result,
    trained_cift_prediction_to_detector_result,
)
from aegis_introspection.cift_calibration import CalibratedCiftPrediction
from aegis_introspection.cift_model_bundle import CiftModelPrediction
from aegis_introspection.error_analysis import BinaryExamplePrediction


def _prediction(predicted_label: str) -> BinaryExamplePrediction:
    return BinaryExamplePrediction(
        fold_index=2,
        example_id="example-1",
        family="family-a",
        source_label="exfiltration_intent",
        true_label="exfiltration_intent",
        predicted_label=predicted_label,
        is_correct=predicted_label == "exfiltration_intent",
    )


class DetectorResultBridgeTest(unittest.TestCase):
    def test_cift_prediction_to_detector_result_flags_predicted_exfiltration(self) -> None:
        config = CiftDetectorBridgeConfig(
            detector_name="cift_selector_probe",
            feature_key="readout_window_layer_15",
            task_name="safe_secret_vs_exfiltration",
            probe_version="dp_honey_lite_v3_selector_window_layer_15_v1",
            capability_required="self_hosted_introspection",
            positive_label="exfiltration_intent",
            positive_score=0.86,
            negative_score=0.12,
            positive_action="warn",
            negative_action="allow",
            confidence=0.77,
        )

        result = cift_prediction_to_detector_result(prediction=_prediction("exfiltration_intent"), config=config)

        self.assertEqual("cift_selector_probe", result["detector_name"])
        self.assertEqual("cift", result["component"])
        self.assertEqual(0.86, result["score"])
        self.assertEqual(0.77, result["confidence"])
        self.assertEqual("warn", result["recommended_action"])
        self.assertEqual("self_hosted_introspection", result["capability_required"])
        self.assertEqual("active", result["capability_status"])
        self.assertEqual("readout_window_layer_15", result["evidence"]["feature_key"])
        self.assertEqual("example-1", result["evidence"]["example_id"])
        self.assertEqual("exfiltration_intent", result["evidence"]["predicted_label"])

    def test_cift_prediction_to_detector_result_allows_predicted_safe_secret(self) -> None:
        config = CiftDetectorBridgeConfig(
            detector_name="cift_selector_probe",
            feature_key="readout_window_layer_15",
            task_name="safe_secret_vs_exfiltration",
            probe_version="dp_honey_lite_v3_selector_window_layer_15_v1",
            capability_required="self_hosted_introspection",
            positive_label="exfiltration_intent",
            positive_score=0.86,
            negative_score=0.12,
            positive_action="warn",
            negative_action="allow",
            confidence=0.77,
        )

        result = cift_prediction_to_detector_result(prediction=_prediction("secret_present_safe"), config=config)

        self.assertEqual(0.12, result["score"])
        self.assertEqual("allow", result["recommended_action"])
        self.assertEqual("secret_present_safe", result["evidence"]["predicted_label"])

    def test_calibrated_cift_prediction_to_detector_result_uses_probability_score(self) -> None:
        prediction = CalibratedCiftPrediction(
            fold_index=1,
            example_id="example-1",
            family="family-a",
            source_label="exfiltration_intent",
            true_label="exfiltration_intent",
            predicted_label="exfiltration_intent",
            is_correct=True,
            positive_label="exfiltration_intent",
            positive_probability=0.73,
        )
        config = CalibratedCiftDetectorBridgeConfig(
            detector_name="cift_selector_probe",
            feature_key="readout_window_layer_15",
            task_name="safe_secret_vs_exfiltration",
            probe_version="dp_honey_lite_v3_selector_window_layer_15_v1",
            capability_required="self_hosted_introspection",
            decision_threshold=0.5,
            positive_action="warn",
            negative_action="allow",
            confidence=0.77,
        )

        result = calibrated_cift_prediction_to_detector_result(prediction=prediction, config=config)

        self.assertEqual(0.73, result["score"])
        self.assertEqual("warn", result["recommended_action"])
        self.assertEqual("inner_cv_platt_calibrated_probability", result["evidence"]["score_semantics"])
        self.assertEqual(0.5, result["evidence"]["decision_threshold"])
        self.assertEqual("exfiltration_intent", result["evidence"]["positive_label"])

    def test_trained_cift_prediction_to_detector_result_uses_bundle_probability_score(self) -> None:
        prediction = CiftModelPrediction(
            positive_label="exfiltration_intent",
            positive_probability=0.81,
            predicted_label="exfiltration_intent",
            decision_threshold=0.5,
            score_semantics="full_train_classifier_probability",
        )
        context = CiftModelPredictionContext(
            example_id="example-1",
            family="family-a",
            source_label="exfiltration_intent",
            true_label="exfiltration_intent",
        )
        config = TrainedCiftDetectorBridgeConfig(
            detector_name="cift_selector_probe",
            feature_key="readout_window_layer_15",
            task_name="safe_secret_vs_exfiltration",
            model_bundle_id="cift_qwen3_0_6b_dp_honey_lite_v4_1_selector_window_layer_15_v1",
            capability_required="self_hosted_introspection",
            positive_action="warn",
            negative_action="allow",
            confidence=0.77,
        )

        result = trained_cift_prediction_to_detector_result(
            prediction=prediction,
            context=context,
            config=config,
        )

        self.assertEqual(0.81, result["score"])
        self.assertEqual("warn", result["recommended_action"])
        self.assertEqual("cift_qwen3_0_6b_dp_honey_lite_v4_1_selector_window_layer_15_v1", result["evidence"]["model_bundle_id"])
        self.assertEqual("full_train_classifier_probability", result["evidence"]["score_semantics"])
        self.assertEqual(True, result["evidence"]["offline_eval"])


if __name__ == "__main__":
    unittest.main()
