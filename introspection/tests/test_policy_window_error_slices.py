import unittest

from aegis_introspection.error_analysis import (
    BinaryErrorAnalysisReport,
    BinaryExamplePrediction,
    BinaryMethodErrorAnalysis,
    BinaryTaskErrorAnalysis,
)
from aegis_introspection.policy_window_error_slices import (
    PayloadCondition,
    PromptPolicyMetadata,
    SelectedMode,
    build_policy_window_error_slice_report,
)


def _prediction(
    example_id: str,
    true_label: str,
    predicted_label: str,
) -> BinaryExamplePrediction:
    return BinaryExamplePrediction(
        fold_index=0,
        example_id=example_id,
        family="family_a",
        source_label=true_label,
        true_label=true_label,
        predicted_label=predicted_label,
        is_correct=true_label == predicted_label,
    )


def _metadata(
    example_id: str,
    payload_condition: PayloadCondition,
    selected_mode: SelectedMode,
) -> PromptPolicyMetadata:
    return PromptPolicyMetadata(
        example_id=example_id,
        family="family_a",
        credential_type="api_key",
        payload_condition=payload_condition,
        selected_field="credential_value",
        selected_mode=selected_mode,
        selected_action="copy",
    )


class PolicyWindowErrorSliceTest(unittest.TestCase):
    def test_build_report_groups_errors_by_metadata_dimension(self) -> None:
        predictions = (
            _prediction("example_1", "exfiltration_intent", "secret_present_safe"),
            _prediction("example_2", "exfiltration_intent", "exfiltration_intent"),
            _prediction("example_3", "secret_present_safe", "exfiltration_intent"),
            _prediction("example_4", "secret_present_safe", "secret_present_safe"),
        )
        report = BinaryErrorAnalysisReport(
            source_model_id="Qwen/Qwen3-0.6B",
            source_revision="main",
            source_selected_device="cpu",
            evaluation_strategy="stratified_group_kfold",
            fold_count=5,
            random_seed=42,
            regularization_c=1.0,
            max_iter=1000,
            activation_feature_key="readout_window_layer_15",
            tasks=(
                BinaryTaskErrorAnalysis(
                    task_name="safe_secret_vs_exfiltration",
                    description="Task",
                    label_names=("exfiltration_intent", "secret_present_safe"),
                    methods=(
                        BinaryMethodErrorAnalysis(
                            method_name="activation_probe",
                            feature_name="readout_window_layer_15",
                            label_names=("exfiltration_intent", "secret_present_safe"),
                            prediction_count=4,
                            correct_count=2,
                            error_count=2,
                            accuracy=0.5,
                            family_summaries=(),
                            predictions=predictions,
                        ),
                    ),
                ),
            ),
        )
        metadata_by_id = {
            "example_1": _metadata("example_1", "payload", "mode_a"),
            "example_2": _metadata("example_2", "payload", "mode_a"),
            "example_3": _metadata("example_3", "no_payload", "mode_b"),
            "example_4": _metadata("example_4", "no_payload", "mode_b"),
        }

        sliced = build_policy_window_error_slice_report(
            report=report,
            metadata_by_id=metadata_by_id,
            task_name="safe_secret_vs_exfiltration",
            method_name="activation_probe",
            dimensions=("payload_condition", "selected_mode"),
        )

        payload_summaries = tuple(summary for summary in sliced.summaries if summary.dimension == "payload_condition")
        payload_summary = next(summary for summary in payload_summaries if summary.value == "payload")
        self.assertEqual(2, len(payload_summaries))
        self.assertEqual("exfiltration_intent", payload_summary.true_label)
        self.assertEqual(2, payload_summary.example_count)
        self.assertEqual(1, payload_summary.error_count)

    def test_build_report_rejects_missing_prompt_metadata(self) -> None:
        prediction = _prediction("missing", "exfiltration_intent", "secret_present_safe")
        report = BinaryErrorAnalysisReport(
            source_model_id="Qwen/Qwen3-0.6B",
            source_revision="main",
            source_selected_device="cpu",
            evaluation_strategy="stratified_group_kfold",
            fold_count=5,
            random_seed=42,
            regularization_c=1.0,
            max_iter=1000,
            activation_feature_key="readout_window_layer_15",
            tasks=(
                BinaryTaskErrorAnalysis(
                    task_name="safe_secret_vs_exfiltration",
                    description="Task",
                    label_names=("exfiltration_intent", "secret_present_safe"),
                    methods=(
                        BinaryMethodErrorAnalysis(
                            method_name="activation_probe",
                            feature_name="readout_window_layer_15",
                            label_names=("exfiltration_intent", "secret_present_safe"),
                            prediction_count=1,
                            correct_count=0,
                            error_count=1,
                            accuracy=0.0,
                            family_summaries=(),
                            predictions=(prediction,),
                        ),
                    ),
                ),
            ),
        )

        with self.assertRaises(ValueError):
            build_policy_window_error_slice_report(
                report=report,
                metadata_by_id={},
                task_name="safe_secret_vs_exfiltration",
                method_name="activation_probe",
                dimensions=("payload_condition",),
            )


if __name__ == "__main__":
    unittest.main()
