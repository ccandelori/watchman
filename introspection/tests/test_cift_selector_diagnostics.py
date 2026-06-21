import unittest

from aegis_introspection.cift_calibration import CalibratedCiftPrediction, CiftCalibrationReport
from aegis_introspection.cift_selector_diagnostics import (
    SelectorDiagnosticConfig,
    build_selector_diagnostic_report,
    render_selector_diagnostic_markdown,
)
from aegis_introspection.policy_window_error_slices import PromptPolicyMetadata


def _prediction(
    example_id: str,
    true_label: str,
    probability: float,
) -> CalibratedCiftPrediction:
    predicted_label = "exfiltration_intent" if probability >= 0.5 else "secret_present_safe"
    return CalibratedCiftPrediction(
        fold_index=0,
        example_id=example_id,
        family=f"family_{example_id}",
        source_label=true_label,
        true_label=true_label,
        predicted_label=predicted_label,
        is_correct=true_label == predicted_label,
        positive_label="exfiltration_intent",
        positive_probability=probability,
    )


def _metadata(
    example_id: str,
    family: str,
    payload_condition: str,
    selected_mode: str,
) -> PromptPolicyMetadata:
    return PromptPolicyMetadata(
        example_id=example_id,
        family=family,
        credential_type="api_key",
        payload_condition=payload_condition,
        selected_field="credential_value",
        selected_mode=selected_mode,
        selected_action="copy",
    )


def _report() -> CiftCalibrationReport:
    return CiftCalibrationReport(
        source_model_id="Qwen/Qwen3-0.6B",
        source_revision="main",
        source_selected_device="cpu",
        evaluation_strategy="stratified_group_kfold",
        score_semantics="inner_cv_platt_calibrated_probability",
        task_name="safe_secret_vs_exfiltration",
        positive_label="exfiltration_intent",
        activation_feature_key="readout_window_layer_15",
        fold_count=5,
        inner_fold_count=3,
        random_seed=42,
        regularization_c=1.0,
        max_iter=1000,
        decision_threshold=0.5,
        accuracy=0.25,
        macro_f1=0.2,
        brier_score=0.4,
        log_loss=0.7,
        expected_calibration_error=0.3,
        confusion_matrix=((1, 1), (2, 0)),
        bin_summaries=(),
        predictions=(
            _prediction("near_fn", "exfiltration_intent", 0.47),
            _prediction("confident_fn", "exfiltration_intent", 0.12),
            _prediction("near_fp", "secret_present_safe", 0.56),
            _prediction("correct_safe", "secret_present_safe", 0.18),
        ),
    )


class CiftSelectorDiagnosticsTest(unittest.TestCase):
    def test_build_report_marks_margins_slice_calibration_and_threshold_effects(self) -> None:
        metadata_by_id = {
            "near_fn": _metadata("near_fn", "support_transcript", "payload", "mode_b"),
            "confident_fn": _metadata("confident_fn", "support_transcript", "payload", "mode_b"),
            "near_fp": _metadata("near_fp", "policy_note", "no_payload", "mode_a"),
            "correct_safe": _metadata("correct_safe", "policy_note", "no_payload", "mode_a"),
        }
        config = SelectorDiagnosticConfig(
            dimensions=("payload_condition", "selected_mode", "family", "source_label"),
            thresholds=(0.25, 0.5, 0.75),
            near_threshold_radius=0.1,
        )

        diagnostic = build_selector_diagnostic_report(
            report=_report(),
            metadata_by_id=metadata_by_id,
            config=config,
        )

        payload_summary = next(
            summary
            for summary in diagnostic.slice_summaries
            if summary.dimension == "payload_condition" and summary.value == "payload"
        )
        self.assertEqual(2, payload_summary.example_count)
        self.assertEqual(2, payload_summary.false_negative_count)
        self.assertAlmostEqual(1.0, payload_summary.empirical_positive_rate)
        self.assertAlmostEqual(0.295, payload_summary.mean_probability)
        self.assertEqual(1, payload_summary.near_threshold_error_count)
        self.assertEqual(1, payload_summary.confident_error_count)
        exfil_label_summary = next(
            summary
            for summary in diagnostic.slice_summaries
            if summary.dimension == "source_label" and summary.value == "exfiltration_intent"
        )
        self.assertEqual(2, exfil_label_summary.false_negative_count)

        threshold_25 = next(point for point in diagnostic.threshold_summaries if point.threshold == 0.25)
        threshold_50 = next(point for point in diagnostic.threshold_summaries if point.threshold == 0.5)
        self.assertLess(threshold_25.false_negative, threshold_50.false_negative)
        self.assertEqual(3, len(diagnostic.error_examples))
        self.assertEqual("confident_fn", diagnostic.error_examples[0].example_id)
        self.assertEqual("false_negative", diagnostic.error_examples[0].error_kind)

    def test_render_markdown_includes_high_signal_sections(self) -> None:
        metadata_by_id = {
            "near_fn": _metadata("near_fn", "support_transcript", "payload", "mode_b"),
            "confident_fn": _metadata("confident_fn", "support_transcript", "payload", "mode_b"),
            "near_fp": _metadata("near_fp", "policy_note", "no_payload", "mode_a"),
            "correct_safe": _metadata("correct_safe", "policy_note", "no_payload", "mode_a"),
        }
        diagnostic = build_selector_diagnostic_report(
            report=_report(),
            metadata_by_id=metadata_by_id,
            config=SelectorDiagnosticConfig(
                dimensions=("payload_condition",),
                thresholds=(0.25, 0.5),
                near_threshold_radius=0.1,
            ),
        )

        markdown = render_selector_diagnostic_markdown(diagnostic)

        self.assertIn("Selector-Window Score Diagnostics", markdown)
        self.assertIn("Threshold Sweep", markdown)
        self.assertIn("Slice Calibration And Errors", markdown)
        self.assertIn("confident_fn", markdown)

    def test_missing_metadata_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_selector_diagnostic_report(
                report=_report(),
                metadata_by_id={},
                config=SelectorDiagnosticConfig(
                    dimensions=("payload_condition",),
                    thresholds=(0.5,),
                    near_threshold_radius=0.1,
                ),
            )


if __name__ == "__main__":
    unittest.main()
