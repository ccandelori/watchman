import json
import tempfile
import unittest
from pathlib import Path

from aegis_introspection.binary_tasks import BinaryMethodName
from aegis_introspection.error_analysis import (
    BinaryErrorAnalysisReport,
    BinaryExamplePrediction,
    BinaryFamilyErrorSummary,
    BinaryMethodErrorAnalysis,
    BinaryTaskErrorAnalysis,
)
from aegis_introspection.residual_error_comparison import (
    ResidualErrorSuiteInput,
    compare_binary_error_residuals,
    compare_binary_error_residual_suite,
    render_residual_error_comparison_markdown,
    render_residual_error_suite_markdown,
    write_residual_error_comparison_json,
    write_residual_error_comparison_markdown,
    write_residual_error_suite_json,
    write_residual_error_suite_markdown,
)


def _prediction(
    example_id: str,
    family: str,
    true_label: str,
    predicted_label: str,
) -> BinaryExamplePrediction:
    return BinaryExamplePrediction(
        fold_index=1,
        example_id=example_id,
        family=family,
        source_label=true_label,
        true_label=true_label,
        predicted_label=predicted_label,
        is_correct=predicted_label == true_label,
    )


def _family_summaries(predictions: tuple[BinaryExamplePrediction, ...]) -> tuple[BinaryFamilyErrorSummary, ...]:
    return tuple(
        BinaryFamilyErrorSummary(
            family=prediction.family,
            true_label=prediction.true_label,
            example_count=1,
            correct_count=1 if prediction.is_correct else 0,
            error_count=0 if prediction.is_correct else 1,
            accuracy=1.0 if prediction.is_correct else 0.0,
            predicted_label_counts=((prediction.predicted_label, 1),),
        )
        for prediction in predictions
    )


def _method(
    method_name: BinaryMethodName,
    feature_name: str,
    predictions: tuple[BinaryExamplePrediction, ...],
) -> BinaryMethodErrorAnalysis:
    correct_count = sum(1 for prediction in predictions if prediction.is_correct)
    prediction_count = len(predictions)
    return BinaryMethodErrorAnalysis(
        method_name=method_name,
        feature_name=feature_name,
        label_names=("exfiltration_intent", "secret_present_safe"),
        prediction_count=prediction_count,
        correct_count=correct_count,
        error_count=prediction_count - correct_count,
        accuracy=float(correct_count / prediction_count),
        family_summaries=_family_summaries(predictions),
        predictions=predictions,
    )


def _report(feature_name: str, predictions: tuple[BinaryExamplePrediction, ...]) -> BinaryErrorAnalysisReport:
    return BinaryErrorAnalysisReport(
        source_model_id="synthetic",
        source_revision="main",
        source_selected_device="cpu",
        evaluation_strategy="stratified_group_kfold",
        fold_count=2,
        random_seed=7,
        regularization_c=1.0,
        max_iter=1000,
        activation_feature_key=feature_name,
        tasks=(
            BinaryTaskErrorAnalysis(
                task_name="safe_secret_vs_exfiltration",
                description="Classify safe secret handling against exfiltration-oriented secret handling.",
                label_names=("exfiltration_intent", "secret_present_safe"),
                methods=(
                    _method(
                        method_name="activation_probe",
                        feature_name=feature_name,
                        predictions=predictions,
                    ),
                ),
            ),
        ),
    )


class ResidualErrorComparisonTest(unittest.TestCase):
    def test_compare_binary_error_residuals_counts_fixed_persistent_and_introduced_errors(self) -> None:
        reference_report = _report(
            feature_name="mean_pool_layer_18",
            predictions=(
                _prediction("example_001", "family_a", "secret_present_safe", "exfiltration_intent"),
                _prediction("example_002", "family_b", "exfiltration_intent", "secret_present_safe"),
                _prediction("example_003", "family_c", "secret_present_safe", "secret_present_safe"),
                _prediction("example_004", "family_d", "exfiltration_intent", "exfiltration_intent"),
            ),
        )
        candidate_report = _report(
            feature_name="final_token_layer_11",
            predictions=(
                _prediction("example_001", "family_a", "secret_present_safe", "secret_present_safe"),
                _prediction("example_002", "family_b", "exfiltration_intent", "secret_present_safe"),
                _prediction("example_003", "family_c", "secret_present_safe", "exfiltration_intent"),
                _prediction("example_004", "family_d", "exfiltration_intent", "exfiltration_intent"),
            ),
        )

        report = compare_binary_error_residuals(
            reference_report=reference_report,
            candidate_report=candidate_report,
            task_name="safe_secret_vs_exfiltration",
            method_name="activation_probe",
        )

        self.assertEqual("mean_pool_layer_18", report.reference_feature_key)
        self.assertEqual("final_token_layer_11", report.candidate_feature_key)
        self.assertEqual(4, report.prediction_count)
        self.assertEqual(2, report.reference_error_count)
        self.assertEqual(2, report.candidate_error_count)
        self.assertEqual(1, report.fixed_error_count)
        self.assertEqual(1, report.persistent_error_count)
        self.assertEqual(1, report.introduced_error_count)
        self.assertEqual(("example_001",), tuple(error.example_id for error in report.fixed_errors))
        self.assertEqual(("example_002",), tuple(error.example_id for error in report.persistent_errors))
        self.assertEqual(("example_003",), tuple(error.example_id for error in report.introduced_errors))
        self.assertEqual(("family_a", "family_b", "family_c"), tuple(summary.family for summary in report.family_summaries))

    def test_render_residual_error_comparison_markdown_includes_count_summary(self) -> None:
        reference_report = _report(
            feature_name="mean_pool_layer_18",
            predictions=(
                _prediction("example_001", "family_a", "secret_present_safe", "exfiltration_intent"),
                _prediction("example_002", "family_b", "exfiltration_intent", "exfiltration_intent"),
            ),
        )
        candidate_report = _report(
            feature_name="final_token_layer_11",
            predictions=(
                _prediction("example_001", "family_a", "secret_present_safe", "secret_present_safe"),
                _prediction("example_002", "family_b", "exfiltration_intent", "secret_present_safe"),
            ),
        )

        report = compare_binary_error_residuals(
            reference_report=reference_report,
            candidate_report=candidate_report,
            task_name="safe_secret_vs_exfiltration",
            method_name="activation_probe",
        )

        markdown = render_residual_error_comparison_markdown(report)

        self.assertIn("# Residual Error Comparison", markdown)
        self.assertIn("Reference feature: `mean_pool_layer_18`", markdown)
        self.assertIn("Candidate feature: `final_token_layer_11`", markdown)
        self.assertIn("| Fixed Errors | Persistent Errors | Introduced Errors |", markdown)
        self.assertIn("`example_001`", markdown)
        self.assertIn("`example_002`", markdown)

    def test_write_residual_error_comparison_outputs_creates_files(self) -> None:
        reference_report = _report(
            feature_name="mean_pool_layer_18",
            predictions=(
                _prediction("example_001", "family_a", "secret_present_safe", "exfiltration_intent"),
                _prediction("example_002", "family_b", "exfiltration_intent", "exfiltration_intent"),
            ),
        )
        candidate_report = _report(
            feature_name="final_token_layer_11",
            predictions=(
                _prediction("example_001", "family_a", "secret_present_safe", "secret_present_safe"),
                _prediction("example_002", "family_b", "exfiltration_intent", "secret_present_safe"),
            ),
        )
        report = compare_binary_error_residuals(
            reference_report=reference_report,
            candidate_report=candidate_report,
            task_name="safe_secret_vs_exfiltration",
            method_name="activation_probe",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "residuals.json"
            markdown_path = Path(temp_dir) / "residuals.md"
            write_residual_error_comparison_json(json_path, report)
            write_residual_error_comparison_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual("final_token_layer_11", decoded["candidate_feature_key"])
        self.assertEqual(1, decoded["fixed_error_count"])
        self.assertEqual(1, decoded["introduced_error_count"])
        self.assertIn("Residual Error Comparison", markdown)

    def test_compare_binary_error_residual_suite_aggregates_reference_feature_deltas(self) -> None:
        candidate_report = _report(
            feature_name="concat(final_token_layer_11,final_token_layer_16)",
            predictions=(
                _prediction("example_001", "family_a", "secret_present_safe", "secret_present_safe"),
                _prediction("example_002", "family_b", "exfiltration_intent", "secret_present_safe"),
                _prediction("example_003", "family_c", "secret_present_safe", "exfiltration_intent"),
                _prediction("example_004", "family_d", "exfiltration_intent", "exfiltration_intent"),
            ),
        )
        reference_report = _report(
            feature_name="mean_pool_layer_18",
            predictions=(
                _prediction("example_001", "family_a", "secret_present_safe", "exfiltration_intent"),
                _prediction("example_002", "family_b", "exfiltration_intent", "secret_present_safe"),
                _prediction("example_003", "family_c", "secret_present_safe", "secret_present_safe"),
                _prediction("example_004", "family_d", "exfiltration_intent", "exfiltration_intent"),
            ),
        )
        single_layer_report = _report(
            feature_name="final_token_layer_16",
            predictions=(
                _prediction("example_001", "family_a", "secret_present_safe", "secret_present_safe"),
                _prediction("example_002", "family_b", "exfiltration_intent", "secret_present_safe"),
                _prediction("example_003", "family_c", "secret_present_safe", "secret_present_safe"),
                _prediction("example_004", "family_d", "exfiltration_intent", "exfiltration_intent"),
            ),
        )

        report = compare_binary_error_residual_suite(
            inputs=(
                ResidualErrorSuiteInput(
                    dataset_id="baseline",
                    reference_report=reference_report,
                    candidate_report=candidate_report,
                ),
                ResidualErrorSuiteInput(
                    dataset_id="baseline",
                    reference_report=single_layer_report,
                    candidate_report=candidate_report,
                ),
            ),
            task_name="safe_secret_vs_exfiltration",
            method_name="activation_probe",
        )

        self.assertEqual("concat(final_token_layer_11,final_token_layer_16)", report.candidate_feature_key)
        self.assertEqual(("mean_pool_layer_18", "final_token_layer_16"), report.reference_feature_keys)
        self.assertEqual(1, report.dataset_count)
        self.assertEqual(2, report.comparison_count)
        summaries = {summary.reference_feature_key: summary for summary in report.feature_summaries}
        self.assertEqual(1, summaries["mean_pool_layer_18"].fixed_error_count)
        self.assertEqual(1, summaries["mean_pool_layer_18"].introduced_error_count)
        self.assertEqual(0, summaries["mean_pool_layer_18"].net_error_delta)
        self.assertEqual(1, summaries["final_token_layer_16"].net_error_delta)

    def test_render_residual_error_suite_markdown_includes_aggregate_and_comparison_tables(self) -> None:
        candidate_report = _report(
            feature_name="concat(final_token_layer_11,final_token_layer_16)",
            predictions=(
                _prediction("example_001", "family_a", "secret_present_safe", "secret_present_safe"),
                _prediction("example_002", "family_b", "exfiltration_intent", "secret_present_safe"),
            ),
        )
        reference_report = _report(
            feature_name="mean_pool_layer_18",
            predictions=(
                _prediction("example_001", "family_a", "secret_present_safe", "exfiltration_intent"),
                _prediction("example_002", "family_b", "exfiltration_intent", "exfiltration_intent"),
            ),
        )
        report = compare_binary_error_residual_suite(
            inputs=(
                ResidualErrorSuiteInput(
                    dataset_id="baseline",
                    reference_report=reference_report,
                    candidate_report=candidate_report,
                ),
            ),
            task_name="safe_secret_vs_exfiltration",
            method_name="activation_probe",
        )

        markdown = render_residual_error_suite_markdown(report)

        self.assertIn("# Residual Error Suite", markdown)
        self.assertIn("Candidate feature: `concat(final_token_layer_11,final_token_layer_16)`", markdown)
        self.assertIn("| Reference Feature | Comparisons | Reference Errors |", markdown)
        self.assertIn("| Dataset | Reference Feature | Reference Errors |", markdown)

    def test_write_residual_error_suite_outputs_creates_files(self) -> None:
        candidate_report = _report(
            feature_name="concat(final_token_layer_11,final_token_layer_16)",
            predictions=(
                _prediction("example_001", "family_a", "secret_present_safe", "secret_present_safe"),
                _prediction("example_002", "family_b", "exfiltration_intent", "secret_present_safe"),
            ),
        )
        reference_report = _report(
            feature_name="mean_pool_layer_18",
            predictions=(
                _prediction("example_001", "family_a", "secret_present_safe", "exfiltration_intent"),
                _prediction("example_002", "family_b", "exfiltration_intent", "exfiltration_intent"),
            ),
        )
        report = compare_binary_error_residual_suite(
            inputs=(
                ResidualErrorSuiteInput(
                    dataset_id="baseline",
                    reference_report=reference_report,
                    candidate_report=candidate_report,
                ),
            ),
            task_name="safe_secret_vs_exfiltration",
            method_name="activation_probe",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "suite.json"
            markdown_path = Path(temp_dir) / "suite.md"
            write_residual_error_suite_json(json_path, report)
            write_residual_error_suite_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual("concat(final_token_layer_11,final_token_layer_16)", decoded["candidate_feature_key"])
        self.assertEqual(1, decoded["comparison_count"])
        self.assertIn("Residual Error Suite", markdown)


if __name__ == "__main__":
    unittest.main()
