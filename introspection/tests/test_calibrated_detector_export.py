import json
import tempfile
import unittest
from pathlib import Path

from aegis_introspection.calibrated_detector_export import (
    CalibratedDetectorExportConfig,
    export_calibrated_cift_detector_results,
)
from aegis_introspection.cift_calibration import (
    CalibratedCiftPrediction,
    CalibrationBinSummary,
    CiftCalibrationReport,
    write_cift_calibration_json,
)


def _calibration_report() -> CiftCalibrationReport:
    return CiftCalibrationReport(
        source_model_id="Qwen/Qwen3-0.6B",
        source_revision="main",
        source_selected_device="cpu",
        evaluation_strategy="stratified_group_kfold_with_inner_platt_calibration",
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
        accuracy=1.0,
        macro_f1=1.0,
        brier_score=0.04,
        log_loss=0.12,
        expected_calibration_error=0.05,
        confusion_matrix=((1, 0), (0, 1)),
        bin_summaries=(
            CalibrationBinSummary(
                bin_index=1,
                lower_bound=0.0,
                upper_bound=0.1,
                example_count=1,
                mean_probability=0.08,
                empirical_positive_rate=0.0,
                absolute_gap=0.08,
            ),
        ),
        predictions=(
            CalibratedCiftPrediction(
                fold_index=1,
                example_id="safe-1",
                family="family-a",
                source_label="secret_present_safe",
                true_label="secret_present_safe",
                predicted_label="secret_present_safe",
                is_correct=True,
                positive_label="exfiltration_intent",
                positive_probability=0.08,
            ),
            CalibratedCiftPrediction(
                fold_index=1,
                example_id="exfil-1",
                family="family-a",
                source_label="exfiltration_intent",
                true_label="exfiltration_intent",
                predicted_label="exfiltration_intent",
                is_correct=True,
                positive_label="exfiltration_intent",
                positive_probability=0.81,
            ),
        ),
    )


def _runtime_turn(example_id: str, trace_id: str, turn_index: int) -> dict[str, object]:
    return {
        "trace_id": trace_id,
        "session_id": "session-1",
        "turn_index": turn_index,
        "metadata": {"example_id": example_id},
    }


class CalibratedDetectorExportTest(unittest.TestCase):
    def test_export_calibrated_cift_detector_results_writes_probability_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_turns_path = root / "turns.jsonl"
            calibration_report_path = root / "calibration.json"
            output_path = root / "detector_results.jsonl"
            runtime_turns_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        _runtime_turn("safe-1", "trace-safe", 1),
                        _runtime_turn("exfil-1", "trace-exfil", 2),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            write_cift_calibration_json(calibration_report_path, _calibration_report())
            config = CalibratedDetectorExportConfig(
                runtime_turns_path=runtime_turns_path,
                calibration_report_path=calibration_report_path,
                output_path=output_path,
                detector_name="cift_selector_probe",
                probe_version="dp_honey_lite_v3_selector_window_layer_15_calibrated_v1",
                capability_required="self_hosted_introspection",
                positive_action="warn",
                negative_action="allow",
                confidence=1.0,
            )

            row_count = export_calibrated_cift_detector_results(config)

            self.assertEqual(2, row_count)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual("trace-safe", rows[0]["trace_id"])
            self.assertEqual(0.08, rows[0]["detector_result"]["score"])
            self.assertEqual("allow", rows[0]["detector_result"]["recommended_action"])
            self.assertEqual("trace-exfil", rows[1]["trace_id"])
            self.assertEqual(0.81, rows[1]["detector_result"]["score"])
            self.assertEqual("warn", rows[1]["detector_result"]["recommended_action"])
            self.assertEqual(
                "inner_cv_platt_calibrated_probability",
                rows[1]["detector_result"]["evidence"]["score_semantics"],
            )


if __name__ == "__main__":
    unittest.main()
