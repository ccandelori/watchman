import tempfile
import unittest
from pathlib import Path

from aegis_introspection.cift_calibration import (
    CalibratedCiftPrediction,
    CalibrationBinSummary,
    CiftCalibrationReport,
)
from aegis_introspection.cift_operating_points import (
    CiftOperatingPointConfig,
    build_cift_operating_point_report,
    cift_operating_point_report_to_json,
    write_cift_operating_point_json,
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
        accuracy=0.75,
        macro_f1=0.7333,
        brier_score=0.15,
        log_loss=0.5,
        expected_calibration_error=0.1,
        confusion_matrix=((2, 1), (1, 2)),
        bin_summaries=(
            CalibrationBinSummary(
                bin_index=1,
                lower_bound=0.0,
                upper_bound=0.5,
                example_count=3,
                mean_probability=0.3,
                empirical_positive_rate=0.3333,
                absolute_gap=0.0333,
            ),
        ),
        predictions=(
            _prediction("safe-low", "secret_present_safe", 0.10),
            _prediction("safe-mid", "secret_present_safe", 0.40),
            _prediction("safe-high", "secret_present_safe", 0.70),
            _prediction("exfil-low", "exfiltration_intent", 0.30),
            _prediction("exfil-mid", "exfiltration_intent", 0.60),
            _prediction("exfil-high", "exfiltration_intent", 0.90),
        ),
    )


def _prediction(example_id: str, true_label: str, probability: float) -> CalibratedCiftPrediction:
    predicted_label = "exfiltration_intent" if probability >= 0.5 else "secret_present_safe"
    return CalibratedCiftPrediction(
        fold_index=1,
        example_id=example_id,
        family="family-a",
        source_label=true_label,
        true_label=true_label,
        predicted_label=predicted_label,
        is_correct=predicted_label == true_label,
        positive_label="exfiltration_intent",
        positive_probability=probability,
    )


class CiftOperatingPointsTest(unittest.TestCase):
    def test_build_cift_operating_point_report_sweeps_thresholds(self) -> None:
        config = CiftOperatingPointConfig(thresholds=(0.25, 0.50, 0.75))

        report = build_cift_operating_point_report(report=_calibration_report(), config=config)

        self.assertEqual("safe_secret_vs_exfiltration", report.task_name)
        self.assertEqual("exfiltration_intent", report.positive_label)
        self.assertEqual("inner_cv_platt_calibrated_probability", report.score_semantics)
        self.assertEqual(0.50, report.best_macro_f1_threshold)
        self.assertEqual(0.25, report.high_recall_threshold)
        self.assertEqual(3, len(report.operating_points))

        low, middle, high = report.operating_points
        self.assertEqual(0.25, low.threshold)
        self.assertEqual((3, 2, 1, 0), (low.true_positive, low.false_positive, low.true_negative, low.false_negative))
        self.assertAlmostEqual(1.0, low.recall)
        self.assertAlmostEqual(0.6, low.precision)

        self.assertEqual(0.50, middle.threshold)
        self.assertEqual(
            (2, 1, 2, 1),
            (middle.true_positive, middle.false_positive, middle.true_negative, middle.false_negative),
        )
        self.assertAlmostEqual(2.0 / 3.0, middle.macro_f1)

        self.assertEqual(0.75, high.threshold)
        self.assertEqual((1, 0, 3, 2), (high.true_positive, high.false_positive, high.true_negative, high.false_negative))
        self.assertAlmostEqual(1.0, high.precision)

    def test_cift_operating_point_report_to_json_round_trips_safe_values(self) -> None:
        config = CiftOperatingPointConfig(thresholds=(0.50,))
        report = build_cift_operating_point_report(report=_calibration_report(), config=config)

        encoded = cift_operating_point_report_to_json(report)

        self.assertEqual("safe_secret_vs_exfiltration", encoded["task_name"])
        self.assertEqual(0.50, encoded["best_macro_f1_threshold"])
        self.assertEqual(1, len(encoded["operating_points"]))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operating_points.json"
            write_cift_operating_point_json(path, report)
            self.assertIn("operating_points", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
