import unittest
import tempfile
from pathlib import Path

import torch

from aegis_introspection.cift_calibration import (
    CiftCalibrationConfig,
    cift_calibration_report_to_json,
    collect_grouped_calibrated_cift_predictions,
    load_cift_calibration_report_json,
    write_cift_calibration_json,
)


def _artifact() -> dict:
    example_ids: list[str] = []
    labels: list[str] = []
    families: list[str] = []
    texts: list[str] = []
    tags: list[tuple[str, ...]] = []
    rows: list[list[float]] = []
    for family_index in range(8):
        family = f"family_{family_index}"
        example_ids.append(f"{family}_safe")
        labels.append("secret_present_safe")
        families.append(family)
        texts.append("safe")
        tags.append(("test",))
        rows.append([-1.0, float(family_index) / 100.0])

        example_ids.append(f"{family}_exfil")
        labels.append("exfiltration_intent")
        families.append(family)
        texts.append("exfil")
        tags.append(("test",))
        rows.append([1.0, float(family_index) / 100.0])

    return {
        "metadata": {
            "model_id": "test-model",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (15,),
            "pooling_methods": ("readout_window",),
        },
        "example_ids": tuple(example_ids),
        "labels": tuple(labels),
        "families": tuple(families),
        "texts": tuple(texts),
        "tags": tuple(tags),
        "features": {"readout_window_layer_15": torch.tensor(rows, dtype=torch.float32)},
    }


class CiftCalibrationTest(unittest.TestCase):
    def test_collect_grouped_calibrated_cift_predictions_outputs_probabilities(self) -> None:
        config = CiftCalibrationConfig(
            task_name="safe_secret_vs_exfiltration",
            positive_label="exfiltration_intent",
            activation_feature_key="readout_window_layer_15",
            fold_count=2,
            inner_fold_count=2,
            random_seed=42,
            max_iter=1000,
            regularization_c=1.0,
            decision_threshold=0.5,
        )

        report = collect_grouped_calibrated_cift_predictions(artifact=_artifact(), config=config)
        encoded = cift_calibration_report_to_json(report)

        self.assertEqual("safe_secret_vs_exfiltration", report.task_name)
        self.assertEqual("inner_cv_platt_calibrated_probability", report.score_semantics)
        self.assertEqual(16, len(report.predictions))
        self.assertEqual(16, len(encoded["predictions"]))
        self.assertGreaterEqual(report.brier_score, 0.0)
        self.assertLessEqual(report.brier_score, 1.0)
        for prediction in report.predictions:
            self.assertGreaterEqual(prediction.positive_probability, 0.0)
            self.assertLessEqual(prediction.positive_probability, 1.0)

    def test_load_cift_calibration_report_json_round_trips_report(self) -> None:
        config = CiftCalibrationConfig(
            task_name="safe_secret_vs_exfiltration",
            positive_label="exfiltration_intent",
            activation_feature_key="readout_window_layer_15",
            fold_count=2,
            inner_fold_count=2,
            random_seed=42,
            max_iter=1000,
            regularization_c=1.0,
            decision_threshold=0.5,
        )
        report = collect_grouped_calibrated_cift_predictions(artifact=_artifact(), config=config)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            write_cift_calibration_json(path, report)

            loaded = load_cift_calibration_report_json(path)

        self.assertEqual(report.task_name, loaded.task_name)
        self.assertEqual(report.score_semantics, loaded.score_semantics)
        self.assertEqual(len(report.predictions), len(loaded.predictions))


if __name__ == "__main__":
    unittest.main()
