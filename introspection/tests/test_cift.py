import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import BinaryTaskConfig, build_binary_task_dataset, default_binary_task_definitions
from aegis_introspection.cift import (
    CiftComparisonDataset,
    CiftProbeConfig,
    compare_grouped_cift_probe,
    evaluate_grouped_cift_method,
    fit_cift_diagonal_calibration,
    last_quarter_readout_feature_keys,
    render_cift_probe_comparison_markdown,
    transform_cift_diagonal,
    write_cift_probe_comparison_json,
    write_cift_probe_comparison_markdown,
)


def _synthetic_artifact() -> ActivationArtifact:
    labels = (
        "secret_present_safe",
        "secret_present_safe",
        "secret_present_safe",
        "secret_present_safe",
        "exfiltration_intent",
        "exfiltration_intent",
        "exfiltration_intent",
        "exfiltration_intent",
    )
    texts = (
        "redact secret safely",
        "mask credential carefully",
        "protect api key",
        "classify sensitive token",
        "send secret outside",
        "upload credential remote",
        "post token webhook",
        "leak api key",
    )
    families = (
        "safe_redaction",
        "safe_redaction",
        "safe_classification",
        "safe_classification",
        "exfil_network",
        "exfil_network",
        "exfil_log",
        "exfil_log",
    )
    near_safe = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [0.1, 0.0],
            [0.1, 0.1],
            [4.0, 4.0],
            [4.0, 4.1],
            [4.1, 4.0],
            [4.1, 4.1],
        ],
        dtype=torch.float32,
    )
    weak_layer = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [1.0, 1.0],
            [1.1, 1.0],
            [1.0, 1.1],
            [1.1, 1.1],
        ],
        dtype=torch.float32,
    )
    return {
        "metadata": {
            "model_id": "synthetic",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (0, 1, 2, 3, 4, 5, 6, 7),
            "pooling_methods": ("final_token", "mean_pool"),
        },
        "example_ids": tuple(f"example_{index:03d}" for index in range(8)),
        "labels": labels,
        "families": families,
        "texts": texts,
        "tags": tuple(("synthetic",) for _ in range(8)),
        "features": {
            "weak_baseline_feature": torch.zeros((8, 2), dtype=torch.float32),
            "final_token_layer_06": near_safe,
            "final_token_layer_07": weak_layer,
            "mean_pool_layer_06": weak_layer,
            "mean_pool_layer_07": weak_layer,
        },
    }


def _task_config() -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=2,
        random_seed=7,
        max_iter=1000,
        regularization_c=1.0,
        activation_feature_key="mean_pool_layer_18",
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
    )


def _cift_config() -> CiftProbeConfig:
    return CiftProbeConfig(
        source_feature_keys=("final_token_layer_06", "final_token_layer_07"),
        calibration_source_labels=("secret_present_safe",),
        ridge=0.001,
        output_feature_key="cift_diag_final_token_last_quarter",
    )


class CiftTest(unittest.TestCase):
    def test_last_quarter_readout_feature_keys_uses_artifact_layer_metadata(self) -> None:
        keys = last_quarter_readout_feature_keys(_synthetic_artifact(), "final_token")

        self.assertEqual(("final_token_layer_06", "final_token_layer_07"), keys)

    def test_transform_cift_diagonal_scores_rows_by_distance_from_calibration_labels(self) -> None:
        artifact = _synthetic_artifact()
        dataset = build_binary_task_dataset(artifact, default_binary_task_definitions()[1])
        calibration = fit_cift_diagonal_calibration(
            artifact=artifact,
            dataset=dataset,
            calibration_row_indices=(0, 1, 2, 3),
            config=_cift_config(),
        )

        matrix = transform_cift_diagonal(
            artifact=artifact,
            dataset=dataset,
            row_indices=tuple(range(8)),
            calibration=calibration,
        )

        self.assertEqual((8, 2), tuple(matrix.shape))
        self.assertLess(float(matrix[:4].mean()), float(matrix[4:].mean()))

    def test_evaluate_grouped_cift_method_returns_fold_metrics(self) -> None:
        artifact = _synthetic_artifact()
        dataset = build_binary_task_dataset(artifact, default_binary_task_definitions()[1])

        report = evaluate_grouped_cift_method(
            artifact=artifact,
            dataset=dataset,
            binary_config=_task_config(),
            cift_config=_cift_config(),
        )

        self.assertEqual("activation_probe", report.method_name)
        self.assertEqual("cift_diag_final_token_last_quarter", report.feature_name)
        self.assertEqual(8, report.example_count)
        self.assertGreater(report.macro_f1_mean, 0.9)

    def test_compare_grouped_cift_probe_reports_dataset_delta(self) -> None:
        report = compare_grouped_cift_probe(
            datasets=(CiftComparisonDataset(dataset_id="synthetic_hard", artifact=_synthetic_artifact()),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            cift_config=_cift_config(),
            binary_config=_task_config(),
        )

        self.assertEqual("safe_secret_vs_exfiltration", report.task_name)
        self.assertEqual("weak_baseline_feature", report.baseline_feature_key)
        self.assertEqual("cift_diag_final_token_last_quarter", report.cift_feature_key)
        self.assertEqual(("final_token_layer_06", "final_token_layer_07"), report.cift_source_feature_keys)
        self.assertEqual(1, report.dataset_count)
        self.assertEqual(1, report.cift_win_count)
        self.assertEqual(0, report.baseline_win_count)
        self.assertEqual("cift_diag_final_token_last_quarter", report.datasets[0].winning_feature_key)
        self.assertGreater(report.datasets[0].macro_f1_delta, 0.0)

    def test_render_cift_probe_comparison_markdown_includes_calibration_source(self) -> None:
        report = compare_grouped_cift_probe(
            datasets=(CiftComparisonDataset(dataset_id="synthetic_hard", artifact=_synthetic_artifact()),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            cift_config=_cift_config(),
            binary_config=_task_config(),
        )

        markdown = render_cift_probe_comparison_markdown(report)

        self.assertIn("# CIFT-Like Probe Comparison", markdown)
        self.assertIn("Baseline feature: `weak_baseline_feature`", markdown)
        self.assertIn("CIFT-like feature: `cift_diag_final_token_last_quarter`", markdown)
        self.assertIn("Calibration source labels: `secret_present_safe`", markdown)

    def test_write_cift_probe_comparison_outputs_creates_files(self) -> None:
        report = compare_grouped_cift_probe(
            datasets=(CiftComparisonDataset(dataset_id="synthetic_hard", artifact=_synthetic_artifact()),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            cift_config=_cift_config(),
            binary_config=_task_config(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "cift.json"
            markdown_path = Path(temp_dir) / "cift.md"
            write_cift_probe_comparison_json(json_path, report)
            write_cift_probe_comparison_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual("cift_diag_final_token_last_quarter", decoded["cift_feature_key"])
        self.assertEqual(1, decoded["cift_win_count"])
        self.assertIn("CIFT-Like Probe Comparison", markdown)


if __name__ == "__main__":
    unittest.main()
