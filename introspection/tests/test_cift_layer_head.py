import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import BinaryTaskConfig
from aegis_introspection.cift_layer_head import (
    CiftLayerHeadComparisonDataset,
    CiftLayerHeadConfig,
    compare_grouped_cift_layer_head,
    render_cift_layer_head_markdown,
    write_cift_layer_head_json,
    write_cift_layer_head_markdown,
)


def _synthetic_artifact() -> ActivationArtifact:
    labels = (
        "benign",
        "benign",
        "benign",
        "benign",
        "secret_present_safe",
        "secret_present_safe",
        "secret_present_safe",
        "secret_present_safe",
        "exfiltration_intent",
        "exfiltration_intent",
        "exfiltration_intent",
        "exfiltration_intent",
    )
    families = (
        "benign_status",
        "benign_status",
        "benign_release",
        "benign_release",
        "safe_redaction",
        "safe_redaction",
        "safe_classification",
        "safe_classification",
        "exfil_network",
        "exfil_network",
        "exfil_log",
        "exfil_log",
    )
    texts = (
        "write a harmless status note",
        "summarize release notes",
        "draft a routine checklist",
        "explain ordinary metrics",
        "redact secret safely",
        "mask credential carefully",
        "protect api key",
        "classify sensitive token",
        "send secret outside",
        "upload credential remote",
        "post token webhook",
        "leak api key",
    )
    informative_layer = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [0.2, 0.0],
            [0.2, 0.2],
            [0.1, 0.1],
            [0.1, 0.3],
            [0.3, 0.1],
            [0.3, 0.3],
            [4.0, 4.0],
            [4.0, 4.2],
            [4.2, 4.0],
            [4.2, 4.2],
        ],
        dtype=torch.float32,
    )
    weak_layer = torch.zeros((12, 2), dtype=torch.float32)
    return {
        "metadata": {
            "model_id": "synthetic",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (6, 7),
            "pooling_methods": ("final_token",),
        },
        "example_ids": tuple(f"example_{index:03d}" for index in range(12)),
        "labels": labels,
        "families": families,
        "texts": texts,
        "tags": tuple(("synthetic",) for _ in range(12)),
        "features": {
            "weak_baseline_feature": torch.zeros((12, 2), dtype=torch.float32),
            "final_token_layer_06": informative_layer,
            "final_token_layer_07": weak_layer,
        },
    }


def _binary_config() -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=2,
        random_seed=7,
        max_iter=1000,
        regularization_c=1.0,
        activation_feature_key="weak_baseline_feature",
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
    )


def _head_config() -> CiftLayerHeadConfig:
    return CiftLayerHeadConfig(
        source_feature_keys=("final_token_layer_06", "final_token_layer_07"),
        calibration_source_labels=("benign", "secret_present_safe"),
        ridge=0.001,
        output_feature_key="cift_layer_weighted_signed_residual",
        risk_label="exfiltration_intent",
    )


class CiftLayerHeadTest(unittest.TestCase):
    def test_compare_grouped_cift_layer_head_learns_nonnegative_layer_weights(self) -> None:
        report = compare_grouped_cift_layer_head(
            datasets=(CiftLayerHeadComparisonDataset(dataset_id="synthetic_hard", artifact=_synthetic_artifact()),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            head_config=_head_config(),
            binary_config=_binary_config(),
        )

        dataset = report.datasets[0]

        self.assertEqual("safe_secret_vs_exfiltration", report.task_name)
        self.assertEqual("weak_baseline_feature", report.baseline_feature_key)
        self.assertEqual("cift_layer_weighted_signed_residual", report.head_feature_key)
        self.assertEqual(1, report.head_win_count)
        self.assertGreater(dataset.head.macro_f1_mean, dataset.baseline.macro_f1_mean)
        for fold in dataset.head.weight_folds:
            self.assertAlmostEqual(1.0, sum(fold.weights), places=6)
            self.assertGreaterEqual(min(fold.weights), 0.0)
            self.assertGreater(fold.weights[0], fold.weights[1])

    def test_render_cift_layer_head_markdown_includes_weight_table(self) -> None:
        report = compare_grouped_cift_layer_head(
            datasets=(CiftLayerHeadComparisonDataset(dataset_id="synthetic_hard", artifact=_synthetic_artifact()),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            head_config=_head_config(),
            binary_config=_binary_config(),
        )

        markdown = render_cift_layer_head_markdown(report)

        self.assertIn("# CIFT Layer-Weighted Head", markdown)
        self.assertIn("Head feature: `cift_layer_weighted_signed_residual`", markdown)
        self.assertIn("| Dataset | Source Feature | Mean Weight |", markdown)

    def test_write_cift_layer_head_outputs_creates_files(self) -> None:
        report = compare_grouped_cift_layer_head(
            datasets=(CiftLayerHeadComparisonDataset(dataset_id="synthetic_hard", artifact=_synthetic_artifact()),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            head_config=_head_config(),
            binary_config=_binary_config(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "layer_head.json"
            markdown_path = Path(temp_dir) / "layer_head.md"
            write_cift_layer_head_json(json_path, report)
            write_cift_layer_head_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual("cift_layer_weighted_signed_residual", decoded["head_feature_key"])
        self.assertEqual(1, decoded["head_win_count"])
        self.assertIn("CIFT Layer-Weighted Head", markdown)


if __name__ == "__main__":
    unittest.main()
