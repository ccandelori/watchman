import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_layer_sweep import (
    evaluate_grouped_binary_layer_sweep,
    render_binary_layer_sweep_markdown,
    write_binary_layer_sweep_json,
    write_binary_layer_sweep_markdown,
)
from aegis_introspection.binary_tasks import BinaryTaskConfig


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
    strong_feature = torch.tensor(
        [
            [4.0, 4.0],
            [4.0, 4.2],
            [4.2, 4.0],
            [4.2, 4.2],
            [-4.0, 4.0],
            [-4.0, 4.2],
            [-4.2, 4.0],
            [-4.2, 4.2],
        ],
        dtype=torch.float32,
    )
    weak_feature = torch.zeros((8, 2), dtype=torch.float32)
    return {
        "metadata": {
            "model_id": "synthetic",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (1, 2),
            "pooling_methods": ("mean_pool",),
        },
        "example_ids": tuple(f"example_{index:03d}" for index in range(8)),
        "labels": labels,
        "families": families,
        "texts": texts,
        "tags": tuple(("synthetic",) for _ in range(8)),
        "features": {
            "mean_pool_layer_1": weak_feature,
            "mean_pool_layer_2": strong_feature,
        },
    }


def _config() -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=2,
        random_seed=7,
        max_iter=1000,
        regularization_c=1.0,
        activation_feature_key="mean_pool_layer_1",
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
    )


class BinaryLayerSweepTest(unittest.TestCase):
    def test_evaluate_grouped_layer_sweep_ranks_activation_features(self) -> None:
        report = evaluate_grouped_binary_layer_sweep(
            artifact=_synthetic_artifact(),
            task_name="safe_secret_vs_exfiltration",
            config=_config(),
        )

        self.assertEqual("synthetic", report.source_model_id)
        self.assertEqual("safe_secret_vs_exfiltration", report.task_name)
        self.assertEqual("stratified_group_kfold", report.evaluation_strategy)
        self.assertEqual("mean_pool_layer_1", report.reference_feature_key)
        self.assertEqual("mean_pool_layer_2", report.best_feature_key)
        self.assertEqual(("mean_pool_layer_2", "mean_pool_layer_1"), tuple(feature.feature_name for feature in report.features))
        self.assertEqual((1, 2), tuple(feature.rank for feature in report.features))

    def test_render_layer_sweep_markdown_includes_reference_feature(self) -> None:
        report = evaluate_grouped_binary_layer_sweep(
            artifact=_synthetic_artifact(),
            task_name="safe_secret_vs_exfiltration",
            config=_config(),
        )

        markdown = render_binary_layer_sweep_markdown(report)

        self.assertIn("# Binary Layer Sweep", markdown)
        self.assertIn("Reference feature: `mean_pool_layer_1`", markdown)
        self.assertIn("Best feature: `mean_pool_layer_2`", markdown)
        self.assertIn("| Rank | Feature | Macro F1 | Accuracy |", markdown)

    def test_write_layer_sweep_outputs_creates_files(self) -> None:
        report = evaluate_grouped_binary_layer_sweep(
            artifact=_synthetic_artifact(),
            task_name="safe_secret_vs_exfiltration",
            config=_config(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "layer_sweep.json"
            markdown_path = Path(temp_dir) / "layer_sweep.md"
            write_binary_layer_sweep_json(json_path, report)
            write_binary_layer_sweep_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual("mean_pool_layer_2", decoded["best_feature_key"])
        self.assertEqual(2, len(decoded["features"]))
        self.assertIn("Binary Layer Sweep", markdown)


if __name__ == "__main__":
    unittest.main()
