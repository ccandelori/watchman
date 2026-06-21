import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_feature_crosscheck import FeatureCrosscheckDataset
from aegis_introspection.binary_feature_stability import (
    compare_grouped_binary_feature_stability,
    render_feature_stability_markdown,
    write_feature_stability_json,
    write_feature_stability_markdown,
)
from aegis_introspection.binary_tasks import BinaryTaskConfig


FEATURE_KEYS = ("mean_pool_layer_18", "final_token_layer_11", "final_token_layer_16")


def _separating_feature() -> torch.Tensor:
    return torch.tensor(
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


def _flat_feature() -> torch.Tensor:
    return torch.zeros((8, 2), dtype=torch.float32)


def _synthetic_artifact(model_id: str, winning_feature_key: str) -> ActivationArtifact:
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
    features = {
        feature_key: _separating_feature() if feature_key == winning_feature_key else _flat_feature()
        for feature_key in FEATURE_KEYS
    }
    return {
        "metadata": {
            "model_id": model_id,
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (11, 16, 18),
            "pooling_methods": ("final_token", "mean_pool"),
        },
        "example_ids": tuple(f"{model_id}_example_{index:03d}" for index in range(8)),
        "labels": labels,
        "families": families,
        "texts": texts,
        "tags": tuple(("synthetic",) for _ in range(8)),
        "features": features,
    }


def _config() -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=2,
        random_seed=7,
        max_iter=1000,
        regularization_c=1.0,
        activation_feature_key="mean_pool_layer_18",
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
    )


class BinaryFeatureStabilityTest(unittest.TestCase):
    def test_compare_grouped_binary_feature_stability_reports_wins_and_aggregate_metrics(self) -> None:
        report = compare_grouped_binary_feature_stability(
            datasets=(
                FeatureCrosscheckDataset(
                    dataset_id="baseline",
                    artifact=_synthetic_artifact("baseline", "final_token_layer_11"),
                ),
                FeatureCrosscheckDataset(
                    dataset_id="hard_v3",
                    artifact=_synthetic_artifact("hard_v3", "final_token_layer_16"),
                ),
            ),
            task_name="safe_secret_vs_exfiltration",
            feature_keys=FEATURE_KEYS,
            config=_config(),
        )

        self.assertEqual("safe_secret_vs_exfiltration", report.task_name)
        self.assertEqual(2, report.dataset_count)
        self.assertEqual(3, report.feature_count)
        self.assertEqual(("final_token_layer_11",), report.datasets[0].winning_feature_keys)
        self.assertEqual(("final_token_layer_16",), report.datasets[1].winning_feature_keys)
        win_counts = {summary.feature_key: summary.win_count for summary in report.feature_summaries}
        self.assertEqual(1, win_counts["final_token_layer_11"])
        self.assertEqual(1, win_counts["final_token_layer_16"])
        self.assertEqual(0, win_counts["mean_pool_layer_18"])
        self.assertEqual(3, len(report.datasets[0].metrics))

    def test_render_feature_stability_markdown_includes_summary_tables(self) -> None:
        report = compare_grouped_binary_feature_stability(
            datasets=(
                FeatureCrosscheckDataset(
                    dataset_id="baseline",
                    artifact=_synthetic_artifact("baseline", "final_token_layer_11"),
                ),
                FeatureCrosscheckDataset(
                    dataset_id="hard_v3",
                    artifact=_synthetic_artifact("hard_v3", "final_token_layer_16"),
                ),
            ),
            task_name="safe_secret_vs_exfiltration",
            feature_keys=FEATURE_KEYS,
            config=_config(),
        )

        markdown = render_feature_stability_markdown(report)

        self.assertIn("# Binary Feature Stability", markdown)
        self.assertIn("Feature count: `3`", markdown)
        self.assertIn("| Rank | Feature | Wins | Mean Macro F1 |", markdown)
        self.assertIn("| Dataset | Winner | `mean_pool_layer_18` |", markdown)

    def test_write_feature_stability_outputs_creates_files(self) -> None:
        report = compare_grouped_binary_feature_stability(
            datasets=(
                FeatureCrosscheckDataset(
                    dataset_id="baseline",
                    artifact=_synthetic_artifact("baseline", "final_token_layer_11"),
                ),
                FeatureCrosscheckDataset(
                    dataset_id="hard_v3",
                    artifact=_synthetic_artifact("hard_v3", "final_token_layer_16"),
                ),
            ),
            task_name="safe_secret_vs_exfiltration",
            feature_keys=FEATURE_KEYS,
            config=_config(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "stability.json"
            markdown_path = Path(temp_dir) / "stability.md"
            write_feature_stability_json(json_path, report)
            write_feature_stability_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(3, decoded["feature_count"])
        self.assertEqual("final_token_layer_11", decoded["datasets"][0]["winning_feature_keys"][0])
        self.assertIn("Binary Feature Stability", markdown)


if __name__ == "__main__":
    unittest.main()
