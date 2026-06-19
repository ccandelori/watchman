import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_feature_crosscheck import (
    FeatureCrosscheckDataset,
    compare_grouped_binary_features,
    render_feature_crosscheck_markdown,
    write_feature_crosscheck_json,
    write_feature_crosscheck_markdown,
)
from aegis_introspection.binary_tasks import BinaryTaskConfig


def _synthetic_artifact(model_id: str) -> ActivationArtifact:
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
    candidate_feature = torch.tensor(
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
    reference_feature = torch.zeros((8, 2), dtype=torch.float32)
    return {
        "metadata": {
            "model_id": model_id,
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (11, 18),
            "pooling_methods": ("final_token", "mean_pool"),
        },
        "example_ids": tuple(f"{model_id}_example_{index:03d}" for index in range(8)),
        "labels": labels,
        "families": families,
        "texts": texts,
        "tags": tuple(("synthetic",) for _ in range(8)),
        "features": {
            "mean_pool_layer_18": reference_feature,
            "final_token_layer_11": candidate_feature,
        },
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


class BinaryFeatureCrosscheckTest(unittest.TestCase):
    def test_compare_grouped_binary_features_reports_deltas_per_dataset(self) -> None:
        report = compare_grouped_binary_features(
            datasets=(
                FeatureCrosscheckDataset(dataset_id="baseline", artifact=_synthetic_artifact("baseline")),
                FeatureCrosscheckDataset(dataset_id="hard_v2", artifact=_synthetic_artifact("hard_v2")),
            ),
            task_name="safe_secret_vs_exfiltration",
            reference_feature_key="mean_pool_layer_18",
            candidate_feature_key="final_token_layer_11",
            config=_config(),
        )

        self.assertEqual("safe_secret_vs_exfiltration", report.task_name)
        self.assertEqual("mean_pool_layer_18", report.reference_feature_key)
        self.assertEqual("final_token_layer_11", report.candidate_feature_key)
        self.assertEqual(2, report.dataset_count)
        self.assertEqual(2, report.candidate_win_count)
        self.assertEqual(0, report.reference_win_count)
        self.assertEqual(("baseline", "hard_v2"), tuple(dataset.dataset_id for dataset in report.datasets))
        for dataset in report.datasets:
            self.assertEqual("final_token_layer_11", dataset.winning_feature_key)
            self.assertGreater(dataset.macro_f1_delta, 0.0)
            self.assertGreater(dataset.accuracy_delta, 0.0)

    def test_render_feature_crosscheck_markdown_includes_winner_table(self) -> None:
        report = compare_grouped_binary_features(
            datasets=(
                FeatureCrosscheckDataset(dataset_id="baseline", artifact=_synthetic_artifact("baseline")),
                FeatureCrosscheckDataset(dataset_id="hard_v2", artifact=_synthetic_artifact("hard_v2")),
            ),
            task_name="safe_secret_vs_exfiltration",
            reference_feature_key="mean_pool_layer_18",
            candidate_feature_key="final_token_layer_11",
            config=_config(),
        )

        markdown = render_feature_crosscheck_markdown(report)

        self.assertIn("# Binary Feature Crosscheck", markdown)
        self.assertIn("Reference feature: `mean_pool_layer_18`", markdown)
        self.assertIn("Candidate feature: `final_token_layer_11`", markdown)
        self.assertIn("| Dataset | Reference Macro F1 | Candidate Macro F1 | Delta Macro F1 | Winner |", markdown)

    def test_write_feature_crosscheck_outputs_creates_files(self) -> None:
        report = compare_grouped_binary_features(
            datasets=(
                FeatureCrosscheckDataset(dataset_id="baseline", artifact=_synthetic_artifact("baseline")),
                FeatureCrosscheckDataset(dataset_id="hard_v2", artifact=_synthetic_artifact("hard_v2")),
            ),
            task_name="safe_secret_vs_exfiltration",
            reference_feature_key="mean_pool_layer_18",
            candidate_feature_key="final_token_layer_11",
            config=_config(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "crosscheck.json"
            markdown_path = Path(temp_dir) / "crosscheck.md"
            write_feature_crosscheck_json(json_path, report)
            write_feature_crosscheck_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual("final_token_layer_11", decoded["candidate_feature_key"])
        self.assertEqual(2, decoded["candidate_win_count"])
        self.assertIn("Binary Feature Crosscheck", markdown)


if __name__ == "__main__":
    unittest.main()
