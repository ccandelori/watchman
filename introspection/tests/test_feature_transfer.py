import json
import tempfile
import unittest
from pathlib import Path

import torch
from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.feature_transfer import (
    FeatureTransferConfig,
    FeatureTransferDataset,
    FeatureTransferError,
    evaluate_feature_transfer,
    render_feature_transfer_markdown,
    write_feature_transfer_json,
    write_feature_transfer_markdown,
)


def _artifact(dataset_id: str, feature: torch.Tensor) -> ActivationArtifact:
    labels = (
        "secret_present_safe",
        "secret_present_safe",
        "exfiltration_intent",
        "exfiltration_intent",
    )
    return {
        "metadata": {
            "model_id": "Qwen/Qwen3-4B",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (20,),
            "pooling_methods": ("readout_window",),
        },
        "example_ids": tuple(f"{dataset_id}_example_{index:03d}" for index in range(4)),
        "labels": labels,
        "families": ("safe-a", "safe-b", "exfil-a", "exfil-b"),
        "texts": (
            "keep credential inside the review",
            "summarize without exposing token",
            "move credential into the report",
            "send token to an outside tool",
        ),
        "tags": tuple(("test",) for _ in range(4)),
        "features": {
            "readout_window_layer_20": feature,
        },
    }


def _config() -> FeatureTransferConfig:
    return FeatureTransferConfig(
        task_name="safe_secret_vs_exfiltration",
        activation_feature_key="readout_window_layer_20",
        positive_label="exfiltration_intent",
        decision_threshold=0.5,
        random_seed=42,
        max_iter=1000,
        regularization_c=10.0,
    )


class FeatureTransferTest(unittest.TestCase):
    def test_evaluate_feature_transfer_scores_held_out_profile(self) -> None:
        train_artifact = _artifact(
            dataset_id="train_profile",
            feature=torch.tensor(
                (
                    (0.0, 0.0),
                    (0.2, 0.1),
                    (4.0, 4.0),
                    (4.2, 4.1),
                ),
                dtype=torch.float32,
            ),
        )
        test_artifact = _artifact(
            dataset_id="test_profile",
            feature=torch.tensor(
                (
                    (0.1, 0.0),
                    (0.0, 0.2),
                    (3.9, 4.1),
                    (4.1, 3.9),
                ),
                dtype=torch.float32,
            ),
        )

        report = evaluate_feature_transfer(
            train_datasets=(FeatureTransferDataset(dataset_id="train_profile", artifact=train_artifact),),
            test_datasets=(FeatureTransferDataset(dataset_id="test_profile", artifact=test_artifact),),
            config=_config(),
        )

        self.assertEqual("train_profiles_to_test_profiles", report.evaluation_strategy)
        self.assertEqual(("train_profile",), report.train_dataset_ids)
        self.assertEqual(("test_profile",), report.test_dataset_ids)
        self.assertEqual(4, report.train_example_count)
        self.assertEqual(2, report.feature_count)
        self.assertEqual(("exfiltration_intent", "secret_present_safe"), report.label_names)
        self.assertEqual(1.0, report.tests[0].accuracy)
        self.assertEqual(1.0, report.tests[0].macro_f1)
        self.assertEqual(((2, 0), (0, 2)), report.tests[0].confusion_matrix)

    def test_feature_transfer_outputs_are_json_safe_and_renderable(self) -> None:
        report = evaluate_feature_transfer(
            train_datasets=(
                FeatureTransferDataset(
                    dataset_id="train_profile",
                    artifact=_artifact(
                        dataset_id="train_profile",
                        feature=torch.tensor(
                            (
                                (0.0, 0.0),
                                (0.2, 0.1),
                                (4.0, 4.0),
                                (4.2, 4.1),
                            ),
                            dtype=torch.float32,
                        ),
                    ),
                ),
            ),
            test_datasets=(
                FeatureTransferDataset(
                    dataset_id="test_profile",
                    artifact=_artifact(
                        dataset_id="test_profile",
                        feature=torch.tensor(
                            (
                                (0.1, 0.0),
                                (0.0, 0.2),
                                (3.9, 4.1),
                                (4.1, 3.9),
                            ),
                            dtype=torch.float32,
                        ),
                    ),
                ),
            ),
            config=_config(),
        )

        markdown = render_feature_transfer_markdown(report)

        self.assertIn("# Feature Transfer Evaluation", markdown)
        self.assertIn("Train datasets: `train_profile`", markdown)
        self.assertIn("| `test_profile` | 4 | 1.0000 | 1.0000 |", markdown)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / "transfer.json"
            markdown_path = root / "transfer.md"
            write_feature_transfer_json(json_path, report)
            write_feature_transfer_markdown(markdown_path, report)
            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            written_markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual("readout_window_layer_20", decoded["activation_feature_key"])
        self.assertEqual(1.0, decoded["tests"][0]["macro_f1"])
        self.assertIn("Feature Transfer Evaluation", written_markdown)

    def test_evaluate_feature_transfer_rejects_mismatched_feature_counts(self) -> None:
        train_artifact = _artifact(
            dataset_id="train_profile",
            feature=torch.zeros((4, 2), dtype=torch.float32),
        )
        test_artifact = _artifact(
            dataset_id="test_profile",
            feature=torch.zeros((4, 3), dtype=torch.float32),
        )

        with self.assertRaises(FeatureTransferError):
            evaluate_feature_transfer(
                train_datasets=(FeatureTransferDataset(dataset_id="train_profile", artifact=train_artifact),),
                test_datasets=(FeatureTransferDataset(dataset_id="test_profile", artifact=test_artifact),),
                config=_config(),
            )


if __name__ == "__main__":
    unittest.main()
