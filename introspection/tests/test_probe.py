import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.probe import (
    ProbeTrainingConfig,
    ProbeTrainingError,
    encode_labels,
    tensor_to_float_matrix,
    train_probe_report,
    write_probe_report_json,
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
    good_feature = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [0.2, 0.0],
            [0.2, 0.2],
            [5.0, 5.0],
            [5.0, 5.2],
            [5.2, 5.0],
            [5.2, 5.2],
            [-5.0, 5.0],
            [-5.0, 5.2],
            [-5.2, 5.0],
            [-5.2, 5.2],
        ],
        dtype=torch.float32,
    )
    flat_feature = torch.zeros((12, 2), dtype=torch.float32)
    return {
        "metadata": {
            "model_id": "synthetic",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (0,),
            "pooling_methods": ("mean_pool",),
        },
        "example_ids": tuple(f"example_{index:03d}" for index in range(12)),
        "labels": labels,
        "texts": tuple("text" for _ in range(12)),
        "tags": tuple(("synthetic",) for _ in range(12)),
        "features": {
            "good_feature": good_feature,
            "flat_feature": flat_feature,
        },
    }


class ProbeTrainingTest(unittest.TestCase):
    def test_encode_labels_uses_stable_sorted_label_order(self) -> None:
        encoding = encode_labels(("secret_present_safe", "benign", "exfiltration_intent"))

        self.assertEqual(("benign", "exfiltration_intent", "secret_present_safe"), encoding.label_names)
        self.assertEqual((2, 0, 1), tuple(int(value) for value in encoding.encoded_labels))

    def test_tensor_to_float_matrix_rejects_non_matrix_tensor(self) -> None:
        with self.assertRaises(ProbeTrainingError):
            tensor_to_float_matrix(torch.zeros((1, 2, 3)))

    def test_train_probe_report_selects_best_feature(self) -> None:
        config = ProbeTrainingConfig(
            fold_count=2,
            random_seed=7,
            max_iter=1000,
            regularization_c=1.0,
        )

        report = train_probe_report(_synthetic_artifact(), config)

        self.assertEqual("good_feature", report.best_feature_key)
        self.assertEqual(("benign", "exfiltration_intent", "secret_present_safe"), report.label_names)
        self.assertEqual(2, len(report.features))

    def test_train_probe_report_rejects_too_many_folds(self) -> None:
        config = ProbeTrainingConfig(
            fold_count=5,
            random_seed=7,
            max_iter=1000,
            regularization_c=1.0,
        )

        with self.assertRaises(ProbeTrainingError):
            train_probe_report(_synthetic_artifact(), config)

    def test_write_probe_report_json_creates_readable_report(self) -> None:
        config = ProbeTrainingConfig(
            fold_count=2,
            random_seed=7,
            max_iter=1000,
            regularization_c=1.0,
        )
        report = train_probe_report(_synthetic_artifact(), config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "probe_report.json"
            write_probe_report_json(output_path, report)
            decoded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual("synthetic", decoded["source_model_id"])
        self.assertEqual("good_feature", decoded["best_feature_key"])


if __name__ == "__main__":
    unittest.main()
