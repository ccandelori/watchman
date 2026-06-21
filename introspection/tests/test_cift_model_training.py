import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.cift_model_bundle import load_cift_model_bundle, predict_cift_model_bundle
from aegis_introspection.cift_model_training import CiftModelTrainingConfig, train_cift_model_bundle
from aegis_introspection.lineage import sha256_file


def _artifact() -> ActivationArtifact:
    return {
        "metadata": {
            "model_id": "Qwen/Qwen3-0.6B",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (15,),
            "pooling_methods": ("readout_window",),
        },
        "example_ids": ("benign-1", "safe-1", "safe-2", "exfil-1", "exfil-2"),
        "labels": ("benign", "secret_present_safe", "secret_present_safe", "exfiltration_intent", "exfiltration_intent"),
        "families": ("benign", "family-a", "family-b", "family-a", "family-b"),
        "texts": ("benign text", "safe text one", "safe text two", "exfil text one", "exfil text two"),
        "tags": (("test",), ("test",), ("test",), ("test",), ("test",)),
        "features": {
            "readout_window_layer_15": torch.tensor(
                (
                    (9.0, 9.0),
                    (0.0, 0.0),
                    (0.1, 0.0),
                    (2.0, 2.0),
                    (2.2, 2.0),
                ),
                dtype=torch.float32,
            )
        },
    }


class CiftModelTrainingTest(unittest.TestCase):
    def test_train_cift_model_bundle_writes_loadable_full_train_detector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "features.pt"
            bundle_path = root / "cift_model_bundle.pkl"
            torch.save(_artifact(), artifact_path)
            config = CiftModelTrainingConfig(
                artifact_path=artifact_path,
                output_bundle_path=bundle_path,
                training_dataset_id="synthetic_test_dataset",
                task_name="safe_secret_vs_exfiltration",
                positive_label="exfiltration_intent",
                activation_feature_key="readout_window_layer_15",
                decision_threshold=0.5,
                random_seed=42,
                max_iter=1000,
                regularization_c=1.0,
                evaluation_report_ids=("synthetic_eval_report",),
                score_semantics="full_train_classifier_probability",
                candidate_status="offline_research_candidate",
                created_at="2026-06-21T00:00:00Z",
            )

            report = train_cift_model_bundle(config)
            bundle = load_cift_model_bundle(bundle_path)
            predictions = predict_cift_model_bundle(
                bundle=bundle,
                feature_matrix=np.asarray(((2.4, 2.1), (0.0, 0.1)), dtype=np.float32),
            )
            expected_artifact_sha256 = sha256_file(artifact_path)

            self.assertEqual(bundle_path, report.output_bundle_path)
            self.assertEqual(4, report.example_count)
            self.assertEqual(2, report.feature_count)
            self.assertEqual(expected_artifact_sha256, bundle.metadata.source_artifact_sha256)
            self.assertEqual(("exfiltration_intent", "secret_present_safe"), bundle.metadata.label_names)
            self.assertEqual("exfiltration_intent", predictions[0].predicted_label)
            self.assertEqual("secret_present_safe", predictions[1].predicted_label)


if __name__ == "__main__":
    unittest.main()
