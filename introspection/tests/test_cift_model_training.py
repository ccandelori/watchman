from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from aegis_introspection.cift_model_bundle import load_cift_model_bundle, predict_cift_model_bundle
from aegis_introspection.cift_model_training import CiftModelTrainingConfig, train_cift_model_bundle
from aegis_introspection.lineage import sha256_file
from introspection.scripts.train_cift_model_bundle import _parse_args, _training_config
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float32]


class _FakeBfloat16Tensor:
    def __init__(self, matrix: FloatMatrix) -> None:
        self._matrix = matrix
        self._converted_to_float = False

    def detach(self) -> _FakeBfloat16Tensor:
        return self

    def cpu(self) -> _FakeBfloat16Tensor:
        return self

    def float(self) -> _FakeBfloat16Tensor:
        self._converted_to_float = True
        return self

    def numpy(self) -> FloatMatrix:
        if not self._converted_to_float:
            raise TypeError("Got unsupported ScalarType BFloat16")
        return self._matrix


def _artifact() -> dict[str, object]:
    return {
        "metadata": {
            "model_id": "Qwen/Qwen3-0.6B",
            "revision": "main",
            "selected_device": "cpu",
            "hidden_size": 1024,
            "layer_count": 24,
            "tokenizer_fingerprint_sha256": "b" * 64,
            "special_tokens_map_sha256": "c" * 64,
            "chat_template_sha256": "d" * 64,
            "layer_indices": (15,),
            "pooling_methods": ("readout_window",),
        },
        "example_ids": ("benign-1", "safe-1", "safe-2", "exfil-1", "exfil-2"),
        "labels": (
            "benign",
            "secret_present_safe",
            "secret_present_safe",
            "exfiltration_intent",
            "exfiltration_intent",
        ),
        "families": ("benign", "family-a", "family-b", "family-a", "family-b"),
        "texts": ("benign text", "safe text one", "safe text two", "exfil text one", "exfil text two"),
        "tags": (("test",), ("test",), ("test",), ("test",), ("test",)),
        "features": {
            "readout_window_layer_15": np.asarray(
                [
                    (9.0, 9.0),
                    (0.0, 0.0),
                    (0.1, 0.0),
                    (2.0, 2.0),
                    (2.2, 2.0),
                ],
                dtype=np.float32,
            )
        },
    }


def _artifact_with_feature_matrix(feature_matrix: object) -> dict[str, object]:
    artifact = _artifact()
    artifact["features"] = {"readout_window_layer_15": feature_matrix}
    return artifact


class CiftModelTrainingTest(unittest.TestCase):
    def test_train_cift_model_bundle_cli_parses_classifier_family(self) -> None:
        config = _parse_args(
            (
                "--artifact",
                "features.pkl",
                "--output-bundle",
                "bundle.pkl",
                "--classifier-family",
                "mlp_128_64_1",
            )
        )
        training_config = _training_config(config)

        self.assertEqual("mlp_128_64_1", config.classifier_family)
        self.assertEqual("mlp_128_64_1", training_config.classifier_family)

    def test_train_cift_model_bundle_writes_loadable_linear_detector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "features.pkl"
            bundle_path = root / "cift_model_bundle.pkl"
            artifact_path.write_bytes(pickle.dumps(_artifact()))
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
                classifier_family="linear_logistic_regression",
                evaluation_report_ids=("synthetic_eval_report",),
                score_semantics="full_train_classifier_probability",
                candidate_status="offline_research_candidate",
                created_at="2026-06-21T00:00:00Z",
                allow_sealed_holdout=False,
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

    def test_train_cift_model_bundle_casts_tensor_features_to_float32_before_numpy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "features.pkl"
            bundle_path = root / "cift_model_bundle.pkl"
            feature_matrix = np.asarray(
                [
                    (9.0, 9.0),
                    (0.0, 0.0),
                    (0.1, 0.0),
                    (2.0, 2.0),
                    (2.2, 2.0),
                ],
                dtype=np.float32,
            )
            artifact_path.write_bytes(pickle.dumps(_artifact_with_feature_matrix(_FakeBfloat16Tensor(feature_matrix))))
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
                classifier_family="linear_logistic_regression",
                evaluation_report_ids=("synthetic_eval_report",),
                score_semantics="full_train_classifier_probability",
                candidate_status="offline_research_candidate",
                created_at="2026-06-21T00:00:00Z",
                allow_sealed_holdout=False,
            )

            report = train_cift_model_bundle(config)

            self.assertEqual(bundle_path, report.output_bundle_path)
            self.assertEqual(4, report.example_count)
            self.assertEqual(2, report.feature_count)

    def test_train_cift_model_bundle_writes_loadable_paper_mlp_detector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "features.pkl"
            bundle_path = root / "cift_paper_mlp_bundle.pkl"
            artifact_path.write_bytes(pickle.dumps(_artifact()))
            config = CiftModelTrainingConfig(
                artifact_path=artifact_path,
                output_bundle_path=bundle_path,
                training_dataset_id="synthetic_test_dataset",
                task_name="safe_secret_vs_exfiltration",
                positive_label="exfiltration_intent",
                activation_feature_key="readout_window_layer_15",
                decision_threshold=0.5,
                random_seed=42,
                max_iter=220,
                regularization_c=0.0001,
                classifier_family="mlp_128_64_1",
                evaluation_report_ids=("synthetic_eval_report",),
                score_semantics="paper_mlp_probability",
                candidate_status="offline_research_candidate",
                created_at="2026-06-21T00:00:00Z",
                allow_sealed_holdout=False,
            )

            report = train_cift_model_bundle(config)
            bundle = load_cift_model_bundle(bundle_path)
            predictions = predict_cift_model_bundle(
                bundle=bundle,
                feature_matrix=np.asarray(((2.4, 2.1), (0.0, 0.1)), dtype=np.float32),
            )
            layer_weights = bundle.classifier.softplus_layer_weights()

            self.assertEqual(bundle_path, report.output_bundle_path)
            self.assertEqual("paper_mlp_probability", bundle.metadata.score_semantics)
            self.assertEqual("exfiltration_intent", predictions[0].predicted_label)
            self.assertEqual("secret_present_safe", predictions[1].predicted_label)
            self.assertTrue(bool(np.all(layer_weights > 0.0)))


if __name__ == "__main__":
    unittest.main()
