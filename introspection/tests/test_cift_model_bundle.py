import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from aegis_introspection.binary_tasks import BinaryTaskConfig, build_activation_classifier
from aegis_introspection.cift_model_bundle import (
    CiftModelBundle,
    CiftModelBundleError,
    CiftModelBundleMetadata,
    load_cift_model_bundle,
    predict_cift_model_bundle,
    save_cift_model_bundle,
)


def _trained_classifier() -> Pipeline:
    matrix = np.asarray(
        (
            (0.0, 0.0),
            (0.1, 0.0),
            (2.0, 2.0),
            (2.2, 2.0),
        ),
        dtype=np.float32,
    )
    labels = np.asarray((0, 0, 1, 1), dtype=np.int64)
    config = BinaryTaskConfig(
        fold_count=2,
        random_seed=42,
        max_iter=1000,
        regularization_c=1.0,
        activation_feature_key="readout_window_layer_15",
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
    )
    classifier = build_activation_classifier(config)
    classifier.fit(matrix, labels)
    return classifier


def _metadata() -> CiftModelBundleMetadata:
    return CiftModelBundleMetadata(
        schema_version="cift_model_bundle/v1",
        source_model_id="Qwen/Qwen3-0.6B",
        source_revision="main",
        source_selected_device="mps",
        training_dataset_id="dp_honey_lite_v4_1_selector_windows",
        source_artifact_path="data/activations/qwen3_0_6b_dp_honey_lite_v4_1_selector_windows.pt",
        source_artifact_sha256="a" * 64,
        evaluation_report_ids=("dp_honey_lite_v4_1_selector_window_grouped_binary_tasks_readout_window_layer_15_v1",),
        task_name="safe_secret_vs_exfiltration",
        activation_feature_key="readout_window_layer_15",
        feature_count=2,
        label_names=("secret_present_safe", "exfiltration_intent"),
        positive_label="exfiltration_intent",
        decision_threshold=0.5,
        score_semantics="full_train_classifier_probability",
        created_at="2026-06-21T00:00:00Z",
        candidate_status="offline_research_candidate",
    )


class CiftModelBundleTest(unittest.TestCase):
    def test_save_load_and_predict_preserves_trained_detector_metadata(self) -> None:
        bundle = CiftModelBundle(metadata=_metadata(), classifier=_trained_classifier(), calibrator=None)
        matrix = np.asarray(((2.4, 2.1), (0.0, 0.1)), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cift_model_bundle.pkl"

            save_cift_model_bundle(path=path, bundle=bundle)
            loaded = load_cift_model_bundle(path=path)
            predictions = predict_cift_model_bundle(bundle=loaded, feature_matrix=matrix)

        self.assertEqual(bundle.metadata, loaded.metadata)
        self.assertEqual("exfiltration_intent", predictions[0].predicted_label)
        self.assertEqual("secret_present_safe", predictions[1].predicted_label)
        self.assertGreater(predictions[0].positive_probability, predictions[1].positive_probability)
        self.assertEqual("full_train_classifier_probability", predictions[0].score_semantics)

    def test_save_rejects_invalid_decision_threshold(self) -> None:
        metadata = replace(_metadata(), decision_threshold=1.2)
        bundle = CiftModelBundle(metadata=metadata, classifier=_trained_classifier(), calibrator=None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cift_model_bundle.pkl"

            with self.assertRaisesRegex(CiftModelBundleError, "decision_threshold"):
                save_cift_model_bundle(path=path, bundle=bundle)

    def test_predict_rejects_feature_count_mismatch(self) -> None:
        bundle = CiftModelBundle(metadata=_metadata(), classifier=_trained_classifier(), calibrator=None)
        matrix = np.asarray(((2.4, 2.1, 9.9),), dtype=np.float32)

        with self.assertRaisesRegex(CiftModelBundleError, "bundle expects 2"):
            predict_cift_model_bundle(bundle=bundle, feature_matrix=matrix)


if __name__ == "__main__":
    unittest.main()
