import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from aegis_introspection.cift_model_training import CiftModelTrainingConfig, train_cift_model_bundle
from aegis_introspection.trained_detector_export import (
    TrainedDetectorExportConfig,
    export_trained_cift_detector_results,
)


def _artifact() -> dict[str, object]:
    return {
        "metadata": {
            "model_id": "Qwen/Qwen3-0.6B",
            "revision": "main",
            "selected_device": "cpu",
            "hidden_size": 2,
            "layer_count": 16,
            "tokenizer_fingerprint_sha256": "a" * 64,
            "special_tokens_map_sha256": "b" * 64,
            "chat_template_sha256": "c" * 64,
            "layer_indices": (15,),
            "pooling_methods": ("readout_window",),
        },
        "example_ids": ("safe-1", "safe-2", "exfil-1", "exfil-2"),
        "labels": ("secret_present_safe", "secret_present_safe", "exfiltration_intent", "exfiltration_intent"),
        "families": ("family-a", "family-b", "family-a", "family-b"),
        "texts": ("safe text one", "safe text two", "exfil text one", "exfil text two"),
        "tags": (("test",), ("test",), ("test",), ("test",)),
        "features": {
            "readout_window_layer_15": np.asarray(
                [
                    (0.0, 0.0),
                    (0.1, 0.0),
                    (2.0, 2.0),
                    (2.2, 2.0),
                ],
                dtype=np.float32,
            )
        },
    }


def _runtime_turn(example_id: str, turn_index: int) -> dict[str, object]:
    return {
        "trace_id": f"trace-{example_id}",
        "session_id": "session-1",
        "turn_index": turn_index,
        "metadata": {"example_id": example_id},
    }


class TrainedDetectorExportTest(unittest.TestCase):
    def test_export_trained_cift_detector_results_writes_runtime_detector_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "features.pkl"
            bundle_path = root / "cift_model_bundle.pkl"
            runtime_turns_path = root / "runtime_turns.jsonl"
            output_path = root / "detector_results.jsonl"
            artifact_path.write_bytes(pickle.dumps(_artifact()))
            train_cift_model_bundle(
                CiftModelTrainingConfig(
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
            )
            runtime_turns_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        _runtime_turn("safe-1", 1),
                        _runtime_turn("safe-2", 2),
                        _runtime_turn("exfil-1", 3),
                        _runtime_turn("exfil-2", 4),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            config = TrainedDetectorExportConfig(
                runtime_turns_path=runtime_turns_path,
                artifact_path=artifact_path,
                model_bundle_path=bundle_path,
                output_path=output_path,
                detector_name="cift_selector_probe",
                model_bundle_id="synthetic_cift_bundle",
                capability_required="self_hosted_introspection",
                positive_action="warn",
                negative_action="allow",
                confidence=0.77,
                allow_sealed_holdout=False,
            )

            row_count = export_trained_cift_detector_results(config)

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(4, row_count)
        self.assertEqual("trace-safe-1", rows[0]["trace_id"])
        self.assertEqual("allow", rows[0]["detector_result"]["recommended_action"])
        self.assertEqual("trace-exfil-1", rows[2]["trace_id"])
        self.assertEqual("warn", rows[2]["detector_result"]["recommended_action"])
        self.assertEqual("synthetic_cift_bundle", rows[2]["detector_result"]["evidence"]["model_bundle_id"])
        self.assertEqual(
            "full_train_classifier_probability",
            rows[2]["detector_result"]["evidence"]["score_semantics"],
        )

    def test_export_trained_cift_detector_results_supports_paper_mlp_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "features.pkl"
            bundle_path = root / "cift_paper_mlp_bundle.pkl"
            runtime_turns_path = root / "runtime_turns.jsonl"
            output_path = root / "detector_results.jsonl"
            artifact_path.write_bytes(pickle.dumps(_artifact()))
            train_cift_model_bundle(
                CiftModelTrainingConfig(
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
            )
            runtime_turns_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        _runtime_turn("safe-1", 1),
                        _runtime_turn("safe-2", 2),
                        _runtime_turn("exfil-1", 3),
                        _runtime_turn("exfil-2", 4),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            config = TrainedDetectorExportConfig(
                runtime_turns_path=runtime_turns_path,
                artifact_path=artifact_path,
                model_bundle_path=bundle_path,
                output_path=output_path,
                detector_name="cift_selector_probe",
                model_bundle_id="synthetic_cift_paper_mlp_bundle",
                capability_required="self_hosted_introspection",
                positive_action="warn",
                negative_action="allow",
                confidence=0.77,
                allow_sealed_holdout=False,
            )

            row_count = export_trained_cift_detector_results(config)

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(4, row_count)
        self.assertEqual("paper_mlp_probability", rows[2]["detector_result"]["evidence"]["score_semantics"])
        self.assertEqual("synthetic_cift_paper_mlp_bundle", rows[2]["detector_result"]["evidence"]["model_bundle_id"])
        self.assertEqual("warn", rows[2]["detector_result"]["recommended_action"])


if __name__ == "__main__":
    unittest.main()
