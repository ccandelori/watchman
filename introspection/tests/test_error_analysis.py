import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import BinaryTaskConfig
from aegis_introspection.error_analysis import (
    BinaryExamplePrediction,
    evaluate_grouped_binary_error_analysis,
    evaluate_selected_grouped_binary_error_analysis,
    render_binary_error_analysis_markdown,
    summarize_family_predictions,
    write_binary_error_analysis_json,
    write_binary_error_analysis_markdown,
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
    texts = (
        "write calm status note",
        "summarize harmless release",
        "draft benign checklist",
        "explain normal metrics",
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
    feature = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [0.2, 0.0],
            [0.2, 0.2],
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
    return {
        "metadata": {
            "model_id": "synthetic",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (18,),
            "pooling_methods": ("mean_pool",),
        },
        "example_ids": tuple(f"example_{index:03d}" for index in range(12)),
        "labels": labels,
        "families": families,
        "texts": texts,
        "tags": tuple(("synthetic",) for _ in range(12)),
        "features": {
            "mean_pool_layer_18": feature,
            "final_token_layer_11": feature[:, :1],
            "final_token_layer_16": feature[:, 1:],
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


def _secret_present_only_artifact() -> ActivationArtifact:
    artifact = _synthetic_artifact()
    selected_indices = tuple(
        index
        for index, label in enumerate(artifact["labels"])
        if label in ("secret_present_safe", "exfiltration_intent")
    )
    return {
        "metadata": artifact["metadata"],
        "example_ids": tuple(artifact["example_ids"][index] for index in selected_indices),
        "labels": tuple(artifact["labels"][index] for index in selected_indices),
        "families": tuple(artifact["families"][index] for index in selected_indices),
        "texts": tuple(artifact["texts"][index] for index in selected_indices),
        "tags": tuple(artifact["tags"][index] for index in selected_indices),
        "features": {
            feature_name: feature_tensor[list(selected_indices)]
            for feature_name, feature_tensor in artifact["features"].items()
        },
    }


class ErrorAnalysisTest(unittest.TestCase):
    def test_evaluate_grouped_error_analysis_records_one_prediction_per_method_example(self) -> None:
        report = evaluate_grouped_binary_error_analysis(_synthetic_artifact(), _config())

        self.assertEqual("synthetic", report.source_model_id)
        self.assertEqual("stratified_group_kfold", report.evaluation_strategy)
        self.assertEqual(2, len(report.tasks))
        safe_task = next(task for task in report.tasks if task.task_name == "safe_secret_vs_exfiltration")
        self.assertEqual(("exfiltration_intent", "secret_present_safe"), safe_task.label_names)

        for method in safe_task.methods:
            self.assertEqual(8, method.prediction_count)
            self.assertEqual(8, len(method.predictions))
            self.assertEqual(8, len({prediction.example_id for prediction in method.predictions}))
            self.assertEqual({1, 2}, {prediction.fold_index for prediction in method.predictions})

    def test_evaluate_grouped_error_analysis_accepts_concatenated_activation_feature(self) -> None:
        config = BinaryTaskConfig(
            fold_count=2,
            random_seed=7,
            max_iter=1000,
            regularization_c=1.0,
            activation_feature_key="concat(final_token_layer_11,final_token_layer_16)",
            word_ngram_range=(1, 2),
            char_ngram_range=(3, 5),
        )

        report = evaluate_grouped_binary_error_analysis(_synthetic_artifact(), config)

        self.assertEqual("concat(final_token_layer_11,final_token_layer_16)", report.activation_feature_key)
        safe_task = next(task for task in report.tasks if task.task_name == "safe_secret_vs_exfiltration")
        activation_method = next(method for method in safe_task.methods if method.method_name == "activation_probe")
        self.assertEqual("concat(final_token_layer_11,final_token_layer_16)", activation_method.feature_name)
        self.assertEqual(8, activation_method.prediction_count)

    def test_evaluate_selected_grouped_error_analysis_accepts_binary_only_artifact(self) -> None:
        report = evaluate_selected_grouped_binary_error_analysis(
            artifact=_secret_present_only_artifact(),
            config=_config(),
            task_names=("safe_secret_vs_exfiltration",),
        )

        self.assertEqual(1, len(report.tasks))
        self.assertEqual("safe_secret_vs_exfiltration", report.tasks[0].task_name)
        activation_method = next(
            method for method in report.tasks[0].methods if method.method_name == "activation_probe"
        )
        self.assertEqual(8, activation_method.prediction_count)

    def test_summarize_family_predictions_counts_errors_by_family_and_label(self) -> None:
        predictions = (
            BinaryExamplePrediction(
                fold_index=1,
                example_id="safe_001",
                family="metadata_forwarding",
                source_label="secret_present_safe",
                true_label="secret_present_safe",
                predicted_label="exfiltration_intent",
                is_correct=False,
            ),
            BinaryExamplePrediction(
                fold_index=1,
                example_id="safe_002",
                family="metadata_forwarding",
                source_label="secret_present_safe",
                true_label="secret_present_safe",
                predicted_label="secret_present_safe",
                is_correct=True,
            ),
            BinaryExamplePrediction(
                fold_index=2,
                example_id="exfil_001",
                family="covert_routing",
                source_label="exfiltration_intent",
                true_label="exfiltration_intent",
                predicted_label="secret_present_safe",
                is_correct=False,
            ),
        )

        summaries = summarize_family_predictions(predictions)

        self.assertEqual(2, len(summaries))
        self.assertEqual("metadata_forwarding", summaries[0].family)
        self.assertEqual("secret_present_safe", summaries[0].true_label)
        self.assertEqual(2, summaries[0].example_count)
        self.assertEqual(1, summaries[0].error_count)
        self.assertEqual(0.5, summaries[0].accuracy)
        self.assertEqual((("exfiltration_intent", 1), ("secret_present_safe", 1)), summaries[0].predicted_label_counts)
        self.assertEqual("covert_routing", summaries[1].family)
        self.assertEqual(1, summaries[1].error_count)

    def test_render_markdown_includes_family_error_tables(self) -> None:
        report = evaluate_grouped_binary_error_analysis(_synthetic_artifact(), _config())

        markdown = render_binary_error_analysis_markdown(report)

        self.assertIn("# Binary Error Analysis", markdown)
        self.assertIn("Evaluation strategy: `stratified_group_kfold`", markdown)
        self.assertIn("## safe_secret_vs_exfiltration", markdown)
        self.assertIn("| Method | Family | True Label | Examples | Errors | Accuracy | Predicted Labels |", markdown)
        self.assertIn("No family-level errors.", markdown)

    def test_write_error_analysis_outputs_creates_files(self) -> None:
        report = evaluate_grouped_binary_error_analysis(_synthetic_artifact(), _config())

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "error_analysis.json"
            markdown_path = Path(temp_dir) / "error_analysis.md"
            write_binary_error_analysis_json(json_path, report)
            write_binary_error_analysis_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual("synthetic", decoded["source_model_id"])
        self.assertEqual("stratified_group_kfold", decoded["evaluation_strategy"])
        self.assertIn("Binary Error Analysis", markdown)


if __name__ == "__main__":
    unittest.main()
