import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import (
    BinaryTaskConfig,
    BinaryTaskDefinition,
    BinaryTaskError,
    build_binary_task_dataset,
    default_binary_task_definitions,
    evaluate_binary_tasks,
    evaluate_grouped_binary_tasks,
    render_binary_tasks_markdown,
    stratified_group_splits,
    write_binary_tasks_markdown,
    write_binary_tasks_report_json,
)
from aegis_introspection.probe import encode_labels


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


class BinaryTasksTest(unittest.TestCase):
    def test_build_binary_task_dataset_maps_benign_vs_secret_related(self) -> None:
        definition = default_binary_task_definitions()[0]

        dataset = build_binary_task_dataset(_synthetic_artifact(), definition)

        self.assertEqual("benign_vs_secret_related", dataset.name)
        self.assertEqual(12, len(dataset.example_ids))
        self.assertEqual(12, len(dataset.families))
        self.assertEqual(4, dataset.target_labels.count("benign"))
        self.assertEqual(8, dataset.target_labels.count("secret_related"))

    def test_build_binary_task_dataset_filters_safe_vs_exfiltration(self) -> None:
        definition = default_binary_task_definitions()[1]

        dataset = build_binary_task_dataset(_synthetic_artifact(), definition)

        self.assertEqual("safe_secret_vs_exfiltration", dataset.name)
        self.assertEqual(8, len(dataset.example_ids))
        self.assertEqual(4, dataset.target_labels.count("secret_present_safe"))
        self.assertEqual(4, dataset.target_labels.count("exfiltration_intent"))

    def test_build_binary_task_dataset_rejects_non_binary_output(self) -> None:
        definition = BinaryTaskDefinition(
            name="bad_task",
            description="Bad task.",
            source_labels=("benign", "secret_present_safe"),
            target_labels=("same", "same"),
        )

        with self.assertRaises(BinaryTaskError):
            build_binary_task_dataset(_synthetic_artifact(), definition)

    def test_evaluate_binary_tasks_returns_three_methods_per_task(self) -> None:
        report = evaluate_binary_tasks(_synthetic_artifact(), _config())

        self.assertEqual("synthetic", report.source_model_id)
        self.assertEqual("stratified_kfold", report.evaluation_strategy)
        self.assertEqual(2, len(report.tasks))
        for task in report.tasks:
            self.assertEqual(("activation_probe", "word_tfidf", "char_tfidf"), tuple(method.method_name for method in task.methods))

    def test_stratified_group_splits_keep_families_out_of_both_train_and_test(self) -> None:
        artifact = _synthetic_artifact()
        dataset = build_binary_task_dataset(artifact, default_binary_task_definitions()[1])
        label_encoding = encode_labels(dataset.target_labels)

        splits = stratified_group_splits(
            encoded_labels=label_encoding.encoded_labels,
            groups=dataset.families,
            config=_config(),
        )

        self.assertEqual(2, len(splits))
        for split in splits:
            train_groups = {dataset.families[index] for index in split.train_indices.tolist()}
            test_groups = {dataset.families[index] for index in split.test_indices.tolist()}
            self.assertEqual(set(), train_groups.intersection(test_groups))

    def test_evaluate_grouped_binary_tasks_uses_grouped_strategy(self) -> None:
        report = evaluate_grouped_binary_tasks(_synthetic_artifact(), _config())

        self.assertEqual("stratified_group_kfold", report.evaluation_strategy)
        self.assertEqual(2, len(report.tasks))
        for task in report.tasks:
            self.assertEqual(("activation_probe", "word_tfidf", "char_tfidf"), tuple(method.method_name for method in task.methods))

    def test_evaluate_binary_tasks_rejects_missing_activation_feature(self) -> None:
        config = BinaryTaskConfig(
            fold_count=2,
            random_seed=7,
            max_iter=1000,
            regularization_c=1.0,
            activation_feature_key="missing_feature",
            word_ngram_range=(1, 2),
            char_ngram_range=(3, 5),
        )

        with self.assertRaises(BinaryTaskError):
            evaluate_binary_tasks(_synthetic_artifact(), config)

    def test_render_binary_tasks_markdown_includes_all_methods(self) -> None:
        report = evaluate_binary_tasks(_synthetic_artifact(), _config())

        markdown = render_binary_tasks_markdown(report)

        self.assertIn("# Binary Task Evaluation Summary", markdown)
        self.assertIn("Evaluation strategy: `stratified_kfold`", markdown)
        self.assertIn("`activation_probe`", markdown)
        self.assertIn("`word_tfidf`", markdown)
        self.assertIn("`char_tfidf`", markdown)

    def test_write_binary_tasks_outputs_creates_files(self) -> None:
        report = evaluate_binary_tasks(_synthetic_artifact(), _config())

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "binary_tasks.json"
            markdown_path = Path(temp_dir) / "binary_tasks.md"
            write_binary_tasks_report_json(json_path, report)
            write_binary_tasks_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual("synthetic", decoded["source_model_id"])
        self.assertIn("Binary Task Evaluation Summary", markdown)


if __name__ == "__main__":
    unittest.main()
