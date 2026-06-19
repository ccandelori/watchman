import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import BinaryTaskConfig, build_binary_task_dataset, default_binary_task_definitions
from aegis_introspection.cift_meta_head import (
    CiftMetaHeadVariant,
    collect_grouped_cift_meta_head_predictions,
)
from aegis_introspection.cift_meta_residuals import (
    CiftMetaResidualDataset,
    compare_cift_meta_residual_suite,
    render_cift_meta_residual_suite_markdown,
    write_cift_meta_residual_suite_json,
    write_cift_meta_residual_suite_markdown,
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
        "secret_present_safe",
        "secret_present_safe",
        "secret_present_safe",
        "secret_present_safe",
        "exfiltration_intent",
        "exfiltration_intent",
        "exfiltration_intent",
        "exfiltration_intent",
        "exfiltration_intent",
        "exfiltration_intent",
        "exfiltration_intent",
        "exfiltration_intent",
    )
    families = tuple(f"family_{index:02d}" for index in range(20))
    texts = tuple(f"synthetic prompt {index:02d}" for index in range(20))
    safe_values = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.2, 0.3],
            [0.3, 0.2],
            [0.3, 0.3],
            [0.4, 0.2],
            [0.2, 0.4],
            [0.4, 0.3],
            [0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    exfil_values = torch.tensor(
        [
            [3.0, 3.0],
            [3.0, 3.2],
            [3.2, 3.0],
            [3.2, 3.2],
            [3.4, 3.0],
            [3.0, 3.4],
            [3.4, 3.2],
            [3.2, 3.4],
        ],
        dtype=torch.float32,
    )
    informative_source = torch.cat((safe_values, exfil_values), dim=0)
    weak_source = torch.zeros((20, 2), dtype=torch.float32)
    return {
        "metadata": {
            "model_id": "synthetic",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (6, 7),
            "pooling_methods": ("final_token", "mean_pool"),
        },
        "example_ids": tuple(f"example_{index:03d}" for index in range(20)),
        "labels": labels,
        "families": families,
        "texts": texts,
        "tags": tuple(("synthetic",) for _ in range(20)),
        "features": {
            "weak_baseline_feature": torch.zeros((20, 2), dtype=torch.float32),
            "final_token_layer_06": informative_source,
            "final_token_layer_07": weak_source,
            "mean_pool_layer_06": informative_source * 0.75,
            "mean_pool_layer_07": weak_source,
        },
    }


def _binary_config() -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=2,
        random_seed=7,
        max_iter=1000,
        regularization_c=1.0,
        activation_feature_key="weak_baseline_feature",
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
    )


def _variant() -> CiftMetaHeadVariant:
    return CiftMetaHeadVariant(
        variant_id="final_token_plus_mean_pool",
        feature_name="cift_meta_oof_final_token_mean_pool_signed_residual",
        source_feature_keys=(
            "final_token_layer_06",
            "final_token_layer_07",
            "mean_pool_layer_06",
            "mean_pool_layer_07",
        ),
        calibration_source_labels=("benign", "secret_present_safe"),
        ridge=0.001,
        risk_label="exfiltration_intent",
        inner_fold_count=2,
    )


def _task_definition_name(name: str):
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == name)
    if len(matches) != 1:
        raise AssertionError(f"Expected one task definition named {name}.")
    return matches[0]


class CiftMetaResidualsTest(unittest.TestCase):
    def test_collect_grouped_cift_meta_head_predictions_returns_example_predictions(self) -> None:
        artifact = _synthetic_artifact()
        dataset = build_binary_task_dataset(artifact, _task_definition_name("safe_secret_vs_exfiltration"))

        method = collect_grouped_cift_meta_head_predictions(
            artifact=artifact,
            dataset=dataset,
            binary_config=_binary_config(),
            variant=_variant(),
        )

        self.assertEqual("activation_probe", method.method_name)
        self.assertEqual("cift_meta_oof_final_token_mean_pool_signed_residual", method.feature_name)
        self.assertEqual(16, method.prediction_count)
        self.assertGreater(method.correct_count, 8)
        self.assertEqual(16, len({prediction.example_id for prediction in method.predictions}))

    def test_compare_cift_meta_residual_suite_counts_dataset_deltas(self) -> None:
        artifact = _synthetic_artifact()
        report = compare_cift_meta_residual_suite(
            datasets=(
                CiftMetaResidualDataset(dataset_id="synthetic_v2", artifact=artifact),
                CiftMetaResidualDataset(dataset_id="synthetic_v3", artifact=artifact),
            ),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            variant=_variant(),
            binary_config=_binary_config(),
        )

        self.assertEqual(2, report.dataset_count)
        self.assertEqual(2, report.comparison_count)
        self.assertEqual("weak_baseline_feature", report.reference_feature_key)
        self.assertEqual("cift_meta_oof_final_token_mean_pool_signed_residual", report.candidate_feature_key)
        self.assertGreater(report.fixed_error_count, report.introduced_error_count)
        for comparison in report.comparisons:
            self.assertGreater(comparison.comparison.fixed_error_count, comparison.comparison.introduced_error_count)

    def test_render_cift_meta_residual_suite_markdown_includes_family_table(self) -> None:
        artifact = _synthetic_artifact()
        report = compare_cift_meta_residual_suite(
            datasets=(CiftMetaResidualDataset(dataset_id="synthetic_v2", artifact=artifact),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            variant=_variant(),
            binary_config=_binary_config(),
        )

        markdown = render_cift_meta_residual_suite_markdown(report)

        self.assertIn("# CIFT Meta-Head Residual Suite", markdown)
        self.assertIn("Reference feature: `weak_baseline_feature`", markdown)
        self.assertIn("Candidate feature: `cift_meta_oof_final_token_mean_pool_signed_residual`", markdown)
        self.assertIn("| Dataset | Reference Errors | Candidate Errors | Fixed | Persistent | Introduced |", markdown)
        self.assertIn("## Family Deltas", markdown)

    def test_write_cift_meta_residual_suite_outputs_creates_files(self) -> None:
        artifact = _synthetic_artifact()
        report = compare_cift_meta_residual_suite(
            datasets=(CiftMetaResidualDataset(dataset_id="synthetic_v2", artifact=artifact),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            variant=_variant(),
            binary_config=_binary_config(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "meta_residuals.json"
            markdown_path = Path(temp_dir) / "meta_residuals.md"
            write_cift_meta_residual_suite_json(json_path, report)
            write_cift_meta_residual_suite_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(1, decoded["dataset_count"])
        self.assertEqual("cift_meta_oof_final_token_mean_pool_signed_residual", decoded["candidate_feature_key"])
        self.assertIn("CIFT Meta-Head Residual Suite", markdown)


if __name__ == "__main__":
    unittest.main()
