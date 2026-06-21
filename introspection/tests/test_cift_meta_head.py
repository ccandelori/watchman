import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import BinaryTaskConfig
from aegis_introspection.cift_meta_head import (
    CiftMetaHeadComparisonDataset,
    CiftMetaHeadVariant,
    compare_grouped_cift_meta_head,
    render_cift_meta_head_markdown,
    write_cift_meta_head_json,
    write_cift_meta_head_markdown,
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
    families = (
        "benign_status",
        "benign_release",
        "benign_metrics",
        "benign_schedule",
        "safe_redaction_a",
        "safe_redaction_b",
        "safe_routing_a",
        "safe_routing_b",
        "safe_summary_a",
        "safe_summary_b",
        "safe_replacement_a",
        "safe_replacement_b",
        "exfil_network_a",
        "exfil_network_b",
        "exfil_log_a",
        "exfil_log_b",
        "exfil_summary_a",
        "exfil_summary_b",
        "exfil_replacement_a",
        "exfil_replacement_b",
    )
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
    secondary_source = torch.cat((safe_values * 0.5, exfil_values * 0.75), dim=0)
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
            "mean_pool_layer_06": secondary_source,
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


def _variants() -> tuple[CiftMetaHeadVariant, ...]:
    return (
        CiftMetaHeadVariant(
            variant_id="final_token",
            feature_name="cift_meta_oof_final_token",
            source_feature_keys=("final_token_layer_06", "final_token_layer_07"),
            calibration_source_labels=("benign", "secret_present_safe"),
            ridge=0.001,
            risk_label="exfiltration_intent",
            inner_fold_count=2,
            decision_rule="logistic_default",
        ),
        CiftMetaHeadVariant(
            variant_id="final_token_plus_mean_pool",
            feature_name="cift_meta_oof_final_token_plus_mean_pool",
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
            decision_rule="logistic_default",
        ),
    )


class CiftMetaHeadTest(unittest.TestCase):
    def test_compare_grouped_cift_meta_head_reports_oof_meta_folds(self) -> None:
        report = compare_grouped_cift_meta_head(
            datasets=(CiftMetaHeadComparisonDataset(dataset_id="synthetic_hard", artifact=_synthetic_artifact()),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            variants=_variants(),
            binary_config=_binary_config(),
        )

        dataset = report.datasets[0]
        best_variant = dataset.best_variant

        self.assertEqual("safe_secret_vs_exfiltration", report.task_name)
        self.assertEqual(2, report.variant_count)
        self.assertEqual(1, report.meta_head_win_count)
        self.assertGreater(best_variant.macro_f1_mean, dataset.baseline.macro_f1_mean)
        self.assertEqual(2, best_variant.inner_fold_count)
        self.assertEqual(2, len(best_variant.meta_folds))
        for fold in best_variant.meta_folds:
            self.assertEqual(best_variant.source_feature_keys, fold.source_feature_keys)
            self.assertEqual(len(best_variant.source_feature_keys), len(fold.coefficients))

    def test_render_cift_meta_head_markdown_includes_variant_and_coefficient_tables(self) -> None:
        report = compare_grouped_cift_meta_head(
            datasets=(CiftMetaHeadComparisonDataset(dataset_id="synthetic_hard", artifact=_synthetic_artifact()),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            variants=_variants(),
            binary_config=_binary_config(),
        )

        markdown = render_cift_meta_head_markdown(report)

        self.assertIn("# CIFT OOF Meta-Head", markdown)
        self.assertIn("Baseline feature: `weak_baseline_feature`", markdown)
        self.assertIn("| Variant | Source Count | Inner Folds | Mean Macro F1 | Min Macro F1 |", markdown)
        self.assertIn("| Dataset | Variant | Source Feature | Mean Coefficient |", markdown)

    def test_write_cift_meta_head_outputs_creates_files(self) -> None:
        report = compare_grouped_cift_meta_head(
            datasets=(CiftMetaHeadComparisonDataset(dataset_id="synthetic_hard", artifact=_synthetic_artifact()),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            variants=_variants(),
            binary_config=_binary_config(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "meta_head.json"
            markdown_path = Path(temp_dir) / "meta_head.md"
            write_cift_meta_head_json(json_path, report)
            write_cift_meta_head_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(2, decoded["variant_count"])
        self.assertEqual(1, decoded["meta_head_win_count"])
        self.assertIn("meta_folds", decoded["datasets"][0]["best_variant"])
        self.assertIn("CIFT OOF Meta-Head", markdown)


if __name__ == "__main__":
    unittest.main()
