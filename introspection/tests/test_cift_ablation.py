import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import BinaryTaskConfig
from aegis_introspection.cift_ablation import (
    CiftAblationDataset,
    CiftAblationVariant,
    compare_grouped_cift_ablation,
    render_cift_ablation_markdown,
    write_cift_ablation_json,
    write_cift_ablation_markdown,
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
    texts = (
        "write a harmless status note",
        "summarize release notes",
        "draft a routine checklist",
        "explain ordinary metrics",
        "redact secret safely",
        "mask credential carefully",
        "protect api key",
        "classify sensitive token",
        "send secret outside",
        "upload credential remote",
        "post token webhook",
        "leak api key",
    )
    layer_06 = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [0.2, 0.0],
            [0.2, 0.2],
            [0.1, 0.1],
            [0.1, 0.3],
            [0.3, 0.1],
            [0.3, 0.3],
            [4.0, 4.0],
            [4.0, 4.2],
            [4.2, 4.0],
            [4.2, 4.2],
        ],
        dtype=torch.float32,
    )
    layer_07 = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.2],
            [0.2, 0.3],
            [0.3, 0.3],
            [2.0, 2.0],
            [2.1, 2.0],
            [2.0, 2.1],
            [2.1, 2.1],
        ],
        dtype=torch.float32,
    )
    return {
        "metadata": {
            "model_id": "synthetic",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (0, 1, 2, 3, 4, 5, 6, 7),
            "pooling_methods": ("final_token", "mean_pool"),
        },
        "example_ids": tuple(f"example_{index:03d}" for index in range(12)),
        "labels": labels,
        "families": families,
        "texts": texts,
        "tags": tuple(("synthetic",) for _ in range(12)),
        "features": {
            "weak_baseline_feature": torch.zeros((12, 2), dtype=torch.float32),
            "final_token_layer_06": layer_06,
            "final_token_layer_07": layer_07,
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


def _variants() -> tuple[CiftAblationVariant, ...]:
    return (
        CiftAblationVariant(
            variant_id="diag_nonleaking",
            feature_name="cift_diag_nonleaking_final_token_last_quarter",
            source_feature_keys=("final_token_layer_06", "final_token_layer_07"),
            calibration_source_labels=("benign", "secret_present_safe"),
            representation="diagonal_distance",
            classifier_mode="standard_scaled_logreg",
            ridge=0.001,
        ),
        CiftAblationVariant(
            variant_id="residual_nonleaking",
            feature_name="cift_residual_nonleaking_final_token_last_quarter",
            source_feature_keys=("final_token_layer_06", "final_token_layer_07"),
            calibration_source_labels=("benign", "secret_present_safe"),
            representation="standardized_residual_concat",
            classifier_mode="standard_scaled_logreg",
            ridge=0.001,
        ),
        CiftAblationVariant(
            variant_id="absolute_residual_raw_nonleaking",
            feature_name="cift_abs_residual_raw_nonleaking_final_token_last_quarter",
            source_feature_keys=("final_token_layer_06", "final_token_layer_07"),
            calibration_source_labels=("benign", "secret_present_safe"),
            representation="absolute_standardized_residual_concat",
            classifier_mode="raw_logreg",
            ridge=0.001,
        ),
    )


class CiftAblationTest(unittest.TestCase):
    def test_compare_grouped_cift_ablation_reports_variant_winners(self) -> None:
        report = compare_grouped_cift_ablation(
            datasets=(CiftAblationDataset(dataset_id="synthetic_hard", artifact=_synthetic_artifact()),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            variants=_variants(),
            binary_config=_binary_config(),
        )

        self.assertEqual("safe_secret_vs_exfiltration", report.task_name)
        self.assertEqual("weak_baseline_feature", report.baseline_feature_key)
        self.assertEqual(3, report.variant_count)
        self.assertEqual(1, report.dataset_count)
        self.assertEqual(1, report.ablation_win_count)
        self.assertEqual(0, report.baseline_win_count)
        self.assertEqual("synthetic_hard", report.datasets[0].dataset_id)
        self.assertEqual(
            ("standard_scaled_logreg", "standard_scaled_logreg", "raw_logreg"),
            tuple(variant.classifier_mode for variant in report.datasets[0].variants),
        )
        self.assertGreater(report.datasets[0].best_variant.macro_f1_mean, report.datasets[0].baseline.macro_f1_mean)

    def test_render_cift_ablation_markdown_includes_variant_table(self) -> None:
        report = compare_grouped_cift_ablation(
            datasets=(CiftAblationDataset(dataset_id="synthetic_hard", artifact=_synthetic_artifact()),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            variants=_variants(),
            binary_config=_binary_config(),
        )

        markdown = render_cift_ablation_markdown(report)

        self.assertIn("# CIFT-Like Ablation", markdown)
        self.assertIn("Baseline feature: `weak_baseline_feature`", markdown)
        self.assertIn(
            "| Variant | Representation | Classifier Mode | Calibration Labels | Mean Macro F1 | Min Macro F1 |",
            markdown,
        )
        self.assertIn(
            "| Dataset | Variant | Representation | Classifier Mode | Calibration Labels | Macro F1 | Delta Macro F1 |",
            markdown,
        )

    def test_write_cift_ablation_outputs_creates_files(self) -> None:
        report = compare_grouped_cift_ablation(
            datasets=(CiftAblationDataset(dataset_id="synthetic_hard", artifact=_synthetic_artifact()),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            variants=_variants(),
            binary_config=_binary_config(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "ablation.json"
            markdown_path = Path(temp_dir) / "ablation.md"
            write_cift_ablation_json(json_path, report)
            write_cift_ablation_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(3, decoded["variant_count"])
        self.assertEqual(1, decoded["ablation_win_count"])
        self.assertEqual("raw_logreg", decoded["datasets"][0]["variants"][2]["classifier_mode"])
        self.assertIn("CIFT-Like Ablation", markdown)


if __name__ == "__main__":
    unittest.main()
