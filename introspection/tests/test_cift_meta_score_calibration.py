import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import BinaryTaskConfig
from aegis_introspection.cift_meta_score_calibration import (
    CiftMetaScoreCalibrationDataset,
    CiftMetaScoreCalibrationVariant,
    compare_cift_meta_score_calibration,
    render_cift_meta_score_calibration_markdown,
    write_cift_meta_score_calibration_json,
    write_cift_meta_score_calibration_markdown,
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


def _variants() -> tuple[CiftMetaScoreCalibrationVariant, ...]:
    source_feature_keys = (
        "final_token_layer_06",
        "final_token_layer_07",
        "mean_pool_layer_06",
        "mean_pool_layer_07",
    )
    return (
        CiftMetaScoreCalibrationVariant(
            variant_id="raw_probability",
            feature_name="cift_meta_score_calibration_raw_probability",
            source_feature_keys=source_feature_keys,
            calibration_source_labels=("secret_present_safe",),
            ridge=0.001,
            risk_label="exfiltration_intent",
            inner_fold_count=2,
            meta_regularization_c=10.0,
            score_calibration_rule="raw_probability",
        ),
        CiftMetaScoreCalibrationVariant(
            variant_id="clipped_logit",
            feature_name="cift_meta_score_calibration_clipped_logit",
            source_feature_keys=source_feature_keys,
            calibration_source_labels=("secret_present_safe",),
            ridge=0.001,
            risk_label="exfiltration_intent",
            inner_fold_count=2,
            meta_regularization_c=10.0,
            score_calibration_rule="clipped_logit",
        ),
        CiftMetaScoreCalibrationVariant(
            variant_id="platt_probability",
            feature_name="cift_meta_score_calibration_platt_probability",
            source_feature_keys=source_feature_keys,
            calibration_source_labels=("secret_present_safe",),
            ridge=0.001,
            risk_label="exfiltration_intent",
            inner_fold_count=2,
            meta_regularization_c=10.0,
            score_calibration_rule="platt_probability",
        ),
    )


class CiftMetaScoreCalibrationTest(unittest.TestCase):
    def test_compare_cift_meta_score_calibration_reports_rule_tradeoffs(self) -> None:
        artifact = _synthetic_artifact()
        report = compare_cift_meta_score_calibration(
            datasets=(CiftMetaScoreCalibrationDataset(dataset_id="synthetic_v2", artifact=artifact),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            variants=_variants(),
            binary_config=_binary_config(),
        )

        self.assertEqual(3, report.variant_count)
        self.assertEqual("weak_baseline_feature", report.baseline_feature_key)
        self.assertEqual(
            ("raw_probability", "clipped_logit", "platt_probability"),
            tuple(summary.score_calibration_rule for summary in report.variant_summaries),
        )
        self.assertEqual(10.0, report.meta_regularization_c)

    def test_render_cift_meta_score_calibration_markdown_includes_rule_table(self) -> None:
        artifact = _synthetic_artifact()
        report = compare_cift_meta_score_calibration(
            datasets=(CiftMetaScoreCalibrationDataset(dataset_id="synthetic_v2", artifact=artifact),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            variants=_variants(),
            binary_config=_binary_config(),
        )

        markdown = render_cift_meta_score_calibration_markdown(report)

        self.assertIn("# CIFT Meta-Head Source-Score Calibration", markdown)
        self.assertIn("| Variant | Calibration Rule | Meta C | Source Count |", markdown)
        self.assertIn("| Dataset | Variant | Candidate Errors | Fixed | Persistent | Introduced |", markdown)

    def test_write_cift_meta_score_calibration_outputs_creates_files(self) -> None:
        artifact = _synthetic_artifact()
        report = compare_cift_meta_score_calibration(
            datasets=(CiftMetaScoreCalibrationDataset(dataset_id="synthetic_v2", artifact=artifact),),
            task_name="safe_secret_vs_exfiltration",
            baseline_feature_key="weak_baseline_feature",
            variants=_variants(),
            binary_config=_binary_config(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "score_calibration.json"
            markdown_path = Path(temp_dir) / "score_calibration.md"
            write_cift_meta_score_calibration_json(json_path, report)
            write_cift_meta_score_calibration_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(3, decoded["variant_count"])
        self.assertIn("best_variant_summary", decoded)
        self.assertIn("score_calibration_rule", decoded["variant_summaries"][0])
        self.assertIn("CIFT Meta-Head Source-Score Calibration", markdown)


if __name__ == "__main__":
    unittest.main()
