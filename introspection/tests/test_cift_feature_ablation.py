import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import BinaryTaskConfig, BinaryTaskError
from aegis_introspection.cift_feature_ablation import (
    CiftFeatureAblationVariant,
    evaluate_grouped_cift_feature_ablation,
    render_cift_feature_ablation_markdown,
    write_cift_feature_ablation_json,
    write_cift_feature_ablation_markdown,
)
from introspection.scripts.ablate_cift_selector_window_features import _default_selector_window_variants, _parse_args


def _synthetic_artifact() -> ActivationArtifact:
    labels = (
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
        "mask selected credential",
        "protect selected credential",
        "summarize without copying",
        "redact selected field",
        "copy selected credential",
        "forward selected credential",
        "include selected credential",
        "send selected field",
    )
    families = (
        "safe_a",
        "safe_a",
        "safe_b",
        "safe_b",
        "exfil_a",
        "exfil_a",
        "exfil_b",
        "exfil_b",
    )
    weak_feature = torch.zeros((8, 2), dtype=torch.float32)
    strong_feature = torch.tensor(
        [
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
            "layer_indices": (14, 15),
            "pooling_methods": ("readout_window",),
        },
        "example_ids": tuple(f"example_{index:03d}" for index in range(8)),
        "labels": labels,
        "families": families,
        "texts": texts,
        "tags": tuple(("synthetic",) for _ in range(8)),
        "features": {
            "readout_window_layer_14": strong_feature,
            "readout_window_layer_15": weak_feature,
        },
    }


def _config() -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=2,
        random_seed=7,
        max_iter=1000,
        regularization_c=1.0,
        activation_feature_key="readout_window_layer_15",
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
    )


class CiftFeatureAblationTest(unittest.TestCase):
    def test_grouped_ablation_ranks_named_feature_variants_against_baseline(self) -> None:
        report = evaluate_grouped_cift_feature_ablation(
            artifact=_synthetic_artifact(),
            task_name="safe_secret_vs_exfiltration",
            variants=(
                CiftFeatureAblationVariant(
                    variant_id="baseline_layer_15",
                    feature_key="readout_window_layer_15",
                ),
                CiftFeatureAblationVariant(
                    variant_id="local_concat_14_15",
                    feature_key="concat(readout_window_layer_14,readout_window_layer_15)",
                ),
            ),
            baseline_variant_id="baseline_layer_15",
            config=_config(),
        )

        self.assertEqual("synthetic", report.source_model_id)
        self.assertEqual("safe_secret_vs_exfiltration", report.task_name)
        self.assertEqual("baseline_layer_15", report.baseline_variant_id)
        self.assertEqual("local_concat_14_15", report.best_variant_id)
        self.assertEqual("local_concat_14_15", report.variants[0].variant_id)
        self.assertFalse(report.variants[0].is_baseline)
        self.assertTrue(report.variants[1].is_baseline)
        self.assertGreater(report.variants[0].macro_f1_mean, report.variants[1].macro_f1_mean)

    def test_render_markdown_includes_ranked_variants(self) -> None:
        report = evaluate_grouped_cift_feature_ablation(
            artifact=_synthetic_artifact(),
            task_name="safe_secret_vs_exfiltration",
            variants=(
                CiftFeatureAblationVariant(
                    variant_id="baseline_layer_15",
                    feature_key="readout_window_layer_15",
                ),
                CiftFeatureAblationVariant(
                    variant_id="local_concat_14_15",
                    feature_key="concat(readout_window_layer_14,readout_window_layer_15)",
                ),
            ),
            baseline_variant_id="baseline_layer_15",
            config=_config(),
        )

        markdown = render_cift_feature_ablation_markdown(report)

        self.assertIn("CIFT Feature Ablation", markdown)
        self.assertIn("Best variant: `local_concat_14_15`", markdown)
        self.assertIn("baseline_layer_15", markdown)

    def test_write_ablation_outputs_creates_json_and_markdown(self) -> None:
        report = evaluate_grouped_cift_feature_ablation(
            artifact=_synthetic_artifact(),
            task_name="safe_secret_vs_exfiltration",
            variants=(
                CiftFeatureAblationVariant(
                    variant_id="baseline_layer_15",
                    feature_key="readout_window_layer_15",
                ),
                CiftFeatureAblationVariant(
                    variant_id="local_concat_14_15",
                    feature_key="concat(readout_window_layer_14,readout_window_layer_15)",
                ),
            ),
            baseline_variant_id="baseline_layer_15",
            config=_config(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "ablation.json"
            markdown_path = Path(temp_dir) / "ablation.md"
            write_cift_feature_ablation_json(json_path, report)
            write_cift_feature_ablation_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual("local_concat_14_15", decoded["best_variant_id"])
        self.assertEqual(2, len(decoded["variants"]))
        self.assertIn("CIFT Feature Ablation", markdown)

    def test_duplicate_variant_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(BinaryTaskError, "variant ids must be unique"):
            evaluate_grouped_cift_feature_ablation(
                artifact=_synthetic_artifact(),
                task_name="safe_secret_vs_exfiltration",
                variants=(
                    CiftFeatureAblationVariant(
                        variant_id="duplicate",
                        feature_key="readout_window_layer_14",
                    ),
                    CiftFeatureAblationVariant(
                        variant_id="duplicate",
                        feature_key="readout_window_layer_15",
                    ),
                ),
                baseline_variant_id="duplicate",
                config=_config(),
            )

    def test_script_defaults_include_baseline_and_multilayer_selector_variants(self) -> None:
        config = _parse_args(())
        variants = _default_selector_window_variants()

        self.assertEqual("baseline_layer_15", config.baseline_variant_id)
        self.assertEqual("dp_honey_lite_v3_selector_window_feature_ablation_v1.json", config.output_json_path.name)
        self.assertEqual("baseline_layer_15", variants[0].variant_id)
        self.assertEqual("readout_window_layer_15", variants[0].feature_key)
        self.assertIn("local_concat_14_15_16", tuple(variant.variant_id for variant in variants))
        self.assertIn("all_readout_layers_00_28", tuple(variant.variant_id for variant in variants))


if __name__ == "__main__":
    unittest.main()
