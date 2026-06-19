import unittest

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.cift_meta_source_ablation import build_targeted_cift_meta_source_ablation_variants
from aegis_introspection.features import build_feature_key


def _artifact_with_layers(layer_indices: tuple[int, ...]) -> ActivationArtifact:
    features = {
        build_feature_key(pooling_method, layer_index): torch.zeros((4, 2), dtype=torch.float32)
        for pooling_method in ("final_token", "mean_pool")
        for layer_index in layer_indices
    }
    return {
        "metadata": {
            "model_id": "synthetic",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": layer_indices,
            "pooling_methods": ("final_token", "mean_pool"),
        },
        "example_ids": tuple(f"example_{index:03d}" for index in range(4)),
        "labels": ("benign", "secret_present_safe", "secret_present_safe", "exfiltration_intent"),
        "families": tuple(f"family_{index:02d}" for index in range(4)),
        "texts": tuple(f"synthetic prompt {index:02d}" for index in range(4)),
        "tags": tuple(("synthetic",) for _ in range(4)),
        "features": features,
    }


class CiftMetaSourceAblationTest(unittest.TestCase):
    def test_build_targeted_source_ablation_variants_removes_late_sources(self) -> None:
        artifact = _artifact_with_layers((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))

        variants = build_targeted_cift_meta_source_ablation_variants(
            artifact=artifact,
            calibration_source_labels=("secret_present_safe",),
            ridge=0.001,
            risk_label="exfiltration_intent",
            inner_fold_count=3,
            decision_rule="logistic_default",
        )

        variants_by_id = {variant.variant_id: variant for variant in variants}

        self.assertEqual(len(variants), len(variants_by_id))
        self.assertIn("full_dual_readout", variants_by_id)
        self.assertIn("drop_last_mean_pool", variants_by_id)
        self.assertIn("drop_last_two_mean_pool", variants_by_id)
        self.assertIn("drop_last_final_token", variants_by_id)
        self.assertIn("drop_last_dual_readout_layer", variants_by_id)
        self.assertIn("final_token_only", variants_by_id)
        self.assertIn("mean_pool_only", variants_by_id)
        self.assertNotIn("mean_pool_layer_12", variants_by_id["drop_last_mean_pool"].source_feature_keys)
        self.assertNotIn("mean_pool_layer_12", variants_by_id["drop_last_two_mean_pool"].source_feature_keys)
        self.assertNotIn("mean_pool_layer_11", variants_by_id["drop_last_two_mean_pool"].source_feature_keys)
        self.assertNotIn("final_token_layer_12", variants_by_id["drop_last_final_token"].source_feature_keys)
        self.assertNotIn("final_token_layer_12", variants_by_id["drop_last_dual_readout_layer"].source_feature_keys)
        self.assertNotIn("mean_pool_layer_12", variants_by_id["drop_last_dual_readout_layer"].source_feature_keys)


if __name__ == "__main__":
    unittest.main()
