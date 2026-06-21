import unittest

import torch

from aegis_introspection.artifacts import ActivationArtifactError, validate_activation_artifact


class ActivationArtifactTest(unittest.TestCase):
    def test_validate_activation_artifact_accepts_expected_schema(self) -> None:
        artifact = validate_activation_artifact(
            {
                "metadata": {
                    "model_id": "Qwen/Qwen3-0.6B",
                    "revision": "main",
                    "selected_device": "cpu",
                    "layer_indices": (0, 28),
                    "pooling_methods": ("final_token",),
                },
                "example_ids": ("example_001",),
                "labels": ("benign",),
                "families": ("benign_release_notes",),
                "texts": ("Text.",),
                "tags": (("tag",),),
                "features": {
                    "final_token_layer_00": torch.zeros((1, 2)),
                },
            }
        )

        self.assertEqual("Qwen/Qwen3-0.6B", artifact["metadata"]["model_id"])
        self.assertEqual(("benign_release_notes",), artifact["families"])
        self.assertEqual((1, 2), tuple(artifact["features"]["final_token_layer_00"].shape))

    def test_validate_activation_artifact_rejects_missing_families(self) -> None:
        with self.assertRaises(ActivationArtifactError):
            validate_activation_artifact(
                {
                    "metadata": {
                        "model_id": "Qwen/Qwen3-0.6B",
                        "revision": "main",
                        "selected_device": "cpu",
                        "layer_indices": (0,),
                        "pooling_methods": ("final_token",),
                    },
                    "example_ids": ("example_001",),
                    "labels": ("benign",),
                    "texts": ("Text.",),
                    "tags": (("tag",),),
                    "features": {
                        "final_token_layer_00": torch.zeros((1, 2)),
                    },
                }
            )

    def test_validate_activation_artifact_rejects_row_count_mismatch(self) -> None:
        with self.assertRaises(ActivationArtifactError):
            validate_activation_artifact(
                {
                    "metadata": {
                        "model_id": "Qwen/Qwen3-0.6B",
                        "revision": "main",
                        "selected_device": "cpu",
                        "layer_indices": (0,),
                        "pooling_methods": ("final_token",),
                    },
                    "example_ids": ("example_001",),
                    "labels": ("benign",),
                    "families": ("benign_release_notes", "benign_release_notes"),
                    "texts": ("Text.",),
                    "tags": (("tag",),),
                    "features": {
                        "final_token_layer_00": torch.zeros((2, 2)),
                    },
                }
            )

    def test_validate_activation_artifact_rejects_non_tensor_feature(self) -> None:
        with self.assertRaises(ActivationArtifactError):
            validate_activation_artifact(
                {
                    "metadata": {
                        "model_id": "Qwen/Qwen3-0.6B",
                        "revision": "main",
                        "selected_device": "cpu",
                        "layer_indices": (0,),
                        "pooling_methods": ("final_token",),
                    },
                    "example_ids": ("example_001",),
                    "labels": ("benign",),
                    "families": ("benign_release_notes",),
                    "texts": ("Text.",),
                    "tags": (("tag",),),
                    "features": {
                        "final_token_layer_00": [[0.0, 1.0]],
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
