import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import (
    ActivationArtifactError,
    load_activation_artifact,
    load_activation_artifact_allowing_sealed_holdout,
    validate_activation_artifact,
)


def _valid_artifact(tags: tuple[tuple[str, ...], ...]) -> dict[str, object]:
    return {
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
        "tags": tags,
        "features": {
            "final_token_layer_00": torch.zeros((1, 2)),
        },
    }


class ActivationArtifactTest(unittest.TestCase):
    def test_validate_activation_artifact_accepts_expected_schema(self) -> None:
        artifact = validate_activation_artifact(_valid_artifact(tags=(("tag",),)))

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

    def test_load_activation_artifact_rejects_sealed_tags_even_with_neutral_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "features.pt"
            torch.save(_valid_artifact(tags=(("sealed_holdout",),)), artifact_path)

            with self.assertRaises(ActivationArtifactError):
                load_activation_artifact(artifact_path)

    def test_load_activation_artifact_allowing_sealed_holdout_reads_sealed_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "features.pt"
            torch.save(_valid_artifact(tags=(("sealed_holdout",),)), artifact_path)

            artifact = load_activation_artifact_allowing_sealed_holdout(artifact_path)

        self.assertEqual((("sealed_holdout",),), artifact["tags"])


if __name__ == "__main__":
    unittest.main()
