from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from aegis_introspection.cift_model_training import load_cift_training_artifact_with_unseal_policy
from aegis_introspection.cift_training_artifact_conversion import (
    CiftTrainingArtifactConversionConfig,
    CiftTrainingArtifactConversionError,
    convert_cift_training_artifact,
)
from introspection.scripts.convert_cift_training_artifact import _conversion_config, _parse_args


def _artifact(tags: tuple[tuple[str, ...], ...]) -> dict[str, object]:
    return {
        "metadata": {
            "model_id": "Qwen/Qwen3-0.6B",
            "revision": "main",
            "selected_device": "cpu",
            "hidden_size": 2,
            "layer_count": 16,
            "tokenizer_fingerprint_sha256": "a" * 64,
            "special_tokens_map_sha256": "b" * 64,
            "chat_template_sha256": "c" * 64,
            "layer_indices": (15,),
            "pooling_methods": ("readout_window",),
        },
        "example_ids": ("safe-1", "exfil-1"),
        "labels": ("secret_present_safe", "exfiltration_intent"),
        "families": ("family-a", "family-b"),
        "texts": ("safe text", "exfil text"),
        "tags": tags,
        "features": {
            "readout_window_layer_15": np.asarray(
                [
                    (0.0, 0.0),
                    (2.0, 2.0),
                ],
                dtype=np.float64,
            )
        },
    }


class CiftTrainingArtifactConversionTest(unittest.TestCase):
    def test_convert_cift_training_artifact_cli_parses_paths_and_unseal_flag(self) -> None:
        cli_config = _parse_args(
            (
                "--source",
                "source.pt",
                "--output",
                "converted.pkl",
                "--allow-sealed-holdout",
            )
        )
        conversion_config = _conversion_config(cli_config)

        self.assertEqual(Path("source.pt"), conversion_config.source_path)
        self.assertEqual(Path("converted.pkl"), conversion_config.output_path)
        self.assertTrue(conversion_config.allow_sealed_holdout)

    def test_convert_cift_training_artifact_writes_dependency_clean_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.pkl"
            output_path = root / "converted.pkl"
            source_path.write_bytes(pickle.dumps(_artifact(tags=(("test",), ("test",)))))

            report = convert_cift_training_artifact(
                CiftTrainingArtifactConversionConfig(
                    source_path=source_path,
                    output_path=output_path,
                    allow_sealed_holdout=False,
                )
            )
            converted = load_cift_training_artifact_with_unseal_policy(
                path=output_path,
                allow_sealed_holdout=False,
                context="test",
            )

        self.assertEqual(output_path, report.output_path)
        self.assertEqual(2, report.example_count)
        self.assertEqual(("readout_window_layer_15",), report.feature_keys)
        self.assertEqual(np.float32, converted.features["readout_window_layer_15"].dtype)
        self.assertEqual((2, 2), converted.features["readout_window_layer_15"].shape)

    def test_convert_cift_training_artifact_rejects_sealed_tags_without_unseal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.pkl"
            output_path = root / "converted.pkl"
            source_path.write_bytes(pickle.dumps(_artifact(tags=(("test",), ("sealed_holdout",)))))

            with self.assertRaisesRegex(CiftTrainingArtifactConversionError, "sealed holdout"):
                convert_cift_training_artifact(
                    CiftTrainingArtifactConversionConfig(
                        source_path=source_path,
                        output_path=output_path,
                        allow_sealed_holdout=False,
                    )
                )


if __name__ == "__main__":
    unittest.main()
