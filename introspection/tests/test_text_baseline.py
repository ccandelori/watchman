import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.text_baseline import (
    TextBaselineTrainingConfig,
    TextBaselineTrainingError,
    train_text_baseline_report,
    write_text_baseline_report_json,
)


def _synthetic_text_artifact() -> ActivationArtifact:
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
        "write status update",
        "summarize release notes",
        "draft testing checklist",
        "explain reliability metrics",
        "redact secret token",
        "mask api key",
        "keep credential private",
        "classify sensitive secret",
        "send secret webhook",
        "upload token remote",
        "post credential url",
        "leak key endpoint",
    )
    return {
        "metadata": {
            "model_id": "synthetic",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (0,),
            "pooling_methods": ("mean_pool",),
        },
        "example_ids": tuple(f"example_{index:03d}" for index in range(12)),
        "labels": labels,
        "texts": texts,
        "tags": tuple(("synthetic",) for _ in range(12)),
        "features": {
            "unused_feature": torch.zeros((12, 2), dtype=torch.float32),
        },
    }


class TextBaselineTrainingTest(unittest.TestCase):
    def test_train_text_baseline_report_uses_text_path(self) -> None:
        config = TextBaselineTrainingConfig(
            fold_count=2,
            random_seed=7,
            max_iter=1000,
            regularization_c=1.0,
            lowercase=True,
            min_df=1,
            ngram_range=(1, 2),
        )

        report = train_text_baseline_report(_synthetic_text_artifact(), config)

        self.assertEqual("tfidf_logistic_regression", report.baseline_name)
        self.assertEqual("synthetic", report.source_model_id)
        self.assertEqual(("benign", "exfiltration_intent", "secret_present_safe"), report.label_names)
        self.assertEqual(12, report.example_count)
        self.assertEqual(2, len(report.folds))

    def test_train_text_baseline_report_rejects_too_many_folds(self) -> None:
        config = TextBaselineTrainingConfig(
            fold_count=5,
            random_seed=7,
            max_iter=1000,
            regularization_c=1.0,
            lowercase=True,
            min_df=1,
            ngram_range=(1, 2),
        )

        with self.assertRaises(TextBaselineTrainingError):
            train_text_baseline_report(_synthetic_text_artifact(), config)

    def test_train_text_baseline_report_rejects_invalid_ngram_range(self) -> None:
        config = TextBaselineTrainingConfig(
            fold_count=2,
            random_seed=7,
            max_iter=1000,
            regularization_c=1.0,
            lowercase=True,
            min_df=1,
            ngram_range=(2, 1),
        )

        with self.assertRaises(TextBaselineTrainingError):
            train_text_baseline_report(_synthetic_text_artifact(), config)

    def test_write_text_baseline_report_json_creates_readable_report(self) -> None:
        config = TextBaselineTrainingConfig(
            fold_count=2,
            random_seed=7,
            max_iter=1000,
            regularization_c=1.0,
            lowercase=True,
            min_df=1,
            ngram_range=(1, 2),
        )
        report = train_text_baseline_report(_synthetic_text_artifact(), config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "text_baseline.json"
            write_text_baseline_report_json(output_path, report)
            decoded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual("tfidf_logistic_regression", decoded["baseline_name"])
        self.assertEqual("synthetic", decoded["source_model_id"])


if __name__ == "__main__":
    unittest.main()
