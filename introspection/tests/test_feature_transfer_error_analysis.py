import json
import tempfile
import unittest
from pathlib import Path

import torch
from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.feature_transfer_error_analysis import (
    StructuredPromptRecord,
    TransferErrorAnalysisConfig,
    TransferErrorDataset,
    analyze_feature_transfer_errors,
    render_feature_transfer_error_analysis_markdown,
    write_feature_transfer_error_analysis_json,
    write_feature_transfer_error_analysis_markdown,
)


def _artifact(dataset_id: str, feature: torch.Tensor) -> ActivationArtifact:
    labels = (
        "secret_present_safe",
        "secret_present_safe",
        "exfiltration_intent",
        "exfiltration_intent",
    )
    return {
        "metadata": {
            "model_id": "Qwen/Qwen3-4B",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (20,),
            "pooling_methods": ("readout_window",),
        },
        "example_ids": tuple(f"{dataset_id}_example_{index}" for index in range(4)),
        "labels": labels,
        "families": ("family-a", "family-b", "family-a", "family-b"),
        "texts": ("safe a", "safe b", "exfil a", "exfil b"),
        "tags": tuple(("test",) for _ in range(4)),
        "features": {"readout_window_layer_20": feature},
    }


def _records(dataset_id: str) -> tuple[StructuredPromptRecord, ...]:
    records: list[StructuredPromptRecord] = []
    for index in range(4):
        records.append(
            StructuredPromptRecord(
                example_id=f"{dataset_id}_example_{index}",
                rendered_prompt=(
                    "[message:system:0]\n"
                    "credential: SECRET-VALUE\n"
                    "[message:user:1]\n"
                    f"example {index} should be assessed without exposing the credential"
                ),
                label="secret_present_safe" if index < 2 else "exfiltration_intent",
                family="family-a" if index in (0, 2) else "family-b",
                secret_char_span=(30, 42),
                secret_token_span=(2, 5),
                query_token_span=(8, 16),
                payload_token_span=(18, 21),
                readout_token_indices=(18, 19, 20),
                tool_call_name="send_security_report",
                tool_argument_path="arguments.credential",
            )
        )
    return tuple(records)


def _config() -> TransferErrorAnalysisConfig:
    return TransferErrorAnalysisConfig(
        task_name="safe_secret_vs_exfiltration",
        activation_feature_key="readout_window_layer_20",
        positive_label="exfiltration_intent",
        decision_threshold=0.5,
        random_seed=42,
        max_iter=1000,
        regularization_c=10.0,
        max_error_examples=10,
    )


class FeatureTransferErrorAnalysisTest(unittest.TestCase):
    def test_analyze_feature_transfer_errors_counts_errors_and_redacts_examples(self) -> None:
        train_artifact = _artifact(
            dataset_id="train",
            feature=torch.tensor(
                (
                    (0.0, 0.0),
                    (0.1, 0.2),
                    (4.0, 4.0),
                    (4.1, 4.2),
                ),
                dtype=torch.float32,
            ),
        )
        test_artifact = _artifact(
            dataset_id="test",
            feature=torch.tensor(
                (
                    (4.1, 4.1),
                    (0.1, 0.1),
                    (4.2, 4.0),
                    (0.0, 0.1),
                ),
                dtype=torch.float32,
            ),
        )

        report = analyze_feature_transfer_errors(
            train_datasets=(TransferErrorDataset(dataset_id="train", artifact=train_artifact),),
            test_dataset=TransferErrorDataset(dataset_id="test", artifact=test_artifact),
            structured_prompt_records=_records("test"),
            config=_config(),
        )

        self.assertEqual(4, report.test_example_count)
        self.assertEqual(2, report.error_count)
        self.assertEqual(1, report.false_positive_count)
        self.assertEqual(1, report.false_negative_count)
        self.assertEqual(("exfiltration_intent", "secret_present_safe"), report.label_names)
        self.assertEqual(2, len(report.error_examples))
        for example in report.error_examples:
            self.assertIn("[message:user:1]", example.redacted_excerpt)
            self.assertNotIn("SECRET-VALUE", example.redacted_excerpt)

    def test_feature_transfer_error_analysis_writes_json_and_markdown(self) -> None:
        train_artifact = _artifact(
            dataset_id="train",
            feature=torch.tensor(
                (
                    (0.0, 0.0),
                    (0.1, 0.2),
                    (4.0, 4.0),
                    (4.1, 4.2),
                ),
                dtype=torch.float32,
            ),
        )
        test_artifact = _artifact(
            dataset_id="test",
            feature=torch.tensor(
                (
                    (4.1, 4.1),
                    (0.1, 0.1),
                    (4.2, 4.0),
                    (0.0, 0.1),
                ),
                dtype=torch.float32,
            ),
        )
        report = analyze_feature_transfer_errors(
            train_datasets=(TransferErrorDataset(dataset_id="train", artifact=train_artifact),),
            test_dataset=TransferErrorDataset(dataset_id="test", artifact=test_artifact),
            structured_prompt_records=_records("test"),
            config=_config(),
        )
        markdown = render_feature_transfer_error_analysis_markdown(report)

        self.assertIn("# Feature Transfer Error Analysis", markdown)
        self.assertIn("False positives", markdown)
        self.assertIn("[message:user:1]", markdown)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / "errors.json"
            markdown_path = root / "errors.md"
            write_feature_transfer_error_analysis_json(json_path, report)
            write_feature_transfer_error_analysis_markdown(markdown_path, report)
            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            written_markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(2, decoded["error_count"])
        self.assertEqual(1, decoded["false_positive_count"])
        self.assertIn("Feature Transfer Error Analysis", written_markdown)


if __name__ == "__main__":
    unittest.main()
