# ruff: noqa: E402

import json
import tempfile
import unittest
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from aegis_introspection.calibrated_detector_export import (
    CalibratedDetectorExportConfig,
    export_calibrated_cift_detector_results,
)
from aegis_introspection.sealed_holdout import (
    SealedHoldoutError,
    assert_unsealed_activation_artifact_path,
    assert_unsealed_activation_artifact_tags,
    assert_unsealed_jsonl_tags,
    assert_unsealed_path,
    assert_unsealed_tag_rows,
    path_is_sealed_holdout,
    tag_rows_are_sealed_holdout,
)
from introspection.scripts.analyze_binary_errors import AnalyzeBinaryErrorsScriptConfig, run_analysis
from introspection.scripts.diagnose_cift_selector_window_scores import DiagnoseSelectorScoresConfig, run_diagnostics
from introspection.scripts.export_cift_detector_results import ExportCiftDetectorResultsConfig, run_export
from introspection.scripts.export_runtime_turns import ExportRuntimeTurnsConfig
from introspection.scripts.export_runtime_turns import run_export as run_runtime_export
from introspection.scripts.export_trained_cift_detector_results import (
    ExportTrainedCiftDetectorResultsCliConfig,
)
from introspection.scripts.export_trained_cift_detector_results import (
    run_export as run_trained_export,
)
from introspection.scripts.summarize_cift_operating_points import SummarizeCiftOperatingPointsCliConfig, run_summary
from introspection.scripts.summarize_policy_window_errors import (
    SummarizePolicyWindowErrorsConfig,
)
from introspection.scripts.summarize_policy_window_errors import (
    run_summary as run_policy_window_summary,
)


def _activation_artifact(tags: tuple[tuple[str, ...], ...]) -> dict[str, object]:
    return {
        "metadata": {
            "model_id": "Qwen/Qwen3-0.6B",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (15,),
            "pooling_methods": ("readout_window",),
        },
        "example_ids": ("row-1",),
        "labels": ("exfiltration_intent",),
        "families": ("family-1",),
        "texts": ("text",),
        "tags": tags,
        "features": {"readout_window_layer_15": torch.zeros((1, 1))},
    }


class SealedHoldoutTest(unittest.TestCase):
    def test_path_is_sealed_holdout_detects_sealed_filename(self) -> None:
        self.assertTrue(path_is_sealed_holdout(Path("prompts_dp_honey_lite_v4_3_sealed.jsonl")))
        self.assertFalse(path_is_sealed_holdout(Path("prompts_dp_honey_lite_v4_1.jsonl")))
        self.assertFalse(path_is_sealed_holdout(Path("unsealed_report.jsonl")))

    def test_tag_rows_are_sealed_holdout_detects_row_tag(self) -> None:
        self.assertTrue(tag_rows_are_sealed_holdout((("dp_honey_lite", "sealed_holdout"),)))
        self.assertFalse(tag_rows_are_sealed_holdout((("dp_honey_lite", "hard_v4_1"),)))

    def test_assert_unsealed_path_rejects_without_override(self) -> None:
        with self.assertRaises(SealedHoldoutError):
            assert_unsealed_path(
                path=Path("qwen3_0_6b_dp_honey_lite_v4_3_sealed_selector_windows.pt"),
                allow_sealed_holdout=False,
                context="activation extraction",
            )

    def test_assert_unsealed_path_allows_with_override(self) -> None:
        assert_unsealed_path(
            path=Path("qwen3_0_6b_dp_honey_lite_v4_3_sealed_selector_windows.pt"),
            allow_sealed_holdout=True,
            context="activation extraction",
        )

    def test_assert_unsealed_tag_rows_rejects_without_override(self) -> None:
        with self.assertRaises(SealedHoldoutError):
            assert_unsealed_tag_rows(
                tag_rows=(("dp_honey_lite", "sealed_holdout"),),
                allow_sealed_holdout=False,
                context="model training",
            )

    def test_assert_unsealed_jsonl_tags_rejects_sealed_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text('{"id": "row-1", "tags": ["dp_honey_lite", "sealed_holdout"]}\n', encoding="utf-8")

            with self.assertRaises(SealedHoldoutError):
                assert_unsealed_jsonl_tags(
                    path=path,
                    allow_sealed_holdout=False,
                    context="detector export",
                )

    def test_assert_unsealed_jsonl_tags_rejects_runtime_turn_eval_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime_turns.jsonl"
            path.write_text(
                json.dumps({"metadata": {"example_id": "row-1", "eval": {"tags": ["sealed_holdout"]}}}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(SealedHoldoutError):
                assert_unsealed_jsonl_tags(
                    path=path,
                    allow_sealed_holdout=False,
                    context="detector export",
                )

    def test_assert_unsealed_jsonl_tags_allows_non_sealed_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text('{"id": "row-1", "tags": ["dp_honey_lite", "hard_v4_1"]}\n', encoding="utf-8")

            assert_unsealed_jsonl_tags(
                path=path,
                allow_sealed_holdout=False,
                context="detector export",
            )

    def test_assert_unsealed_activation_artifact_tags_rejects_sealed_tags(self) -> None:
        with self.assertRaises(SealedHoldoutError):
            assert_unsealed_activation_artifact_tags(
                artifact=_activation_artifact(tags=(("dp_honey_lite", "sealed_holdout"),)),
                allow_sealed_holdout=False,
                context="model training",
            )

    def test_assert_unsealed_activation_artifact_path_rejects_renamed_sealed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "renamed_artifact.pt"
            torch.save(_activation_artifact(tags=(("dp_honey_lite", "sealed_holdout"),)), artifact_path)

            with self.assertRaises(SealedHoldoutError):
                assert_unsealed_activation_artifact_path(
                    path=artifact_path,
                    allow_sealed_holdout=False,
                    context="model training",
                )

    def test_error_analysis_rejects_renamed_sealed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "artifact.pt"
            torch.save(_activation_artifact(tags=(("dp_honey_lite", "sealed_holdout"),)), artifact_path)

            with self.assertRaises(SealedHoldoutError):
                run_analysis(
                    AnalyzeBinaryErrorsScriptConfig(
                        artifact_path=artifact_path,
                        output_json_path=root / "errors.json",
                        output_markdown_path=root / "errors.md",
                        fold_count=2,
                        random_seed=42,
                        max_iter=100,
                        regularization_c=1.0,
                        activation_feature_key="readout_window_layer_15",
                        word_ngram_range=(1, 2),
                        char_ngram_range=(3, 5),
                        task_names=(),
                        allow_sealed_holdout=False,
                    )
                )

    def test_trained_export_rejects_renamed_sealed_runtime_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_turns_path = root / "runtime_turns.jsonl"
            runtime_turns_path.write_text(
                json.dumps({"metadata": {"example_id": "row-1", "eval": {"tags": ["sealed_holdout"]}}}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(SealedHoldoutError):
                run_trained_export(
                    ExportTrainedCiftDetectorResultsCliConfig(
                        runtime_turns_path=runtime_turns_path,
                        artifact_path=root / "artifact.pt",
                        model_bundle_path=root / "model.pkl",
                        output_path=root / "detector_results.jsonl",
                        detector_name="cift_selector_probe",
                        model_bundle_id="bundle",
                        capability_required="self_hosted_introspection",
                        positive_action="warn",
                        negative_action="allow",
                        confidence=0.5,
                        allow_sealed_holdout=False,
                    )
                )

    def test_legacy_export_rejects_renamed_sealed_runtime_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_turns_path = root / "runtime_turns.jsonl"
            runtime_turns_path.write_text(
                json.dumps({"metadata": {"example_id": "row-1", "eval": {"tags": ["sealed_holdout"]}}}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(SealedHoldoutError):
                run_export(
                    ExportCiftDetectorResultsConfig(
                        runtime_turns_path=runtime_turns_path,
                        error_report_path=root / "error_report.json",
                        output_path=root / "detector_results.jsonl",
                        task_name="safe_secret_vs_exfiltration",
                        method_name="activation_probe",
                        detector_name="cift_selector_probe",
                        feature_key="readout_window_layer_15",
                        probe_version="probe",
                        capability_required="self_hosted_introspection",
                        positive_label="exfiltration_intent",
                        positive_score=1.0,
                        negative_score=0.0,
                        positive_action="warn",
                        negative_action="allow",
                        confidence=0.5,
                        allow_sealed_holdout=False,
                    )
                )

    def test_calibrated_export_rejects_renamed_sealed_runtime_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_turns_path = root / "runtime_turns.jsonl"
            runtime_turns_path.write_text(
                json.dumps({"metadata": {"example_id": "row-1", "eval": {"tags": ["sealed_holdout"]}}}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(SealedHoldoutError):
                export_calibrated_cift_detector_results(
                    CalibratedDetectorExportConfig(
                        runtime_turns_path=runtime_turns_path,
                        calibration_report_path=root / "calibration.json",
                        output_path=root / "detector_results.jsonl",
                        detector_name="cift_selector_probe",
                        probe_version="probe",
                        capability_required="self_hosted_introspection",
                        positive_action="warn",
                        negative_action="allow",
                        confidence=0.5,
                        allow_sealed_holdout=False,
                    )
                )

    def test_operating_point_summary_rejects_sealed_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            with self.assertRaises(SealedHoldoutError):
                run_summary(
                    SummarizeCiftOperatingPointsCliConfig(
                        calibration_report_path=root / "v4_3_sealed_calibration.json",
                        output_json_path=root / "operating_points.json",
                        output_markdown_path=root / "operating_points.md",
                        thresholds=(0.5,),
                        allow_sealed_holdout=False,
                    )
                )

    def test_selector_score_diagnostics_rejects_renamed_sealed_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts_path = root / "prompts.jsonl"
            prompts_path.write_text(
                json.dumps({"id": "row-1", "tags": ["sealed_holdout"]}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(SealedHoldoutError):
                run_diagnostics(
                    DiagnoseSelectorScoresConfig(
                        prompts_path=prompts_path,
                        calibration_report_path=root / "calibration.json",
                        output_json_path=root / "diagnostics.json",
                        output_markdown_path=root / "diagnostics.md",
                        dimensions=("family",),
                        thresholds=(0.5,),
                        near_threshold_radius=0.1,
                        allow_sealed_holdout=False,
                    )
                )

    def test_policy_window_summary_rejects_renamed_sealed_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts_path = root / "prompts.jsonl"
            prompts_path.write_text(
                json.dumps({"id": "row-1", "tags": ["sealed_holdout"]}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(SealedHoldoutError):
                run_policy_window_summary(
                    SummarizePolicyWindowErrorsConfig(
                        prompts_path=prompts_path,
                        error_report_path=root / "error_report.json",
                        output_json_path=root / "slices.json",
                        output_markdown_path=root / "slices.md",
                        task_name="safe_secret_vs_exfiltration",
                        method_name="activation_probe",
                        dimensions=("family",),
                        allow_sealed_holdout=False,
                    )
                )

    def test_runtime_turn_export_rejects_renamed_sealed_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts_path = root / "prompts.jsonl"
            prompts_path.write_text(
                json.dumps({"id": "row-1", "tags": ["sealed_holdout"]}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(SealedHoldoutError):
                run_runtime_export(
                    ExportRuntimeTurnsConfig(
                        input_path=prompts_path,
                        output_path=root / "runtime_turns.jsonl",
                        capability_mode="offline_eval",
                        model_provider="huggingface",
                        model_id="Qwen/Qwen3-0.6B",
                        revision="main",
                        selected_device="cpu",
                        sensitive_source="dp_honey_lite",
                        session_id="session",
                        allow_sealed_holdout=False,
                    )
                )


if __name__ == "__main__":
    unittest.main()
