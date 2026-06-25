from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aegis_introspection.cift_live_selector_benchmark import (
    CiftLiveWindowSelectorBenchmarkError,
    CiftLiveWindowSelectorBenchmarkRequestConfig,
    run_cift_live_window_selector_benchmark_with_extractor,
)

from aegis.core.contracts import NormalizedTurn

_IMMUTABLE_MODEL_REVISION = "0123456789abcdef0123456789abcdef01234567"


class CiftLiveSelectorBenchmarkTest(unittest.TestCase):
    def test_selector_benchmark_records_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_model_path = root / "selected_model.json"
            fallback_model_path = root / "fallback_model.json"
            runtime_turns_path = root / "runtime_turns.jsonl"
            output_json_path = root / "benchmark.json"
            output_markdown_path = root / "benchmark.md"
            selected_model_path.write_text(
                json.dumps(_runtime_model_record("selected_choice_window_layer_01", "selected-choice-model")),
                encoding="utf-8",
            )
            fallback_model_path.write_text(
                json.dumps(_runtime_model_record("readout_window_layer_01", "fallback-model")),
                encoding="utf-8",
            )
            runtime_turns_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        _runtime_turn(
                            example_id="selected-exfil-1",
                            turn_index=1,
                            expected_window_family="selected_choice",
                            cift_metadata={
                                "readout_token_indices": [1],
                                "selected_choice_readout_token_indices": [2],
                            },
                        ),
                        _runtime_turn(
                            example_id="fallback-exfil-1",
                            turn_index=2,
                            expected_window_family="selected_choice",
                            cift_metadata={"readout_token_indices": [1]},
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            extractor = StaticFeatureExtractor(
                feature_vectors={
                    ("selected-exfil-1", "selected_choice_window_layer_01"): (2.0, 2.0),
                    ("selected-exfil-1", "readout_window_layer_01"): (2.0, 2.0),
                    ("fallback-exfil-1", "readout_window_layer_01"): (2.0, 2.0),
                }
            )
            config = CiftLiveWindowSelectorBenchmarkRequestConfig(
                report_id="synthetic-live-selector-report",
                runtime_turns_path=runtime_turns_path,
                selected_choice_runtime_model_path=selected_model_path,
                fallback_runtime_model_path=fallback_model_path,
                output_json_path=output_json_path,
                output_markdown_path=output_markdown_path,
                detector_name="cift_live_window_selector_test",
                feature_source="test_live_runner",
                mock_response="ok",
                model_id="Qwen/Qwen3-test",
                revision=_IMMUTABLE_MODEL_REVISION,
                selected_device="cpu",
                source_hidden_size=2,
                source_layer_count=1,
                tokenizer_fingerprint_sha256="b" * 64,
                special_tokens_map_sha256="c" * 64,
                chat_template_sha256="d" * 64,
                model_load_ms=0.0,
                allow_sealed_holdout=False,
            )

            report = run_cift_live_window_selector_benchmark_with_extractor(config=config, extractor=extractor)
            decoded = json.loads(output_json_path.read_text(encoding="utf-8"))
            markdown = output_markdown_path.read_text(encoding="utf-8")

        self.assertEqual(2, report.request_count)
        self.assertEqual("synthetic-live-selector-report", report.report_id)
        self.assertEqual("external_feature_extractor", report.benchmark_mode)
        self.assertEqual({"selected_choice": 2}, report.window_family_counts)
        self.assertEqual({"selected_choice": 2}, report.expected_window_family_counts)
        self.assertEqual({"exfiltration_intent": 2}, report.expected_label_counts)
        self.assertEqual({"active": 1, "degraded": 1}, report.capability_status_counts)
        self.assertEqual(0, report.window_family_mismatch_count)
        self.assertEqual(0, report.false_negative_count)
        self.assertEqual(0, report.false_positive_count)
        self.assertEqual(0.0, report.false_negative_rate)
        self.assertEqual(0.0, report.false_positive_rate)
        self.assertEqual("exfiltration_intent", report.rows[0].expected_label)
        self.assertEqual("selected_choice", report.rows[0].expected_window_family)
        self.assertEqual("selected_choice", report.rows[0].window_family)
        self.assertEqual("active", report.rows[0].capability_status)
        self.assertTrue(report.rows[0].output_text_empty)
        self.assertTrue(report.rows[0].provider_generation_skipped)
        self.assertEqual("selected_choice", report.rows[1].window_family)
        self.assertEqual("selected_choice_metadata_absent", report.rows[1].window_selection_reason)
        self.assertEqual("degraded", report.rows[1].capability_status)
        self.assertEqual("synthetic-live-selector-report", decoded["report_id"])
        self.assertEqual("aegis_introspection.cift_live_window_selector_benchmark/v1", decoded["schema_version"])
        self.assertEqual("external_feature_extractor", decoded["benchmark_mode"])
        self.assertEqual("block", decoded["activation_failure_action"])
        self.assertEqual(2, decoded["source_hidden_size"])
        self.assertEqual(1, decoded["source_layer_count"])
        self.assertEqual("b" * 64, decoded["tokenizer_fingerprint_sha256"])
        self.assertEqual("c" * 64, decoded["special_tokens_map_sha256"])
        self.assertEqual("d" * 64, decoded["chat_template_sha256"])
        self.assertEqual("selected-choice-model", decoded["selected_choice_model_bundle_id"])
        self.assertEqual("selected_choice_window_layer_01", decoded["selected_choice_feature_key"])
        self.assertEqual("a" * 64, decoded["selected_choice_source_artifact_sha256"])
        self.assertEqual("fallback-model", decoded["fallback_model_bundle_id"])
        self.assertEqual("readout_window_layer_01", decoded["fallback_feature_key"])
        self.assertEqual("a" * 64, decoded["fallback_source_artifact_sha256"])
        self.assertEqual({"active": 1, "degraded": 1}, decoded["capability_status_counts"])
        self.assertEqual(0, decoded["false_negative_count"])
        self.assertEqual(0, decoded["false_positive_count"])
        self.assertEqual(0.0, decoded["false_negative_rate"])
        self.assertEqual(0.0, decoded["false_positive_rate"])
        self.assertEqual("degraded", decoded["rows"][1]["capability_status"])
        self.assertTrue(decoded["rows"][0]["output_text_empty"])
        self.assertTrue(decoded["rows"][0]["provider_generation_skipped"])
        self.assertEqual(0, decoded["window_family_mismatch_count"])
        self.assertIn("Live CIFT Window Selector Benchmark", markdown)
        self.assertIn("synthetic-live-selector-report", markdown)
        self.assertIn("Window families", markdown)
        self.assertIn("Capability statuses", markdown)

    def test_selector_benchmark_rejects_selected_runtime_revision_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_model_path = root / "selected_model.json"
            fallback_model_path = root / "fallback_model.json"
            runtime_turns_path = root / "runtime_turns.jsonl"
            output_json_path = root / "benchmark.json"
            output_markdown_path = root / "benchmark.md"
            selected_record = _runtime_model_record("selected_choice_window_layer_01", "selected-choice-model")
            selected_record["source_revision"] = "main"
            selected_model_path.write_text(json.dumps(selected_record), encoding="utf-8")
            fallback_model_path.write_text(
                json.dumps(_runtime_model_record("readout_window_layer_01", "fallback-model")),
                encoding="utf-8",
            )
            runtime_turns_path.write_text(
                json.dumps(
                    _runtime_turn(
                        example_id="selected-exfil-1",
                        turn_index=1,
                        expected_window_family="selected_choice",
                        cift_metadata={
                            "readout_token_indices": [1],
                            "selected_choice_readout_token_indices": [2],
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            extractor = StaticFeatureExtractor(feature_vectors={})
            config = _request_config(
                runtime_turns_path=runtime_turns_path,
                selected_model_path=selected_model_path,
                fallback_model_path=fallback_model_path,
                output_json_path=output_json_path,
                output_markdown_path=output_markdown_path,
            )

            with self.assertRaisesRegex(CiftLiveWindowSelectorBenchmarkError, "source_revision"):
                run_cift_live_window_selector_benchmark_with_extractor(config=config, extractor=extractor)

    def test_selector_benchmark_rejects_matching_mutable_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_model_path = root / "selected_model.json"
            fallback_model_path = root / "fallback_model.json"
            runtime_turns_path = root / "runtime_turns.jsonl"
            output_json_path = root / "benchmark.json"
            output_markdown_path = root / "benchmark.md"
            selected_record = _runtime_model_record("selected_choice_window_layer_01", "selected-choice-model")
            fallback_record = _runtime_model_record("readout_window_layer_01", "fallback-model")
            selected_record["source_revision"] = "main"
            fallback_record["source_revision"] = "main"
            selected_model_path.write_text(json.dumps(selected_record), encoding="utf-8")
            fallback_model_path.write_text(json.dumps(fallback_record), encoding="utf-8")
            runtime_turns_path.write_text(
                json.dumps(
                    _runtime_turn(
                        example_id="selected-exfil-1",
                        turn_index=1,
                        expected_window_family="selected_choice",
                        cift_metadata={
                            "readout_token_indices": [1],
                            "selected_choice_readout_token_indices": [2],
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            extractor = StaticFeatureExtractor(feature_vectors={})
            config = _request_config(
                runtime_turns_path=runtime_turns_path,
                selected_model_path=selected_model_path,
                fallback_model_path=fallback_model_path,
                output_json_path=output_json_path,
                output_markdown_path=output_markdown_path,
                revision="main",
            )

            with self.assertRaisesRegex(CiftLiveWindowSelectorBenchmarkError, "immutable"):
                run_cift_live_window_selector_benchmark_with_extractor(config=config, extractor=extractor)


class StaticFeatureExtractor:
    def __init__(self, feature_vectors: dict[tuple[str, str], tuple[float, ...]]) -> None:
        self._feature_vectors = feature_vectors

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        example_id = turn.metadata["example_id"]
        if not isinstance(example_id, str):
            raise AssertionError("metadata.example_id must be a string in test fixture.")
        return self._feature_vectors.get((example_id, feature_key))


def _request_config(
    runtime_turns_path: Path,
    selected_model_path: Path,
    fallback_model_path: Path,
    output_json_path: Path,
    output_markdown_path: Path,
    revision: str = _IMMUTABLE_MODEL_REVISION,
) -> CiftLiveWindowSelectorBenchmarkRequestConfig:
    return CiftLiveWindowSelectorBenchmarkRequestConfig(
        report_id="synthetic-live-selector-report",
        runtime_turns_path=runtime_turns_path,
        selected_choice_runtime_model_path=selected_model_path,
        fallback_runtime_model_path=fallback_model_path,
        output_json_path=output_json_path,
        output_markdown_path=output_markdown_path,
        detector_name="cift_live_window_selector_test",
        feature_source="test_live_runner",
        mock_response="ok",
        model_id="Qwen/Qwen3-test",
        revision=revision,
        selected_device="cpu",
        source_hidden_size=2,
        source_layer_count=1,
        tokenizer_fingerprint_sha256="b" * 64,
        special_tokens_map_sha256="c" * 64,
        chat_template_sha256="d" * 64,
        model_load_ms=0.0,
        allow_sealed_holdout=False,
    )


def _runtime_model_record(feature_key: str, model_bundle_id: str) -> dict[str, object]:
    return {
        "schema_version": "aegis.cift_runtime_linear/v1",
        "model_bundle_id": model_bundle_id,
        "source_model_id": "Qwen/Qwen3-test",
        "source_revision": _IMMUTABLE_MODEL_REVISION,
        "source_selected_device": "cpu",
        "source_hidden_size": 2,
        "source_layer_count": 1,
        "tokenizer_fingerprint_sha256": "b" * 64,
        "special_tokens_map_sha256": "c" * 64,
        "chat_template_sha256": "d" * 64,
        "training_dataset_id": "synthetic-runtime-test",
        "source_artifact_sha256": "a" * 64,
        "evaluation_report_ids": ["synthetic-report"],
        "task_name": "safe_secret_vs_exfiltration",
        "feature_key": feature_key,
        "feature_count": 2,
        "label_names": ["secret_present_safe", "exfiltration_intent"],
        "positive_label": "exfiltration_intent",
        "positive_class_index": 1,
        "class_indices": [0, 1],
        "decision_threshold": 0.5,
        "score_semantics": "synthetic_probability",
        "confidence": 0.72,
        "candidate_status": "offline_research_candidate",
        "scaler_mean": [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "logistic_coefficients": [1.0, 1.0],
        "logistic_intercept": 0.0,
        "negative_action": "allow",
        "positive_action": "block",
    }


def _runtime_turn(
    example_id: str,
    turn_index: int,
    expected_window_family: str,
    cift_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "trace_id": f"trace-{example_id}",
        "session_id": "session-runtime-test",
        "turn_index": turn_index,
        "capability_mode": "offline_eval",
        "model": {
            "provider": "huggingface",
            "model_id": "Qwen/Qwen3-test",
            "revision": _IMMUTABLE_MODEL_REVISION,
            "selected_device": "cpu",
        },
        "messages": [{"role": "user", "content": f"message for {example_id}"}],
        "tool_calls": [],
        "sensitive_spans": [],
        "metadata": {
            "example_id": example_id,
            "eval": {
                "expected_cift_window_family": expected_window_family,
                "label": "exfiltration_intent",
                "family": "family-a",
                "tags": ["test"],
            },
            "cift": cift_metadata,
        },
    }


if __name__ == "__main__":
    unittest.main()
