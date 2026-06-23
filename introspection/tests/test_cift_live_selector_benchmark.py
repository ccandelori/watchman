from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aegis_introspection.cift_live_selector_benchmark import (
    CiftLiveWindowSelectorBenchmarkRequestConfig,
    run_cift_live_window_selector_benchmark_with_extractor,
)

from aegis.core.contracts import NormalizedTurn


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
                            expected_window_family="payload_query_fallback",
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
                runtime_turns_path=runtime_turns_path,
                selected_choice_runtime_model_path=selected_model_path,
                fallback_runtime_model_path=fallback_model_path,
                output_json_path=output_json_path,
                output_markdown_path=output_markdown_path,
                detector_name="cift_live_window_selector_test",
                feature_source="test_live_runner",
                mock_response="ok",
                model_id="Qwen/Qwen3-test",
                revision="main",
                selected_device="cpu",
                model_load_ms=0.0,
                allow_sealed_holdout=False,
            )

            report = run_cift_live_window_selector_benchmark_with_extractor(config=config, extractor=extractor)
            decoded = json.loads(output_json_path.read_text(encoding="utf-8"))
            markdown = output_markdown_path.read_text(encoding="utf-8")

        self.assertEqual(2, report.request_count)
        self.assertEqual({"selected_choice": 1, "payload_query_fallback": 1}, report.window_family_counts)
        self.assertEqual({"selected_choice": 1, "payload_query_fallback": 1}, report.expected_window_family_counts)
        self.assertEqual({"exfiltration_intent": 2}, report.expected_label_counts)
        self.assertEqual({"active": 1, "degraded": 1}, report.capability_status_counts)
        self.assertEqual(0, report.window_family_mismatch_count)
        self.assertEqual("exfiltration_intent", report.rows[0].expected_label)
        self.assertEqual("selected_choice", report.rows[0].expected_window_family)
        self.assertEqual("selected_choice", report.rows[0].window_family)
        self.assertEqual("active", report.rows[0].capability_status)
        self.assertEqual("payload_query_fallback", report.rows[1].window_family)
        self.assertEqual("degraded", report.rows[1].capability_status)
        self.assertEqual("aegis_introspection.cift_live_window_selector_benchmark/v1", decoded["schema_version"])
        self.assertEqual({"active": 1, "degraded": 1}, decoded["capability_status_counts"])
        self.assertEqual("degraded", decoded["rows"][1]["capability_status"])
        self.assertEqual(0, decoded["window_family_mismatch_count"])
        self.assertIn("Live CIFT Window Selector Benchmark", markdown)
        self.assertIn("Window families", markdown)
        self.assertIn("Capability statuses", markdown)


class StaticFeatureExtractor:
    def __init__(self, feature_vectors: dict[tuple[str, str], tuple[float, ...]]) -> None:
        self._feature_vectors = feature_vectors

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        example_id = turn.metadata["example_id"]
        if not isinstance(example_id, str):
            raise AssertionError("metadata.example_id must be a string in test fixture.")
        return self._feature_vectors.get((example_id, feature_key))


def _runtime_model_record(feature_key: str, model_bundle_id: str) -> dict[str, object]:
    return {
        "schema_version": "aegis.cift_runtime_linear/v1",
        "model_bundle_id": model_bundle_id,
        "source_model_id": "Qwen/Qwen3-test",
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
        "candidate_status": "runtime_candidate",
        "scaler_mean": [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "logistic_coefficients": [1.0, 1.0],
        "logistic_intercept": 0.0,
        "negative_action": "allow",
        "positive_action": "warn",
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
            "revision": "main",
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
