from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from aegis_introspection.activations import HiddenStateForwardPass
from aegis_introspection.cift_live_runtime_eval import (
    CiftLiveWindowSelectorRequestEvalConfig,
    run_cift_live_window_selector_runtime_eval_with_runner,
)


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


def _runtime_turn(example_id: str, turn_index: int, cift_metadata: dict[str, object]) -> dict[str, object]:
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
                "label": "exfiltration_intent",
                "family": "family-a",
                "tags": ["test"],
            },
            "cift": cift_metadata,
        },
    }


class CiftLiveRuntimeEvalTest(unittest.TestCase):
    def test_live_window_selector_eval_routes_selected_and_fallback_rows_with_one_forward_each(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_model_path = root / "selected_model.json"
            fallback_model_path = root / "fallback_model.json"
            runtime_turns_path = root / "runtime_turns.jsonl"
            output_path = root / "runtime_eval.jsonl"
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
                            cift_metadata={
                                "readout_token_indices": [1],
                                "selected_choice_readout_token_indices": [2],
                            },
                        ),
                        _runtime_turn(
                            example_id="fallback-exfil-1",
                            turn_index=2,
                            cift_metadata={"readout_token_indices": [1]},
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            runner = RecordingRunner(
                forward_pass=_forward_pass(
                    hidden_states=(
                        _hidden_state(((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))),
                        _hidden_state(((0.0, 0.0), (2.0, 2.0), (2.0, 2.0))),
                    )
                )
            )
            config = CiftLiveWindowSelectorRequestEvalConfig(
                runtime_turns_path=runtime_turns_path,
                selected_choice_runtime_model_path=selected_model_path,
                fallback_runtime_model_path=fallback_model_path,
                output_path=output_path,
                detector_name="cift_live_window_selector_test",
                feature_source="test_live_runner",
                mock_response="ok",
                allow_sealed_holdout=False,
            )

            summary = run_cift_live_window_selector_runtime_eval_with_runner(config=config, runner=runner)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(2, summary.request_count)
        self.assertEqual({"warn": 2}, summary.detector_action_counts)
        self.assertEqual(("message for selected-exfil-1", "message for fallback-exfil-1"), tuple(runner.prompts))
        self.assertEqual("selected_choice", rows[0]["detector_result"]["evidence"]["cift_window_family"])
        self.assertEqual("payload_query_fallback", rows[1]["detector_result"]["evidence"]["cift_window_family"])


class RecordingRunner:
    def __init__(self, forward_pass: HiddenStateForwardPass) -> None:
        self.prompts: list[str] = []
        self._forward_pass = forward_pass

    def run(self, prompt: str) -> HiddenStateForwardPass:
        self.prompts.append(prompt)
        return self._forward_pass


def _forward_pass(hidden_states: tuple[torch.Tensor, ...]) -> HiddenStateForwardPass:
    return HiddenStateForwardPass(
        prompt="unused",
        input_ids=torch.tensor(((1, 2, 3),), dtype=torch.int64),
        attention_mask=torch.tensor(((1, 1, 1),), dtype=torch.int64),
        hidden_states=hidden_states,
    )


def _hidden_state(values: tuple[tuple[float, float], ...]) -> torch.Tensor:
    return torch.tensor((values,), dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
