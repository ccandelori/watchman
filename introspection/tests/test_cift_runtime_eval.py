from __future__ import annotations

# ruff: noqa: E402
import json
import tempfile
import unittest
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.cift_runtime_eval import (
    ActivationArtifactFeatureExtractor,
    CiftRuntimeEvalConfig,
    CiftRuntimeEvalError,
    CiftWindowSelectorRuntimeEvalConfig,
    run_cift_runtime_eval,
    run_cift_window_selector_runtime_eval,
)

from aegis.core.contracts import Action, CapabilityMode, Message, ModelInfo, NormalizedTurn


def _artifact() -> ActivationArtifact:
    return {
        "metadata": {
            "model_id": "Qwen/Qwen3-test",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (15,),
            "pooling_methods": ("readout_window",),
        },
        "example_ids": ("safe-1", "exfil-1"),
        "labels": ("secret_present_safe", "exfiltration_intent"),
        "families": ("family-a", "family-a"),
        "texts": ("safe text", "exfil text"),
        "tags": (("test",), ("test",)),
        "features": {
            "readout_window_layer_15": torch.tensor(
                (
                    (0.0, 0.0),
                    (2.0, 2.0),
                ),
                dtype=torch.float32,
            )
        },
    }


def _selector_artifact() -> ActivationArtifact:
    return {
        "metadata": {
            "model_id": "Qwen/Qwen3-test",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (15,),
            "pooling_methods": ("readout_window", "selected_choice_window"),
        },
        "example_ids": ("selected-exfil-1", "fallback-exfil-1"),
        "labels": ("exfiltration_intent", "exfiltration_intent"),
        "families": ("family-a", "family-a"),
        "texts": ("selected text", "fallback text"),
        "tags": (("test",), ("test",)),
        "features": {
            "selected_choice_window_layer_15": torch.tensor(
                (
                    (2.0, 2.0),
                    (-2.0, -2.0),
                ),
                dtype=torch.float32,
            ),
            "readout_window_layer_15": torch.tensor(
                (
                    (-2.0, -2.0),
                    (2.0, 2.0),
                ),
                dtype=torch.float32,
            ),
        },
    }


def _runtime_model_record() -> dict[str, object]:
    return {
        "schema_version": "aegis.cift_runtime_linear/v1",
        "model_bundle_id": "synthetic-runtime-cift",
        "source_model_id": "Qwen/Qwen3-test",
        "training_dataset_id": "synthetic-runtime-test",
        "source_artifact_sha256": "a" * 64,
        "evaluation_report_ids": ["synthetic-report"],
        "task_name": "safe_secret_vs_exfiltration",
        "feature_key": "readout_window_layer_15",
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
        "logistic_coefficients": [2.0, 2.0],
        "logistic_intercept": -3.0,
        "negative_action": "allow",
        "positive_action": "warn",
    }


def _selector_runtime_model_record(
    feature_key: str,
    model_bundle_id: str,
    logistic_coefficients: list[float],
) -> dict[str, object]:
    record = _runtime_model_record()
    record["model_bundle_id"] = model_bundle_id
    record["feature_key"] = feature_key
    record["logistic_coefficients"] = logistic_coefficients
    record["logistic_intercept"] = 0.0
    record["candidate_status"] = "runtime_candidate"
    return record


def _runtime_turn(example_id: str, turn_index: int) -> dict[str, object]:
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
                "label": "exfiltration_intent" if example_id == "exfil-1" else "secret_present_safe",
                "family": "family-a",
                "tags": ["test"],
            },
        },
    }


def _selector_runtime_turn(example_id: str, turn_index: int, cift_metadata: dict[str, object]) -> dict[str, object]:
    row = _runtime_turn(example_id=example_id, turn_index=turn_index)
    metadata = row["metadata"]
    if not isinstance(metadata, dict):
        raise AssertionError("metadata must be a dict.")
    metadata["cift"] = cift_metadata
    return row


class CiftRuntimeEvalTest(unittest.TestCase):
    def test_feature_extractor_reads_activation_rows_by_runtime_example_id(self) -> None:
        extractor = ActivationArtifactFeatureExtractor(artifact=_artifact(), feature_key="readout_window_layer_15")

        vector = extractor.extract_feature_vector(
            turn=_normalized_turn("exfil-1"),
            feature_key="readout_window_layer_15",
        )

        self.assertEqual((2.0, 2.0), vector)

    def test_feature_extractor_rejects_mismatched_feature_key(self) -> None:
        extractor = ActivationArtifactFeatureExtractor(artifact=_artifact(), feature_key="readout_window_layer_15")

        with self.assertRaisesRegex(CiftRuntimeEvalError, "initialized"):
            extractor.extract_feature_vector(
                turn=_normalized_turn("safe-1"),
                feature_key="other_feature",
            )

    def test_run_cift_runtime_eval_writes_detector_and_policy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "artifact.pt"
            runtime_model_path = root / "runtime_model.json"
            runtime_turns_path = root / "runtime_turns.jsonl"
            output_path = root / "runtime_eval.jsonl"
            torch.save(_artifact(), artifact_path)
            runtime_model_path.write_text(json.dumps(_runtime_model_record()), encoding="utf-8")
            runtime_turns_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        _runtime_turn("safe-1", 1),
                        _runtime_turn("exfil-1", 2),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            config = CiftRuntimeEvalConfig(
                runtime_turns_path=runtime_turns_path,
                activation_artifact_path=artifact_path,
                runtime_model_path=runtime_model_path,
                output_path=output_path,
                detector_name="cift_runtime_test",
                feature_source="test_activation_artifact",
                mock_response="ok",
                allow_sealed_holdout=False,
            )

            summary = run_cift_runtime_eval(config)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(2, summary.request_count)
        self.assertEqual({"allow": 1, "warn": 1}, summary.detector_action_counts)
        self.assertEqual({"allow": 1, "warn": 1}, summary.policy_action_counts)
        self.assertEqual({"active": 2}, summary.capability_status_counts)
        self.assertEqual("allow", rows[0]["detector_result"]["recommended_action"])
        self.assertEqual("warn", rows[1]["detector_result"]["recommended_action"])
        self.assertEqual(Action.WARN.value, rows[1]["policy_decision"]["final_action"])

    def test_run_cift_window_selector_runtime_eval_writes_route_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "artifact.pt"
            selected_model_path = root / "selected_model.json"
            fallback_model_path = root / "fallback_model.json"
            runtime_turns_path = root / "runtime_turns.jsonl"
            output_path = root / "runtime_eval.jsonl"
            torch.save(_selector_artifact(), artifact_path)
            selected_model_path.write_text(
                json.dumps(
                    _selector_runtime_model_record(
                        feature_key="selected_choice_window_layer_15",
                        model_bundle_id="selected-choice-model",
                        logistic_coefficients=[1.0, 1.0],
                    )
                ),
                encoding="utf-8",
            )
            fallback_model_path.write_text(
                json.dumps(
                    _selector_runtime_model_record(
                        feature_key="readout_window_layer_15",
                        model_bundle_id="fallback-model",
                        logistic_coefficients=[1.0, 1.0],
                    )
                ),
                encoding="utf-8",
            )
            runtime_turns_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        _selector_runtime_turn(
                            example_id="selected-exfil-1",
                            turn_index=1,
                            cift_metadata={"selected_choice_readout_token_indices": [8, 9]},
                        ),
                        _selector_runtime_turn(
                            example_id="fallback-exfil-1",
                            turn_index=2,
                            cift_metadata={"readout_token_indices": [3, 4]},
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            config = CiftWindowSelectorRuntimeEvalConfig(
                runtime_turns_path=runtime_turns_path,
                activation_artifact_path=artifact_path,
                selected_choice_runtime_model_path=selected_model_path,
                fallback_runtime_model_path=fallback_model_path,
                output_path=output_path,
                detector_name="cift_runtime_selector_test",
                feature_source="test_activation_artifact",
                mock_response="ok",
                allow_sealed_holdout=False,
            )

            summary = run_cift_window_selector_runtime_eval(config)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(2, summary.request_count)
        self.assertEqual({"warn": 2}, summary.detector_action_counts)
        self.assertEqual({"warn": 2}, summary.policy_action_counts)
        self.assertEqual({"active": 1, "degraded": 1}, summary.capability_status_counts)
        self.assertEqual("active", rows[0]["detector_result"]["capability_status"])
        self.assertEqual("selected_choice", rows[0]["detector_result"]["evidence"]["cift_window_family"])
        self.assertEqual("primary", rows[0]["detector_result"]["evidence"]["cift_window_coverage"])
        self.assertEqual("selected-choice-model", rows[0]["detector_result"]["evidence"]["model_bundle_id"])
        self.assertEqual(0.72, rows[0]["detector_result"]["confidence"])
        self.assertEqual("degraded", rows[1]["detector_result"]["capability_status"])
        self.assertEqual("payload_query_fallback", rows[1]["detector_result"]["evidence"]["cift_window_family"])
        self.assertEqual("degraded_fallback", rows[1]["detector_result"]["evidence"]["cift_window_coverage"])
        self.assertEqual(
            "selected_choice_metadata_required_for_primary_cift",
            rows[1]["detector_result"]["evidence"]["degradation_reason"],
        )
        self.assertEqual("fallback-model", rows[1]["detector_result"]["evidence"]["model_bundle_id"])
        self.assertEqual(0.35, rows[1]["detector_result"]["confidence"])


def _normalized_turn(example_id: str) -> NormalizedTurn:
    return NormalizedTurn(
        trace_id=f"trace-{example_id}",
        session_id="session-runtime-test",
        turn_index=1,
        capability_mode=CapabilityMode.OFFLINE_EVAL,
        model=ModelInfo(provider="huggingface", model_id="Qwen/Qwen3-test", revision="main", selected_device="cpu"),
        messages=(Message(role="user", content=f"message for {example_id}"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={"example_id": example_id},
    )


if __name__ == "__main__":
    unittest.main()
