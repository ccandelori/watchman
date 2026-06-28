from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from aegis_introspection.cift_sealed_holdout_metric import (
    CiftSealedHoldoutMetricConfig,
    CiftSealedHoldoutMetricError,
    materialize_cift_sealed_holdout_metric,
)

from aegis.core.contracts import Action
from aegis.detectors.cift_runtime import CiftRuntimeLinearModel, cift_runtime_model_to_dict


class CiftSealedHoldoutMetricTest(unittest.TestCase):
    def test_materializes_sealed_selected_choice_metric_from_live_runtime_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_model_path = _write_runtime_model(root)
            runtime_turns_path = _write_runtime_turns(
                root,
                sealed=True,
                selected_choice=True,
                expected_window_family=None,
            )
            runtime_report_path = _write_runtime_report(root, benchmark_mode="live_hidden_state_runner")
            output_path = root / "sealed_metric.json"

            record = materialize_cift_sealed_holdout_metric(
                _config(
                    runtime_report_path=runtime_report_path,
                    runtime_turns_path=runtime_turns_path,
                    runtime_model_path=runtime_model_path,
                    output_path=output_path,
                    allow_sealed_holdout=True,
                )
            )
            self.assertTrue(output_path.exists())

        self.assertEqual("aegis_introspection.cift_sealed_holdout_metric/v1", record["schema_version"])
        self.assertEqual("synthetic-sealed-report", record["report_id"])
        self.assertEqual("synthetic-cift-lab/sealed-selected-choice", record["sealed_holdout_split_id"])
        self.assertEqual(1.0, record["metric_value"])
        self.assertEqual(0, record["false_negative_count"])
        self.assertEqual(0, record["false_positive_count"])
        self.assertEqual("cpu", record["source_selected_device"])
        self.assertEqual("synthetic-runtime-cift", record["selected_choice_model_bundle_id"])
        self.assertEqual("selected_choice", record["window_family"])

    def test_materializes_sealed_freeform_query_tail_metric_from_live_runtime_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_model_path = _write_runtime_model(
                root,
                feature_key="query_tail_window_layer_15",
                task_name="non_exfiltration_vs_exfiltration",
            )
            runtime_turns_path = _write_runtime_turns(
                root,
                sealed=True,
                selected_choice=False,
                expected_window_family=None,
            )
            runtime_report_path = _write_runtime_report(
                root,
                benchmark_mode="live_hidden_state_runner",
                feature_key="query_tail_window_layer_15",
                window_family="freeform_query_tail",
                task_name="non_exfiltration_vs_exfiltration",
            )
            output_path = root / "sealed_metric.json"

            record = materialize_cift_sealed_holdout_metric(
                _config(
                    runtime_report_path=runtime_report_path,
                    runtime_turns_path=runtime_turns_path,
                    runtime_model_path=runtime_model_path,
                    output_path=output_path,
                    allow_sealed_holdout=True,
                )
            )
            self.assertTrue(output_path.exists())

        self.assertEqual("aegis_introspection.cift_sealed_holdout_metric/v1", record["schema_version"])
        self.assertEqual("freeform_query_tail", record["window_family"])
        self.assertEqual("synthetic-runtime-cift", record["fallback_model_bundle_id"])
        self.assertEqual(1.0, record["metric_value"])
        self.assertEqual(0, record["false_negative_count"])
        self.assertEqual(0, record["false_positive_count"])
        self.assertEqual({"benign": 1, "exfiltration_intent": 1}, record["expected_label_counts"])

    def test_materializes_sealed_freeform_final_token_metric_from_live_runtime_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_model_path = _write_runtime_model(
                root,
                feature_key="final_token_layer_12",
                task_name="non_exfiltration_vs_exfiltration",
            )
            runtime_turns_path = _write_runtime_turns(
                root,
                sealed=True,
                selected_choice=False,
                expected_window_family="freeform_final_token",
            )
            runtime_report_path = _write_runtime_report(
                root,
                benchmark_mode="live_hidden_state_runner",
                feature_key="final_token_layer_12",
                window_family="freeform_final_token",
                task_name="non_exfiltration_vs_exfiltration",
            )
            output_path = root / "sealed_metric.json"

            record = materialize_cift_sealed_holdout_metric(
                _config(
                    runtime_report_path=runtime_report_path,
                    runtime_turns_path=runtime_turns_path,
                    runtime_model_path=runtime_model_path,
                    output_path=output_path,
                    allow_sealed_holdout=True,
                )
            )
            self.assertTrue(output_path.exists())

        self.assertEqual("aegis_introspection.cift_sealed_holdout_metric/v1", record["schema_version"])
        self.assertEqual("freeform_final_token", record["window_family"])
        self.assertEqual("synthetic-runtime-cift", record["fallback_model_bundle_id"])
        self.assertEqual(1.0, record["metric_value"])
        self.assertEqual(0, record["false_negative_count"])
        self.assertEqual(0, record["false_positive_count"])

    def test_rejects_without_explicit_unseal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_model_path = _write_runtime_model(root)
            runtime_turns_path = _write_runtime_turns(
                root,
                sealed=True,
                selected_choice=True,
                expected_window_family=None,
            )
            runtime_report_path = _write_runtime_report(root, benchmark_mode="live_hidden_state_runner")
            output_path = root / "sealed_metric.json"

            with self.assertRaisesRegex(CiftSealedHoldoutMetricError, "--allow-sealed-holdout"):
                materialize_cift_sealed_holdout_metric(
                    _config(
                        runtime_report_path=runtime_report_path,
                        runtime_turns_path=runtime_turns_path,
                        runtime_model_path=runtime_model_path,
                        output_path=output_path,
                        allow_sealed_holdout=False,
                    )
                )

        self.assertFalse(output_path.exists())

    def test_rejects_unsealed_runtime_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_model_path = _write_runtime_model(root)
            runtime_turns_path = _write_runtime_turns(
                root,
                sealed=False,
                selected_choice=True,
                expected_window_family=None,
            )
            runtime_report_path = _write_runtime_report(root, benchmark_mode="live_hidden_state_runner")
            output_path = root / "sealed_metric.json"

            with self.assertRaisesRegex(CiftSealedHoldoutMetricError, "sealed_holdout"):
                materialize_cift_sealed_holdout_metric(
                    _config(
                        runtime_report_path=runtime_report_path,
                        runtime_turns_path=runtime_turns_path,
                        runtime_model_path=runtime_model_path,
                        output_path=output_path,
                        allow_sealed_holdout=True,
                    )
                )

    def test_rejects_runtime_turns_without_selected_choice_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_model_path = _write_runtime_model(root)
            runtime_turns_path = _write_runtime_turns(
                root,
                sealed=True,
                selected_choice=False,
                expected_window_family=None,
            )
            runtime_report_path = _write_runtime_report(root, benchmark_mode="live_hidden_state_runner")
            output_path = root / "sealed_metric.json"

            with self.assertRaisesRegex(CiftSealedHoldoutMetricError, "selected_choice"):
                materialize_cift_sealed_holdout_metric(
                    _config(
                        runtime_report_path=runtime_report_path,
                        runtime_turns_path=runtime_turns_path,
                        runtime_model_path=runtime_model_path,
                        output_path=output_path,
                        allow_sealed_holdout=True,
                    )
                )

    def test_rejects_replay_runtime_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_model_path = _write_runtime_model(root)
            runtime_turns_path = _write_runtime_turns(
                root,
                sealed=True,
                selected_choice=True,
                expected_window_family=None,
            )
            runtime_report_path = _write_runtime_report(root, benchmark_mode="offline_replay")
            output_path = root / "sealed_metric.json"

            with self.assertRaisesRegex(CiftSealedHoldoutMetricError, "live_hidden_state_runner"):
                materialize_cift_sealed_holdout_metric(
                    _config(
                        runtime_report_path=runtime_report_path,
                        runtime_turns_path=runtime_turns_path,
                        runtime_model_path=runtime_model_path,
                        output_path=output_path,
                        allow_sealed_holdout=True,
                    )
                )


def _config(
    runtime_report_path: Path,
    runtime_turns_path: Path,
    runtime_model_path: Path,
    output_path: Path,
    allow_sealed_holdout: bool,
) -> CiftSealedHoldoutMetricConfig:
    return CiftSealedHoldoutMetricConfig(
        runtime_report_path=runtime_report_path,
        runtime_turns_path=runtime_turns_path,
        selected_choice_runtime_model_path=runtime_model_path,
        output_path=output_path,
        report_id="synthetic-sealed-report",
        sealed_holdout_split_id="synthetic-cift-lab/sealed-selected-choice",
        metric_name="sealed_holdout_macro_f1",
        created_at="2026-06-24T00:00:00Z",
        allow_sealed_holdout=allow_sealed_holdout,
    )


def _write_runtime_model(
    root: Path,
    feature_key: str = "selected_choice_window_layer_15",
    task_name: str = "safe_secret_vs_exfiltration",
) -> Path:
    path = root / "runtime_model.json"
    model = CiftRuntimeLinearModel(
        schema_version="aegis.cift_runtime_linear/v1",
        model_bundle_id="synthetic-runtime-cift",
        source_model_id="Qwen/Qwen3-test",
        source_revision="main",
        source_selected_device="cpu",
        source_hidden_size=4096,
        source_layer_count=36,
        tokenizer_fingerprint_sha256="b" * 64,
        special_tokens_map_sha256="c" * 64,
        chat_template_sha256="d" * 64,
        training_dataset_id="synthetic-cift-lab",
        source_artifact_sha256="a" * 64,
        evaluation_report_ids=("synthetic-report",),
        task_name=task_name,
        feature_key=feature_key,
        feature_count=2,
        label_names=("secret_present_safe", "exfiltration_intent"),
        positive_label="exfiltration_intent",
        positive_class_index=1,
        class_indices=(0, 1),
        decision_threshold=0.5,
        score_semantics="full_train_classifier_probability",
        confidence=0.9,
        candidate_status="offline_research_candidate",
        scaler_mean=(0.0, 0.0),
        scaler_scale=(1.0, 1.0),
        logistic_coefficients=(1.0, 1.0),
        logistic_intercept=-1.0,
        negative_action=Action.ALLOW,
        positive_action=Action.BLOCK,
    )
    path.write_text(json.dumps(cift_runtime_model_to_dict(model), sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_runtime_turns(
    root: Path,
    sealed: bool,
    selected_choice: bool,
    expected_window_family: str | None,
) -> Path:
    path = root / "runtime_turns_sealed_selected_choice.jsonl"
    turns = (
        _runtime_turn(
            example_id="safe-001",
            label="secret_present_safe",
            sealed=sealed,
            selected_choice=selected_choice,
            expected_window_family=expected_window_family,
        ),
        _runtime_turn(
            example_id="exfil-001",
            label="exfiltration_intent",
            sealed=sealed,
            selected_choice=selected_choice,
            expected_window_family=expected_window_family,
        ),
    )
    path.write_text("".join(json.dumps(turn, sort_keys=True) + "\n" for turn in turns), encoding="utf-8")
    return path


def _runtime_turn(
    example_id: str,
    label: str,
    sealed: bool,
    selected_choice: bool,
    expected_window_family: str | None,
) -> dict[str, object]:
    tags = ["synthetic"]
    if sealed:
        tags.append("sealed_holdout")
    resolved_window_family = expected_window_family
    if resolved_window_family is None:
        resolved_window_family = "selected_choice" if selected_choice else "freeform_query_tail"
    cift_metadata: dict[str, object] = {
        "readout_token_indices": [10, 11],
        "secret_token_span": [1, 3],
        "query_token_span": [4, 9],
    }
    if selected_choice:
        cift_metadata["selected_choice_char_span"] = [20, 25]
        cift_metadata["selected_choice_token_span"] = [6, 9]
        cift_metadata["selected_choice_readout_token_indices"] = [6, 7, 8]
    else:
        cift_metadata["query_tail_readout_token_indices"] = [6, 7, 8]
    return {
        "trace_id": f"trace-{example_id}",
        "turn_index": 1,
        "messages": [{"role": "user", "content": "synthetic"}],
        "metadata": {
            "example_id": example_id,
            "eval": {
                "label": label,
                "family": "synthetic-family",
                "tags": tags,
                "expected_cift_window_family": resolved_window_family,
            },
            "cift": cift_metadata,
        },
    }


def _write_runtime_report(
    root: Path,
    benchmark_mode: str,
    feature_key: str = "selected_choice_window_layer_15",
    window_family: str = "selected_choice",
    task_name: str = "safe_secret_vs_exfiltration",
) -> Path:
    path = root / "runtime_report.json"
    route_fields = (
        {
            "selected_choice_model_bundle_id": "synthetic-runtime-cift",
            "selected_choice_feature_key": feature_key,
            "selected_choice_source_artifact_sha256": "a" * 64,
        }
        if window_family == "selected_choice"
        else {
            "fallback_model_bundle_id": "synthetic-runtime-cift",
            "fallback_feature_key": feature_key,
            "fallback_source_artifact_sha256": "a" * 64,
        }
    )
    report = {
        "report_id": "synthetic-runtime-report",
        "schema_version": "aegis_introspection.cift_live_window_selector_benchmark/v1",
        "benchmark_mode": benchmark_mode,
        "activation_failure_action": "block",
        "model_id": "Qwen/Qwen3-test",
        "revision": "main",
        "selected_device": "cpu",
        "source_hidden_size": 4096,
        "source_layer_count": 36,
        "tokenizer_fingerprint_sha256": "b" * 64,
        "special_tokens_map_sha256": "c" * 64,
        "chat_template_sha256": "d" * 64,
        **route_fields,
        "request_count": 2,
        "task_name": task_name,
        "window_family_mismatch_count": 0,
        "false_negative_count": 0,
        "false_positive_count": 0,
        "false_negative_rate": 0.0,
        "false_positive_rate": 0.0,
        "rows": [
            {
                "example_id": "safe-001",
                "expected_label": "secret_present_safe" if window_family == "selected_choice" else "benign",
                "expected_window_family": window_family,
                "window_family": window_family,
                "detector_action": "allow",
                **_receipt_fields(window_family),
            },
            {
                "example_id": "exfil-001",
                "expected_label": "exfiltration_intent",
                "expected_window_family": window_family,
                "window_family": window_family,
                "detector_action": "block",
                **_receipt_fields(window_family),
            },
        ],
    }
    path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _receipt_fields(window_family: str) -> dict[str, object]:
    token_indices = [6, 7, 8]
    fields: dict[str, object] = {
        "extractor_extraction_receipt_schema_version": "aegis.cift_extraction_receipt/v1",
        "extractor_feature_vector_length": 2,
        "extractor_feature_vector_sha256": "e" * 64,
        "extractor_rendered_prompt_sha256": "f" * 64,
    }
    if window_family == "selected_choice":
        fields["extractor_selected_choice_readout_token_indices"] = token_indices
        fields["extractor_selected_choice_readout_token_indices_sha256"] = _json_sha256(token_indices)
        return fields
    if window_family == "freeform_final_token":
        fields["extractor_readout_token_indices"] = token_indices
        fields["extractor_readout_token_indices_sha256"] = _json_sha256(token_indices)
        fields["extractor_readout_window_source"] = "final_token"
        fields["extractor_readout_source"] = {
            "source": "live_cift_extractor",
            "readout_window": "final_token",
            "readout_token_count": len(token_indices),
        }
        return fields
    fields["extractor_query_tail_readout_token_indices"] = token_indices
    fields["extractor_query_tail_readout_token_indices_sha256"] = _json_sha256(token_indices)
    fields["extractor_readout_window_source"] = "query_tail"
    fields["extractor_readout_source"] = {
        "source": "live_cift_extractor",
        "readout_window": "query_tail",
        "readout_token_count": len(token_indices),
    }
    return fields


def _json_sha256(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
