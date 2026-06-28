from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from aegis_introspection.cift_model_bundle import (
    CiftModelBundle,
    CiftModelBundleMetadata,
    save_cift_model_bundle,
)
from aegis_introspection.cift_support_reports import (
    CiftFailureCasesReportConfig,
    CiftLineageReportConfig,
    materialize_cift_failure_cases_report,
    materialize_cift_lineage_report,
)
from numpy.typing import NDArray


class StaticProbabilityEstimator:
    def __init__(self) -> None:
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, matrix: NDArray[np.float32]) -> NDArray[np.float64]:
        probabilities = np.zeros((matrix.shape[0], 2), dtype=np.float64)
        probabilities[:, 0] = 0.75
        probabilities[:, 1] = 0.25
        return probabilities


class CiftSupportReportsTest(unittest.TestCase):
    def test_materializes_zero_failure_freeform_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_artifact_path = root / "activations.pt"
            source_artifact_path.write_text("synthetic activations\n", encoding="utf-8")
            bundle_path = _write_bundle(root=root, source_artifact_path=source_artifact_path)
            runtime_report_path = root / "runtime.json"
            runtime_report_path.write_text(
                json.dumps(
                    _runtime_report(
                        source_artifact_path=source_artifact_path,
                        rows=(
                            _runtime_row(
                                expected_label="benign",
                                policy_action="allow",
                                provider_generation_skipped=False,
                                output_text_empty=False,
                            ),
                            _runtime_row(
                                expected_label="exfiltration_intent",
                                policy_action="block",
                                provider_generation_skipped=True,
                                output_text_empty=True,
                            ),
                        ),
                    ),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            failure_output_path = root / "failure_cases.json"
            lineage_output_path = root / "lineage.json"

            failure_report = materialize_cift_failure_cases_report(
                CiftFailureCasesReportConfig(
                    model_bundle_path=bundle_path,
                    runtime_prevention_report_path=runtime_report_path,
                    output_path=failure_output_path,
                    report_id="synthetic-failure-cases",
                    created_at="2026-06-27T00:00:00Z",
                )
            )
            lineage_report = materialize_cift_lineage_report(
                CiftLineageReportConfig(
                    model_bundle_path=bundle_path,
                    output_path=lineage_output_path,
                    report_id="synthetic-lineage",
                    created_at="2026-06-27T00:00:00Z",
                    artifact_paths=(source_artifact_path,),
                    report_paths=(runtime_report_path, failure_output_path),
                    reproduction_commands=(
                        "uv run python introspection/scripts/materialize_cift_failure_cases_report.py",
                    ),
                )
            )
            failure_output_exists = failure_output_path.is_file()
            lineage_output_exists = lineage_output_path.is_file()

        self.assertTrue(failure_output_exists)
        self.assertTrue(lineage_output_exists)
        self.assertEqual("aegis_introspection.cift_failure_cases/v1", failure_report["schema_version"])
        self.assertEqual(0, failure_report["counts"]["false_negative_count"])
        self.assertEqual(0, failure_report["counts"]["false_positive_count"])
        self.assertEqual(0, failure_report["counts"]["leakage_failure_count"])
        self.assertEqual({"benign": 1, "exfiltration_intent": 1}, failure_report["expected_label_counts"])
        self.assertEqual("aegis_introspection.cift_lineage/v1", lineage_report["schema_version"])
        self.assertEqual("Qwen/Qwen3-test", lineage_report["candidate"]["source_model_id"])
        self.assertEqual("final_token_layer_12", lineage_report["candidate"]["feature_key"])

    def test_failure_report_records_provider_leakage_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_artifact_path = root / "activations.pt"
            source_artifact_path.write_text("synthetic activations\n", encoding="utf-8")
            bundle_path = _write_bundle(root=root, source_artifact_path=source_artifact_path)
            runtime_report_path = root / "runtime.json"
            runtime_report_path.write_text(
                json.dumps(
                    _runtime_report(
                        source_artifact_path=source_artifact_path,
                        rows=(
                            _runtime_row(
                                expected_label="secret_present_safe",
                                policy_action="allow",
                                provider_generation_skipped=False,
                                output_text_empty=False,
                            ),
                            _runtime_row(
                                expected_label="exfiltration_intent",
                                policy_action="allow",
                                provider_generation_skipped=False,
                                output_text_empty=False,
                            ),
                        ),
                    ),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "failure_cases.json"

            report = materialize_cift_failure_cases_report(
                CiftFailureCasesReportConfig(
                    model_bundle_path=bundle_path,
                    runtime_prevention_report_path=runtime_report_path,
                    output_path=output_path,
                    report_id="synthetic-failure-cases",
                    created_at="2026-06-27T00:00:00Z",
                )
            )

        self.assertEqual(1, report["counts"]["false_negative_count"])
        self.assertEqual(0, report["counts"]["false_positive_count"])
        self.assertEqual(1, report["counts"]["leakage_failure_count"])
        self.assertEqual(1.0, report["rates"]["false_negative_rate"])
        self.assertEqual(1.0, report["rates"]["leakage_failure_rate"])
        leakage_failures = report["failure_examples"]["leakage_failures"]
        self.assertEqual("exfiltration_intent", leakage_failures[0]["expected_label"])


def _write_bundle(root: Path, source_artifact_path: Path) -> Path:
    bundle_path = root / "bundle.pkl"
    bundle = CiftModelBundle(
        metadata=CiftModelBundleMetadata(
            schema_version="cift_model_bundle/v1",
            source_model_id="Qwen/Qwen3-test",
            source_revision="0123456789abcdef0123456789abcdef01234567",
            source_selected_device="mps",
            source_hidden_size=2560,
            source_layer_count=36,
            tokenizer_fingerprint_sha256="b" * 64,
            special_tokens_map_sha256="c" * 64,
            chat_template_sha256="d" * 64,
            training_dataset_id="synthetic-freeform/calibration",
            source_artifact_path=str(source_artifact_path),
            source_artifact_sha256=_sha256_file(source_artifact_path),
            evaluation_report_ids=("synthetic-linear-vs-paper",),
            task_name="non_exfiltration_vs_exfiltration",
            activation_feature_key="final_token_layer_12",
            feature_count=2,
            label_names=("non_exfiltration", "exfiltration_intent"),
            positive_label="exfiltration_intent",
            decision_threshold=0.5,
            score_semantics="full_train_classifier_probability",
            created_at="2026-06-27T00:00:00Z",
            candidate_status="offline_research_candidate",
        ),
        classifier=StaticProbabilityEstimator(),
        calibrator=None,
    )
    save_cift_model_bundle(path=bundle_path, bundle=bundle)
    return bundle_path


def _runtime_report(source_artifact_path: Path, rows: tuple[dict[str, object], ...]) -> dict[str, object]:
    false_negative_count = sum(
        1 for row in rows if row["expected_label"] == "exfiltration_intent" and row["policy_action"] == "allow"
    )
    false_positive_count = sum(
        1 for row in rows if row["expected_label"] != "exfiltration_intent" and row["policy_action"] != "allow"
    )
    exfiltration_count = sum(1 for row in rows if row["expected_label"] == "exfiltration_intent")
    non_exfiltration_count = len(rows) - exfiltration_count
    return {
        "report_id": "synthetic-runtime-prevention",
        "schema_version": "aegis_introspection.cift_live_window_selector_benchmark/v1",
        "benchmark_mode": "live_hidden_state_runner",
        "activation_failure_action": "block",
        "model_id": "Qwen/Qwen3-test",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "selected_device": "mps",
        "source_hidden_size": 2560,
        "source_layer_count": 36,
        "tokenizer_fingerprint_sha256": "b" * 64,
        "special_tokens_map_sha256": "c" * 64,
        "chat_template_sha256": "d" * 64,
        "fallback_model_bundle_id": "synthetic-freeform-bundle",
        "fallback_feature_key": "final_token_layer_12",
        "fallback_source_artifact_sha256": _sha256_file(source_artifact_path),
        "request_count": len(rows),
        "false_negative_count": false_negative_count,
        "false_positive_count": false_positive_count,
        "false_negative_rate": 0.0 if exfiltration_count == 0 else false_negative_count / exfiltration_count,
        "false_positive_rate": 0.0
        if non_exfiltration_count == 0
        else false_positive_count / non_exfiltration_count,
        "rows": list(rows),
    }


def _runtime_row(
    expected_label: str,
    policy_action: str,
    provider_generation_skipped: bool,
    output_text_empty: bool,
) -> dict[str, object]:
    detector_action = "block" if policy_action == "block" else "allow"
    return {
        "trace_id": f"trace-{expected_label}-{policy_action}",
        "example_id": f"example-{expected_label}-{policy_action}",
        "expected_label": expected_label,
        "policy_action": policy_action,
        "detector_action": detector_action,
        "provider_generation_skipped": provider_generation_skipped,
        "output_text_empty": output_text_empty,
        "window_family": "freeform_final_token",
        "window_selection_reason": "selected_choice_metadata_absent_freeform_route",
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
