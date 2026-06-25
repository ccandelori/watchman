from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from aegis_introspection.cift_certification_workflow import (
    CiftCertificationEvidenceManifestConfig,
    CiftCertificationWorkflowConfig,
    CiftCertificationWorkflowError,
    build_cift_certification_evidence_manifest,
    build_cift_certification_workflow_manifest,
)
from aegis_introspection.cift_model_metadata import CiftModelMetadataReport

from aegis.cift_contract import (
    CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
    CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
)

_IMMUTABLE_MODEL_REVISION = "0123456789abcdef0123456789abcdef01234567"


class CiftCertificationWorkflowTest(unittest.TestCase):
    def test_evidence_manifest_binds_explicit_current_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            fixture = _write_evidence_fixture(repository_root)

            manifest = build_cift_certification_evidence_manifest(
                CiftCertificationEvidenceManifestConfig(
                    certification_id="synthetic_qwen3_4b_current",
                    repository_root=repository_root,
                    created_at="2026-06-24T00:00:00Z",
                    behavior_id="secret-exfiltration-intent",
                    behavior_description="User request attempts to move a protected secret into an external channel.",
                    requested_device="mps",
                    prompt_renderer=CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
                    selected_choice_geometry=CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
                    selected_choice_readout_token_count=4,
                    dtype_name="device",
                    metric_threshold=1.0,
                    ablation_delta_threshold=0.0,
                    model_metadata_report_path=fixture["model_metadata"],
                    activation_artifact_path=fixture["activation"],
                    linear_bundle_path=fixture["bundle"],
                    grouped_head_to_head_report_path=fixture["grouped_head_to_head"],
                    calibration_report_path=fixture["calibration"],
                    feature_ablation_report_path=fixture["feature_ablation"],
                    patching_report_path=fixture["patching"],
                    failure_cases_report_path=fixture["failure_cases"],
                    lineage_report_path=fixture["lineage"],
                    device_preflight_report_path=fixture["device_preflight"],
                    live_runtime_prevention_report_path=fixture["runtime_prevention"],
                    sealed_holdout_metric_path=fixture["sealed_metric"],
                    gateway_smoke_report_path=fixture["gateway_smoke"],
                    paper_mlp_runtime_prevention_report_path=fixture["paper_runtime_prevention"],
                    paper_mlp_sealed_holdout_metric_path=fixture["paper_sealed_metric"],
                    live_head_to_head_report_path=fixture["live_head_to_head"],
                    promotion_evidence_path=fixture["promotion_evidence"],
                    promoted_runtime_artifact_path=fixture["runtime"],
                    promotion_report_output_dir=repository_root / "reports" / "cift_promotion",
                    evidence_chain_verification_report_path=fixture["evidence_chain"],
                )
            )

        self.assertEqual("evidence_bound", manifest["status"])
        self.assertEqual("Qwen/Qwen3-test", manifest["model_identity"]["model_id"])
        self.assertEqual("synthetic-training", manifest["training"]["training_dataset_id"])
        self.assertEqual("mps", manifest["training"]["requested_device"])
        self.assertEqual(CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1, manifest["training"]["prompt_renderer"])
        self.assertEqual(
            CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
            manifest["training"]["selected_choice_geometry"],
        )
        self.assertEqual(4, manifest["training"]["selected_choice_readout_token_count"])
        self.assertEqual("device", manifest["training"]["dtype_name"])
        self.assertEqual([21], manifest["training"]["layer_indices"])
        self.assertEqual(["selected_choice_window"], manifest["training"]["pooling_methods"])
        artifacts_by_role = {artifact["role"]: artifact for artifact in manifest["required_evidence_artifacts"]}
        self.assertEqual("cift_model_bundle/v1", artifacts_by_role["linear_candidate_bundle"]["schema_version"])
        self.assertEqual("synthetic-sealed", artifacts_by_role["linear_sealed_holdout_metric"]["report_id"])
        self.assertEqual(
            "aegis.proxy.cift_gateway_smoke/v1",
            artifacts_by_role["linear_gateway_smoke"]["schema_version"],
        )
        self.assertEqual(
            "synthetic-paper-runtime-prevention",
            artifacts_by_role["paper_mlp_live_runtime_prevention"]["report_id"],
        )
        self.assertEqual(
            "synthetic-paper-sealed",
            artifacts_by_role["paper_mlp_sealed_holdout_metric"]["report_id"],
        )
        self.assertEqual(
            hashlib.sha256(b"activation-bytes").hexdigest(),
            artifacts_by_role["calibration_activation_artifact"]["sha256"],
        )
        self.assertEqual(
            "aegis_introspection.device_preflight/v1",
            artifacts_by_role["device_preflight"]["schema_version"],
        )

    def test_evidence_manifest_rejects_missing_required_output_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            fixture = _write_evidence_fixture(repository_root)
            fixture["evidence_chain"].unlink()

            with self.assertRaisesRegex(CiftCertificationWorkflowError, "evidence_chain_verification_report_path"):
                build_cift_certification_evidence_manifest(
                    CiftCertificationEvidenceManifestConfig(
                        certification_id="synthetic_qwen3_4b_current",
                        repository_root=repository_root,
                        created_at="2026-06-24T00:00:00Z",
                        behavior_id="secret-exfiltration-intent",
                        behavior_description=(
                            "User request attempts to move a protected secret into an external channel."
                        ),
                        requested_device="mps",
                        prompt_renderer=CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
                        selected_choice_geometry=CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
                        selected_choice_readout_token_count=4,
                        dtype_name="device",
                        metric_threshold=1.0,
                        ablation_delta_threshold=0.0,
                        model_metadata_report_path=fixture["model_metadata"],
                        activation_artifact_path=fixture["activation"],
                        linear_bundle_path=fixture["bundle"],
                        grouped_head_to_head_report_path=fixture["grouped_head_to_head"],
                        calibration_report_path=fixture["calibration"],
                        feature_ablation_report_path=fixture["feature_ablation"],
                        patching_report_path=fixture["patching"],
                        failure_cases_report_path=fixture["failure_cases"],
                        lineage_report_path=fixture["lineage"],
                        device_preflight_report_path=fixture["device_preflight"],
                        live_runtime_prevention_report_path=fixture["runtime_prevention"],
                        sealed_holdout_metric_path=fixture["sealed_metric"],
                        gateway_smoke_report_path=fixture["gateway_smoke"],
                        paper_mlp_runtime_prevention_report_path=fixture["paper_runtime_prevention"],
                        paper_mlp_sealed_holdout_metric_path=fixture["paper_sealed_metric"],
                        live_head_to_head_report_path=fixture["live_head_to_head"],
                        promotion_evidence_path=fixture["promotion_evidence"],
                        promoted_runtime_artifact_path=fixture["runtime"],
                        promotion_report_output_dir=repository_root / "reports" / "cift_promotion",
                        evidence_chain_verification_report_path=fixture["evidence_chain"],
                    )
                )

    def test_evidence_manifest_rejects_missing_grouped_head_to_head_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            fixture = _write_evidence_fixture(repository_root)
            fixture["grouped_head_to_head"].unlink()

            with self.assertRaisesRegex(CiftCertificationWorkflowError, "grouped_head_to_head_report_path"):
                build_cift_certification_evidence_manifest(
                    _evidence_manifest_config(repository_root=repository_root, fixture=fixture)
                )

    def test_evidence_manifest_rejects_runtime_prevention_device_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            fixture = _write_evidence_fixture(repository_root)
            runtime_prevention = _runtime_prevention_record()
            runtime_prevention["selected_device"] = "cpu"
            _write_json(fixture["runtime_prevention"], runtime_prevention)

            with self.assertRaisesRegex(CiftCertificationWorkflowError, "selected_device"):
                build_cift_certification_evidence_manifest(
                    CiftCertificationEvidenceManifestConfig(
                        certification_id="synthetic_qwen3_4b_current",
                        repository_root=repository_root,
                        created_at="2026-06-24T00:00:00Z",
                        behavior_id="secret-exfiltration-intent",
                        behavior_description=(
                            "User request attempts to move a protected secret into an external channel."
                        ),
                        requested_device="mps",
                        prompt_renderer=CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
                        selected_choice_geometry=CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
                        selected_choice_readout_token_count=4,
                        dtype_name="device",
                        metric_threshold=1.0,
                        ablation_delta_threshold=0.0,
                        model_metadata_report_path=fixture["model_metadata"],
                        activation_artifact_path=fixture["activation"],
                        linear_bundle_path=fixture["bundle"],
                        grouped_head_to_head_report_path=fixture["grouped_head_to_head"],
                        calibration_report_path=fixture["calibration"],
                        feature_ablation_report_path=fixture["feature_ablation"],
                        patching_report_path=fixture["patching"],
                        failure_cases_report_path=fixture["failure_cases"],
                        lineage_report_path=fixture["lineage"],
                        device_preflight_report_path=fixture["device_preflight"],
                        live_runtime_prevention_report_path=fixture["runtime_prevention"],
                        sealed_holdout_metric_path=fixture["sealed_metric"],
                        gateway_smoke_report_path=fixture["gateway_smoke"],
                        paper_mlp_runtime_prevention_report_path=fixture["paper_runtime_prevention"],
                        paper_mlp_sealed_holdout_metric_path=fixture["paper_sealed_metric"],
                        live_head_to_head_report_path=fixture["live_head_to_head"],
                        promotion_evidence_path=fixture["promotion_evidence"],
                        promoted_runtime_artifact_path=fixture["runtime"],
                        promotion_report_output_dir=repository_root / "reports" / "cift_promotion",
                        evidence_chain_verification_report_path=fixture["evidence_chain"],
                    )
                )

    def test_evidence_manifest_rejects_mutable_model_metadata_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            fixture = _write_evidence_fixture(repository_root)
            metadata = _metadata_record()
            metadata["revision"] = "main"
            _write_json(fixture["model_metadata"], metadata)

            with self.assertRaisesRegex(CiftCertificationWorkflowError, "model metadata revision"):
                build_cift_certification_evidence_manifest(
                    _evidence_manifest_config(repository_root=repository_root, fixture=fixture)
                )

    def test_evidence_manifest_rejects_promoted_runtime_feature_layer_outside_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            fixture = _write_evidence_fixture(repository_root)
            metadata = _metadata_record()
            metadata["layer_count"] = 21
            _write_json(fixture["model_metadata"], metadata)

            with self.assertRaisesRegex(
                CiftCertificationWorkflowError,
                "promoted runtime artifact feature_key",
            ):
                build_cift_certification_evidence_manifest(
                    _evidence_manifest_config(repository_root=repository_root, fixture=fixture)
                )

    def test_evidence_manifest_rejects_unknown_dtype_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            fixture = _write_evidence_fixture(repository_root)
            config = _evidence_manifest_config(repository_root=repository_root, fixture=fixture, dtype_name="mps_auto")

            with self.assertRaisesRegex(CiftCertificationWorkflowError, "dtype_name"):
                build_cift_certification_evidence_manifest(config)

    def test_manifest_binds_model_identity_and_input_hashes_without_certifying_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            corpus_path = repository_root / "data" / "calibration.jsonl"
            runtime_turns_path = repository_root / "data" / "runtime_turns.jsonl"
            fallback_runtime_model_path = repository_root / "models" / "fallback_runtime.json"
            corpus_path.parent.mkdir(parents=True)
            fallback_runtime_model_path.parent.mkdir(parents=True)
            corpus_path.write_text('{"id":"safe"}\n', encoding="utf-8")
            runtime_turns_path.write_text('{"trace_id":"one"}\n', encoding="utf-8")
            fallback_runtime_model_path.write_text("{}\n", encoding="utf-8")

            manifest = build_cift_certification_workflow_manifest(
                config=_config(
                    repository_root=repository_root,
                    corpus_path=corpus_path,
                    runtime_turns_path=runtime_turns_path,
                    fallback_runtime_model_path=fallback_runtime_model_path,
                ),
                model_metadata=_metadata(),
            )

        self.assertEqual("aegis_introspection.cift_certification_workflow/v1", manifest["schema_version"])
        self.assertEqual("not_certified_until_release_gate_passes", manifest["support_claim_status"])
        self.assertEqual("Qwen/Qwen3-4B", manifest["model_identity"]["model_id"])
        self.assertEqual(2560, manifest["model_identity"]["hidden_size"])
        self.assertEqual(36, manifest["model_identity"]["layer_count"])
        self.assertEqual(
            hashlib.sha256(b'{"id":"safe"}\n').hexdigest(),
            manifest["corpus"]["sha256"],
        )
        self.assertEqual(
            hashlib.sha256(b'{"trace_id":"one"}\n').hexdigest(),
            manifest["runtime_turns"]["sha256"],
        )
        self.assertEqual(
            "introspection/data/activations/qwen3_4b_watchman_v4_windows_l19_l22.pt",
            manifest["planned_artifacts"]["activation_artifact_path"],
        )
        self.assertEqual(
            "introspection/data/reports/qwen3_4b_watchman_v4_device_preflight_v1.json",
            manifest["planned_artifacts"]["device_preflight_report_path"],
        )
        self.assertEqual(
            "introspection/data/reports/qwen3_4b_watchman_v4_certification_workflow_v1.json",
            manifest["planned_artifacts"]["certification_manifest_path"],
        )
        self.assertEqual(
            "introspection/data/reports/qwen3_4b_watchman_v4_certification_workflow_run_v1.json",
            manifest["planned_artifacts"]["certification_report_path"],
        )
        self.assertEqual(
            "introspection/data/reports/qwen3_4b_watchman_v4_strict_deployment_env.sh",
            manifest["planned_artifacts"]["deployment_env_path"],
        )
        self.assertEqual(
            "introspection/data/reports/qwen3_4b_watchman_v4_release_gate_v1.json",
            manifest["planned_artifacts"]["release_gate_report_path"],
        )
        self.assertEqual("mps", manifest["training"]["requested_device"])
        self.assertEqual("float16", manifest["training"]["dtype_name"])
        self.assertEqual(CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1, manifest["training"]["prompt_renderer"])
        self.assertEqual(
            CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
            manifest["training"]["selected_choice_geometry"],
        )
        self.assertEqual(4, manifest["training"]["selected_choice_readout_token_count"])
        self.assertIn("live_hidden_state_runtime_prevention", manifest["required_release_evidence"])
        self.assertIn("live_gateway_sidecar_runtime_prevention", manifest["required_release_evidence"])
        self.assertIn("hardened_release_gate_pass", manifest["required_release_evidence"])
        required_artifacts = {artifact["role"]: artifact for artifact in manifest["required_evidence_artifacts"]}
        self.assertEqual(
            "aegis.proxy.cift_gateway_smoke/v1",
            required_artifacts["linear_gateway_smoke"]["schema_version"],
        )
        self.assertEqual(
            "aegis_introspection.cift_live_window_selector_benchmark/v1",
            required_artifacts["linear_live_runtime_prevention"]["schema_version"],
        )
        self.assertEqual(
            "aegis_introspection.device_preflight/v1",
            required_artifacts["device_preflight"]["schema_version"],
        )
        self.assertEqual("cift_model_bundle/v1", required_artifacts["linear_candidate_bundle"]["schema_version"])
        self.assertEqual("planned", required_artifacts["linear_sealed_holdout_metric"]["status"])
        command_plan = manifest["command_plan"]
        step_ids = [step["step_id"] for step in command_plan]
        self.assertLess(step_ids.index("run_device_preflight"), step_ids.index("extract_activation_artifact"))
        self.assertLess(
            step_ids.index("export_linear_runtime_bootstrap_for_live_evidence"),
            step_ids.index("run_linear_live_runtime_prevention"),
        )
        self.assertLess(
            step_ids.index("run_linear_live_runtime_prevention"),
            step_ids.index("run_linear_gateway_smoke"),
        )
        self.assertLess(step_ids.index("run_linear_gateway_smoke"), step_ids.index("materialize_promotion_evidence"))
        self.assertLess(step_ids.index("materialize_promotion_evidence"), step_ids.index("export_promoted_runtime"))
        self.assertLess(step_ids.index("export_promoted_runtime"), step_ids.index("verify_evidence_chain_identity"))
        self.assertLess(
            step_ids.index("verify_evidence_chain_identity"),
            step_ids.index("materialize_certification_manifest"),
        )
        self.assertLess(
            step_ids.index("materialize_certification_manifest"),
            step_ids.index("verify_certification_workflow_run"),
        )
        self.assertLess(
            step_ids.index("verify_certification_workflow_run"),
            step_ids.index("run_hardened_release_gate"),
        )
        steps_by_id = {step["step_id"]: step for step in command_plan}
        self.assertIn(
            manifest["planned_artifacts"]["device_preflight_report_path"],
            steps_by_id["run_device_preflight"]["argv"],
        )
        promoted_runtime_path = manifest["planned_artifacts"]["promoted_runtime_artifact_path"]
        self.assertIn(promoted_runtime_path, steps_by_id["run_linear_live_runtime_prevention"]["argv"])
        self.assertIn("mps", steps_by_id["verify_evidence_chain_identity"]["argv"])
        self.assertIn(
            manifest["planned_artifacts"]["certification_manifest_path"],
            steps_by_id["materialize_certification_manifest"]["argv"],
        )
        self.assertIn("--dtype", steps_by_id["materialize_certification_manifest"]["argv"])
        self.assertIn("float16", steps_by_id["materialize_certification_manifest"]["argv"])
        self.assertIn(
            manifest["planned_artifacts"]["certification_report_path"],
            steps_by_id["verify_certification_workflow_run"]["argv"],
        )
        self.assertIn("--command-timeout-seconds", steps_by_id["verify_certification_workflow_run"]["argv"])
        self.assertIn("30.0", steps_by_id["verify_certification_workflow_run"]["argv"])
        hardened_gate_argv = steps_by_id["run_hardened_release_gate"]["argv_template"]
        self.assertIn("introspection/scripts/materialize_cift_deployment_env.py", hardened_gate_argv)
        self.assertIn("mps", hardened_gate_argv)
        self.assertIn("${extractor_id}", hardened_gate_argv)
        self.assertIn("${sidecar_base_url}", hardened_gate_argv)
        self.assertIn(manifest["planned_artifacts"]["deployment_env_path"], hardened_gate_argv)
        self.assertIn(manifest["planned_artifacts"]["release_gate_report_path"], hardened_gate_argv)
        self.assertNotIn("--allow-embedded-artifact-only", hardened_gate_argv)
        self.assertEqual(
            [
                manifest["planned_artifacts"]["deployment_env_path"],
                manifest["planned_artifacts"]["release_gate_report_path"],
            ],
            steps_by_id["run_hardened_release_gate"]["produces"],
        )
        self.assertIn("${extractor_id}", steps_by_id["run_linear_gateway_smoke"]["argv_template"])
        self.assertIn(
            manifest["planned_artifacts"]["gateway_smoke_report_path"],
            steps_by_id["run_linear_gateway_smoke"]["argv_template"],
        )
        self.assertIn(
            "${linear_sealed_holdout_metric.metric_value}",
            steps_by_id["materialize_promotion_evidence"]["argv_template"],
        )

    def test_manifest_rejects_model_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            corpus_path = repository_root / "data" / "calibration.jsonl"
            runtime_turns_path = repository_root / "data" / "runtime_turns.jsonl"
            fallback_runtime_model_path = repository_root / "models" / "fallback_runtime.json"
            corpus_path.parent.mkdir(parents=True)
            fallback_runtime_model_path.parent.mkdir(parents=True)
            corpus_path.write_text("{}\n", encoding="utf-8")
            runtime_turns_path.write_text("{}\n", encoding="utf-8")
            fallback_runtime_model_path.write_text("{}\n", encoding="utf-8")
            metadata = _metadata(model_id="Qwen/Qwen3-0.6B")

            with self.assertRaisesRegex(CiftCertificationWorkflowError, "model_id"):
                build_cift_certification_workflow_manifest(
                    config=_config(
                        repository_root=repository_root,
                        corpus_path=corpus_path,
                        runtime_turns_path=runtime_turns_path,
                        fallback_runtime_model_path=fallback_runtime_model_path,
                    ),
                    model_metadata=metadata,
                )

    def test_manifest_rejects_qwen3_4b_without_mps_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            corpus_path = repository_root / "data" / "calibration.jsonl"
            runtime_turns_path = repository_root / "data" / "runtime_turns.jsonl"
            fallback_runtime_model_path = repository_root / "models" / "fallback_runtime.json"
            corpus_path.parent.mkdir(parents=True)
            fallback_runtime_model_path.parent.mkdir(parents=True)
            corpus_path.write_text("{}\n", encoding="utf-8")
            runtime_turns_path.write_text("{}\n", encoding="utf-8")
            fallback_runtime_model_path.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(CiftCertificationWorkflowError, "Qwen/Qwen3-4B"):
                build_cift_certification_workflow_manifest(
                    config=_config(
                        repository_root=repository_root,
                        corpus_path=corpus_path,
                        runtime_turns_path=runtime_turns_path,
                        fallback_runtime_model_path=fallback_runtime_model_path,
                        requested_device="cpu",
                    ),
                    model_metadata=_metadata(),
                )

    def test_manifest_rejects_mutable_model_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            corpus_path = repository_root / "data" / "calibration.jsonl"
            runtime_turns_path = repository_root / "data" / "runtime_turns.jsonl"
            fallback_runtime_model_path = repository_root / "models" / "fallback_runtime.json"
            corpus_path.parent.mkdir(parents=True)
            fallback_runtime_model_path.parent.mkdir(parents=True)
            corpus_path.write_text("{}\n", encoding="utf-8")
            runtime_turns_path.write_text("{}\n", encoding="utf-8")
            fallback_runtime_model_path.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(CiftCertificationWorkflowError, "immutable"):
                build_cift_certification_workflow_manifest(
                    config=_config(
                        repository_root=repository_root,
                        corpus_path=corpus_path,
                        runtime_turns_path=runtime_turns_path,
                        fallback_runtime_model_path=fallback_runtime_model_path,
                        revision="main",
                    ),
                    model_metadata=_metadata(revision="main"),
                )

    def test_manifest_rejects_layer_indices_outside_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            corpus_path = repository_root / "data" / "calibration.jsonl"
            runtime_turns_path = repository_root / "data" / "runtime_turns.jsonl"
            fallback_runtime_model_path = repository_root / "models" / "fallback_runtime.json"
            corpus_path.parent.mkdir(parents=True)
            fallback_runtime_model_path.parent.mkdir(parents=True)
            corpus_path.write_text("{}\n", encoding="utf-8")
            runtime_turns_path.write_text("{}\n", encoding="utf-8")
            fallback_runtime_model_path.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(CiftCertificationWorkflowError, "layer_indices"):
                build_cift_certification_workflow_manifest(
                    config=_config(
                        repository_root=repository_root,
                        corpus_path=corpus_path,
                        runtime_turns_path=runtime_turns_path,
                        fallback_runtime_model_path=fallback_runtime_model_path,
                    ),
                    model_metadata=_metadata(layer_count=20),
                )

    def test_manifest_rejects_candidate_feature_layer_outside_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            corpus_path = repository_root / "data" / "calibration.jsonl"
            runtime_turns_path = repository_root / "data" / "runtime_turns.jsonl"
            fallback_runtime_model_path = repository_root / "models" / "fallback_runtime.json"
            corpus_path.parent.mkdir(parents=True)
            fallback_runtime_model_path.parent.mkdir(parents=True)
            corpus_path.write_text("{}\n", encoding="utf-8")
            runtime_turns_path.write_text("{}\n", encoding="utf-8")
            fallback_runtime_model_path.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(CiftCertificationWorkflowError, "candidate_feature_key"):
                build_cift_certification_workflow_manifest(
                    config=_config(
                        repository_root=repository_root,
                        corpus_path=corpus_path,
                        runtime_turns_path=runtime_turns_path,
                        fallback_runtime_model_path=fallback_runtime_model_path,
                        layer_indices=(0, 1),
                        candidate_feature_key="selected_choice_window_layer_20",
                    ),
                    model_metadata=_metadata(layer_count=20),
                )


def _config(
    repository_root: Path,
    corpus_path: Path,
    runtime_turns_path: Path,
    fallback_runtime_model_path: Path,
    requested_device: str = "mps",
    revision: str = _IMMUTABLE_MODEL_REVISION,
    layer_indices: tuple[int, ...] = (19, 20, 21, 22),
    candidate_feature_key: str = "selected_choice_window_layer_19",
) -> CiftCertificationWorkflowConfig:
    return CiftCertificationWorkflowConfig(
        certification_id="qwen3_4b_watchman_v4",
        repository_root=repository_root,
        model_id="Qwen/Qwen3-4B",
        revision=revision,
        corpus_path=corpus_path,
        runtime_turns_path=runtime_turns_path,
        fallback_runtime_model_path=fallback_runtime_model_path,
        output_dir=repository_root / "introspection" / "data",
        training_dataset_id="watchman_semantic_v4_480_selected_clause_l19_raw",
        task_name="safe_secret_vs_exfiltration",
        positive_label="exfiltration_intent",
        behavior_id="secret-exfiltration-intent",
        behavior_description="User request attempts to move a protected secret into an external channel.",
        layer_indices=layer_indices,
        pooling_methods=("readout_window", "query_tail_window", "selected_choice_window", "combined_readout_window"),
        candidate_feature_key=candidate_feature_key,
        requested_device=requested_device,
        prompt_renderer=CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
        selected_choice_geometry=CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
        selected_choice_readout_token_count=4,
        dtype_name="float16",
        metric_threshold=1.0,
        ablation_delta_threshold=0.0,
        allow_download=False,
        trust_remote_code=False,
        created_at="2026-06-24T00:00:00Z",
    )


def _write_evidence_fixture(repository_root: Path) -> dict[str, Path]:
    reports_root = repository_root / "reports"
    models_root = repository_root / "models"
    activations_root = repository_root / "activations"
    reports_root.mkdir(parents=True)
    models_root.mkdir(parents=True)
    activations_root.mkdir(parents=True)
    fixture = {
        "model_metadata": reports_root / "model_metadata.json",
        "activation": activations_root / "activation.pt",
        "bundle": models_root / "bundle.pkl",
        "grouped_head_to_head": reports_root / "grouped_head_to_head.json",
        "calibration": reports_root / "calibration.json",
        "feature_ablation": reports_root / "feature_ablation.json",
        "patching": reports_root / "patching.json",
        "failure_cases": reports_root / "failure_cases.json",
        "lineage": reports_root / "lineage.json",
        "device_preflight": reports_root / "device_preflight.json",
        "runtime_prevention": reports_root / "runtime_prevention.json",
        "sealed_metric": reports_root / "sealed_metric.json",
        "gateway_smoke": reports_root / "gateway_smoke.json",
        "paper_runtime_prevention": reports_root / "paper_runtime_prevention.json",
        "paper_sealed_metric": reports_root / "paper_sealed_metric.json",
        "live_head_to_head": reports_root / "live_head_to_head.json",
        "promotion_evidence": reports_root / "promotion_evidence.json",
        "runtime": models_root / "runtime.json",
        "evidence_chain": reports_root / "evidence_chain.json",
    }
    fixture["activation"].write_bytes(b"activation-bytes")
    fixture["bundle"].write_bytes(b"bundle-bytes")
    _write_json(fixture["model_metadata"], _metadata_record())
    _write_json(fixture["runtime"], _runtime_record())
    _write_json(fixture["device_preflight"], _device_preflight_record())
    _write_json(fixture["runtime_prevention"], _runtime_prevention_record())
    _write_json(
        fixture["paper_runtime_prevention"],
        _report_record(
            "synthetic-paper-runtime-prevention",
            "aegis_introspection.cift_live_window_selector_benchmark/v1",
            extra={
                "activation_failure_action": "block",
                "benchmark_mode": "live_hidden_state_runner",
                "false_negative_count": 1,
                "false_negative_rate": 0.01,
                "false_positive_count": 0,
                "false_positive_rate": 0.0,
                "selected_device": "mps",
                "window_family_mismatch_count": 0,
            },
        ),
    )
    _write_json(
        fixture["grouped_head_to_head"],
        _report_record("synthetic-grouped", "cift_probe_competition/v1", extra=None),
    )
    _write_json(
        fixture["calibration"],
        _report_record("synthetic-calibration", "aegis_introspection.cift_calibration/v1", extra=None),
    )
    _write_json(
        fixture["feature_ablation"],
        _report_record("synthetic-ablation", "aegis_introspection.cift_feature_ablation/v1", extra=None),
    )
    _write_json(
        fixture["patching"],
        _report_record("synthetic-patching", "aegis_introspection.cift_counterfactual_patching/v1", extra=None),
    )
    _write_json(
        fixture["failure_cases"],
        _report_record("synthetic-failure-cases", "aegis_introspection.cift_failure_cases/v1", extra=None),
    )
    _write_json(
        fixture["lineage"],
        _report_record("synthetic-lineage", "aegis_introspection.cift_lineage/v1", extra=None),
    )
    _write_json(
        fixture["sealed_metric"],
        _report_record(
            "synthetic-sealed",
            "aegis_introspection.cift_sealed_holdout_metric/v1",
            extra={
                "false_negative_count": 0,
                "false_negative_rate": 0.0,
                "false_positive_count": 0,
                "false_positive_rate": 0.0,
                "metric_value": 1.0,
                "source_selected_device": "mps",
            },
        ),
    )
    _write_json(
        fixture["gateway_smoke"],
        _report_record("synthetic-gateway-smoke", "aegis.proxy.cift_gateway_smoke/v1", extra={"status": "ok"}),
    )
    _write_json(
        fixture["paper_sealed_metric"],
        _report_record(
            "synthetic-paper-sealed",
            "aegis_introspection.cift_sealed_holdout_metric/v1",
            extra={
                "false_negative_count": 1,
                "false_negative_rate": 0.01,
                "false_positive_count": 0,
                "false_positive_rate": 0.0,
                "metric_value": 0.99,
                "source_selected_device": "mps",
            },
        ),
    )
    _write_json(
        fixture["live_head_to_head"],
        _report_record(
            "synthetic-live-head-to-head",
            "aegis_introspection.cift_live_probe_competition/v1",
            extra=None,
        ),
    )
    _write_json(fixture["promotion_evidence"], {"schema_version": "cift_promotion_evidence/v1"})
    _write_json(
        fixture["evidence_chain"],
        {"schema_version": "aegis_introspection.cift_evidence_chain_verification/v1", "eligible": True},
    )
    return fixture


def _write_json(path: Path, record: dict[str, object]) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evidence_manifest_config(
    repository_root: Path,
    fixture: dict[str, Path],
    dtype_name: str = "device",
) -> CiftCertificationEvidenceManifestConfig:
    return CiftCertificationEvidenceManifestConfig(
        certification_id="synthetic_qwen3_4b_current",
        repository_root=repository_root,
        created_at="2026-06-24T00:00:00Z",
        behavior_id="secret-exfiltration-intent",
        behavior_description="User request attempts to move a protected secret into an external channel.",
        requested_device="mps",
        prompt_renderer=CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
        selected_choice_geometry=CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
        selected_choice_readout_token_count=4,
        dtype_name=dtype_name,
        metric_threshold=1.0,
        ablation_delta_threshold=0.0,
        model_metadata_report_path=fixture["model_metadata"],
        activation_artifact_path=fixture["activation"],
        linear_bundle_path=fixture["bundle"],
        grouped_head_to_head_report_path=fixture["grouped_head_to_head"],
        calibration_report_path=fixture["calibration"],
        feature_ablation_report_path=fixture["feature_ablation"],
        patching_report_path=fixture["patching"],
        failure_cases_report_path=fixture["failure_cases"],
        lineage_report_path=fixture["lineage"],
        device_preflight_report_path=fixture["device_preflight"],
        live_runtime_prevention_report_path=fixture["runtime_prevention"],
        sealed_holdout_metric_path=fixture["sealed_metric"],
        gateway_smoke_report_path=fixture["gateway_smoke"],
        paper_mlp_runtime_prevention_report_path=fixture["paper_runtime_prevention"],
        paper_mlp_sealed_holdout_metric_path=fixture["paper_sealed_metric"],
        live_head_to_head_report_path=fixture["live_head_to_head"],
        promotion_evidence_path=fixture["promotion_evidence"],
        promoted_runtime_artifact_path=fixture["runtime"],
        promotion_report_output_dir=repository_root / "reports" / "cift_promotion",
        evidence_chain_verification_report_path=fixture["evidence_chain"],
    )


def _metadata_record() -> dict[str, object]:
    return {
        "schema_version": "aegis_introspection.cift_model_metadata/v1",
        "model_id": "Qwen/Qwen3-test",
        "revision": _IMMUTABLE_MODEL_REVISION,
        "model_type": "qwen3",
        "hidden_size": 2560,
        "layer_count": 36,
        "tokenizer_class": "FakeTokenizer",
        "tokenizer_vocab_size": 100,
        "tokenizer_fingerprint_sha256": "a" * 64,
        "special_tokens_map_sha256": "b" * 64,
        "chat_template_present": True,
        "chat_template_sha256": "c" * 64,
    }


def _runtime_record() -> dict[str, object]:
    return {
        "schema_version": "aegis.cift_runtime_linear/v1",
        "source_revision": _IMMUTABLE_MODEL_REVISION,
        "training_dataset_id": "synthetic-training",
        "task_name": "safe_secret_vs_exfiltration",
        "positive_label": "exfiltration_intent",
        "feature_key": "selected_choice_window_layer_21",
    }


def _runtime_prevention_record() -> dict[str, object]:
    return {
        "schema_version": "aegis_introspection.cift_live_window_selector_benchmark/v1",
        "report_id": "synthetic-runtime-prevention",
        "activation_failure_action": "block",
        "benchmark_mode": "live_hidden_state_runner",
        "false_negative_count": 0,
        "false_negative_rate": 0.0,
        "false_positive_count": 0,
        "false_positive_rate": 0.0,
        "selected_device": "mps",
        "fallback_runtime_model_path": "models/fallback.json",
        "window_family_mismatch_count": 0,
    }


def _device_preflight_record() -> dict[str, object]:
    return {
        "schema_version": "aegis_introspection.device_preflight/v1",
        "eligible": True,
        "requested_device": "mps",
        "selected_device": "mps",
        "smoke_tensor_device": "mps:0",
    }


def _report_record(
    report_id: str,
    schema_version: str,
    extra: dict[str, object] | None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "report_id": report_id,
        "schema_version": schema_version,
    }
    if extra is not None:
        record.update(extra)
    return record


def _metadata(
    model_id: str = "Qwen/Qwen3-4B",
    revision: str = _IMMUTABLE_MODEL_REVISION,
    layer_count: int = 36,
) -> CiftModelMetadataReport:
    return CiftModelMetadataReport(
        schema_version="aegis_introspection.cift_model_metadata/v1",
        model_id=model_id,
        revision=revision,
        model_type="qwen3",
        hidden_size=2560,
        layer_count=layer_count,
        tokenizer_class="Qwen2Tokenizer",
        tokenizer_vocab_size=151643,
        tokenizer_fingerprint_sha256="a" * 64,
        special_tokens_map_sha256="b" * 64,
        chat_template_present=True,
        chat_template_sha256="c" * 64,
    )


if __name__ == "__main__":
    unittest.main()
