from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import ModuleType

from aegis_introspection.cift_model_metadata import CiftModelMetadataReport

from aegis.cift_contract import (
    CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
    CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
)

_IMMUTABLE_MODEL_REVISION = "0123456789abcdef0123456789abcdef01234567"
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "certify_cift_local_model.py"


class CertifyCiftLocalModelCliTest(unittest.TestCase):
    def test_dry_run_plans_generic_model_specific_certification_workflow(self) -> None:
        module = _load_cli_module()
        original_discover = module.discover_cift_model_metadata
        module.discover_cift_model_metadata = _fake_discover_cift_model_metadata
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                corpus_path = root / "data" / "calibration.jsonl"
                runtime_turns_path = root / "data" / "runtime_turns.jsonl"
                fallback_runtime_path = root / "models" / "fallback_runtime.json"
                workflow_manifest_path = root / "reports" / "workflow.json"
                run_report_path = root / "reports" / "workflow_run.json"
                output_dir = root / "introspection" / "data"
                corpus_path.parent.mkdir(parents=True)
                fallback_runtime_path.parent.mkdir(parents=True)
                corpus_path.write_text('{"id":"calibration-safe"}\n', encoding="utf-8")
                runtime_turns_path.write_text('{"trace_id":"runtime-safe"}\n', encoding="utf-8")
                fallback_runtime_path.write_text("{}\n", encoding="utf-8")

                exit_code = module.main(
                    (
                        "--repository-root",
                        str(root),
                        "--certification-id",
                        "synthetic_local_model_cert",
                        "--model-id",
                        "Local/Test-Model",
                        "--revision",
                        _IMMUTABLE_MODEL_REVISION,
                        "--corpus",
                        str(corpus_path),
                        "--runtime-turns",
                        str(runtime_turns_path),
                        "--fallback-runtime-model",
                        str(fallback_runtime_path),
                        "--output-dir",
                        str(output_dir),
                        "--workflow-manifest",
                        str(workflow_manifest_path),
                        "--run-report",
                        str(run_report_path),
                        "--training-dataset-id",
                        "synthetic-training",
                        "--task",
                        "safe_secret_vs_exfiltration",
                        "--positive-label",
                        "exfiltration_intent",
                        "--behavior-id",
                        "secret-exfiltration-intent",
                        "--behavior-description",
                        "User request attempts to move a protected secret into an external channel.",
                        "--layers",
                        "1,2",
                        "--pooling",
                        "selected_choice_window",
                        "--candidate-feature",
                        "selected_choice_window_layer_1",
                        "--device",
                        "mps",
                        "--prompt-renderer",
                        CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
                        "--selected-choice-geometry",
                        CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
                        "--selected-choice-readout-token-count",
                        "4",
                        "--dtype",
                        "device",
                        "--metric-threshold",
                        "1.0",
                        "--ablation-delta-threshold",
                        "0.0",
                        "--created-at",
                        "2026-06-25T00:00:00Z",
                        "--command-timeout-seconds",
                        "30",
                    )
                )

                workflow_manifest = json.loads(workflow_manifest_path.read_text(encoding="utf-8"))
                run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
        finally:
            module.discover_cift_model_metadata = original_discover

        self.assertEqual(0, exit_code)
        self.assertEqual("calibration-ready", workflow_manifest["support_state"])
        self.assertEqual("not_certified_until_release_gate_passes", workflow_manifest["support_claim_status"])
        self.assertEqual("calibration-ready", workflow_manifest["model_identity"]["support_state"])
        self.assertEqual("Local/Test-Model", workflow_manifest["model_identity"]["model_id"])
        self.assertEqual("selected_choice_window_layer_1", workflow_manifest["training"]["candidate_feature_key"])
        self.assertEqual("dry_run", run_report["mode"])
        self.assertFalse(run_report["certification_eligible"])

    def test_verify_existing_certification_validates_model_bound_evidence(self) -> None:
        module = _load_cli_module()
        original_binding = module.validate_cift_certification_binding
        original_release_gate = module.evaluate_cift_release_gate
        calls: dict[str, object] = {}
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                runtime_path = root / "introspection" / "data" / "models" / "runtime.json"
                manifest_path = root / "introspection" / "data" / "reports" / "workflow.json"
                certification_report_path = root / "introspection" / "data" / "reports" / "workflow_run.json"
                release_gate_report_path = root / "introspection" / "data" / "reports" / "release_gate.json"
                verification_report_path = root / "introspection" / "data" / "reports" / "verification.json"
                runtime_path.parent.mkdir(parents=True)
                manifest_path.parent.mkdir(parents=True)
                runtime_path.write_text(
                    json.dumps(
                        {
                            "candidate_status": "runtime_candidate",
                            "chat_template_sha256": "d" * 64,
                            "feature_key": "selected_choice_window_layer_1",
                            "model_bundle_id": "synthetic_bundle",
                            "schema_version": "aegis.cift_runtime_linear/v1",
                            "source_hidden_size": 128,
                            "source_layer_count": 4,
                            "source_model_id": "Local/Test-Model",
                            "source_revision": _IMMUTABLE_MODEL_REVISION,
                            "source_selected_device": "mps",
                            "special_tokens_map_sha256": "c" * 64,
                            "tokenizer_fingerprint_sha256": "b" * 64,
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                manifest_path.write_text(
                    json.dumps(
                        {
                            "model_identity": {
                                "chat_template_sha256": "d" * 64,
                                "hidden_size": 128,
                                "layer_count": 4,
                                "model_id": "Local/Test-Model",
                                "revision": _IMMUTABLE_MODEL_REVISION,
                                "special_tokens_map_sha256": "c" * 64,
                                "tokenizer_fingerprint_sha256": "b" * 64,
                            },
                            "training": {
                                "candidate_feature_key": "selected_choice_window_layer_1",
                                "dtype_name": "device",
                                "pooling_methods": ["selected_choice_window"],
                                "prompt_renderer": CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
                                "requested_device": "mps",
                                "selected_choice_geometry": CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
                                "selected_choice_readout_token_count": 4,
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                certification_report_path.write_text("{}\n", encoding="utf-8")
                release_gate_report_path.write_text("{}\n", encoding="utf-8")
                runtime_sha256 = _sha256_file(runtime_path)

                def fake_validate_binding(config: object) -> object:
                    calls["binding_config"] = config
                    return types.SimpleNamespace(
                        certification_id="synthetic_local_model_cert",
                        runtime_sha256=runtime_sha256,
                    )

                def fake_evaluate_release_gate(config: object) -> object:
                    calls["release_gate_config"] = config
                    return types.SimpleNamespace(
                        eligible=True,
                        evidence_mode="certification_bound",
                        failed_requirements=(),
                    )

                module.validate_cift_certification_binding = fake_validate_binding
                module.evaluate_cift_release_gate = fake_evaluate_release_gate

                exit_code = module.main(
                    _verify_existing_args(
                        root=root,
                        runtime_path=runtime_path,
                        runtime_sha256=runtime_sha256,
                        manifest_path=manifest_path,
                        certification_report_path=certification_report_path,
                        release_gate_report_path=release_gate_report_path,
                        verification_report_path=verification_report_path,
                    )
                )
                verification_report = json.loads(verification_report_path.read_text(encoding="utf-8"))
        finally:
            module.validate_cift_certification_binding = original_binding
            module.evaluate_cift_release_gate = original_release_gate

        self.assertEqual(0, exit_code)
        self.assertEqual("certified", verification_report["status"])
        self.assertEqual("runtime-enforceable", verification_report["support_state"])
        self.assertEqual("certification_bound", verification_report["release_gate"]["evidence_mode"])
        self.assertEqual(runtime_sha256, verification_report["runtime_binding"]["runtime_model_sha256"])
        binding_config = calls["binding_config"]
        release_gate_config = calls["release_gate_config"]
        self.assertEqual("mps", binding_config.required_device)
        self.assertEqual("trusted-activation-sidecar", binding_config.expected_extractor_id)
        self.assertEqual("a" * 64, binding_config.expected_manifest_sha256)
        self.assertEqual("f" * 64, binding_config.expected_release_gate_report_sha256)
        self.assertEqual("mps", release_gate_config.required_runtime_prevention_device)
        self.assertFalse(release_gate_config.allow_embedded_artifact_only)

    def test_certification_writes_failed_report_when_discovery_fails(self) -> None:
        module = _load_cli_module()
        original_discover = module.discover_cift_model_metadata

        def fail_discovery(config: object) -> object:
            raise ValueError("model files are not available locally")

        module.discover_cift_model_metadata = fail_discovery
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                corpus_path = root / "data" / "calibration.jsonl"
                runtime_turns_path = root / "data" / "runtime_turns.jsonl"
                fallback_runtime_path = root / "models" / "fallback_runtime.json"
                workflow_manifest_path = root / "reports" / "workflow.json"
                run_report_path = root / "reports" / "workflow_run.json"
                output_dir = root / "introspection" / "data"
                corpus_path.parent.mkdir(parents=True)
                fallback_runtime_path.parent.mkdir(parents=True)
                corpus_path.write_text('{"id":"calibration-safe"}\n', encoding="utf-8")
                runtime_turns_path.write_text('{"trace_id":"runtime-safe"}\n', encoding="utf-8")
                fallback_runtime_path.write_text("{}\n", encoding="utf-8")

                exit_code = module.run_cli(
                    module.CertifyCiftLocalModelCliConfig(
                        repository_root=root,
                        certification_id="qwen3_0_6b_portability_check",
                        model_id="Qwen/Qwen3-0.6B",
                        revision=_IMMUTABLE_MODEL_REVISION,
                        corpus_path=corpus_path,
                        runtime_turns_path=runtime_turns_path,
                        selected_choice_runtime_model_path=fallback_runtime_path,
                        output_dir=output_dir,
                        workflow_manifest_path=workflow_manifest_path,
                        run_report_path=run_report_path,
                        training_dataset_id="synthetic-training",
                        task_name="safe_secret_vs_exfiltration",
                        positive_label="exfiltration_intent",
                        behavior_id="secret-exfiltration-intent",
                        behavior_description=(
                            "User request attempts to move a protected secret into an external channel."
                        ),
                        layer_indices=(1, 2),
                        pooling_methods=("selected_choice_window",),
                        candidate_feature_key="selected_choice_window_layer_1",
                        requested_device="mps",
                        prompt_renderer=CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
                        selected_choice_geometry=CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
                        selected_choice_readout_token_count=4,
                        dtype_name="device",
                        metric_threshold=1.0,
                        ablation_delta_threshold=0.0,
                        created_at="2026-06-25T00:00:00Z",
                        allow_download=False,
                        trust_remote_code=False,
                        execute=False,
                        allow_sealed_holdout_execution=False,
                        overwrite_existing_outputs=False,
                        template_values={},
                        command_timeout_seconds=30.0,
                    )
                )
                run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
                workflow_manifest_exists = workflow_manifest_path.exists()
        finally:
            module.discover_cift_model_metadata = original_discover

        self.assertEqual(1, exit_code)
        self.assertFalse(workflow_manifest_exists)
        self.assertEqual("failed certification", run_report["support_state"])
        self.assertFalse(run_report["certification_eligible"])
        self.assertIn("model_metadata_discovery_failed", run_report["failed_requirements"][0])
        self.assertEqual("Qwen/Qwen3-0.6B", run_report["model_identity"]["model_id"])

    def test_verify_existing_certification_fails_on_model_identity_mismatch(self) -> None:
        module = _load_cli_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runtime_path = root / "runtime.json"
            manifest_path = root / "workflow.json"
            certification_report_path = root / "workflow_run.json"
            release_gate_report_path = root / "release_gate.json"
            verification_report_path = root / "verification.json"
            runtime_path.write_text(
                json.dumps(
                    {
                        "chat_template_sha256": "d" * 64,
                        "feature_key": "selected_choice_window_layer_1",
                        "source_hidden_size": 128,
                        "source_layer_count": 4,
                        "source_model_id": "Local/Wrong-Model",
                        "source_revision": _IMMUTABLE_MODEL_REVISION,
                        "source_selected_device": "mps",
                        "special_tokens_map_sha256": "c" * 64,
                        "tokenizer_fingerprint_sha256": "b" * 64,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "model_identity": {
                            "chat_template_sha256": "d" * 64,
                            "hidden_size": 128,
                            "layer_count": 4,
                            "model_id": "Local/Test-Model",
                            "revision": _IMMUTABLE_MODEL_REVISION,
                            "special_tokens_map_sha256": "c" * 64,
                            "tokenizer_fingerprint_sha256": "b" * 64,
                        },
                        "training": {
                            "candidate_feature_key": "selected_choice_window_layer_1",
                            "dtype_name": "device",
                            "pooling_methods": ["selected_choice_window"],
                            "prompt_renderer": CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
                            "requested_device": "mps",
                            "selected_choice_geometry": CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
                            "selected_choice_readout_token_count": 4,
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            certification_report_path.write_text("{}\n", encoding="utf-8")
            release_gate_report_path.write_text("{}\n", encoding="utf-8")

            exit_code = module.main(
                _verify_existing_args(
                    root=root,
                    runtime_path=runtime_path,
                    runtime_sha256=_sha256_file(runtime_path),
                    manifest_path=manifest_path,
                    certification_report_path=certification_report_path,
                    release_gate_report_path=release_gate_report_path,
                    verification_report_path=verification_report_path,
                )
            )

        self.assertEqual(1, exit_code)
        self.assertFalse(verification_report_path.exists())


def _verify_existing_args(
    root: Path,
    runtime_path: Path,
    runtime_sha256: str,
    manifest_path: Path,
    certification_report_path: Path,
    release_gate_report_path: Path,
    verification_report_path: Path,
) -> tuple[str, ...]:
    return (
        "verify-existing",
        "--repository-root",
        str(root),
        "--runtime-model",
        str(runtime_path),
        "--expected-runtime-sha256",
        runtime_sha256,
        "--certification-manifest",
        str(manifest_path),
        "--certification-report",
        str(certification_report_path),
        "--certification-artifact-root",
        str(root),
        "--release-gate-report",
        str(release_gate_report_path),
        "--verification-report",
        str(verification_report_path),
        "--certification-manifest-sha256",
        "a" * 64,
        "--certification-report-sha256",
        "e" * 64,
        "--release-gate-report-sha256",
        "f" * 64,
        "--model-id",
        "Local/Test-Model",
        "--revision",
        _IMMUTABLE_MODEL_REVISION,
        "--required-device",
        "mps",
        "--expected-hidden-size",
        "128",
        "--expected-layer-count",
        "4",
        "--expected-tokenizer-sha256",
        "b" * 64,
        "--expected-special-tokens-sha256",
        "c" * 64,
        "--expected-chat-template-sha256",
        "d" * 64,
        "--expected-feature-key",
        "selected_choice_window_layer_1",
        "--expected-pooling-method",
        "selected_choice_window",
        "--expected-dtype-name",
        "device",
        "--expected-detector-name",
        "cift_runtime",
        "--expected-extractor-id",
        "trusted-activation-sidecar",
        "--expected-feature-source",
        "self_hosted_activation_extractor",
        "--expected-prompt-renderer",
        CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
        "--expected-selected-choice-geometry",
        CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
        "--expected-selected-choice-readout-token-count",
        "4",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cli_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("certify_cift_local_model_test_module", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("Unable to load certify_cift_local_model.py.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["certify_cift_local_model_test_module"] = module
    spec.loader.exec_module(module)
    return module


def _fake_discover_cift_model_metadata(config: object) -> CiftModelMetadataReport:
    return CiftModelMetadataReport(
        schema_version="aegis_introspection.cift_model_metadata/v1",
        support_state="calibration-ready",
        model_id="Local/Test-Model",
        revision=_IMMUTABLE_MODEL_REVISION,
        resolved_revision=_IMMUTABLE_MODEL_REVISION,
        model_type="local_test",
        hidden_size=128,
        layer_count=4,
        requested_device="mps",
        selected_device="mps",
        dtype_name="device",
        resolved_torch_dtype="torch.float16",
        hidden_state_support="configurable_output_hidden_states",
        hidden_state_capable=True,
        selected_readout_candidates=("selected_choice_window_layer_1",),
        failure_reason=None,
        tokenizer_class="LocalTokenizer",
        tokenizer_vocab_size=32000,
        tokenizer_fingerprint_sha256="b" * 64,
        special_tokens_map_sha256="c" * 64,
        chat_template_present=True,
        chat_template_sha256="d" * 64,
    )
