from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
INTROSPECTION_SRC_PATH = INTROSPECTION_ROOT / "src"
RUNTIME_SRC_PATH = WORKSPACE_ROOT / "src"
for source_path in (INTROSPECTION_SRC_PATH, RUNTIME_SRC_PATH):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from aegis_introspection.cift_certification_workflow import (  # noqa: E402
    CiftCertificationWorkflowConfig,
    build_cift_certification_workflow_manifest,
)
from aegis_introspection.cift_certification_workflow_runner import (  # noqa: E402
    CiftCertificationWorkflowRunnerConfig,
    CiftCertificationWorkflowRunnerError,
    run_cift_certification_workflow,
)
from aegis_introspection.cift_model_metadata import CiftModelMetadataConfig, discover_cift_model_metadata  # noqa: E402
from aegis_introspection.cift_release_gate import CiftReleaseGateConfig, evaluate_cift_release_gate  # noqa: E402

from aegis.proxy.cift_certification import (  # noqa: E402
    CiftCertificationBindingConfig,
    validate_cift_certification_binding,
)

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
_VERIFY_EXISTING_SCHEMA_VERSION = "aegis_introspection.cift_local_model_certification_verification/v1"


@dataclass(frozen=True)
class CertifyCiftLocalModelCliConfig:
    repository_root: Path
    certification_id: str
    model_id: str
    revision: str
    corpus_path: Path
    runtime_turns_path: Path
    fallback_runtime_model_path: Path
    output_dir: Path
    workflow_manifest_path: Path
    run_report_path: Path
    training_dataset_id: str
    task_name: str
    positive_label: str
    behavior_id: str
    behavior_description: str
    layer_indices: tuple[int, ...]
    pooling_methods: tuple[str, ...]
    candidate_feature_key: str
    requested_device: str
    prompt_renderer: str
    selected_choice_geometry: str
    selected_choice_readout_token_count: int
    dtype_name: str
    metric_threshold: float
    ablation_delta_threshold: float
    created_at: str
    allow_download: bool
    trust_remote_code: bool
    execute: bool
    allow_sealed_holdout_execution: bool
    overwrite_existing_outputs: bool
    template_values: Mapping[str, str]
    command_timeout_seconds: float


@dataclass(frozen=True)
class VerifyExistingCiftCertificationCliConfig:
    repository_root: Path
    runtime_model_path: Path
    expected_runtime_sha256: str
    certification_manifest_path: Path
    certification_report_path: Path
    certification_artifact_root: Path
    release_gate_report_path: Path
    verification_report_path: Path
    expected_manifest_sha256: str
    expected_report_sha256: str
    expected_release_gate_report_sha256: str
    model_id: str
    revision: str
    required_device: str
    expected_hidden_size: int
    expected_layer_count: int
    expected_tokenizer_sha256: str
    expected_special_tokens_sha256: str
    expected_chat_template_sha256: str
    expected_feature_key: str
    expected_pooling_method: str
    expected_dtype_name: str
    expected_detector_name: str
    expected_extractor_id: str
    expected_feature_source: str
    expected_prompt_renderer: str
    expected_selected_choice_geometry: str
    expected_selected_choice_readout_token_count: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan and run a model-specific CIFT certification workflow for a local hidden-state-capable model."
        )
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--certification-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--runtime-turns", required=True)
    parser.add_argument("--fallback-runtime-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workflow-manifest", required=True)
    parser.add_argument("--run-report", required=True)
    parser.add_argument("--training-dataset-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--positive-label", required=True)
    parser.add_argument("--behavior-id", required=True)
    parser.add_argument("--behavior-description", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--pooling", required=True)
    parser.add_argument("--candidate-feature", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--prompt-renderer", required=True)
    parser.add_argument("--selected-choice-geometry", required=True)
    parser.add_argument("--selected-choice-readout-token-count", required=True, type=int)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--metric-threshold", required=True, type=float)
    parser.add_argument("--ablation-delta-threshold", required=True, type=float)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-sealed-holdout-execution", action="store_true")
    parser.add_argument("--overwrite-existing-outputs", action="store_true")
    parser.add_argument("--command-timeout-seconds", required=True, type=float)
    parser.add_argument(
        "--template-value",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Resolve an operator-supplied workflow template placeholder. May be repeated.",
    )
    return parser


def _build_verify_existing_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an existing model-bound CIFT certification evidence chain without replaying offline evidence."
        )
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--runtime-model", required=True)
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("--certification-manifest", required=True)
    parser.add_argument("--certification-report", required=True)
    parser.add_argument("--certification-artifact-root", required=True)
    parser.add_argument("--release-gate-report", required=True)
    parser.add_argument("--verification-report", required=True)
    parser.add_argument("--certification-manifest-sha256", required=True)
    parser.add_argument("--certification-report-sha256", required=True)
    parser.add_argument("--release-gate-report-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--required-device", required=True)
    parser.add_argument("--expected-hidden-size", required=True, type=int)
    parser.add_argument("--expected-layer-count", required=True, type=int)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--expected-special-tokens-sha256", required=True)
    parser.add_argument("--expected-chat-template-sha256", required=True)
    parser.add_argument("--expected-feature-key", required=True)
    parser.add_argument("--expected-pooling-method", required=True)
    parser.add_argument("--expected-dtype-name", required=True)
    parser.add_argument("--expected-detector-name", required=True)
    parser.add_argument("--expected-extractor-id", required=True)
    parser.add_argument("--expected-feature-source", required=True)
    parser.add_argument("--expected-prompt-renderer", required=True)
    parser.add_argument("--expected-selected-choice-geometry", required=True)
    parser.add_argument("--expected-selected-choice-readout-token-count", required=True, type=int)
    return parser


def _parse_args(argv: Sequence[str]) -> CertifyCiftLocalModelCliConfig:
    namespace = _build_parser().parse_args(argv)
    return CertifyCiftLocalModelCliConfig(
        repository_root=Path(str(namespace.repository_root)),
        certification_id=str(namespace.certification_id),
        model_id=str(namespace.model_id),
        revision=str(namespace.revision),
        corpus_path=Path(str(namespace.corpus)),
        runtime_turns_path=Path(str(namespace.runtime_turns)),
        fallback_runtime_model_path=Path(str(namespace.fallback_runtime_model)),
        output_dir=Path(str(namespace.output_dir)),
        workflow_manifest_path=Path(str(namespace.workflow_manifest)),
        run_report_path=Path(str(namespace.run_report)),
        training_dataset_id=str(namespace.training_dataset_id),
        task_name=str(namespace.task),
        positive_label=str(namespace.positive_label),
        behavior_id=str(namespace.behavior_id),
        behavior_description=str(namespace.behavior_description),
        layer_indices=_parse_layer_indices(str(namespace.layers)),
        pooling_methods=_parse_pooling_methods(str(namespace.pooling)),
        candidate_feature_key=str(namespace.candidate_feature),
        requested_device=str(namespace.device),
        prompt_renderer=str(namespace.prompt_renderer),
        selected_choice_geometry=str(namespace.selected_choice_geometry),
        selected_choice_readout_token_count=_positive_int(
            raw_value=namespace.selected_choice_readout_token_count,
            field_name="--selected-choice-readout-token-count",
        ),
        dtype_name=str(namespace.dtype),
        metric_threshold=float(namespace.metric_threshold),
        ablation_delta_threshold=float(namespace.ablation_delta_threshold),
        created_at=str(namespace.created_at),
        allow_download=bool(namespace.allow_download),
        trust_remote_code=bool(namespace.trust_remote_code),
        execute=bool(namespace.execute),
        allow_sealed_holdout_execution=bool(namespace.allow_sealed_holdout_execution),
        overwrite_existing_outputs=bool(namespace.overwrite_existing_outputs),
        template_values=_parse_template_values(tuple(str(value) for value in namespace.template_value)),
        command_timeout_seconds=_positive_float(
            raw_value=namespace.command_timeout_seconds,
            field_name="--command-timeout-seconds",
        ),
    )


def _parse_verify_existing_args(argv: Sequence[str]) -> VerifyExistingCiftCertificationCliConfig:
    namespace = _build_verify_existing_parser().parse_args(argv)
    return VerifyExistingCiftCertificationCliConfig(
        repository_root=Path(str(namespace.repository_root)),
        runtime_model_path=Path(str(namespace.runtime_model)),
        expected_runtime_sha256=_expected_sha256_arg(
            value=str(namespace.expected_runtime_sha256),
            field_name="--expected-runtime-sha256",
        ),
        certification_manifest_path=Path(str(namespace.certification_manifest)),
        certification_report_path=Path(str(namespace.certification_report)),
        certification_artifact_root=Path(str(namespace.certification_artifact_root)),
        release_gate_report_path=Path(str(namespace.release_gate_report)),
        verification_report_path=Path(str(namespace.verification_report)),
        expected_manifest_sha256=_expected_sha256_arg(
            value=str(namespace.certification_manifest_sha256),
            field_name="--certification-manifest-sha256",
        ),
        expected_report_sha256=_expected_sha256_arg(
            value=str(namespace.certification_report_sha256),
            field_name="--certification-report-sha256",
        ),
        expected_release_gate_report_sha256=_expected_sha256_arg(
            value=str(namespace.release_gate_report_sha256),
            field_name="--release-gate-report-sha256",
        ),
        model_id=str(namespace.model_id),
        revision=str(namespace.revision),
        required_device=str(namespace.required_device),
        expected_hidden_size=_positive_int(
            raw_value=namespace.expected_hidden_size,
            field_name="--expected-hidden-size",
        ),
        expected_layer_count=_positive_int(
            raw_value=namespace.expected_layer_count,
            field_name="--expected-layer-count",
        ),
        expected_tokenizer_sha256=_expected_sha256_arg(
            value=str(namespace.expected_tokenizer_sha256),
            field_name="--expected-tokenizer-sha256",
        ),
        expected_special_tokens_sha256=_expected_sha256_arg(
            value=str(namespace.expected_special_tokens_sha256),
            field_name="--expected-special-tokens-sha256",
        ),
        expected_chat_template_sha256=_expected_sha256_arg(
            value=str(namespace.expected_chat_template_sha256),
            field_name="--expected-chat-template-sha256",
        ),
        expected_feature_key=str(namespace.expected_feature_key),
        expected_pooling_method=str(namespace.expected_pooling_method),
        expected_dtype_name=str(namespace.expected_dtype_name),
        expected_detector_name=str(namespace.expected_detector_name),
        expected_extractor_id=str(namespace.expected_extractor_id),
        expected_feature_source=str(namespace.expected_feature_source),
        expected_prompt_renderer=str(namespace.expected_prompt_renderer),
        expected_selected_choice_geometry=str(namespace.expected_selected_choice_geometry),
        expected_selected_choice_readout_token_count=_positive_int(
            raw_value=namespace.expected_selected_choice_readout_token_count,
            field_name="--expected-selected-choice-readout-token-count",
        ),
    )


def _parse_layer_indices(value: str) -> tuple[int, ...]:
    layer_indices = tuple(int(item.strip()) for item in value.split(",") if item.strip() != "")
    if len(layer_indices) == 0:
        raise CiftCertificationWorkflowRunnerError("--layers must contain at least one integer.")
    return layer_indices


def _parse_pooling_methods(value: str) -> tuple[str, ...]:
    pooling_methods = tuple(item.strip() for item in value.split(",") if item.strip() != "")
    if len(pooling_methods) == 0:
        raise CiftCertificationWorkflowRunnerError("--pooling must contain at least one method.")
    return pooling_methods


def _parse_template_values(raw_values: tuple[str, ...]) -> Mapping[str, str]:
    parsed_values: dict[str, str] = {}
    for raw_value in raw_values:
        if "=" not in raw_value:
            raise CiftCertificationWorkflowRunnerError("--template-value must use NAME=VALUE.")
        name, value = raw_value.split("=", maxsplit=1)
        if name == "" or value == "":
            raise CiftCertificationWorkflowRunnerError("--template-value must include non-empty NAME and VALUE.")
        if name in parsed_values:
            raise CiftCertificationWorkflowRunnerError(f"duplicate --template-value for {name}.")
        parsed_values[name] = value
    return parsed_values


def _positive_int(raw_value: object, field_name: str) -> int:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise CiftCertificationWorkflowRunnerError(f"{field_name} must be an integer.")
    if raw_value < 1:
        raise CiftCertificationWorkflowRunnerError(f"{field_name} must be positive.")
    return raw_value


def _positive_float(raw_value: object, field_name: str) -> float:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        raise CiftCertificationWorkflowRunnerError(f"{field_name} must be numeric.")
    value = float(raw_value)
    if not math.isfinite(value) or value <= 0.0:
        raise CiftCertificationWorkflowRunnerError(f"{field_name} must be a finite positive number.")
    return value


def _expected_sha256_arg(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CiftCertificationWorkflowRunnerError(f"{field_name} must be a lowercase 64-character SHA-256 digest.")
    return value


def _workflow_config(config: CertifyCiftLocalModelCliConfig) -> CiftCertificationWorkflowConfig:
    return CiftCertificationWorkflowConfig(
        certification_id=config.certification_id,
        repository_root=config.repository_root,
        model_id=config.model_id,
        revision=config.revision,
        corpus_path=config.corpus_path,
        runtime_turns_path=config.runtime_turns_path,
        fallback_runtime_model_path=config.fallback_runtime_model_path,
        output_dir=config.output_dir,
        training_dataset_id=config.training_dataset_id,
        task_name=config.task_name,
        positive_label=config.positive_label,
        behavior_id=config.behavior_id,
        behavior_description=config.behavior_description,
        layer_indices=config.layer_indices,
        pooling_methods=config.pooling_methods,
        candidate_feature_key=config.candidate_feature_key,
        requested_device=config.requested_device,
        prompt_renderer=config.prompt_renderer,
        selected_choice_geometry=config.selected_choice_geometry,
        selected_choice_readout_token_count=config.selected_choice_readout_token_count,
        dtype_name=config.dtype_name,
        metric_threshold=config.metric_threshold,
        ablation_delta_threshold=config.ablation_delta_threshold,
        allow_download=config.allow_download,
        trust_remote_code=config.trust_remote_code,
        created_at=config.created_at,
    )


def _runner_config(config: CertifyCiftLocalModelCliConfig) -> CiftCertificationWorkflowRunnerConfig:
    return CiftCertificationWorkflowRunnerConfig(
        repository_root=config.repository_root,
        workflow_manifest_path=config.workflow_manifest_path,
        output_path=config.run_report_path,
        execute=config.execute,
        allow_sealed_holdout_execution=config.allow_sealed_holdout_execution,
        overwrite_existing_outputs=config.overwrite_existing_outputs,
        template_values=config.template_values,
        command_timeout_seconds=config.command_timeout_seconds,
    )


def _release_gate_config(config: VerifyExistingCiftCertificationCliConfig) -> CiftReleaseGateConfig:
    return CiftReleaseGateConfig(
        runtime_model_path=config.runtime_model_path,
        repository_root=config.repository_root,
        required_runtime_prevention_device=config.required_device,
        certification_manifest_path=config.certification_manifest_path,
        certification_report_path=config.certification_report_path,
        certification_artifact_root=config.certification_artifact_root,
        certification_manifest_sha256=config.expected_manifest_sha256,
        certification_report_sha256=config.expected_report_sha256,
        expected_detector_name=config.expected_detector_name,
        expected_extractor_id=config.expected_extractor_id,
        expected_feature_source=config.expected_feature_source,
        expected_selected_choice_readout_token_count=config.expected_selected_choice_readout_token_count,
        allow_embedded_artifact_only=False,
    )


def _binding_config(config: VerifyExistingCiftCertificationCliConfig) -> CiftCertificationBindingConfig:
    return CiftCertificationBindingConfig(
        runtime_model_path=config.runtime_model_path,
        certification_manifest_path=config.certification_manifest_path,
        certification_report_path=config.certification_report_path,
        certification_artifact_root=config.certification_artifact_root,
        release_gate_report_path=config.release_gate_report_path,
        required_device=config.required_device,
        expected_manifest_sha256=config.expected_manifest_sha256,
        expected_report_sha256=config.expected_report_sha256,
        expected_release_gate_report_sha256=config.expected_release_gate_report_sha256,
        expected_detector_name=config.expected_detector_name,
        expected_extractor_id=config.expected_extractor_id,
        expected_feature_source=config.expected_feature_source,
        expected_prompt_renderer=config.expected_prompt_renderer,
        expected_selected_choice_geometry=config.expected_selected_choice_geometry,
        expected_selected_choice_readout_token_count=config.expected_selected_choice_readout_token_count,
    )


def run_cli(config: CertifyCiftLocalModelCliConfig) -> int:
    model_metadata = discover_cift_model_metadata(
        CiftModelMetadataConfig(
            model_id=config.model_id,
            revision=config.revision,
            local_files_only=not config.allow_download,
            trust_remote_code=config.trust_remote_code,
        )
    )
    manifest = build_cift_certification_workflow_manifest(
        config=_workflow_config(config),
        model_metadata=model_metadata,
    )
    config.workflow_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.workflow_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote CIFT certification workflow manifest to {config.workflow_manifest_path}")
    report = run_cift_certification_workflow(_runner_config(config))
    mode = "execute" if config.execute else "dry-run"
    if report.eligible:
        print(f"CIFT local-model certification {mode} passed: {config.model_id}@{config.revision}")
        print(f"Run report: {config.run_report_path}")
        return 0
    if not config.execute and report.plan_eligible:
        print(f"CIFT local-model certification {mode} plan passed: {config.model_id}@{config.revision}")
        print(f"Run report: {config.run_report_path}")
        print("Evidence certification: not eligible")
        for failed_requirement in report.failed_requirements:
            print(f"- {failed_requirement}")
        return 0
    print(f"CIFT local-model certification {mode} failed: {config.model_id}@{config.revision}")
    print(f"Run report: {config.run_report_path}")
    for failed_requirement in report.failed_requirements:
        print(f"- {failed_requirement}")
    return 1


def run_verify_existing_cli(config: VerifyExistingCiftCertificationCliConfig) -> int:
    _expect_file_sha256(
        path=config.runtime_model_path,
        expected_sha256=config.expected_runtime_sha256,
        label="runtime_model",
    )
    runtime_record = _load_json_mapping(path=config.runtime_model_path, label="runtime_model")
    manifest_record = _load_json_mapping(path=config.certification_manifest_path, label="certification_manifest")
    _validate_existing_identity(runtime_record=runtime_record, manifest_record=manifest_record, config=config)
    binding = validate_cift_certification_binding(_binding_config(config))
    release_gate_report = evaluate_cift_release_gate(_release_gate_config(config))
    verification_report = _verification_report_to_json(
        config=config,
        runtime_record=runtime_record,
        binding_runtime_sha256=binding.runtime_sha256,
        binding_certification_id=binding.certification_id,
        release_gate_eligible=release_gate_report.eligible,
        release_gate_evidence_mode=release_gate_report.evidence_mode,
        release_gate_failed_requirements=release_gate_report.failed_requirements,
    )
    _write_verification_report(config=config, verification_report=verification_report)
    if release_gate_report.eligible:
        print(f"CIFT existing certification verified: {config.model_id}@{config.revision}")
        print(f"Verification report: {config.verification_report_path}")
        return 0
    print(f"CIFT existing certification failed release gate: {config.model_id}@{config.revision}")
    print(f"Verification report: {config.verification_report_path}")
    for failed_requirement in release_gate_report.failed_requirements:
        print(f"- {failed_requirement}")
    return 1


def _validate_existing_identity(
    runtime_record: Mapping[str, object],
    manifest_record: Mapping[str, object],
    config: VerifyExistingCiftCertificationCliConfig,
) -> None:
    _expect_string_field(runtime_record, "source_model_id", "runtime_model", config.model_id)
    _expect_string_field(runtime_record, "source_revision", "runtime_model", config.revision)
    _expect_string_field(runtime_record, "source_selected_device", "runtime_model", config.required_device)
    _expect_int_field(runtime_record, "source_hidden_size", "runtime_model", config.expected_hidden_size)
    _expect_int_field(runtime_record, "source_layer_count", "runtime_model", config.expected_layer_count)
    _expect_string_field(
        runtime_record,
        "tokenizer_fingerprint_sha256",
        "runtime_model",
        config.expected_tokenizer_sha256,
    )
    _expect_string_field(
        runtime_record,
        "special_tokens_map_sha256",
        "runtime_model",
        config.expected_special_tokens_sha256,
    )
    _expect_string_field(runtime_record, "chat_template_sha256", "runtime_model", config.expected_chat_template_sha256)
    _expect_string_field(runtime_record, "feature_key", "runtime_model", config.expected_feature_key)
    model_identity = _required_mapping(manifest_record.get("model_identity"), "certification_manifest.model_identity")
    _expect_string_field(model_identity, "model_id", "certification_manifest.model_identity", config.model_id)
    _expect_string_field(model_identity, "revision", "certification_manifest.model_identity", config.revision)
    _expect_int_field(
        model_identity,
        "hidden_size",
        "certification_manifest.model_identity",
        config.expected_hidden_size,
    )
    _expect_int_field(
        model_identity,
        "layer_count",
        "certification_manifest.model_identity",
        config.expected_layer_count,
    )
    _expect_string_field(
        model_identity,
        "tokenizer_fingerprint_sha256",
        "certification_manifest.model_identity",
        config.expected_tokenizer_sha256,
    )
    _expect_string_field(
        model_identity,
        "special_tokens_map_sha256",
        "certification_manifest.model_identity",
        config.expected_special_tokens_sha256,
    )
    _expect_string_field(
        model_identity,
        "chat_template_sha256",
        "certification_manifest.model_identity",
        config.expected_chat_template_sha256,
    )
    training = _required_mapping(manifest_record.get("training"), "certification_manifest.training")
    _expect_string_field(training, "requested_device", "certification_manifest.training", config.required_device)
    _expect_string_field(
        training,
        "candidate_feature_key",
        "certification_manifest.training",
        config.expected_feature_key,
    )
    _expect_string_field(training, "dtype_name", "certification_manifest.training", config.expected_dtype_name)
    _expect_string_field(
        training,
        "prompt_renderer",
        "certification_manifest.training",
        config.expected_prompt_renderer,
    )
    _expect_string_field(
        training,
        "selected_choice_geometry",
        "certification_manifest.training",
        config.expected_selected_choice_geometry,
    )
    _expect_int_field(
        training,
        "selected_choice_readout_token_count",
        "certification_manifest.training",
        config.expected_selected_choice_readout_token_count,
    )
    pooling_methods = training.get("pooling_methods")
    if pooling_methods != [config.expected_pooling_method]:
        raise CiftCertificationWorkflowRunnerError(
            "certification_manifest.training.pooling_methods must exactly match "
            f"[{config.expected_pooling_method!r}]."
        )


def _verification_report_to_json(
    config: VerifyExistingCiftCertificationCliConfig,
    runtime_record: Mapping[str, object],
    binding_runtime_sha256: str,
    binding_certification_id: str,
    release_gate_eligible: bool,
    release_gate_evidence_mode: str,
    release_gate_failed_requirements: tuple[str, ...],
) -> dict[str, JsonValue]:
    return {
        "schema_version": _VERIFY_EXISTING_SCHEMA_VERSION,
        "status": "certified" if release_gate_eligible else "failed",
        "support_claim_status": "model_specific_certified_reference_only",
        "unsupported_model_policy": (
            "Other local models are unsupported until they pass their own calibration, sealed holdout, "
            "live runtime, gateway smoke, and hardened release-gate certification."
        ),
        "certification_id": binding_certification_id,
        "model_identity": {
            "model_id": config.model_id,
            "revision": config.revision,
            "hidden_size": config.expected_hidden_size,
            "layer_count": config.expected_layer_count,
            "tokenizer_fingerprint_sha256": config.expected_tokenizer_sha256,
            "special_tokens_map_sha256": config.expected_special_tokens_sha256,
            "chat_template_sha256": config.expected_chat_template_sha256,
        },
        "runtime_binding": {
            "runtime_model_path": str(config.runtime_model_path),
            "runtime_model_sha256": binding_runtime_sha256,
            "expected_runtime_model_sha256": config.expected_runtime_sha256,
            "model_bundle_id": _optional_string(runtime_record, "model_bundle_id"),
            "candidate_status": _optional_string(runtime_record, "candidate_status"),
            "source_selected_device": config.required_device,
            "feature_key": config.expected_feature_key,
            "pooling_method": config.expected_pooling_method,
            "dtype_name": config.expected_dtype_name,
            "selected_choice_readout_token_count": config.expected_selected_choice_readout_token_count,
        },
        "certification_binding": {
            "certification_manifest_path": str(config.certification_manifest_path),
            "certification_manifest_sha256": config.expected_manifest_sha256,
            "certification_report_path": str(config.certification_report_path),
            "certification_report_sha256": config.expected_report_sha256,
            "certification_artifact_root": str(config.certification_artifact_root),
            "release_gate_report_path": str(config.release_gate_report_path),
            "release_gate_report_sha256": config.expected_release_gate_report_sha256,
        },
        "release_gate": {
            "eligible": release_gate_eligible,
            "production_release_eligible": release_gate_eligible,
            "evidence_mode": release_gate_evidence_mode,
            "failed_requirements": list(release_gate_failed_requirements),
        },
    }


def _write_verification_report(
    config: VerifyExistingCiftCertificationCliConfig,
    verification_report: Mapping[str, JsonValue],
) -> None:
    report_path = _path_inside_repository_root(
        repository_root=config.repository_root,
        path=config.verification_report_path,
        label="verification_report",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(verification_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_inside_repository_root(repository_root: Path, path: Path, label: str) -> Path:
    root = repository_root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise CiftCertificationWorkflowRunnerError(f"{label} must stay inside repository root.") from exc
    return resolved_path


def _expect_file_sha256(path: Path, expected_sha256: str, label: str) -> None:
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise CiftCertificationWorkflowRunnerError(
            f"{label} sha256 mismatch: expected {expected_sha256}, got {actual_sha256}."
        )


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise CiftCertificationWorkflowRunnerError(f"required file does not exist: {path}.")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_mapping(path: Path, label: str) -> Mapping[str, object]:
    if not path.is_file():
        raise CiftCertificationWorkflowRunnerError(f"{label} does not exist: {path}.")
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CiftCertificationWorkflowRunnerError(f"{label} contains invalid JSON: {exc.msg}.") from exc
    if not isinstance(decoded, dict):
        raise CiftCertificationWorkflowRunnerError(f"{label} must contain a JSON object.")
    return decoded


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CiftCertificationWorkflowRunnerError(f"{label} must be a JSON object.")
    return value


def _expect_string_field(record: Mapping[str, object], field_name: str, label: str, expected_value: str) -> None:
    value = record.get(field_name)
    if value != expected_value:
        raise CiftCertificationWorkflowRunnerError(
            f"{label}.{field_name} must be {expected_value!r}, got {value!r}."
        )


def _expect_int_field(record: Mapping[str, object], field_name: str, label: str, expected_value: int) -> None:
    value = record.get(field_name)
    if value != expected_value:
        raise CiftCertificationWorkflowRunnerError(f"{label}.{field_name} must be {expected_value}, got {value!r}.")


def _optional_string(record: Mapping[str, object], field_name: str) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CiftCertificationWorkflowRunnerError(f"{field_name} must be a string when present.")
    return value


def main(argv: Sequence[str]) -> int:
    try:
        if len(argv) > 0 and argv[0] == "verify-existing":
            return run_verify_existing_cli(_parse_verify_existing_args(argv[1:]))
        return run_cli(_parse_args(argv))
    except (CiftCertificationWorkflowRunnerError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
