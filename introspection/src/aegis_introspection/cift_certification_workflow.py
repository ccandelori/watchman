from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

from aegis.cift_contract import (
    CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
    CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
    CIFT_SUPPORT_STATE_CALIBRATION_READY,
    CIFT_SUPPORT_STATE_CERTIFIED,
    is_cift_immutable_model_revision,
)
from aegis_introspection.cift_model_metadata import (
    CiftModelMetadataReport,
    cift_model_metadata_report_to_json,
)

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
_CERTIFICATION_WORKFLOW_COMMAND_TIMEOUT_SECONDS = "30.0"


class CiftCertificationWorkflowError(ValueError):
    """Raised when a CIFT certification workflow manifest cannot be built."""


@dataclass(frozen=True)
class CiftCertificationWorkflowConfig:
    certification_id: str
    repository_root: Path
    model_id: str
    revision: str
    corpus_path: Path
    runtime_turns_path: Path
    selected_choice_runtime_model_path: Path
    output_dir: Path
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
    allow_download: bool
    trust_remote_code: bool
    created_at: str


@dataclass(frozen=True)
class CiftCertificationEvidenceManifestConfig:
    certification_id: str
    repository_root: Path
    created_at: str
    behavior_id: str
    behavior_description: str
    requested_device: str
    prompt_renderer: str
    selected_choice_geometry: str
    selected_choice_readout_token_count: int
    dtype_name: str
    metric_threshold: float
    ablation_delta_threshold: float
    model_metadata_report_path: Path
    activation_artifact_path: Path
    linear_bundle_path: Path
    grouped_head_to_head_report_path: Path
    calibration_report_path: Path
    feature_ablation_report_path: Path
    patching_report_path: Path
    failure_cases_report_path: Path
    lineage_report_path: Path
    device_preflight_report_path: Path
    live_runtime_prevention_report_path: Path
    sealed_holdout_metric_path: Path
    gateway_smoke_report_path: Path
    paper_mlp_runtime_prevention_report_path: Path
    paper_mlp_sealed_holdout_metric_path: Path
    live_head_to_head_report_path: Path
    promotion_evidence_path: Path
    promoted_runtime_artifact_path: Path
    promotion_report_output_dir: Path
    evidence_chain_verification_report_path: Path


def build_cift_certification_workflow_manifest(
    config: CiftCertificationWorkflowConfig,
    model_metadata: CiftModelMetadataReport,
) -> dict[str, JsonValue]:
    _validate_config(config)
    _validate_model_metadata(config=config, model_metadata=model_metadata)
    artifact_paths = _planned_artifact_paths(config)
    command_plan = _command_plan(config=config, model_metadata=model_metadata, artifact_paths=artifact_paths)
    required_evidence_artifacts = _required_evidence_artifacts(config=config, artifact_paths=artifact_paths)
    return {
        "schema_version": "aegis_introspection.cift_certification_workflow/v1",
        "certification_id": config.certification_id,
        "created_at": config.created_at,
        "status": "planned",
        "support_state": CIFT_SUPPORT_STATE_CALIBRATION_READY,
        "support_claim_status": "not_certified_until_release_gate_passes",
        "model_identity": cift_model_metadata_report_to_json(model_metadata),
        "corpus": {
            "path": _repository_relative_path(config.repository_root, config.corpus_path),
            "sha256": _sha256_file(config.corpus_path),
            "role": "calibration_and_grouped_cv_source",
        },
        "runtime_turns": {
            "path": _repository_relative_path(config.repository_root, config.runtime_turns_path),
            "sha256": _sha256_file(config.runtime_turns_path),
            "role": "live_runtime_prevention_source",
        },
        "training": {
            "training_dataset_id": config.training_dataset_id,
            "task_name": config.task_name,
            "positive_label": config.positive_label,
            "behavior_id": config.behavior_id,
            "behavior_description": config.behavior_description,
            "layer_indices": list(config.layer_indices),
            "pooling_methods": list(config.pooling_methods),
            "candidate_feature_key": config.candidate_feature_key,
            "requested_device": config.requested_device,
            "prompt_renderer": config.prompt_renderer,
            "selected_choice_geometry": config.selected_choice_geometry,
            "selected_choice_readout_token_count": config.selected_choice_readout_token_count,
            "dtype_name": config.dtype_name,
            "metric_threshold": config.metric_threshold,
            "ablation_delta_threshold": config.ablation_delta_threshold,
            "selected_choice_runtime_model_path": _repository_relative_path(
                config.repository_root,
                config.selected_choice_runtime_model_path,
            ),
        },
        "planned_artifacts": artifact_paths,
        "required_evidence_artifacts": required_evidence_artifacts,
        "command_plan": command_plan,
        "required_release_evidence": [
            "device_preflight_passed",
            "model_metadata_discovered",
            "activation_artifact_bound_to_model_identity",
            "layer_window_sweep",
            "linear_vs_paper_mlp_head_to_head",
            "grouped_cv_metrics",
            "sealed_holdout_metrics",
            "false_negative_rate_reported",
            "false_positive_rate_reported",
            "live_hidden_state_runtime_prevention",
            "live_gateway_sidecar_runtime_prevention",
            "fail_closed_hidden_state_and_selected_choice_metadata",
            "promotion_evidence_materialized",
            "runtime_artifact_bound_to_model_tokenizer_template_hashes",
            "evidence_chain_identity_verification",
            "hardened_release_gate_pass",
        ],
        "release_rule": (
            "Do not claim support for this model until every required_release_evidence item is present "
            "and the promoted runtime artifact passes the hardened release gate."
        ),
    }


def build_cift_certification_evidence_manifest(
    config: CiftCertificationEvidenceManifestConfig,
) -> dict[str, JsonValue]:
    _validate_evidence_manifest_config(config)
    runtime_model = _json_object_from_path(config.promoted_runtime_artifact_path, "promoted runtime artifact")
    runtime_prevention = _json_object_from_path(config.live_runtime_prevention_report_path, "runtime prevention report")
    paper_mlp_runtime_prevention = _json_object_from_path(
        config.paper_mlp_runtime_prevention_report_path,
        "paper MLP runtime prevention report",
    )
    _validate_requested_runtime_prevention_device(
        requested_device=config.requested_device,
        runtime_prevention=runtime_prevention,
        label="runtime prevention report",
    )
    _validate_requested_runtime_prevention_device(
        requested_device=config.requested_device,
        runtime_prevention=paper_mlp_runtime_prevention,
        label="paper MLP runtime prevention report",
    )
    model_metadata = _model_metadata_report_from_path(config.model_metadata_report_path)
    _validate_model_device_policy(model_id=model_metadata.model_id, requested_device=config.requested_device)
    _validate_model_revision_policy(model_metadata.revision, "model metadata revision")
    _validate_runtime_model_revision_binding(runtime_model=runtime_model, model_metadata=model_metadata)
    training_dataset_id = _json_string(runtime_model, "training_dataset_id", "promoted runtime artifact")
    task_name = _json_string(runtime_model, "task_name", "promoted runtime artifact")
    positive_label = _json_string(runtime_model, "positive_label", "promoted runtime artifact")
    candidate_feature_key = _json_string(runtime_model, "feature_key", "promoted runtime artifact")
    _validate_candidate_feature_layer(
        candidate_feature_key=candidate_feature_key,
        model_metadata=model_metadata,
        field_name="promoted runtime artifact feature_key",
    )
    planned_artifacts = _evidence_manifest_planned_artifacts(config)
    return {
        "schema_version": "aegis_introspection.cift_certification_workflow/v1",
        "certification_id": config.certification_id,
        "created_at": config.created_at,
        "status": "evidence_bound",
        "support_state": CIFT_SUPPORT_STATE_CERTIFIED,
        "support_claim_status": "model_specific_certified_only_if_release_gate_passes",
        "model_identity": cift_model_metadata_report_to_json(model_metadata),
        "training": {
            "training_dataset_id": training_dataset_id,
            "task_name": task_name,
            "positive_label": positive_label,
            "behavior_id": config.behavior_id,
            "behavior_description": config.behavior_description,
            "layer_indices": [_layer_index_from_feature_key(candidate_feature_key)],
            "pooling_methods": [_pooling_method_from_feature_key(candidate_feature_key)],
            "candidate_feature_key": candidate_feature_key,
            "requested_device": config.requested_device,
            "prompt_renderer": config.prompt_renderer,
            "selected_choice_geometry": config.selected_choice_geometry,
            "selected_choice_readout_token_count": config.selected_choice_readout_token_count,
            "dtype_name": config.dtype_name,
            "metric_threshold": config.metric_threshold,
            "ablation_delta_threshold": config.ablation_delta_threshold,
            "selected_choice_runtime_model_path": _json_string(
                runtime_prevention,
                "selected_choice_runtime_model_path",
                "runtime prevention report",
            ),
            "fallback_runtime_model_path": _json_string(
                runtime_prevention,
                "fallback_runtime_model_path",
                "runtime prevention report",
            ),
        },
        "planned_artifacts": planned_artifacts,
        "required_evidence_artifacts": _evidence_manifest_artifacts(config),
        "command_plan": [],
        "required_release_evidence": [
            "device_preflight_passed",
            "model_metadata_discovered",
            "activation_artifact_bound_to_model_identity",
            "linear_vs_paper_mlp_head_to_head",
            "grouped_cv_metrics",
            "sealed_holdout_metrics",
            "paper_mlp_sealed_holdout_metrics",
            "false_negative_rate_reported",
            "false_positive_rate_reported",
            "live_hidden_state_runtime_prevention",
            "paper_mlp_live_hidden_state_runtime_prevention",
            "live_gateway_sidecar_runtime_prevention",
            "fail_closed_hidden_state_and_selected_choice_metadata",
            "promotion_evidence_materialized",
            "runtime_artifact_bound_to_model_tokenizer_template_hashes",
            "evidence_chain_identity_verification",
            "hardened_release_gate_pass",
        ],
        "release_rule": (
            "This manifest binds exact model-specific CIFT evidence. Do not claim broader model support unless that "
            "model has its own calibration evidence and release-gate pass."
        ),
    }


def _planned_artifact_paths(config: CiftCertificationWorkflowConfig) -> dict[str, JsonValue]:
    layer_suffix = _layer_suffix(config.layer_indices)
    model_prefix = f"cift_{config.certification_id}"
    activation_path = config.output_dir / "activations" / f"{config.certification_id}_windows_{layer_suffix}.pt"
    reports_dir = config.output_dir / "reports"
    models_dir = config.output_dir / "models"
    promotion_reports_dir = reports_dir / "cift_promotion" / f"{config.certification_id}_linear_promoted_runtime"
    return {
        "model_metadata_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_model_metadata_v1.json",
        ),
        "activation_artifact_path": _repository_relative_path(config.repository_root, activation_path),
        "feature_ablation_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_feature_ablation_v1.json",
        ),
        "calibration_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_calibration_v1.json",
        ),
        "head_to_head_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_linear_vs_paper_mlp_v1.json",
        ),
        "linear_bundle_path": _repository_relative_path(
            config.repository_root,
            models_dir / f"{model_prefix}_linear_candidate_v1.pkl",
        ),
        "paper_mlp_bundle_path": _repository_relative_path(
            config.repository_root,
            models_dir / f"{model_prefix}_paper_mlp_candidate_v1.pkl",
        ),
        "paper_mlp_runtime_preview_path": _repository_relative_path(
            config.repository_root,
            models_dir / f"{model_prefix}_paper_mlp_runtime_preview_v1.json",
        ),
        "live_runtime_prevention_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_linear_runtime_prevention_v1.json",
        ),
        "live_runtime_prevention_summary_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_linear_runtime_prevention_v1_summary.md",
        ),
        "gateway_smoke_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_gateway_smoke_v1.json",
        ),
        "paper_mlp_runtime_prevention_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_paper_mlp_runtime_prevention_v1.json",
        ),
        "paper_mlp_runtime_prevention_summary_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_paper_mlp_runtime_prevention_v1_summary.md",
        ),
        "linear_sealed_holdout_metric_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_linear_sealed_holdout_metric_v1.json",
        ),
        "paper_mlp_sealed_holdout_metric_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_paper_mlp_sealed_holdout_metric_v1.json",
        ),
        "live_sealed_head_to_head_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_linear_vs_paper_mlp_live_sealed_v1.json",
        ),
        "patching_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_causal_patching_v1.json",
        ),
        "failure_cases_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_failure_cases_v1.json",
        ),
        "lineage_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_lineage_v1.json",
        ),
        "device_preflight_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_device_preflight_v1.json",
        ),
        "promotion_report_output_dir": _repository_relative_path(
            config.repository_root,
            promotion_reports_dir,
        ),
        "promotion_evidence_path": _repository_relative_path(
            config.repository_root,
            reports_dir
            / "cift_promotion"
            / f"{config.certification_id}_linear_promoted_runtime_promotion_evidence_v1.json",
        ),
        "evidence_chain_verification_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_evidence_chain_verification_v1.json",
        ),
        "certification_manifest_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_certification_workflow_v1.json",
        ),
        "certification_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_certification_workflow_run_v1.json",
        ),
        "deployment_env_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_strict_deployment_env.sh",
        ),
        "release_gate_report_path": _repository_relative_path(
            config.repository_root,
            reports_dir / f"{config.certification_id}_release_gate_v1.json",
        ),
        "promoted_runtime_artifact_path": _repository_relative_path(
            config.repository_root,
            models_dir / f"{model_prefix}_promoted_runtime_v1.json",
        ),
    }


def _evidence_manifest_planned_artifacts(config: CiftCertificationEvidenceManifestConfig) -> dict[str, JsonValue]:
    return {
        "model_metadata_report_path": _repository_relative_path(
            config.repository_root, config.model_metadata_report_path
        ),
        "activation_artifact_path": _repository_relative_path(config.repository_root, config.activation_artifact_path),
        "linear_bundle_path": _repository_relative_path(config.repository_root, config.linear_bundle_path),
        "head_to_head_report_path": _repository_relative_path(
            config.repository_root,
            config.grouped_head_to_head_report_path,
        ),
        "calibration_report_path": _repository_relative_path(config.repository_root, config.calibration_report_path),
        "feature_ablation_report_path": _repository_relative_path(
            config.repository_root, config.feature_ablation_report_path
        ),
        "patching_report_path": _repository_relative_path(config.repository_root, config.patching_report_path),
        "failure_cases_report_path": _repository_relative_path(
            config.repository_root, config.failure_cases_report_path
        ),
        "lineage_report_path": _repository_relative_path(config.repository_root, config.lineage_report_path),
        "device_preflight_report_path": _repository_relative_path(
            config.repository_root, config.device_preflight_report_path
        ),
        "live_runtime_prevention_report_path": _repository_relative_path(
            config.repository_root,
            config.live_runtime_prevention_report_path,
        ),
        "linear_sealed_holdout_metric_path": _repository_relative_path(
            config.repository_root,
            config.sealed_holdout_metric_path,
        ),
        "gateway_smoke_report_path": _repository_relative_path(
            config.repository_root,
            config.gateway_smoke_report_path,
        ),
        "paper_mlp_runtime_prevention_report_path": _repository_relative_path(
            config.repository_root,
            config.paper_mlp_runtime_prevention_report_path,
        ),
        "paper_mlp_sealed_holdout_metric_path": _repository_relative_path(
            config.repository_root,
            config.paper_mlp_sealed_holdout_metric_path,
        ),
        "live_sealed_head_to_head_report_path": _repository_relative_path(
            config.repository_root,
            config.live_head_to_head_report_path,
        ),
        "promotion_report_output_dir": _repository_relative_path(
            config.repository_root, config.promotion_report_output_dir
        ),
        "promotion_evidence_path": _repository_relative_path(config.repository_root, config.promotion_evidence_path),
        "evidence_chain_verification_report_path": _repository_relative_path(
            config.repository_root,
            config.evidence_chain_verification_report_path,
        ),
        "promoted_runtime_artifact_path": _repository_relative_path(
            config.repository_root,
            config.promoted_runtime_artifact_path,
        ),
    }


def _evidence_manifest_artifacts(config: CiftCertificationEvidenceManifestConfig) -> list[JsonValue]:
    artifacts = [
        _current_artifact(
            config=config,
            artifact_key="model_metadata_report_path",
            artifact_kind="json_report",
            role="model_metadata",
            path=config.model_metadata_report_path,
            producer_step_id="discover_model_metadata",
            consumer_step_ids=("verify_evidence_chain_identity",),
            report_id=None,
            schema_version="aegis_introspection.cift_model_metadata/v1",
        ),
        _current_artifact(
            config=config,
            artifact_key="activation_artifact_path",
            artifact_kind="activation_tensor",
            role="calibration_activation_artifact",
            path=config.activation_artifact_path,
            producer_step_id="extract_activation_artifact",
            consumer_step_ids=("train_linear_runtime_candidate",),
            report_id=None,
            schema_version=None,
        ),
        _current_artifact(
            config=config,
            artifact_key="linear_bundle_path",
            artifact_kind="model_bundle",
            role="linear_candidate_bundle",
            path=config.linear_bundle_path,
            producer_step_id="train_linear_runtime_candidate",
            consumer_step_ids=("materialize_promotion_evidence", "export_promoted_runtime"),
            report_id=None,
            schema_version="cift_model_bundle/v1",
        ),
        _current_artifact(
            config=config,
            artifact_key="calibration_report_path",
            artifact_kind="json_report",
            role="calibration",
            path=config.calibration_report_path,
            producer_step_id="materialize_calibration_report",
            consumer_step_ids=("materialize_promotion_evidence", "run_hardened_release_gate"),
            report_id=_report_id_from_path(config.calibration_report_path),
            schema_version="aegis_introspection.cift_calibration/v1",
        ),
        _current_artifact(
            config=config,
            artifact_key="feature_ablation_report_path",
            artifact_kind="json_report",
            role="feature_ablation",
            path=config.feature_ablation_report_path,
            producer_step_id="materialize_feature_ablation_report",
            consumer_step_ids=("materialize_promotion_evidence", "run_hardened_release_gate"),
            report_id=_report_id_from_path(config.feature_ablation_report_path),
            schema_version="aegis_introspection.cift_feature_ablation/v1",
        ),
        _current_artifact(
            config=config,
            artifact_key="patching_report_path",
            artifact_kind="json_report",
            role="counterfactual_patching",
            path=config.patching_report_path,
            producer_step_id="run_counterfactual_patching",
            consumer_step_ids=("materialize_promotion_evidence", "run_hardened_release_gate"),
            report_id=_report_id_from_path(config.patching_report_path),
            schema_version="aegis_introspection.cift_counterfactual_patching/v1",
        ),
        _current_artifact(
            config=config,
            artifact_key="failure_cases_report_path",
            artifact_kind="json_report",
            role="failure_cases",
            path=config.failure_cases_report_path,
            producer_step_id="materialize_failure_cases_report",
            consumer_step_ids=("materialize_promotion_evidence", "run_hardened_release_gate"),
            report_id=_report_id_from_path(config.failure_cases_report_path),
            schema_version="aegis_introspection.cift_failure_cases/v1",
        ),
        _current_artifact(
            config=config,
            artifact_key="lineage_report_path",
            artifact_kind="json_report",
            role="lineage",
            path=config.lineage_report_path,
            producer_step_id="materialize_lineage_report",
            consumer_step_ids=("materialize_promotion_evidence", "run_hardened_release_gate"),
            report_id=_report_id_from_path(config.lineage_report_path),
            schema_version="aegis_introspection.cift_lineage/v1",
        ),
        _current_artifact(
            config=config,
            artifact_key="device_preflight_report_path",
            artifact_kind="json_report",
            role="device_preflight",
            path=config.device_preflight_report_path,
            producer_step_id="run_device_preflight",
            consumer_step_ids=("extract_activation_artifact", "run_linear_live_runtime_prevention"),
            report_id=None,
            schema_version="aegis_introspection.device_preflight/v1",
        ),
        _current_artifact(
            config=config,
            artifact_key="live_runtime_prevention_report_path",
            artifact_kind="json_report",
            role="linear_live_runtime_prevention",
            path=config.live_runtime_prevention_report_path,
            producer_step_id="run_linear_live_runtime_prevention",
            consumer_step_ids=(
                "materialize_linear_sealed_holdout_metric",
                "materialize_promotion_evidence",
                "verify_evidence_chain_identity",
                "run_hardened_release_gate",
            ),
            report_id=_report_id_from_path(config.live_runtime_prevention_report_path),
            schema_version="aegis_introspection.cift_live_window_selector_benchmark/v1",
        ),
        _current_artifact(
            config=config,
            artifact_key="linear_sealed_holdout_metric_path",
            artifact_kind="json_report",
            role="linear_sealed_holdout_metric",
            path=config.sealed_holdout_metric_path,
            producer_step_id="materialize_linear_sealed_holdout_metric",
            consumer_step_ids=(
                "compare_live_sealed_linear_and_paper_mlp",
                "materialize_promotion_evidence",
                "verify_evidence_chain_identity",
                "run_hardened_release_gate",
            ),
            report_id=_report_id_from_path(config.sealed_holdout_metric_path),
            schema_version="aegis_introspection.cift_sealed_holdout_metric/v1",
        ),
        _current_artifact(
            config=config,
            artifact_key="gateway_smoke_report_path",
            artifact_kind="json_report",
            role="linear_gateway_smoke",
            path=config.gateway_smoke_report_path,
            producer_step_id="run_linear_gateway_smoke",
            consumer_step_ids=("materialize_promotion_evidence", "run_hardened_release_gate"),
            report_id=_report_id_from_path(config.gateway_smoke_report_path),
            schema_version="aegis.proxy.cift_gateway_smoke/v1",
        ),
        _current_artifact(
            config=config,
            artifact_key="paper_mlp_runtime_prevention_report_path",
            artifact_kind="json_report",
            role="paper_mlp_live_runtime_prevention",
            path=config.paper_mlp_runtime_prevention_report_path,
            producer_step_id="run_paper_mlp_live_runtime_prevention",
            consumer_step_ids=(
                "materialize_paper_mlp_sealed_holdout_metric",
                "compare_live_sealed_linear_and_paper_mlp",
            ),
            report_id=_report_id_from_path(config.paper_mlp_runtime_prevention_report_path),
            schema_version="aegis_introspection.cift_live_window_selector_benchmark/v1",
        ),
        _current_artifact(
            config=config,
            artifact_key="paper_mlp_sealed_holdout_metric_path",
            artifact_kind="json_report",
            role="paper_mlp_sealed_holdout_metric",
            path=config.paper_mlp_sealed_holdout_metric_path,
            producer_step_id="materialize_paper_mlp_sealed_holdout_metric",
            consumer_step_ids=("compare_live_sealed_linear_and_paper_mlp",),
            report_id=_report_id_from_path(config.paper_mlp_sealed_holdout_metric_path),
            schema_version="aegis_introspection.cift_sealed_holdout_metric/v1",
        ),
        _current_artifact(
            config=config,
            artifact_key="live_sealed_head_to_head_report_path",
            artifact_kind="json_report",
            role="live_sealed_linear_vs_paper_mlp",
            path=config.live_head_to_head_report_path,
            producer_step_id="compare_live_sealed_linear_and_paper_mlp",
            consumer_step_ids=("materialize_promotion_evidence", "verify_evidence_chain_identity"),
            report_id=_report_id_from_path(config.live_head_to_head_report_path),
            schema_version="aegis_introspection.cift_live_probe_competition/v1",
        ),
        _output_artifact(
            config=config,
            artifact_key="promotion_evidence_path",
            artifact_kind="promotion_evidence",
            role="promotion_evidence",
            path=config.promotion_evidence_path,
            producer_step_id="materialize_promotion_evidence",
            consumer_step_ids=(
                "export_promoted_runtime",
                "verify_evidence_chain_identity",
                "run_hardened_release_gate",
            ),
            report_id=None,
            schema_version="cift_promotion_evidence/v1",
        ),
        _current_artifact(
            config=config,
            artifact_key="promoted_runtime_artifact_path",
            artifact_kind="runtime_model",
            role="promoted_runtime",
            path=config.promoted_runtime_artifact_path,
            producer_step_id="export_promoted_runtime",
            consumer_step_ids=("verify_evidence_chain_identity", "run_hardened_release_gate"),
            report_id=None,
            schema_version="aegis.cift_runtime_linear/v1",
        ),
        _output_artifact(
            config=config,
            artifact_key="evidence_chain_verification_report_path",
            artifact_kind="json_report",
            role="evidence_chain_verification",
            path=config.evidence_chain_verification_report_path,
            producer_step_id="verify_evidence_chain_identity",
            consumer_step_ids=("run_hardened_release_gate",),
            report_id=None,
            schema_version="aegis_introspection.cift_evidence_chain_verification/v1",
        ),
    ]
    artifacts.append(
        _current_artifact(
            config=config,
            artifact_key="head_to_head_report_path",
            artifact_kind="json_report",
            role="grouped_cv_linear_vs_paper_mlp",
            path=config.grouped_head_to_head_report_path,
            producer_step_id="compare_linear_and_paper_mlp_grouped_cv",
            consumer_step_ids=("human_promotion_review",),
            report_id=_report_id_from_path(config.grouped_head_to_head_report_path),
            schema_version="cift_probe_competition/v1",
        )
    )
    return artifacts


def _current_artifact(
    config: CiftCertificationEvidenceManifestConfig,
    artifact_key: str,
    artifact_kind: str,
    role: str,
    path: Path,
    producer_step_id: str,
    consumer_step_ids: tuple[str, ...],
    report_id: str | None,
    schema_version: str | None,
) -> JsonValue:
    _validate_existing_file(config.repository_root, path, artifact_key)
    return {
        "artifact_key": artifact_key,
        "artifact_kind": artifact_kind,
        "role": role,
        "path": _repository_relative_path(config.repository_root, path),
        "report_id": report_id,
        "schema_version": schema_version,
        "sha256": _sha256_file(path),
        "status": "materialized",
        "producer_step_id": producer_step_id,
        "consumer_step_ids": list(consumer_step_ids),
        "required_for_release": True,
    }


def _output_artifact(
    config: CiftCertificationEvidenceManifestConfig,
    artifact_key: str,
    artifact_kind: str,
    role: str,
    path: Path,
    producer_step_id: str,
    consumer_step_ids: tuple[str, ...],
    report_id: str | None,
    schema_version: str | None,
) -> JsonValue:
    resolved_path = _resolved_repository_path(config.repository_root, path, artifact_key)
    _validate_existing_file(config.repository_root, resolved_path, artifact_key)
    return {
        "artifact_key": artifact_key,
        "artifact_kind": artifact_kind,
        "role": role,
        "path": _repository_relative_path(config.repository_root, path),
        "report_id": report_id,
        "schema_version": schema_version,
        "sha256": _sha256_file(resolved_path),
        "status": "materialized",
        "producer_step_id": producer_step_id,
        "consumer_step_ids": list(consumer_step_ids),
        "required_for_release": True,
    }


def _required_evidence_artifacts(
    config: CiftCertificationWorkflowConfig,
    artifact_paths: dict[str, JsonValue],
) -> list[JsonValue]:
    report_ids = _planned_report_ids(config)
    artifact_specs = (
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="device_preflight_report_path",
            artifact_kind="json_report",
            role="device_preflight",
            report_id=None,
            schema_version="aegis_introspection.device_preflight/v1",
            producer_step_id="run_device_preflight",
            consumer_step_ids=(
                "discover_model_metadata",
                "extract_activation_artifact",
                "run_linear_live_runtime_prevention",
                "run_paper_mlp_live_runtime_prevention",
                "verify_evidence_chain_identity",
                "run_hardened_release_gate",
            ),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="model_metadata_report_path",
            artifact_kind="json_report",
            role="model_metadata",
            report_id=None,
            schema_version="aegis_introspection.cift_model_metadata/v1",
            producer_step_id="discover_model_metadata",
            consumer_step_ids=("verify_evidence_chain_identity",),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="activation_artifact_path",
            artifact_kind="activation_tensor",
            role="calibration_activation_artifact",
            report_id=None,
            schema_version=None,
            producer_step_id="extract_activation_artifact",
            consumer_step_ids=(
                "compare_linear_and_paper_mlp_grouped_cv",
                "train_linear_runtime_candidate",
                "train_paper_mlp_reference",
                "materialize_calibration_report",
                "materialize_feature_ablation_report",
                "run_counterfactual_patching",
            ),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="head_to_head_report_path",
            artifact_kind="json_report",
            role="grouped_cv_linear_vs_paper_mlp",
            report_id=report_ids["grouped_head_to_head"],
            schema_version="cift_probe_competition/v1",
            producer_step_id="compare_linear_and_paper_mlp_grouped_cv",
            consumer_step_ids=("human_promotion_review",),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="linear_bundle_path",
            artifact_kind="model_bundle",
            role="linear_candidate_bundle",
            report_id=None,
            schema_version="cift_model_bundle/v1",
            producer_step_id="train_linear_runtime_candidate",
            consumer_step_ids=(
                "export_linear_runtime_bootstrap_for_live_evidence",
                "materialize_promotion_evidence",
                "export_promoted_runtime",
            ),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="calibration_report_path",
            artifact_kind="json_report",
            role="calibration",
            report_id=report_ids["calibration"],
            schema_version="aegis_introspection.cift_calibration/v1",
            producer_step_id="materialize_calibration_report",
            consumer_step_ids=("materialize_promotion_evidence", "run_hardened_release_gate"),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="feature_ablation_report_path",
            artifact_kind="json_report",
            role="feature_ablation",
            report_id=report_ids["feature_ablation"],
            schema_version="aegis_introspection.cift_feature_ablation/v1",
            producer_step_id="materialize_feature_ablation_report",
            consumer_step_ids=("materialize_promotion_evidence", "run_hardened_release_gate"),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="patching_report_path",
            artifact_kind="json_report",
            role="counterfactual_patching",
            report_id=report_ids["patching"],
            schema_version="aegis_introspection.cift_counterfactual_patching/v1",
            producer_step_id="run_counterfactual_patching",
            consumer_step_ids=("materialize_promotion_evidence", "run_hardened_release_gate"),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="failure_cases_report_path",
            artifact_kind="json_report",
            role="failure_cases",
            report_id=report_ids["failure_cases"],
            schema_version="aegis_introspection.cift_failure_cases/v1",
            producer_step_id="materialize_failure_cases_report",
            consumer_step_ids=("materialize_promotion_evidence", "run_hardened_release_gate"),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="lineage_report_path",
            artifact_kind="json_report",
            role="lineage",
            report_id=report_ids["lineage"],
            schema_version="aegis_introspection.cift_lineage/v1",
            producer_step_id="materialize_lineage_report",
            consumer_step_ids=("materialize_promotion_evidence", "run_hardened_release_gate"),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="paper_mlp_runtime_prevention_report_path",
            artifact_kind="json_report",
            role="paper_mlp_live_runtime_prevention",
            report_id=report_ids["paper_mlp_runtime_prevention"],
            schema_version="aegis_introspection.cift_live_window_selector_benchmark/v1",
            producer_step_id="run_paper_mlp_live_runtime_prevention",
            consumer_step_ids=("materialize_paper_mlp_sealed_holdout_metric",),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="paper_mlp_sealed_holdout_metric_path",
            artifact_kind="json_report",
            role="paper_mlp_sealed_holdout_metric",
            report_id=report_ids["paper_mlp_sealed_metric"],
            schema_version="aegis_introspection.cift_sealed_holdout_metric/v1",
            producer_step_id="materialize_paper_mlp_sealed_holdout_metric",
            consumer_step_ids=("compare_live_sealed_linear_and_paper_mlp",),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="live_runtime_prevention_report_path",
            artifact_kind="json_report",
            role="linear_live_runtime_prevention",
            report_id=report_ids["linear_runtime_prevention"],
            schema_version="aegis_introspection.cift_live_window_selector_benchmark/v1",
            producer_step_id="run_linear_live_runtime_prevention",
            consumer_step_ids=(
                "materialize_linear_sealed_holdout_metric",
                "materialize_promotion_evidence",
                "verify_evidence_chain_identity",
                "run_hardened_release_gate",
            ),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="linear_sealed_holdout_metric_path",
            artifact_kind="json_report",
            role="linear_sealed_holdout_metric",
            report_id=report_ids["linear_sealed_metric"],
            schema_version="aegis_introspection.cift_sealed_holdout_metric/v1",
            producer_step_id="materialize_linear_sealed_holdout_metric",
            consumer_step_ids=(
                "compare_live_sealed_linear_and_paper_mlp",
                "materialize_promotion_evidence",
                "verify_evidence_chain_identity",
                "run_hardened_release_gate",
            ),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="gateway_smoke_report_path",
            artifact_kind="json_report",
            role="linear_gateway_smoke",
            report_id=report_ids["gateway_smoke"],
            schema_version="aegis.proxy.cift_gateway_smoke/v1",
            producer_step_id="run_linear_gateway_smoke",
            consumer_step_ids=("materialize_promotion_evidence", "run_hardened_release_gate"),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="live_sealed_head_to_head_report_path",
            artifact_kind="json_report",
            role="live_sealed_linear_vs_paper_mlp",
            report_id=report_ids["live_head_to_head"],
            schema_version="aegis_introspection.cift_live_probe_competition/v1",
            producer_step_id="compare_live_sealed_linear_and_paper_mlp",
            consumer_step_ids=(
                "materialize_promotion_evidence",
                "verify_evidence_chain_identity",
                "run_hardened_release_gate",
            ),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="promotion_evidence_path",
            artifact_kind="promotion_evidence",
            role="promotion_evidence",
            report_id=None,
            schema_version="cift_promotion_evidence/v1",
            producer_step_id="materialize_promotion_evidence",
            consumer_step_ids=(
                "export_promoted_runtime",
                "verify_evidence_chain_identity",
                "run_hardened_release_gate",
            ),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="promoted_runtime_artifact_path",
            artifact_kind="runtime_model",
            role="promoted_runtime",
            report_id=None,
            schema_version="aegis.cift_runtime_linear/v1",
            producer_step_id="export_promoted_runtime",
            consumer_step_ids=(
                "run_linear_live_runtime_prevention",
                "materialize_linear_sealed_holdout_metric",
                "verify_evidence_chain_identity",
                "run_hardened_release_gate",
            ),
            required_for_release=True,
        ),
        _evidence_artifact(
            config=config,
            artifact_paths=artifact_paths,
            artifact_key="evidence_chain_verification_report_path",
            artifact_kind="json_report",
            role="evidence_chain_verification",
            report_id=None,
            schema_version="aegis_introspection.cift_evidence_chain_verification/v1",
            producer_step_id="verify_evidence_chain_identity",
            consumer_step_ids=("run_hardened_release_gate",),
            required_for_release=True,
        ),
    )
    return list(artifact_specs)


def _evidence_artifact(
    config: CiftCertificationWorkflowConfig,
    artifact_paths: dict[str, JsonValue],
    artifact_key: str,
    artifact_kind: str,
    role: str,
    report_id: str | None,
    schema_version: str | None,
    producer_step_id: str,
    consumer_step_ids: tuple[str, ...],
    required_for_release: bool,
) -> JsonValue:
    path = _artifact_path(artifact_paths, artifact_key)
    artifact_path = Path(path)
    resolved_path = _resolved_repository_path(
        config.repository_root,
        artifact_path if artifact_path.is_absolute() else config.repository_root / artifact_path,
        artifact_key,
    )
    materialized = resolved_path.is_file()
    return {
        "artifact_key": artifact_key,
        "artifact_kind": artifact_kind,
        "role": role,
        "path": path,
        "report_id": report_id,
        "schema_version": schema_version,
        "sha256": _sha256_file(resolved_path) if materialized else None,
        "status": "materialized" if materialized else "planned",
        "producer_step_id": producer_step_id,
        "consumer_step_ids": list(consumer_step_ids),
        "required_for_release": required_for_release,
    }


def _command_plan(
    config: CiftCertificationWorkflowConfig,
    model_metadata: CiftModelMetadataReport,
    artifact_paths: dict[str, JsonValue],
) -> list[JsonValue]:
    corpus_path = _repository_relative_path(config.repository_root, config.corpus_path)
    runtime_turns_path = _repository_relative_path(config.repository_root, config.runtime_turns_path)
    selected_choice_runtime_model_path = _repository_relative_path(
        config.repository_root,
        config.selected_choice_runtime_model_path,
    )
    layers = ",".join(str(layer_index) for layer_index in config.layer_indices)
    pooling = ",".join(config.pooling_methods)
    report_ids = _planned_report_ids(config)
    paper_mlp_runtime_preview_path = _artifact_path(artifact_paths, "paper_mlp_runtime_preview_path")
    promoted_runtime_path = _artifact_path(artifact_paths, "promoted_runtime_artifact_path")
    return [
        _command_step(
            step_id="run_device_preflight",
            evidence_item="device_preflight_passed",
            argv=(
                "python",
                "introspection/scripts/check_cift_device_preflight.py",
                "--device",
                config.requested_device,
                "--output",
                _artifact_path(artifact_paths, "device_preflight_report_path"),
            ),
            produces=(_artifact_path(artifact_paths, "device_preflight_report_path"),),
            sealed_holdout_access=False,
        ),
        _command_step(
            step_id="discover_model_metadata",
            evidence_item="model_metadata_discovered",
            argv=(
                "python",
                "introspection/scripts/discover_cift_model_metadata.py",
                "--model-id",
                config.model_id,
                "--revision",
                config.revision,
                "--output",
                _artifact_path(artifact_paths, "model_metadata_report_path"),
                "--device",
                config.requested_device,
                "--dtype",
                config.dtype_name,
                "--selected-readout-candidate",
                config.candidate_feature_key,
                *_model_access_flags(config),
            ),
            produces=(_artifact_path(artifact_paths, "model_metadata_report_path"),),
            sealed_holdout_access=False,
        ),
        _command_step(
            step_id="extract_activation_artifact",
            evidence_item="activation_artifact_bound_to_model_identity",
            argv=(
                "python",
                "introspection/scripts/extract_activations.py",
                "--prompts",
                corpus_path,
                "--output",
                _artifact_path(artifact_paths, "activation_artifact_path"),
                "--model-id",
                config.model_id,
                "--revision",
                config.revision,
                "--device",
                config.requested_device,
                "--dtype",
                config.dtype_name,
                "--layers",
                layers,
                "--pooling",
                pooling,
                *_model_access_flags(config),
            ),
            produces=(_artifact_path(artifact_paths, "activation_artifact_path"),),
            sealed_holdout_access=False,
        ),
        _command_step(
            step_id="compare_linear_and_paper_mlp_grouped_cv",
            evidence_item="linear_vs_paper_mlp_head_to_head",
            argv=(
                "python",
                "introspection/scripts/compare_cift_probe_head_to_head.py",
                "--artifact",
                _artifact_path(artifact_paths, "activation_artifact_path"),
                "--output-json",
                _artifact_path(artifact_paths, "head_to_head_report_path"),
                "--report-id",
                report_ids["grouped_head_to_head"],
                "--training-dataset-id",
                config.training_dataset_id,
                "--task",
                config.task_name,
                "--positive-label",
                config.positive_label,
                "--feature-representation",
                "raw_activation",
                "--activation-feature",
                config.candidate_feature_key,
                "--source-feature",
                config.candidate_feature_key,
                "--ridge",
                "0.001",
                "--fold-count",
                "5",
                "--random-seeds",
                "11,17,23",
                "--decision-threshold",
                "0.5",
                "--linear-max-epochs",
                "1000",
                "--linear-regularization-c",
                "1.0",
                "--paper-mlp-max-epochs",
                "1000",
                "--paper-mlp-learning-rate",
                "0.001",
                "--paper-mlp-l1-softplus-weight",
                "0.0001",
                "--paper-mlp-batch-size",
                "32",
                "--paper-hyperparameter-search-trials",
                "1",
                "--candidate-hyperparameter-search-trials",
                "1",
                "--evaluation-split-id",
                f"{config.training_dataset_id}/grouped-cv",
                "--evaluation-split-manifest-id",
                _artifact_path(artifact_paths, "activation_artifact_path"),
                "--metric-name",
                "grouped_cv_macro_f1",
                "--created-at",
                config.created_at,
            ),
            produces=(_artifact_path(artifact_paths, "head_to_head_report_path"),),
            sealed_holdout_access=False,
        ),
        _train_bundle_step(
            step_id="train_linear_runtime_candidate",
            classifier_family="linear_logistic_regression",
            output_bundle_path=_artifact_path(artifact_paths, "linear_bundle_path"),
            config=config,
            artifact_paths=artifact_paths,
            report_ids=report_ids,
        ),
        _train_bundle_step(
            step_id="train_paper_mlp_reference",
            classifier_family="mlp_128_64_1",
            output_bundle_path=_artifact_path(artifact_paths, "paper_mlp_bundle_path"),
            config=config,
            artifact_paths=artifact_paths,
            report_ids=report_ids,
        ),
        _export_runtime_step(
            step_id="export_linear_runtime_bootstrap_for_live_evidence",
            bundle_path=_artifact_path(artifact_paths, "linear_bundle_path"),
            output_path=promoted_runtime_path,
            model_bundle_id=f"cift_{config.certification_id}_linear_v1",
            promotion_evidence_path=None,
        ),
        _export_runtime_step(
            step_id="export_paper_mlp_runtime_preview",
            bundle_path=_artifact_path(artifact_paths, "paper_mlp_bundle_path"),
            output_path=paper_mlp_runtime_preview_path,
            model_bundle_id=f"cift_{config.certification_id}_paper_mlp_v1",
            promotion_evidence_path=None,
        ),
        _live_benchmark_step(
            step_id="run_linear_live_runtime_prevention",
            report_id=report_ids["linear_runtime_prevention"],
            selected_choice_runtime_model_path=selected_choice_runtime_model_path,
            fallback_runtime_model_path=promoted_runtime_path,
            output_json_path=_artifact_path(artifact_paths, "live_runtime_prevention_report_path"),
            output_markdown_path=_artifact_path(artifact_paths, "live_runtime_prevention_summary_path"),
            runtime_turns_path=runtime_turns_path,
            config=config,
        ),
        _gateway_smoke_step(
            config=config,
            model_metadata=model_metadata,
            artifact_paths=artifact_paths,
            report_ids=report_ids,
        ),
        _live_benchmark_step(
            step_id="run_paper_mlp_live_runtime_prevention",
            report_id=report_ids["paper_mlp_runtime_prevention"],
            selected_choice_runtime_model_path=selected_choice_runtime_model_path,
            fallback_runtime_model_path=paper_mlp_runtime_preview_path,
            output_json_path=_artifact_path(artifact_paths, "paper_mlp_runtime_prevention_report_path"),
            output_markdown_path=_artifact_path(artifact_paths, "paper_mlp_runtime_prevention_summary_path"),
            runtime_turns_path=runtime_turns_path,
            config=config,
        ),
        _sealed_metric_step(
            step_id="materialize_linear_sealed_holdout_metric",
            report_id=report_ids["linear_sealed_metric"],
            runtime_report_path=_artifact_path(artifact_paths, "live_runtime_prevention_report_path"),
            runtime_turns_path=runtime_turns_path,
            runtime_model_path=promoted_runtime_path,
            output_path=_artifact_path(artifact_paths, "linear_sealed_holdout_metric_path"),
            config=config,
        ),
        _sealed_metric_step(
            step_id="materialize_paper_mlp_sealed_holdout_metric",
            report_id=report_ids["paper_mlp_sealed_metric"],
            runtime_report_path=_artifact_path(artifact_paths, "paper_mlp_runtime_prevention_report_path"),
            runtime_turns_path=runtime_turns_path,
            runtime_model_path=paper_mlp_runtime_preview_path,
            output_path=_artifact_path(artifact_paths, "paper_mlp_sealed_holdout_metric_path"),
            config=config,
        ),
        _command_step(
            step_id="compare_live_sealed_linear_and_paper_mlp",
            evidence_item="linear_vs_paper_mlp_head_to_head",
            argv=(
                "python",
                "introspection/scripts/materialize_cift_live_probe_competition.py",
                "--paper-metric-report",
                _artifact_path(artifact_paths, "paper_mlp_sealed_holdout_metric_path"),
                "--candidate-metric-report",
                _artifact_path(artifact_paths, "linear_sealed_holdout_metric_path"),
                "--evaluation-split-manifest",
                runtime_turns_path,
                "--output",
                _artifact_path(artifact_paths, "live_sealed_head_to_head_report_path"),
                "--report-id",
                report_ids["live_head_to_head"],
                "--feature-representation",
                "raw_activation",
                "--activation-feature-key",
                config.candidate_feature_key,
                "--candidate-probe-architecture",
                "linear_logistic_regression",
                "--candidate-training-loss",
                "regularized_logistic_loss",
                "--paper-operating-threshold",
                "0.5",
                "--candidate-operating-threshold",
                "0.5",
                "--created-at",
                config.created_at,
            ),
            produces=(_artifact_path(artifact_paths, "live_sealed_head_to_head_report_path"),),
            sealed_holdout_access=True,
        ),
        _promotion_evidence_step(config=config, artifact_paths=artifact_paths, report_ids=report_ids),
        _export_runtime_step(
            step_id="export_promoted_runtime",
            bundle_path=_artifact_path(artifact_paths, "linear_bundle_path"),
            output_path=promoted_runtime_path,
            model_bundle_id=f"cift_{config.certification_id}_linear_v1",
            promotion_evidence_path=_artifact_path(artifact_paths, "promotion_evidence_path"),
        ),
        _command_step(
            step_id="verify_evidence_chain_identity",
            evidence_item="evidence_chain_identity_verification",
            argv=(
                "python",
                "introspection/scripts/verify_cift_evidence_chain.py",
                "--repository-root",
                ".",
                "--runtime-model",
                promoted_runtime_path,
                "--runtime-prevention-report",
                _artifact_path(artifact_paths, "live_runtime_prevention_report_path"),
                "--gateway-smoke-report",
                _artifact_path(artifact_paths, "gateway_smoke_report_path"),
                "--sealed-holdout-report",
                _artifact_path(artifact_paths, "linear_sealed_holdout_metric_path"),
                "--head-to-head-report",
                _artifact_path(artifact_paths, "live_sealed_head_to_head_report_path"),
                "--promotion-evidence",
                _artifact_path(artifact_paths, "promotion_evidence_path"),
                "--model-metadata-report",
                _artifact_path(artifact_paths, "model_metadata_report_path"),
                *_required_runtime_prevention_device_flags(config),
                "--output",
                _artifact_path(artifact_paths, "evidence_chain_verification_report_path"),
            ),
            produces=(_artifact_path(artifact_paths, "evidence_chain_verification_report_path"),),
            sealed_holdout_access=True,
        ),
        _certification_manifest_step(config=config, artifact_paths=artifact_paths),
        _certification_workflow_run_step(artifact_paths=artifact_paths),
        _strict_deployment_env_step(config=config, artifact_paths=artifact_paths),
    ]


def _certification_manifest_step(
    config: CiftCertificationWorkflowConfig,
    artifact_paths: dict[str, JsonValue],
) -> JsonValue:
    return _command_step(
        step_id="materialize_certification_manifest",
        evidence_item="evidence_chain_identity_verification",
        argv=(
            "python",
            "introspection/scripts/materialize_cift_certification_manifest.py",
            "--certification-id",
            config.certification_id,
            "--repository-root",
            ".",
            "--output",
            _artifact_path(artifact_paths, "certification_manifest_path"),
            "--created-at",
            config.created_at,
            "--behavior-id",
            config.behavior_id,
            "--behavior-description",
            config.behavior_description,
            "--requested-device",
            config.requested_device,
            "--prompt-renderer",
            config.prompt_renderer,
            "--selected-choice-geometry",
            config.selected_choice_geometry,
            "--selected-choice-readout-token-count",
            str(config.selected_choice_readout_token_count),
            "--dtype",
            config.dtype_name,
            "--metric-threshold",
            str(config.metric_threshold),
            "--ablation-delta-threshold",
            str(config.ablation_delta_threshold),
            "--model-metadata-report",
            _artifact_path(artifact_paths, "model_metadata_report_path"),
            "--activation-artifact",
            _artifact_path(artifact_paths, "activation_artifact_path"),
            "--linear-bundle",
            _artifact_path(artifact_paths, "linear_bundle_path"),
            "--grouped-head-to-head-report",
            _artifact_path(artifact_paths, "head_to_head_report_path"),
            "--calibration-report",
            _artifact_path(artifact_paths, "calibration_report_path"),
            "--feature-ablation-report",
            _artifact_path(artifact_paths, "feature_ablation_report_path"),
            "--patching-report",
            _artifact_path(artifact_paths, "patching_report_path"),
            "--failure-cases-report",
            _artifact_path(artifact_paths, "failure_cases_report_path"),
            "--lineage-report",
            _artifact_path(artifact_paths, "lineage_report_path"),
            "--device-preflight-report",
            _artifact_path(artifact_paths, "device_preflight_report_path"),
            "--live-runtime-prevention-report",
            _artifact_path(artifact_paths, "live_runtime_prevention_report_path"),
            "--sealed-holdout-metric",
            _artifact_path(artifact_paths, "linear_sealed_holdout_metric_path"),
            "--gateway-smoke-report",
            _artifact_path(artifact_paths, "gateway_smoke_report_path"),
            "--paper-mlp-runtime-prevention-report",
            _artifact_path(artifact_paths, "paper_mlp_runtime_prevention_report_path"),
            "--paper-mlp-sealed-holdout-metric",
            _artifact_path(artifact_paths, "paper_mlp_sealed_holdout_metric_path"),
            "--live-head-to-head-report",
            _artifact_path(artifact_paths, "live_sealed_head_to_head_report_path"),
            "--promotion-evidence",
            _artifact_path(artifact_paths, "promotion_evidence_path"),
            "--promoted-runtime-artifact",
            _artifact_path(artifact_paths, "promoted_runtime_artifact_path"),
            "--promotion-report-output-dir",
            _artifact_path(artifact_paths, "promotion_report_output_dir"),
            "--evidence-chain-verification-report",
            _artifact_path(artifact_paths, "evidence_chain_verification_report_path"),
        ),
        produces=(_artifact_path(artifact_paths, "certification_manifest_path"),),
        sealed_holdout_access=True,
    )


def _certification_workflow_run_step(artifact_paths: dict[str, JsonValue]) -> JsonValue:
    return _command_step(
        step_id="verify_certification_workflow_run",
        evidence_item="evidence_chain_identity_verification",
        argv=(
            "python",
            "introspection/scripts/run_cift_certification_workflow.py",
            "--repository-root",
            ".",
            "--workflow-manifest",
            _artifact_path(artifact_paths, "certification_manifest_path"),
            "--output",
            _artifact_path(artifact_paths, "certification_report_path"),
            "--command-timeout-seconds",
            _CERTIFICATION_WORKFLOW_COMMAND_TIMEOUT_SECONDS,
        ),
        produces=(_artifact_path(artifact_paths, "certification_report_path"),),
        sealed_holdout_access=True,
    )


def _strict_deployment_env_step(
    config: CiftCertificationWorkflowConfig,
    artifact_paths: dict[str, JsonValue],
) -> JsonValue:
    return _templated_command_step(
        step_id="run_hardened_release_gate",
        evidence_item="hardened_release_gate_pass",
        argv_template=(
            "python",
            "introspection/scripts/materialize_cift_deployment_env.py",
            _artifact_path(artifact_paths, "promoted_runtime_artifact_path"),
            "--repository-root",
            ".",
            "--certification-manifest",
            _artifact_path(artifact_paths, "certification_manifest_path"),
            "--certification-report",
            _artifact_path(artifact_paths, "certification_report_path"),
            "--certification-artifact-root",
            ".",
            "--required-device",
            config.requested_device,
            "--expected-detector-name",
            "cift_runtime",
            "--expected-extractor-id",
            "${extractor_id}",
            "--expected-feature-source",
            "self_hosted_activation_extractor",
            "--expected-selected-choice-readout-token-count",
            str(config.selected_choice_readout_token_count),
            "--extractor-base-url",
            "${sidecar_base_url}",
            "--extractor-timeout-seconds",
            "30.0",
            "--extractor-api-key-env-var",
            "AEGIS_CIFT_EXTRACTOR_API_KEY",
            "--release-gate-report-output",
            _artifact_path(artifact_paths, "release_gate_report_path"),
            "--output",
            _artifact_path(artifact_paths, "deployment_env_path"),
        ),
        template_inputs=(
            {
                "name": "extractor_id",
                "derivation": "operator-supplied trusted activation sidecar identifier",
            },
            {
                "name": "sidecar_base_url",
                "derivation": "operator-supplied running CIFT activation sidecar base URL",
            },
        ),
        produces=(
            _artifact_path(artifact_paths, "deployment_env_path"),
            _artifact_path(artifact_paths, "release_gate_report_path"),
        ),
        consumes=(
            _artifact_path(artifact_paths, "promoted_runtime_artifact_path"),
            _artifact_path(artifact_paths, "certification_manifest_path"),
            _artifact_path(artifact_paths, "certification_report_path"),
        ),
        sealed_holdout_access=False,
    )


def _promotion_evidence_step(
    config: CiftCertificationWorkflowConfig,
    artifact_paths: dict[str, JsonValue],
    report_ids: dict[str, str],
) -> JsonValue:
    report_source_args = _promotion_report_source_args(artifact_paths=artifact_paths, report_ids=report_ids)
    return _templated_command_step(
        step_id="materialize_promotion_evidence",
        evidence_item="promotion_evidence_materialized",
        argv_template=(
            "python",
            "introspection/scripts/materialize_cift_promotion_evidence.py",
            "--bundle",
            _artifact_path(artifact_paths, "linear_bundle_path"),
            "--repository-root",
            ".",
            "--report-output-dir",
            _artifact_path(artifact_paths, "promotion_report_output_dir"),
            "--evidence-output",
            _artifact_path(artifact_paths, "promotion_evidence_path"),
            "--evidence-id",
            f"{config.certification_id}_linear_promoted_runtime_promotion_evidence_v1",
            "--behavior-id",
            config.behavior_id,
            "--behavior-description",
            config.behavior_description,
            "--train-split-id",
            f"{config.training_dataset_id}/train",
            "--calibration-split-id",
            f"{config.training_dataset_id}/calibration",
            "--heldout-split-id",
            f"{config.training_dataset_id}/grouped-cv",
            "--sealed-holdout-split-id",
            f"{config.training_dataset_id}/sealed-holdout",
            "--sealed-holdout-report-id",
            report_ids["linear_sealed_metric"],
            "--metric-report-id",
            report_ids["linear_sealed_metric"],
            "--metric-name",
            "sealed_holdout_macro_f1",
            "--metric-value",
            "${linear_sealed_holdout_metric.metric_value}",
            "--metric-threshold",
            str(config.metric_threshold),
            "--calibration-report-id",
            report_ids["calibration"],
            "--ablation-report-id",
            report_ids["feature_ablation"],
            "--ablation-delta",
            "${feature_ablation_report.ablation_delta}",
            "--ablation-delta-threshold",
            str(config.ablation_delta_threshold),
            "--patching-report-id",
            report_ids["patching"],
            "--failure-case-report-id",
            report_ids["failure_cases"],
            "--runtime-prevention-report-id",
            report_ids["linear_runtime_prevention"],
            "--gateway-smoke-report-id",
            report_ids["gateway_smoke"],
            "--lineage-report-id",
            report_ids["lineage"],
            "--head-to-head-report-id",
            report_ids["live_head_to_head"],
            "--created-at",
            config.created_at,
            *report_source_args,
        ),
        template_inputs=(
            {
                "name": "linear_sealed_holdout_metric.metric_value",
                "path": _artifact_path(artifact_paths, "linear_sealed_holdout_metric_path"),
                "json_pointer": "/metric_value",
            },
            {
                "name": "feature_ablation_report.ablation_delta",
                "path": _artifact_path(artifact_paths, "feature_ablation_report_path"),
                "derivation": "best_variant_macro_f1 - selected_candidate_feature_macro_f1",
            },
        ),
        produces=(_artifact_path(artifact_paths, "promotion_evidence_path"),),
        consumes=_promotion_source_paths(artifact_paths=artifact_paths, report_ids=report_ids),
        sealed_holdout_access=True,
    )


def _gateway_smoke_step(
    config: CiftCertificationWorkflowConfig,
    model_metadata: CiftModelMetadataReport,
    artifact_paths: dict[str, JsonValue],
    report_ids: dict[str, str],
) -> JsonValue:
    return _templated_command_step(
        step_id="run_linear_gateway_smoke",
        evidence_item="live_gateway_sidecar_runtime_prevention",
        argv_template=(
            "aegis-proxy-cift-smoke",
            "--url",
            "${gateway_base_url}",
            "--sidecar-url",
            "${sidecar_base_url}",
            "--gateway-model",
            "${gateway_model}",
            "--report-id",
            report_ids["gateway_smoke"],
            "--timeout",
            "60",
            "--detector-name",
            "cift_runtime",
            "--sidecar-feature-key",
            config.candidate_feature_key,
            "--expected-gateway-feature-source",
            "self_hosted_activation_extractor",
            "--expected-extractor-id",
            "${extractor_id}",
            "--expected-sidecar-model-id",
            config.model_id,
            "--expected-sidecar-revision",
            config.revision,
            "--expected-sidecar-device",
            config.requested_device,
            "--expected-sidecar-hidden-size",
            str(model_metadata.hidden_size),
            "--expected-sidecar-layer-count",
            str(model_metadata.layer_count),
            "--expected-sidecar-tokenizer-fingerprint-sha256",
            model_metadata.tokenizer_fingerprint_sha256,
            "--expected-sidecar-special-tokens-map-sha256",
            model_metadata.special_tokens_map_sha256,
            "--expected-sidecar-chat-template-sha256",
            model_metadata.chat_template_sha256,
            "--selected-choice-readout-token-count",
            str(config.selected_choice_readout_token_count),
            "--sidecar-api-key-env-var",
            "AEGIS_CIFT_EXTRACTOR_API_KEY",
            "--output",
            _artifact_path(artifact_paths, "gateway_smoke_report_path"),
        ),
        template_inputs=(
            {
                "name": "gateway_base_url",
                "derivation": "operator-supplied running gateway base URL for the promoted runtime candidate",
            },
            {
                "name": "sidecar_base_url",
                "derivation": "operator-supplied running CIFT activation sidecar base URL",
            },
            {
                "name": "gateway_model",
                "derivation": "operator-supplied gateway model name routed through the promoted runtime candidate",
            },
            {
                "name": "extractor_id",
                "derivation": "operator-supplied trusted activation sidecar identifier",
            },
        ),
        produces=(_artifact_path(artifact_paths, "gateway_smoke_report_path"),),
        consumes=(_artifact_path(artifact_paths, "promoted_runtime_artifact_path"),),
        sealed_holdout_access=False,
    )


def _train_bundle_step(
    step_id: str,
    classifier_family: str,
    output_bundle_path: str,
    config: CiftCertificationWorkflowConfig,
    artifact_paths: dict[str, JsonValue],
    report_ids: dict[str, str],
) -> JsonValue:
    return _command_step(
        step_id=step_id,
        evidence_item="grouped_cv_metrics",
        argv=(
            "python",
            "introspection/scripts/train_cift_model_bundle.py",
            "--artifact",
            _artifact_path(artifact_paths, "activation_artifact_path"),
            "--output-bundle",
            output_bundle_path,
            "--training-dataset-id",
            config.training_dataset_id,
            "--task",
            config.task_name,
            "--positive-label",
            config.positive_label,
            "--activation-feature",
            config.candidate_feature_key,
            "--decision-threshold",
            "0.5",
            "--seed",
            "42",
            "--max-iter",
            "1000",
            "--regularization-c",
            "1.0",
            "--classifier-family",
            classifier_family,
            "--evaluation-report-ids",
            ",".join(_promotion_report_ids(report_ids)),
            "--score-semantics",
            "full_train_classifier_probability",
            "--candidate-status",
            "runtime_candidate",
            "--created-at",
            config.created_at,
        ),
        produces=(output_bundle_path,),
        sealed_holdout_access=False,
    )


def _export_runtime_step(
    step_id: str,
    bundle_path: str,
    output_path: str,
    model_bundle_id: str,
    promotion_evidence_path: str | None,
) -> JsonValue:
    promotion_args: tuple[str, ...]
    if promotion_evidence_path is None:
        promotion_args = ("--allow-preview-without-promotion",)
    else:
        promotion_args = ("--promotion-evidence", promotion_evidence_path)
    return _command_step(
        step_id=step_id,
        evidence_item="runtime_artifact_bound_to_model_tokenizer_template_hashes",
        argv=(
            "python",
            "introspection/scripts/export_cift_runtime_model.py",
            "--bundle",
            bundle_path,
            "--output",
            output_path,
            "--model-bundle-id",
            model_bundle_id,
            "--confidence",
            "0.99",
            "--negative-action",
            "allow",
            "--positive-action",
            "block",
            *promotion_args,
        ),
        produces=(output_path,),
        sealed_holdout_access=False,
    )


def _live_benchmark_step(
    step_id: str,
    report_id: str,
    selected_choice_runtime_model_path: str,
    fallback_runtime_model_path: str,
    output_json_path: str,
    output_markdown_path: str,
    runtime_turns_path: str,
    config: CiftCertificationWorkflowConfig,
) -> JsonValue:
    return _command_step(
        step_id=step_id,
        evidence_item="live_hidden_state_runtime_prevention",
        argv=(
            "python",
            "introspection/scripts/benchmark_live_cift_window_selector.py",
            "--report-id",
            report_id,
            "--runtime-turns",
            runtime_turns_path,
            "--selected-choice-runtime-model",
            selected_choice_runtime_model_path,
            "--fallback-runtime-model",
            fallback_runtime_model_path,
            "--output-json",
            output_json_path,
            "--output-markdown",
            output_markdown_path,
            "--detector-name",
            f"{config.certification_id}_window_selector",
            "--feature-source",
            f"live_hidden_state_{config.certification_id}",
            "--mock-response",
            "ok",
            "--model-id",
            config.model_id,
            "--revision",
            config.revision,
            "--device",
            config.requested_device,
            "--dtype",
            config.dtype_name,
            "--allow-sealed-holdout",
        ),
        produces=(output_json_path, output_markdown_path),
        sealed_holdout_access=True,
    )


def _sealed_metric_step(
    step_id: str,
    report_id: str,
    runtime_report_path: str,
    runtime_turns_path: str,
    runtime_model_path: str,
    output_path: str,
    config: CiftCertificationWorkflowConfig,
) -> JsonValue:
    return _command_step(
        step_id=step_id,
        evidence_item="sealed_holdout_metrics",
        argv=(
            "python",
            "introspection/scripts/materialize_cift_sealed_holdout_metric.py",
            "--runtime-report",
            runtime_report_path,
            "--runtime-turns",
            runtime_turns_path,
            "--runtime-model",
            runtime_model_path,
            "--output",
            output_path,
            "--report-id",
            report_id,
            "--sealed-holdout-split-id",
            f"{config.training_dataset_id}/sealed-holdout",
            "--metric-name",
            "sealed_holdout_macro_f1",
            "--created-at",
            config.created_at,
            "--allow-sealed-holdout",
        ),
        produces=(output_path,),
        sealed_holdout_access=True,
    )


def _command_step(
    step_id: str,
    evidence_item: str,
    argv: tuple[str, ...],
    produces: tuple[str, ...],
    sealed_holdout_access: bool,
) -> JsonValue:
    return {
        "step_id": step_id,
        "evidence_item": evidence_item,
        "argv": list(argv),
        "produces": list(produces),
        "sealed_holdout_access": sealed_holdout_access,
    }


def _templated_command_step(
    step_id: str,
    evidence_item: str,
    argv_template: tuple[str, ...],
    template_inputs: tuple[dict[str, JsonValue], ...],
    produces: tuple[str, ...],
    consumes: tuple[str, ...],
    sealed_holdout_access: bool,
) -> JsonValue:
    return {
        "step_id": step_id,
        "evidence_item": evidence_item,
        "argv_template": list(argv_template),
        "template_inputs": list(template_inputs),
        "produces": list(produces),
        "consumes": list(consumes),
        "sealed_holdout_access": sealed_holdout_access,
    }


def _promotion_report_source_args(
    artifact_paths: dict[str, JsonValue],
    report_ids: dict[str, str],
) -> tuple[str, ...]:
    args: list[str] = []
    for report_id, schema_version, source_path in _promotion_report_sources(
        artifact_paths=artifact_paths,
        report_ids=report_ids,
    ):
        args.extend(("--report-source", f"{report_id}:{schema_version}:{source_path}"))
    return tuple(args)


def _promotion_source_paths(
    artifact_paths: dict[str, JsonValue],
    report_ids: dict[str, str],
) -> tuple[str, ...]:
    return tuple(
        source_path
        for _, _, source_path in _promotion_report_sources(
            artifact_paths=artifact_paths,
            report_ids=report_ids,
        )
    )


def _promotion_report_sources(
    artifact_paths: dict[str, JsonValue],
    report_ids: dict[str, str],
) -> tuple[tuple[str, str, str], ...]:
    return (
        (
            report_ids["linear_sealed_metric"],
            "aegis_introspection.cift_sealed_holdout_metric/v1",
            _artifact_path(artifact_paths, "linear_sealed_holdout_metric_path"),
        ),
        (
            report_ids["calibration"],
            "aegis_introspection.cift_calibration/v1",
            _artifact_path(artifact_paths, "calibration_report_path"),
        ),
        (
            report_ids["feature_ablation"],
            "aegis_introspection.cift_feature_ablation/v1",
            _artifact_path(artifact_paths, "feature_ablation_report_path"),
        ),
        (
            report_ids["patching"],
            "aegis_introspection.cift_counterfactual_patching/v1",
            _artifact_path(artifact_paths, "patching_report_path"),
        ),
        (
            report_ids["failure_cases"],
            "aegis_introspection.cift_failure_cases/v1",
            _artifact_path(artifact_paths, "failure_cases_report_path"),
        ),
        (
            report_ids["linear_runtime_prevention"],
            "aegis_introspection.cift_live_window_selector_benchmark/v1",
            _artifact_path(artifact_paths, "live_runtime_prevention_report_path"),
        ),
        (
            report_ids["gateway_smoke"],
            "aegis.proxy.cift_gateway_smoke/v1",
            _artifact_path(artifact_paths, "gateway_smoke_report_path"),
        ),
        (
            report_ids["lineage"],
            "aegis_introspection.cift_lineage/v1",
            _artifact_path(artifact_paths, "lineage_report_path"),
        ),
        (
            report_ids["live_head_to_head"],
            "aegis_introspection.cift_live_probe_competition/v1",
            _artifact_path(artifact_paths, "live_sealed_head_to_head_report_path"),
        ),
    )


def _planned_report_ids(config: CiftCertificationWorkflowConfig) -> dict[str, str]:
    return {
        "calibration": f"{config.certification_id}_calibration_v1",
        "feature_ablation": f"{config.certification_id}_feature_ablation_v1",
        "grouped_head_to_head": f"{config.certification_id}_linear_vs_paper_mlp_v1",
        "patching": f"{config.certification_id}_causal_patching_v1",
        "failure_cases": f"{config.certification_id}_failure_cases_v1",
        "lineage": f"{config.certification_id}_lineage_v1",
        "linear_runtime_prevention": f"{config.certification_id}_linear_runtime_prevention_v1",
        "gateway_smoke": f"{config.certification_id}_gateway_smoke_v1",
        "paper_mlp_runtime_prevention": f"{config.certification_id}_paper_mlp_runtime_prevention_v1",
        "linear_sealed_metric": f"{config.certification_id}_linear_sealed_holdout_metric_v1",
        "paper_mlp_sealed_metric": f"{config.certification_id}_paper_mlp_sealed_holdout_metric_v1",
        "live_head_to_head": f"{config.certification_id}_linear_vs_paper_mlp_live_sealed_v1",
    }


def _promotion_report_ids(report_ids: dict[str, str]) -> tuple[str, ...]:
    return (
        report_ids["linear_sealed_metric"],
        report_ids["calibration"],
        report_ids["feature_ablation"],
        report_ids["patching"],
        report_ids["failure_cases"],
        report_ids["linear_runtime_prevention"],
        report_ids["gateway_smoke"],
        report_ids["lineage"],
        report_ids["live_head_to_head"],
    )


def _artifact_path(artifact_paths: dict[str, JsonValue], key: str) -> str:
    value = artifact_paths.get(key)
    if not isinstance(value, str) or value == "":
        raise CiftCertificationWorkflowError(f"planned_artifacts.{key} must be present.")
    return value


def _model_access_flags(config: CiftCertificationWorkflowConfig) -> tuple[str, ...]:
    flags: list[str] = []
    if config.allow_download:
        flags.append("--allow-download")
    if config.trust_remote_code:
        flags.append("--trust-remote-code")
    return tuple(flags)


def _required_runtime_prevention_device_flags(config: CiftCertificationWorkflowConfig) -> tuple[str, ...]:
    if config.requested_device == "auto":
        return ()
    return ("--required-runtime-prevention-device", config.requested_device)


def _validate_requested_runtime_prevention_device(
    requested_device: str,
    runtime_prevention: Mapping[str, object],
    label: str,
) -> None:
    if requested_device == "auto":
        return
    selected_device = _json_string(runtime_prevention, "selected_device", label)
    if selected_device != requested_device:
        raise CiftCertificationWorkflowError(f"{label}.selected_device must match requested_device.")


def _validate_runtime_model_revision_binding(
    runtime_model: Mapping[str, object],
    model_metadata: CiftModelMetadataReport,
) -> None:
    runtime_revision = _json_string(runtime_model, "source_revision", "promoted runtime artifact")
    _validate_model_revision_policy(runtime_revision, "promoted runtime artifact source_revision")
    if runtime_revision != model_metadata.revision:
        raise CiftCertificationWorkflowError(
            "promoted runtime artifact source_revision must match model metadata revision."
        )


def _validate_evidence_manifest_config(config: CiftCertificationEvidenceManifestConfig) -> None:
    _validate_slug(config.certification_id, "certification_id")
    _validate_required_string(config.created_at, "created_at")
    _validate_required_string(config.behavior_id, "behavior_id")
    _validate_required_string(config.behavior_description, "behavior_description")
    _validate_required_string(config.requested_device, "requested_device")
    _validate_cift_contract(
        prompt_renderer=config.prompt_renderer,
        selected_choice_geometry=config.selected_choice_geometry,
        selected_choice_readout_token_count=config.selected_choice_readout_token_count,
    )
    _validate_dtype_name(config.dtype_name)
    _validate_finite_non_negative(config.metric_threshold, "metric_threshold")
    _validate_finite_non_negative(config.ablation_delta_threshold, "ablation_delta_threshold")
    _validate_existing_file(config.repository_root, config.model_metadata_report_path, "model_metadata_report_path")
    _validate_existing_file(config.repository_root, config.activation_artifact_path, "activation_artifact_path")
    _validate_existing_file(config.repository_root, config.linear_bundle_path, "linear_bundle_path")
    if not isinstance(config.grouped_head_to_head_report_path, Path):
        raise CiftCertificationWorkflowError("grouped_head_to_head_report_path must be provided.")
    _validate_existing_file(
        config.repository_root,
        config.grouped_head_to_head_report_path,
        "grouped_head_to_head_report_path",
    )
    _validate_existing_file(config.repository_root, config.calibration_report_path, "calibration_report_path")
    _validate_existing_file(config.repository_root, config.feature_ablation_report_path, "feature_ablation_report_path")
    _validate_existing_file(config.repository_root, config.patching_report_path, "patching_report_path")
    _validate_existing_file(config.repository_root, config.failure_cases_report_path, "failure_cases_report_path")
    _validate_existing_file(config.repository_root, config.lineage_report_path, "lineage_report_path")
    _validate_existing_file(
        config.repository_root,
        config.device_preflight_report_path,
        "device_preflight_report_path",
    )
    _validate_existing_file(
        config.repository_root,
        config.live_runtime_prevention_report_path,
        "live_runtime_prevention_report_path",
    )
    _validate_existing_file(config.repository_root, config.sealed_holdout_metric_path, "sealed_holdout_metric_path")
    _validate_existing_file(config.repository_root, config.gateway_smoke_report_path, "gateway_smoke_report_path")
    _validate_existing_file(
        config.repository_root,
        config.paper_mlp_runtime_prevention_report_path,
        "paper_mlp_runtime_prevention_report_path",
    )
    _validate_existing_file(
        config.repository_root,
        config.paper_mlp_sealed_holdout_metric_path,
        "paper_mlp_sealed_holdout_metric_path",
    )
    _validate_existing_file(
        config.repository_root, config.live_head_to_head_report_path, "live_head_to_head_report_path"
    )
    _validate_existing_file(
        config.repository_root,
        config.promoted_runtime_artifact_path,
        "promoted_runtime_artifact_path",
    )
    _validate_output_dir(config.repository_root, config.promotion_report_output_dir)
    _validate_output_dir(config.repository_root, config.evidence_chain_verification_report_path.parent)


def _validate_config(config: CiftCertificationWorkflowConfig) -> None:
    _validate_slug(config.certification_id, "certification_id")
    _validate_required_string(config.model_id, "model_id")
    _validate_required_string(config.revision, "revision")
    _validate_model_revision_policy(config.revision, "revision")
    _validate_required_string(config.training_dataset_id, "training_dataset_id")
    _validate_required_string(config.task_name, "task_name")
    _validate_required_string(config.positive_label, "positive_label")
    _validate_required_string(config.behavior_id, "behavior_id")
    _validate_required_string(config.behavior_description, "behavior_description")
    _validate_required_string(config.candidate_feature_key, "candidate_feature_key")
    _validate_required_string(config.requested_device, "requested_device")
    _validate_model_device_policy(model_id=config.model_id, requested_device=config.requested_device)
    _validate_cift_contract(
        prompt_renderer=config.prompt_renderer,
        selected_choice_geometry=config.selected_choice_geometry,
        selected_choice_readout_token_count=config.selected_choice_readout_token_count,
    )
    _validate_required_string(config.dtype_name, "dtype_name")
    _validate_dtype_name(config.dtype_name)
    _validate_required_string(config.created_at, "created_at")
    _validate_finite_non_negative(config.metric_threshold, "metric_threshold")
    _validate_finite_non_negative(config.ablation_delta_threshold, "ablation_delta_threshold")
    _validate_existing_file(config.repository_root, config.corpus_path, "corpus_path")
    _validate_existing_file(config.repository_root, config.runtime_turns_path, "runtime_turns_path")
    _validate_existing_file(
        config.repository_root,
        config.selected_choice_runtime_model_path,
        "selected_choice_runtime_model_path",
    )
    _validate_output_dir(config.repository_root, config.output_dir)
    if len(config.layer_indices) == 0:
        raise CiftCertificationWorkflowError("layer_indices must not be empty.")
    for layer_index in config.layer_indices:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int) or layer_index < 0:
            raise CiftCertificationWorkflowError("layer_indices must contain non-negative integers.")
    if len(config.pooling_methods) == 0:
        raise CiftCertificationWorkflowError("pooling_methods must not be empty.")
    for pooling_method in config.pooling_methods:
        _validate_slug(pooling_method, "pooling_methods")


def _validate_cift_contract(
    prompt_renderer: str,
    selected_choice_geometry: str,
    selected_choice_readout_token_count: int,
) -> None:
    if prompt_renderer != CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1:
        raise CiftCertificationWorkflowError(f"prompt_renderer must be {CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1}.")
    if selected_choice_geometry != CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1:
        raise CiftCertificationWorkflowError(
            f"selected_choice_geometry must be {CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1}."
        )
    if (
        isinstance(selected_choice_readout_token_count, bool)
        or not isinstance(selected_choice_readout_token_count, int)
        or selected_choice_readout_token_count < 1
    ):
        raise CiftCertificationWorkflowError("selected_choice_readout_token_count must be a positive integer.")


def _validate_dtype_name(dtype_name: str) -> None:
    _validate_required_string(dtype_name, "dtype_name")
    if dtype_name not in {"auto", "device", "float32", "float16", "bfloat16"}:
        raise CiftCertificationWorkflowError("dtype_name must be one of auto, device, float32, float16, or bfloat16.")


def _validate_model_metadata(
    config: CiftCertificationWorkflowConfig,
    model_metadata: CiftModelMetadataReport,
) -> None:
    _validate_model_device_policy(model_id=model_metadata.model_id, requested_device=config.requested_device)
    if model_metadata.model_id != config.model_id:
        raise CiftCertificationWorkflowError("model_id must match discovered model metadata.")
    if model_metadata.revision != config.revision:
        raise CiftCertificationWorkflowError("revision must match discovered model metadata.")
    if model_metadata.support_state != CIFT_SUPPORT_STATE_CALIBRATION_READY:
        raise CiftCertificationWorkflowError("model metadata support_state must be calibration-ready.")
    if model_metadata.requested_device != config.requested_device:
        raise CiftCertificationWorkflowError("model metadata requested_device must match certification config.")
    if model_metadata.selected_device != config.requested_device and config.requested_device != "auto":
        raise CiftCertificationWorkflowError("model metadata selected_device must match requested_device.")
    if model_metadata.dtype_name != config.dtype_name:
        raise CiftCertificationWorkflowError("model metadata dtype_name must match certification config.")
    if config.candidate_feature_key not in model_metadata.selected_readout_candidates:
        raise CiftCertificationWorkflowError(
            "candidate_feature_key must be present in model metadata selected_readout_candidates."
        )
    _validate_model_revision_policy(model_metadata.revision, "model metadata revision")
    if model_metadata.hidden_size < 1:
        raise CiftCertificationWorkflowError("model metadata hidden_size must be positive.")
    if model_metadata.layer_count < 1:
        raise CiftCertificationWorkflowError("model metadata layer_count must be positive.")
    _validate_config_layers_against_model_metadata(config=config, model_metadata=model_metadata)


def _validate_config_layers_against_model_metadata(
    config: CiftCertificationWorkflowConfig,
    model_metadata: CiftModelMetadataReport,
) -> None:
    for layer_index in config.layer_indices:
        if layer_index >= model_metadata.layer_count:
            raise CiftCertificationWorkflowError("layer_indices must be within discovered model metadata layer_count.")
    _validate_candidate_feature_layer(
        candidate_feature_key=config.candidate_feature_key,
        model_metadata=model_metadata,
        field_name="candidate_feature_key",
    )


def _validate_candidate_feature_layer(
    candidate_feature_key: str,
    model_metadata: CiftModelMetadataReport,
    field_name: str,
) -> None:
    layer_index = _layer_index_from_feature_key(candidate_feature_key)
    if layer_index >= model_metadata.layer_count:
        raise CiftCertificationWorkflowError(f"{field_name} layer index must be within model metadata layer_count.")


def _validate_model_device_policy(model_id: str, requested_device: str) -> None:
    if model_id == "Qwen/Qwen3-4B" and requested_device != "mps":
        raise CiftCertificationWorkflowError("Qwen/Qwen3-4B CIFT certification requires requested_device mps.")


def _validate_model_revision_policy(revision: str, field_name: str) -> None:
    if not is_cift_immutable_model_revision(revision):
        raise CiftCertificationWorkflowError(
            f"{field_name} must be an immutable lowercase 40-character Git commit SHA "
            "or sha256:<64 lowercase hex digest>."
        )


def _validate_required_string(value: str, field_name: str) -> None:
    if value == "":
        raise CiftCertificationWorkflowError(f"{field_name} must not be empty.")


def _validate_finite_non_negative(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise CiftCertificationWorkflowError(f"{field_name} must be finite and non-negative.")


def _validate_slug(value: str, field_name: str) -> None:
    _validate_required_string(value, field_name)
    allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    for character in value:
        if character not in allowed_characters:
            raise CiftCertificationWorkflowError(f"{field_name} contains unsupported character '{character}'.")


def _validate_existing_file(repository_root: Path, path: Path, field_name: str) -> None:
    resolved_path = _resolved_repository_path(repository_root, path, field_name)
    if not resolved_path.is_file():
        raise CiftCertificationWorkflowError(f"{field_name} must be an existing file: {path}.")


def _validate_output_dir(repository_root: Path, path: Path) -> None:
    _resolved_repository_path(repository_root, path, "output_dir")


def _repository_relative_path(repository_root: Path, path: Path) -> str:
    return str(_resolved_repository_path(repository_root, path, "path").relative_to(repository_root.resolve()))


def _optional_repository_relative_path(repository_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    return _repository_relative_path(repository_root, path)


def _model_metadata_report_from_path(path: Path) -> CiftModelMetadataReport:
    record = _json_object_from_path(path, "model metadata report")
    return CiftModelMetadataReport(
        schema_version=_json_string(record, "schema_version", "model metadata report"),
        support_state=_optional_json_string(
            record,
            "support_state",
            "model metadata report",
            CIFT_SUPPORT_STATE_CALIBRATION_READY,
        ),
        model_id=_json_string(record, "model_id", "model metadata report"),
        revision=_json_string(record, "revision", "model metadata report"),
        resolved_revision=_optional_json_string(
            record,
            "resolved_revision",
            "model metadata report",
            _json_string(record, "revision", "model metadata report"),
        ),
        model_type=_json_string(record, "model_type", "model metadata report"),
        hidden_size=_json_int(record, "hidden_size", "model metadata report"),
        layer_count=_json_int(record, "layer_count", "model metadata report"),
        requested_device=_optional_json_string(record, "requested_device", "model metadata report", "mps"),
        selected_device=_optional_json_string(record, "selected_device", "model metadata report", "mps"),
        dtype_name=_optional_json_string(record, "dtype_name", "model metadata report", "device"),
        resolved_torch_dtype=_optional_json_string(
            record,
            "resolved_torch_dtype",
            "model metadata report",
            "torch.float16",
        ),
        hidden_state_support=_optional_json_string(
            record,
            "hidden_state_support",
            "model metadata report",
            "legacy_v1_assumed_configurable_output_hidden_states",
        ),
        hidden_state_capable=_optional_json_bool(record, "hidden_state_capable", "model metadata report", True),
        selected_readout_candidates=_optional_json_string_tuple(
            record,
            "selected_readout_candidates",
            "model metadata report",
        ),
        failure_reason=_optional_json_nullable_string(record, "failure_reason", "model metadata report"),
        tokenizer_class=_json_string(record, "tokenizer_class", "model metadata report"),
        tokenizer_vocab_size=_json_int(record, "tokenizer_vocab_size", "model metadata report"),
        tokenizer_fingerprint_sha256=_json_string(record, "tokenizer_fingerprint_sha256", "model metadata report"),
        special_tokens_map_sha256=_json_string(record, "special_tokens_map_sha256", "model metadata report"),
        chat_template_present=_json_bool(record, "chat_template_present", "model metadata report"),
        chat_template_sha256=_json_string(record, "chat_template_sha256", "model metadata report"),
    )


def _json_object_from_path(path: Path, label: str) -> dict[str, object]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CiftCertificationWorkflowError(f"Invalid {label} JSON in {path}: {exc.msg}.") from exc
    if not isinstance(decoded, dict):
        raise CiftCertificationWorkflowError(f"{label} must contain a JSON object: {path}.")
    return cast(dict[str, object], decoded)


def _report_id_from_path(path: Path) -> str:
    record = _json_object_from_path(path, "report artifact")
    return _json_string(record, "report_id", "report artifact")


def _json_string(record: Mapping[str, object], field_name: str, label: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or value == "":
        raise CiftCertificationWorkflowError(f"{label} {field_name} must be a non-empty string.")
    return value


def _optional_json_string(
    record: Mapping[str, object],
    field_name: str,
    label: str,
    fallback_value: str,
) -> str:
    value = record.get(field_name)
    if value is None:
        return fallback_value
    if not isinstance(value, str) or value == "":
        raise CiftCertificationWorkflowError(f"{label} {field_name} must be a non-empty string when present.")
    return value


def _optional_json_nullable_string(record: Mapping[str, object], field_name: str, label: str) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise CiftCertificationWorkflowError(f"{label} {field_name} must be a non-empty string when present.")
    return value


def _optional_json_bool(
    record: Mapping[str, object],
    field_name: str,
    label: str,
    fallback_value: bool,
) -> bool:
    value = record.get(field_name)
    if value is None:
        return fallback_value
    if not isinstance(value, bool):
        raise CiftCertificationWorkflowError(f"{label} {field_name} must be a boolean when present.")
    return value


def _optional_json_string_tuple(
    record: Mapping[str, object],
    field_name: str,
    label: str,
) -> tuple[str, ...]:
    value = record.get(field_name)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise CiftCertificationWorkflowError(f"{label} {field_name} must be a list when present.")
    candidates: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or item == "":
            raise CiftCertificationWorkflowError(f"{label} {field_name}[{index}] must be a non-empty string.")
        candidates.append(item)
    return tuple(candidates)


def _json_int(record: dict[str, object], field_name: str, label: str) -> int:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftCertificationWorkflowError(f"{label} {field_name} must be an integer.")
    return value


def _json_bool(record: dict[str, object], field_name: str, label: str) -> bool:
    value = record.get(field_name)
    if not isinstance(value, bool):
        raise CiftCertificationWorkflowError(f"{label} {field_name} must be a boolean.")
    return value


def _layer_index_from_feature_key(feature_key: str) -> int:
    marker = "_layer_"
    if marker not in feature_key:
        raise CiftCertificationWorkflowError("feature_key must contain '_layer_'.")
    suffix = feature_key.rsplit(marker, maxsplit=1)[1]
    if not suffix.isdigit():
        raise CiftCertificationWorkflowError("feature_key layer suffix must be an integer.")
    return int(suffix)


def _pooling_method_from_feature_key(feature_key: str) -> str:
    marker = "_layer_"
    if marker not in feature_key:
        raise CiftCertificationWorkflowError("feature_key must contain '_layer_'.")
    pooling_method = feature_key.rsplit(marker, maxsplit=1)[0]
    _validate_slug(pooling_method, "pooling_method")
    return pooling_method


def _resolved_repository_path(repository_root: Path, path: Path, field_name: str) -> Path:
    resolved_root = repository_root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise CiftCertificationWorkflowError(f"{field_name} must stay inside repository_root.")
    return resolved_path


def _layer_suffix(layer_indices: tuple[int, ...]) -> str:
    if len(layer_indices) == 1:
        return f"l{layer_indices[0]}"
    return f"l{layer_indices[0]}_l{layer_indices[-1]}"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
