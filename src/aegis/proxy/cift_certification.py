from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from aegis.cift_contract import (
    CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
    CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
    CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
    is_cift_immutable_model_revision,
)
from aegis.core.action_severity import action_severity
from aegis.core.contracts import Action
from aegis.detectors.cift_runtime import (
    CiftRuntimeDetectorError,
    CiftRuntimeModel,
    cift_runtime_model_to_dict,
    load_cift_runtime_model_with_sha256,
)

_MANIFEST_SCHEMA_VERSION = "aegis_introspection.cift_certification_workflow/v1"
_WORKFLOW_RUN_SCHEMA_VERSION = "aegis_introspection.cift_certification_workflow_run/v1"
_RELEASE_GATE_SCHEMA_VERSION = "aegis_introspection.cift_release_gate/v1"
_RUNTIME_SCHEMA_VERSIONS = ("aegis.cift_runtime_linear/v1", "aegis.cift_runtime_mlp/v1")
_TRUSTED_SELF_HOSTED_FEATURE_SOURCE = "self_hosted_activation_extractor"
_GATEWAY_SMOKE_BOOTSTRAP_CERTIFICATION_MODE = "gateway_smoke_bootstrap"
_STRICT_CERTIFICATION_MODE = "strict"


class CiftCertificationBindingError(ValueError):
    """Raised when a CIFT runtime artifact is not bound to certification evidence."""


@dataclass(frozen=True)
class CiftCertificationBindingConfig:
    runtime_model_path: Path
    certification_manifest_path: Path
    certification_report_path: Path
    certification_artifact_root: Path
    release_gate_report_path: Path | None
    required_device: str
    expected_manifest_sha256: str
    expected_report_sha256: str
    expected_release_gate_report_sha256: str | None
    expected_detector_name: str
    expected_extractor_id: str
    expected_feature_source: str
    expected_prompt_renderer: str
    expected_selected_choice_geometry: str
    expected_selected_choice_readout_token_count: int


@dataclass(frozen=True)
class CiftCertificationBinding:
    runtime_sha256: str
    manifest_sha256: str
    report_sha256: str
    release_gate_report_sha256: str | None
    certification_id: str


@dataclass(frozen=True)
class _RequiredCertificationArtifact:
    role: str
    artifact_kind: str
    schema_versions: tuple[str | None, ...]
    report_id_required: bool


@dataclass(frozen=True)
class _ValidatedCertificationManifest:
    certification_id: str
    artifacts_by_role: dict[str, Mapping[str, object]]


@dataclass(frozen=True)
class _LoadedJsonObject:
    sha256: str
    record: Mapping[str, object]


@dataclass(frozen=True)
class _LoadedArtifactFile:
    sha256: str
    record: Mapping[str, object] | None


@dataclass(frozen=True)
class _PromotionEvidenceReportReference:
    field_label: str
    report_id: str
    manifest_role: str


_REQUIRED_CERTIFICATION_ARTIFACTS = (
    _RequiredCertificationArtifact(
        role="model_metadata",
        artifact_kind="json_report",
        schema_versions=("aegis_introspection.cift_model_metadata/v1",),
        report_id_required=False,
    ),
    _RequiredCertificationArtifact(
        role="device_preflight",
        artifact_kind="json_report",
        schema_versions=("aegis_introspection.device_preflight/v1",),
        report_id_required=False,
    ),
    _RequiredCertificationArtifact(
        role="calibration_activation_artifact",
        artifact_kind="activation_tensor",
        schema_versions=(None,),
        report_id_required=False,
    ),
    _RequiredCertificationArtifact(
        role="linear_candidate_bundle",
        artifact_kind="model_bundle",
        schema_versions=("cift_model_bundle/v1",),
        report_id_required=False,
    ),
    _RequiredCertificationArtifact(
        role="calibration",
        artifact_kind="json_report",
        schema_versions=("aegis_introspection.cift_calibration/v1",),
        report_id_required=True,
    ),
    _RequiredCertificationArtifact(
        role="feature_ablation",
        artifact_kind="json_report",
        schema_versions=("aegis_introspection.cift_feature_ablation/v1",),
        report_id_required=True,
    ),
    _RequiredCertificationArtifact(
        role="counterfactual_patching",
        artifact_kind="json_report",
        schema_versions=("aegis_introspection.cift_counterfactual_patching/v1",),
        report_id_required=True,
    ),
    _RequiredCertificationArtifact(
        role="failure_cases",
        artifact_kind="json_report",
        schema_versions=("aegis_introspection.cift_failure_cases/v1",),
        report_id_required=True,
    ),
    _RequiredCertificationArtifact(
        role="lineage",
        artifact_kind="json_report",
        schema_versions=("aegis_introspection.cift_lineage/v1",),
        report_id_required=True,
    ),
    _RequiredCertificationArtifact(
        role="linear_live_runtime_prevention",
        artifact_kind="json_report",
        schema_versions=("aegis_introspection.cift_live_window_selector_benchmark/v1",),
        report_id_required=True,
    ),
    _RequiredCertificationArtifact(
        role="linear_sealed_holdout_metric",
        artifact_kind="json_report",
        schema_versions=("aegis_introspection.cift_sealed_holdout_metric/v1",),
        report_id_required=True,
    ),
    _RequiredCertificationArtifact(
        role="linear_gateway_smoke",
        artifact_kind="json_report",
        schema_versions=("aegis.proxy.cift_gateway_smoke/v1",),
        report_id_required=True,
    ),
    _RequiredCertificationArtifact(
        role="paper_mlp_live_runtime_prevention",
        artifact_kind="json_report",
        schema_versions=("aegis_introspection.cift_live_window_selector_benchmark/v1",),
        report_id_required=True,
    ),
    _RequiredCertificationArtifact(
        role="paper_mlp_sealed_holdout_metric",
        artifact_kind="json_report",
        schema_versions=("aegis_introspection.cift_sealed_holdout_metric/v1",),
        report_id_required=True,
    ),
    _RequiredCertificationArtifact(
        role="live_sealed_linear_vs_paper_mlp",
        artifact_kind="json_report",
        schema_versions=("aegis_introspection.cift_live_probe_competition/v1",),
        report_id_required=True,
    ),
    _RequiredCertificationArtifact(
        role="promotion_evidence",
        artifact_kind="promotion_evidence",
        schema_versions=("cift_promotion_evidence/v1",),
        report_id_required=False,
    ),
    _RequiredCertificationArtifact(
        role="promoted_runtime",
        artifact_kind="runtime_model",
        schema_versions=_RUNTIME_SCHEMA_VERSIONS,
        report_id_required=False,
    ),
    _RequiredCertificationArtifact(
        role="evidence_chain_verification",
        artifact_kind="json_report",
        schema_versions=("aegis_introspection.cift_evidence_chain_verification/v1",),
        report_id_required=False,
    ),
    _RequiredCertificationArtifact(
        role="grouped_cv_linear_vs_paper_mlp",
        artifact_kind="json_report",
        schema_versions=("cift_probe_competition/v1",),
        report_id_required=True,
    ),
)

_PAPER_MLP_PROBE_ARCHITECTURE = "mlp_128_64_1"
_LINEAR_PROBE_ARCHITECTURE = "linear_logistic_regression"


def validate_cift_certification_binding(config: CiftCertificationBindingConfig) -> CiftCertificationBinding:
    if config.required_device == "":
        raise CiftCertificationBindingError("required_device must not be empty.")
    if config.expected_detector_name == "":
        raise CiftCertificationBindingError("expected_detector_name must not be empty.")
    if config.expected_extractor_id == "":
        raise CiftCertificationBindingError("expected_extractor_id must not be empty.")
    if config.expected_feature_source != _TRUSTED_SELF_HOSTED_FEATURE_SOURCE:
        raise CiftCertificationBindingError(f"expected_feature_source must be {_TRUSTED_SELF_HOSTED_FEATURE_SOURCE}.")
    _validate_expected_cift_contract(config)
    _validate_expected_sha256(value=config.expected_manifest_sha256, label="expected_manifest_sha256")
    _validate_expected_sha256(value=config.expected_report_sha256, label="expected_report_sha256")
    if (config.release_gate_report_path is None) != (config.expected_release_gate_report_sha256 is None):
        raise CiftCertificationBindingError(
            "release_gate_report_path and expected_release_gate_report_sha256 must be provided together."
        )
    if config.expected_release_gate_report_sha256 is not None:
        _validate_expected_sha256(
            value=config.expected_release_gate_report_sha256,
            label="expected_release_gate_report_sha256",
        )
    runtime = _load_json_object(config.runtime_model_path, "runtime_model_path")
    runtime_sha256 = runtime.sha256
    manifest = _load_json_object(config.certification_manifest_path, "certification_manifest_path")
    report = _load_json_object(config.certification_report_path, "certification_report_path")
    release_gate_report = (
        None
        if config.release_gate_report_path is None
        else _load_json_object(config.release_gate_report_path, "release_gate_report_path")
    )
    try:
        runtime_model = load_cift_runtime_model_with_sha256(
            path=config.runtime_model_path,
            expected_sha256=runtime.sha256,
        )
    except CiftRuntimeDetectorError as exc:
        raise CiftCertificationBindingError(str(exc)) from exc
    if runtime_model.source_selected_device != config.required_device:
        raise CiftCertificationBindingError(
            "runtime_model.source_selected_device must match required_device, "
            f"got {runtime_model.source_selected_device}."
        )
    _validate_runtime_candidate_certification_scope(runtime.record)
    _validate_immutable_model_revision(runtime_model.source_revision, "runtime_model.source_revision")
    _validate_model_device_policy(model_id=runtime_model.source_model_id, required_device=config.required_device)
    _expect_sha256(
        actual_sha256=manifest.sha256,
        expected_sha256=config.expected_manifest_sha256,
        label="certification_manifest_path",
    )
    _expect_sha256(
        actual_sha256=report.sha256,
        expected_sha256=config.expected_report_sha256,
        label="certification_report_path",
    )
    if release_gate_report is not None and config.expected_release_gate_report_sha256 is not None:
        _expect_sha256(
            actual_sha256=release_gate_report.sha256,
            expected_sha256=config.expected_release_gate_report_sha256,
            label="release_gate_report_path",
        )
    validated_manifest = _validate_manifest(
        manifest=manifest.record,
        runtime_sha256=runtime_sha256,
        required_device=config.required_device,
        expected_prompt_renderer=config.expected_prompt_renderer,
        expected_selected_choice_geometry=config.expected_selected_choice_geometry,
        expected_selected_choice_readout_token_count=config.expected_selected_choice_readout_token_count,
    )
    _validate_workflow_run_report(
        report=report.record,
        runtime_sha256=runtime_sha256,
        manifest=validated_manifest,
    )
    _validate_materialized_artifact_files(config=config, manifest=validated_manifest)
    _validate_identity_evidence(
        config=config,
        runtime_model=runtime_model,
        manifest=validated_manifest,
    )
    _validate_semantic_evidence(
        config=config,
        runtime_model=runtime_model,
        manifest=validated_manifest,
    )
    if release_gate_report is not None:
        _validate_release_gate_report(
            config=config,
            runtime_model=runtime_model,
            report=release_gate_report.record,
            runtime_sha256=runtime_sha256,
        )
    return CiftCertificationBinding(
        runtime_sha256=runtime_sha256,
        manifest_sha256=manifest.sha256,
        report_sha256=report.sha256,
        release_gate_report_sha256=None if release_gate_report is None else release_gate_report.sha256,
        certification_id=validated_manifest.certification_id,
    )


def _validate_release_gate_report(
    config: CiftCertificationBindingConfig,
    runtime_model: CiftRuntimeModel,
    report: Mapping[str, object],
    runtime_sha256: str,
) -> None:
    label = "release_gate_report"
    _expect_string(report, "schema_version", label, _RELEASE_GATE_SCHEMA_VERSION)
    _expect_string(report, "runtime_model_sha256", label, runtime_sha256)
    _expect_string(report, "model_bundle_id", label, runtime_model.model_bundle_id)
    _expect_string(report, "candidate_status", label, "runtime_candidate")
    _expect_string(report, "required_runtime_prevention_device", label, config.required_device)
    _expect_string(report, "evidence_mode", label, "certification_bound")
    _expect_bool(report, "eligible", label, True)
    _expect_bool(report, "diagnostic_eligible", label, False)
    _expect_bool(report, "production_release_eligible", label, True)
    _expect_empty_string_list(report, "failed_requirements", label)
    _expect_resolved_path(
        config=config,
        record=report,
        field_name="runtime_model_path",
        label=label,
        expected_path=config.runtime_model_path,
    )
    certification_binding = _required_mapping(report.get("certification_binding"), f"{label}.certification_binding")
    _expect_bool(certification_binding, "requested", f"{label}.certification_binding", True)
    _expect_string(
        certification_binding,
        "certification_manifest_sha256",
        f"{label}.certification_binding",
        config.expected_manifest_sha256,
    )
    _expect_string(
        certification_binding,
        "certification_report_sha256",
        f"{label}.certification_binding",
        config.expected_report_sha256,
    )
    _expect_resolved_path(
        config=config,
        record=certification_binding,
        field_name="certification_manifest_path",
        label=f"{label}.certification_binding",
        expected_path=config.certification_manifest_path,
    )
    _expect_resolved_path(
        config=config,
        record=certification_binding,
        field_name="certification_report_path",
        label=f"{label}.certification_binding",
        expected_path=config.certification_report_path,
    )
    _expect_resolved_path(
        config=config,
        record=certification_binding,
        field_name="certification_artifact_root",
        label=f"{label}.certification_binding",
        expected_path=config.certification_artifact_root,
    )
    expected_runtime_contract = _required_mapping(
        report.get("expected_runtime_contract"),
        f"{label}.expected_runtime_contract",
    )
    _expect_string(
        expected_runtime_contract,
        "detector_name",
        f"{label}.expected_runtime_contract",
        config.expected_detector_name,
    )
    _expect_string(
        expected_runtime_contract,
        "extractor_id",
        f"{label}.expected_runtime_contract",
        config.expected_extractor_id,
    )
    _expect_string(
        expected_runtime_contract,
        "feature_source",
        f"{label}.expected_runtime_contract",
        config.expected_feature_source,
    )
    _expect_int(
        expected_runtime_contract,
        "selected_choice_readout_token_count",
        f"{label}.expected_runtime_contract",
        config.expected_selected_choice_readout_token_count,
    )


def _validate_runtime_candidate_certification_scope(runtime_record: Mapping[str, object]) -> None:
    promotion_gates = _required_mapping(runtime_record.get("promotion_gates"), "runtime_model.promotion_gates")
    runtime_candidate = _required_mapping(
        promotion_gates.get("runtime_candidate"),
        "runtime_model.promotion_gates.runtime_candidate",
    )
    label = "runtime_model.promotion_gates.runtime_candidate"
    _expect_string(runtime_candidate, "eligibility_scope", label, "runtime_candidate_promotion_only")
    _expect_bool(runtime_candidate, "production_release_eligible", label, False)
    _expect_bool(runtime_candidate, "requires_certification_binding", label, True)


def _expect_empty_string_list(record: Mapping[str, object], field_name: str, label: str) -> None:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise CiftCertificationBindingError(f"{label}.{field_name} must be a list.")
    if len(value) > 0:
        if any(not isinstance(item, str) for item in value):
            raise CiftCertificationBindingError(f"{label}.{field_name} must contain only strings.")
        raise CiftCertificationBindingError(f"{label}.{field_name} must be empty.")


def _expect_resolved_path(
    config: CiftCertificationBindingConfig,
    record: Mapping[str, object],
    field_name: str,
    label: str,
    expected_path: Path,
) -> None:
    path_text = _required_string(record, field_name, label)
    actual_path = _resolve_manifest_artifact_path(config=config, path_text=path_text)
    if actual_path.resolve() != expected_path.resolve():
        raise CiftCertificationBindingError(f"{label}.{field_name} must match {expected_path}.")


def _validate_expected_cift_contract(config: CiftCertificationBindingConfig) -> None:
    if config.expected_prompt_renderer != CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1:
        raise CiftCertificationBindingError(f"expected_prompt_renderer must be {CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1}.")
    if config.expected_selected_choice_geometry != CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1:
        raise CiftCertificationBindingError(
            f"expected_selected_choice_geometry must be {CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1}."
        )
    if config.expected_selected_choice_readout_token_count < 1:
        raise CiftCertificationBindingError("expected_selected_choice_readout_token_count must be positive.")


def _validate_model_device_policy(model_id: str, required_device: str) -> None:
    if model_id == "Qwen/Qwen3-4B" and required_device != "mps":
        raise CiftCertificationBindingError("Qwen/Qwen3-4B certification requires required_device mps.")


def _validate_immutable_model_revision(revision: str, label: str) -> None:
    if not is_cift_immutable_model_revision(revision):
        raise CiftCertificationBindingError(
            f"{label} must be an immutable lowercase 40-character Git commit SHA or sha256:<64 lowercase hex digest>."
        )


def _validate_manifest(
    manifest: Mapping[str, object],
    runtime_sha256: str,
    required_device: str,
    expected_prompt_renderer: str,
    expected_selected_choice_geometry: str,
    expected_selected_choice_readout_token_count: int,
) -> _ValidatedCertificationManifest:
    _expect_string(manifest, "schema_version", "certification manifest", _MANIFEST_SCHEMA_VERSION)
    _expect_string(manifest, "status", "certification manifest", "evidence_bound")
    certification_id = _required_string(manifest, "certification_id", "certification manifest")
    model_identity = _required_mapping(manifest.get("model_identity"), "certification manifest.model_identity")
    _validate_immutable_model_revision(
        _required_string(model_identity, "revision", "certification manifest.model_identity"),
        "certification manifest.model_identity.revision",
    )
    training = _required_mapping(manifest.get("training"), "certification manifest.training")
    _expect_string(training, "requested_device", "certification manifest.training", required_device)
    _expect_string(training, "prompt_renderer", "certification manifest.training", expected_prompt_renderer)
    _expect_string(
        training,
        "selected_choice_geometry",
        "certification manifest.training",
        expected_selected_choice_geometry,
    )
    _expect_int(
        training,
        "selected_choice_readout_token_count",
        "certification manifest.training",
        expected_selected_choice_readout_token_count,
    )
    artifacts = _required_artifacts(manifest, "certification manifest")
    artifacts_by_role = _artifacts_by_role(artifacts=artifacts, label="certification manifest")
    for spec in _REQUIRED_CERTIFICATION_ARTIFACTS:
        artifact = artifacts_by_role.get(spec.role)
        if artifact is None:
            raise CiftCertificationBindingError(
                f"certification manifest must contain an artifact with role {spec.role}."
            )
        _validate_manifest_artifact_spec(artifact=artifact, spec=spec, runtime_sha256=runtime_sha256)
    for artifact in artifacts_by_role.values():
        _validate_manifest_release_artifact(artifact=artifact, runtime_sha256=runtime_sha256)
    return _ValidatedCertificationManifest(certification_id=certification_id, artifacts_by_role=artifacts_by_role)


def _validate_manifest_release_artifact(
    artifact: Mapping[str, object],
    runtime_sha256: str,
) -> None:
    role = _required_string(artifact, "role", "certification manifest artifact")
    label = f"certification manifest.{role}"
    _required_string(artifact, "artifact_kind", label)
    _expect_string(artifact, "status", label, "materialized")
    _optional_schema_version(artifact, "schema_version", label)
    _required_string(artifact, "path", label)
    if artifact.get("required_for_release") is not True:
        raise CiftCertificationBindingError(f"{label} must be required_for_release.")
    artifact_sha256 = _required_string(artifact, "sha256", label)
    _validate_expected_sha256(value=artifact_sha256, label=f"{label}.sha256")
    if role == "promoted_runtime" and artifact_sha256 != runtime_sha256:
        raise CiftCertificationBindingError(f"{label}.sha256 must be {runtime_sha256}, got {artifact_sha256}.")
    report_id = artifact.get("report_id")
    if report_id is not None and (not isinstance(report_id, str) or report_id == ""):
        raise CiftCertificationBindingError(f"{label}.report_id must be null or a non-empty string.")


def _validate_manifest_artifact_spec(
    artifact: Mapping[str, object],
    spec: _RequiredCertificationArtifact,
    runtime_sha256: str,
) -> None:
    label = f"certification manifest.{spec.role}"
    _expect_string(artifact, "artifact_kind", label, spec.artifact_kind)
    _expect_schema_version(artifact, "schema_version", label, spec.schema_versions)
    if spec.role == "promoted_runtime":
        artifact_sha256 = _required_string(artifact, "sha256", label)
        if artifact_sha256 != runtime_sha256:
            raise CiftCertificationBindingError(f"{label}.sha256 must be {runtime_sha256}, got {artifact_sha256}.")
    _validate_manifest_report_id(artifact=artifact, spec=spec, label=label)


def _validate_manifest_report_id(
    artifact: Mapping[str, object],
    spec: _RequiredCertificationArtifact,
    label: str,
) -> None:
    report_id = artifact.get("report_id")
    if spec.report_id_required:
        if not isinstance(report_id, str) or report_id == "":
            raise CiftCertificationBindingError(f"{label}.report_id must be a non-empty string.")
        return
    if report_id is not None and not isinstance(report_id, str):
        raise CiftCertificationBindingError(f"{label}.report_id must be null or a string.")


def _validate_workflow_run_report(
    report: Mapping[str, object],
    runtime_sha256: str,
    manifest: _ValidatedCertificationManifest,
) -> None:
    _expect_string(report, "schema_version", "certification workflow run", _WORKFLOW_RUN_SCHEMA_VERSION)
    _expect_string(report, "certification_id", "certification workflow run", manifest.certification_id)
    _expect_string(report, "mode", "certification workflow run", "execute")
    command_timeout_seconds = _required_finite_number(
        report,
        "command_timeout_seconds",
        "certification workflow run",
    )
    if command_timeout_seconds <= 0.0:
        raise CiftCertificationBindingError("certification workflow run command_timeout_seconds must be positive.")
    for field_name in ("plan_eligible", "evidence_eligible", "certification_eligible", "eligible"):
        if report.get(field_name) is not True:
            raise CiftCertificationBindingError(f"certification workflow run {field_name} must be true.")
    failed_requirements = report.get("failed_requirements")
    if not isinstance(failed_requirements, list) or len(failed_requirements) > 0:
        raise CiftCertificationBindingError("certification workflow run failed_requirements must be empty.")
    artifacts = _required_artifacts(report, "certification workflow run")
    workflow_artifacts_by_role = _artifacts_by_role(artifacts=artifacts, label="certification workflow run")
    _validate_workflow_artifact_roles(
        workflow_artifacts_by_role=workflow_artifacts_by_role,
        manifest_artifacts_by_role=manifest.artifacts_by_role,
    )
    _validate_artifact_count(
        record=report,
        field_name="artifact_count",
        label="certification workflow run",
        count=len(artifacts),
    )
    for role, manifest_artifact in manifest.artifacts_by_role.items():
        _validate_workflow_run_artifact(
            artifact=workflow_artifacts_by_role[role],
            role=role,
            manifest_artifact=manifest_artifact,
            runtime_sha256=runtime_sha256,
        )


def _validate_workflow_run_artifact(
    artifact: Mapping[str, object],
    role: str,
    manifest_artifact: Mapping[str, object],
    runtime_sha256: str,
) -> None:
    label = f"certification workflow run.{role}"
    _expect_string(artifact, "role", label, role)
    manifest_kind = _required_string(manifest_artifact, "artifact_kind", f"certification manifest.{role}")
    _expect_string(artifact, "artifact_kind", label, manifest_kind)
    _expect_string(artifact, "expected_status", label, "materialized")
    _expect_string(artifact, "actual_status", label, "verified")
    manifest_schema = manifest_artifact.get("schema_version")
    if artifact.get("expected_schema_version") != manifest_schema:
        raise CiftCertificationBindingError(f"{label}.expected_schema_version must match certification manifest.")
    actual_schema = artifact.get("actual_schema_version")
    if _requires_json_identity(manifest_kind):
        if actual_schema != manifest_schema:
            raise CiftCertificationBindingError(f"{label}.actual_schema_version must match certification manifest.")
    elif actual_schema is not None and actual_schema != manifest_schema:
        raise CiftCertificationBindingError(f"{label}.actual_schema_version must match certification manifest.")
    manifest_path = _required_string(manifest_artifact, "path", f"certification manifest.{role}")
    _expect_string(artifact, "path", label, manifest_path)
    expected_sha256 = (
        runtime_sha256
        if role == "promoted_runtime"
        else _required_string(
            manifest_artifact,
            "sha256",
            f"certification manifest.{role}",
        )
    )
    _expect_string(artifact, "expected_sha256", label, expected_sha256)
    _expect_string(artifact, "actual_sha256", label, expected_sha256)
    _validate_workflow_run_report_id(artifact=artifact, manifest_artifact=manifest_artifact, role=role, label=label)
    if artifact.get("required_for_release") is not True:
        raise CiftCertificationBindingError(f"{label} must be required_for_release.")
    if artifact.get("eligible") is not True:
        raise CiftCertificationBindingError(f"{label} must be eligible.")
    artifact_failures = artifact.get("failed_requirements")
    if not isinstance(artifact_failures, list) or len(artifact_failures) > 0:
        raise CiftCertificationBindingError(f"{label} failed_requirements must be empty.")


def _validate_workflow_run_report_id(
    artifact: Mapping[str, object],
    manifest_artifact: Mapping[str, object],
    role: str,
    label: str,
) -> None:
    manifest_report_id = manifest_artifact.get("report_id")
    expected_report_id = artifact.get("expected_report_id")
    actual_report_id = artifact.get("actual_report_id")
    spec = _required_artifact_spec(role)
    if (
        spec is not None
        and spec.report_id_required
        and (not isinstance(manifest_report_id, str) or manifest_report_id == "")
    ):
        raise CiftCertificationBindingError(f"certification manifest.{role}.report_id must be present.")
    if expected_report_id != manifest_report_id:
        raise CiftCertificationBindingError(f"{label}.expected_report_id must match certification manifest.")
    if actual_report_id != manifest_report_id:
        raise CiftCertificationBindingError(f"{label}.actual_report_id must match certification manifest.")


def _validate_materialized_artifact_files(
    config: CiftCertificationBindingConfig,
    manifest: _ValidatedCertificationManifest,
) -> None:
    for role, artifact in manifest.artifacts_by_role.items():
        _load_materialized_artifact_file(config=config, artifact=artifact, role=role)


def _load_materialized_artifact_file(
    config: CiftCertificationBindingConfig,
    artifact: Mapping[str, object],
    role: str,
) -> _LoadedArtifactFile:
    label = f"certification manifest.{role}.path"
    path_text = _required_string(artifact, "path", f"certification manifest.{role}")
    path = _resolve_manifest_artifact_path(config=config, path_text=path_text)
    if not path.is_file():
        raise CiftCertificationBindingError(f"{label} does not exist: {path}.")
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    expected_sha256 = _required_string(artifact, "sha256", f"certification manifest.{role}")
    if sha256 != expected_sha256:
        raise CiftCertificationBindingError(f"{label} sha256 must match certification manifest.")
    record = _optional_json_artifact_record(raw_bytes=raw_bytes, artifact=artifact, role=role, label=label)
    return _LoadedArtifactFile(sha256=sha256, record=record)


def _optional_json_artifact_record(
    raw_bytes: bytes,
    artifact: Mapping[str, object],
    role: str,
    label: str,
) -> Mapping[str, object] | None:
    artifact_kind = _required_string(artifact, "artifact_kind", f"certification manifest.{role}")
    if not _requires_json_identity(artifact_kind):
        return None
    try:
        decoded = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise CiftCertificationBindingError(f"{label} contains invalid JSON: {exc.msg}.") from exc
    except UnicodeDecodeError as exc:
        raise CiftCertificationBindingError(f"{label} must be UTF-8 JSON.") from exc
    if not isinstance(decoded, dict):
        raise CiftCertificationBindingError(f"{label} must contain a JSON object.")
    record = cast(Mapping[str, object], decoded)
    schema_version = artifact.get("schema_version")
    if isinstance(schema_version, str):
        _expect_string(record, "schema_version", label, schema_version)
    report_id = artifact.get("report_id")
    if artifact_kind == "json_report" and isinstance(report_id, str):
        _expect_string(record, "report_id", label, report_id)
    return record


def _validate_identity_evidence(
    config: CiftCertificationBindingConfig,
    runtime_model: CiftRuntimeModel,
    manifest: _ValidatedCertificationManifest,
) -> None:
    runtime_identity = _runtime_identity(runtime_model)
    evidence_chain = _load_manifest_json_artifact(config=config, manifest=manifest, role="evidence_chain_verification")
    gateway_smoke = _load_manifest_json_artifact(config=config, manifest=manifest, role="linear_gateway_smoke")
    _validate_evidence_chain_identity(
        config=config,
        runtime_identity=runtime_identity,
        evidence_chain=evidence_chain.record,
        gateway_smoke_report_id=_required_artifact_report_id(manifest, "linear_gateway_smoke"),
    )
    _validate_gateway_smoke_identity(
        config=config,
        runtime_identity=runtime_identity,
        gateway_smoke=gateway_smoke.record,
    )


def _validate_semantic_evidence(
    config: CiftCertificationBindingConfig,
    runtime_model: CiftRuntimeModel,
    manifest: _ValidatedCertificationManifest,
) -> None:
    runtime_identity = _runtime_identity(runtime_model)
    device_preflight = _load_manifest_json_artifact(config=config, manifest=manifest, role="device_preflight")
    promotion_evidence = _load_manifest_json_artifact(config=config, manifest=manifest, role="promotion_evidence")
    linear_runtime_prevention = _load_manifest_json_artifact(
        config=config,
        manifest=manifest,
        role="linear_live_runtime_prevention",
    )
    paper_mlp_runtime_prevention = _load_manifest_json_artifact(
        config=config,
        manifest=manifest,
        role="paper_mlp_live_runtime_prevention",
    )
    linear_sealed_holdout = _load_manifest_json_artifact(
        config=config,
        manifest=manifest,
        role="linear_sealed_holdout_metric",
    )
    paper_mlp_sealed_holdout = _load_manifest_json_artifact(
        config=config,
        manifest=manifest,
        role="paper_mlp_sealed_holdout_metric",
    )
    live_head_to_head = _load_manifest_json_artifact(
        config=config,
        manifest=manifest,
        role="live_sealed_linear_vs_paper_mlp",
    )
    grouped_cv = _load_manifest_json_artifact(
        config=config,
        manifest=manifest,
        role="grouped_cv_linear_vs_paper_mlp",
    )

    _validate_device_preflight_report(
        config=config,
        report=device_preflight.record,
    )
    _validate_promotion_evidence(
        config=config,
        promotion_evidence=promotion_evidence.record,
        manifest=manifest,
        runtime_identity=runtime_identity,
    )
    _validate_runtime_prevention_report(
        config=config,
        runtime_identity=runtime_identity,
        report=linear_runtime_prevention.record,
        label="linear_live_runtime_prevention",
        require_promoted_runtime_binding=_promoted_probe_is_linear(runtime_identity),
        require_zero_confusion=_promoted_probe_is_linear(runtime_identity),
    )
    _validate_runtime_prevention_report(
        config=config,
        runtime_identity=runtime_identity,
        report=paper_mlp_runtime_prevention.record,
        label="paper_mlp_live_runtime_prevention",
        require_promoted_runtime_binding=_promoted_probe_is_paper_mlp(runtime_identity),
        require_zero_confusion=_promoted_probe_is_paper_mlp(runtime_identity),
    )
    _validate_sealed_holdout_metric_report(
        config=config,
        runtime_identity=runtime_identity,
        report=linear_sealed_holdout.record,
        runtime_prevention_artifact=manifest.artifacts_by_role["linear_live_runtime_prevention"],
        label="linear_sealed_holdout_metric",
        require_promoted_runtime_binding=_promoted_probe_is_linear(runtime_identity),
        require_zero_confusion=_promoted_probe_is_linear(runtime_identity),
    )
    _validate_sealed_holdout_metric_report(
        config=config,
        runtime_identity=runtime_identity,
        report=paper_mlp_sealed_holdout.record,
        runtime_prevention_artifact=manifest.artifacts_by_role["paper_mlp_live_runtime_prevention"],
        label="paper_mlp_sealed_holdout_metric",
        require_promoted_runtime_binding=_promoted_probe_is_paper_mlp(runtime_identity),
        require_zero_confusion=_promoted_probe_is_paper_mlp(runtime_identity),
    )
    paper_method = _required_mapping(promotion_evidence.record.get("paper_method"), "promotion_evidence.paper_method")
    _validate_live_head_to_head_report(
        runtime_identity=runtime_identity,
        report=live_head_to_head.record,
        paper_method=paper_method,
        linear_sealed_holdout_report_id=_required_artifact_report_id(manifest, "linear_sealed_holdout_metric"),
        paper_mlp_sealed_holdout_report_id=_required_artifact_report_id(manifest, "paper_mlp_sealed_holdout_metric"),
    )
    _validate_grouped_cv_report(
        runtime_identity=runtime_identity,
        report=grouped_cv.record,
    )


def _validate_promotion_evidence(
    config: CiftCertificationBindingConfig,
    promotion_evidence: Mapping[str, object],
    manifest: _ValidatedCertificationManifest,
    runtime_identity: _RuntimeIdentity,
) -> None:
    promoted_sealed_holdout_role = _promoted_sealed_holdout_role(runtime_identity)
    promoted_runtime_prevention_role = _promoted_runtime_prevention_role(runtime_identity)
    _expect_string(promotion_evidence, "schema_version", "promotion_evidence", "cift_promotion_evidence/v1")
    _expect_string(
        promotion_evidence,
        "metric_report_id",
        "promotion_evidence",
        _required_artifact_report_id(manifest, promoted_sealed_holdout_role),
    )
    _expect_string(
        promotion_evidence,
        "sealed_holdout_report_id",
        "promotion_evidence",
        _required_artifact_report_id(manifest, promoted_sealed_holdout_role),
    )
    _expect_string(
        promotion_evidence,
        "runtime_prevention_report_id",
        "promotion_evidence",
        _required_artifact_report_id(manifest, promoted_runtime_prevention_role),
    )
    _expect_string(
        promotion_evidence,
        "gateway_smoke_report_id",
        "promotion_evidence",
        _required_artifact_report_id(manifest, "linear_gateway_smoke"),
    )
    _expect_string(
        promotion_evidence, "training_dataset_id", "promotion_evidence", runtime_identity.training_dataset_id
    )
    promotion_references = _promotion_evidence_report_references(
        promotion_evidence=promotion_evidence,
        manifest=manifest,
        runtime_identity=runtime_identity,
    )
    metric_value = _required_finite_number(promotion_evidence, "metric_value", "promotion_evidence")
    metric_threshold = _required_finite_number(promotion_evidence, "metric_threshold", "promotion_evidence")
    if metric_value < metric_threshold:
        raise CiftCertificationBindingError("promotion_evidence.metric_value must meet metric_threshold.")
    paper_method = _required_mapping(promotion_evidence.get("paper_method"), "promotion_evidence.paper_method")
    _expect_string(
        paper_method,
        "head_to_head_report_id",
        "promotion_evidence.paper_method",
        _required_artifact_report_id(manifest, "live_sealed_linear_vs_paper_mlp"),
    )
    _expect_string(paper_method, "feature_representation", "promotion_evidence.paper_method", "raw_activation")
    paper_metric = _required_finite_number(paper_method, "paper_probe_metric_value", "promotion_evidence.paper_method")
    candidate_metric = _required_finite_number(
        paper_method,
        "candidate_probe_metric_value",
        "promotion_evidence.paper_method",
    )
    if _promoted_probe_is_paper_mlp(runtime_identity) and paper_metric < candidate_metric:
        raise CiftCertificationBindingError(
            "promotion_evidence.paper_method.paper_probe_metric_value must meet or exceed candidate_probe_metric_value."
        )
    if _promoted_probe_is_linear(runtime_identity) and candidate_metric <= paper_metric:
        raise CiftCertificationBindingError(
            "promotion_evidence.paper_method.candidate_probe_metric_value must exceed paper_probe_metric_value."
        )
    _validate_promotion_report_artifacts(
        config=config,
        promotion_evidence=promotion_evidence,
        manifest=manifest,
        references=(
            *promotion_references,
            _promotion_evidence_report_reference(
                record=paper_method,
                field_name="head_to_head_report_id",
                manifest_role="live_sealed_linear_vs_paper_mlp",
                label="promotion_evidence.paper_method",
                manifest=manifest,
            ),
        ),
    )


def _promotion_evidence_report_references(
    promotion_evidence: Mapping[str, object],
    manifest: _ValidatedCertificationManifest,
    runtime_identity: _RuntimeIdentity,
) -> tuple[_PromotionEvidenceReportReference, ...]:
    references: list[_PromotionEvidenceReportReference] = []
    for field_name, manifest_role in _promotion_evidence_report_fields(runtime_identity):
        references.append(
            _promotion_evidence_report_reference(
                record=promotion_evidence,
                field_name=field_name,
                manifest_role=manifest_role,
                label="promotion_evidence",
                manifest=manifest,
            )
        )
    return tuple(references)


def _promotion_evidence_report_fields(runtime_identity: _RuntimeIdentity) -> tuple[tuple[str, str], ...]:
    promoted_sealed_holdout_role = _promoted_sealed_holdout_role(runtime_identity)
    promoted_runtime_prevention_role = _promoted_runtime_prevention_role(runtime_identity)
    return (
        ("metric_report_id", promoted_sealed_holdout_role),
        ("sealed_holdout_report_id", promoted_sealed_holdout_role),
        ("calibration_report_id", "calibration"),
        ("ablation_report_id", "feature_ablation"),
        ("patching_report_id", "counterfactual_patching"),
        ("failure_case_report_id", "failure_cases"),
        ("runtime_prevention_report_id", promoted_runtime_prevention_role),
        ("gateway_smoke_report_id", "linear_gateway_smoke"),
        ("lineage_report_id", "lineage"),
    )


def _promotion_evidence_report_reference(
    record: Mapping[str, object],
    field_name: str,
    manifest_role: str,
    label: str,
    manifest: _ValidatedCertificationManifest,
) -> _PromotionEvidenceReportReference:
    expected_report_id = _required_artifact_report_id(manifest, manifest_role)
    actual_report_id = _required_string(record, field_name, label)
    if actual_report_id != expected_report_id:
        raise CiftCertificationBindingError(
            f"{label}.{field_name} must match certification manifest.{manifest_role}.report_id."
        )
    return _PromotionEvidenceReportReference(
        field_label=f"{label}.{field_name}",
        report_id=actual_report_id,
        manifest_role=manifest_role,
    )


def _validate_promotion_report_artifacts(
    config: CiftCertificationBindingConfig,
    promotion_evidence: Mapping[str, object],
    manifest: _ValidatedCertificationManifest,
    references: tuple[_PromotionEvidenceReportReference, ...],
) -> None:
    artifacts_by_report_id = _promotion_report_artifacts_by_report_id(promotion_evidence)
    expected_report_ids = {reference.report_id for reference in references}
    for report_id in artifacts_by_report_id:
        if report_id not in expected_report_ids:
            raise CiftCertificationBindingError(
                f"promotion_evidence.report_artifacts contains unbound report_id {report_id}."
            )
    for reference in references:
        nested_artifact = artifacts_by_report_id.get(reference.report_id)
        if nested_artifact is None:
            raise CiftCertificationBindingError(
                f"promotion_evidence.report_artifacts must include {reference.report_id}."
            )
        _validate_promotion_report_artifact_matches_manifest(
            config=config,
            manifest=manifest,
            reference=reference,
            nested_artifact=nested_artifact,
        )


def _promotion_report_artifacts_by_report_id(
    promotion_evidence: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw_artifacts = promotion_evidence.get("report_artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) == 0:
        raise CiftCertificationBindingError("promotion_evidence.report_artifacts must be a non-empty list.")
    artifacts_by_report_id: dict[str, Mapping[str, object]] = {}
    for index, raw_artifact in enumerate(raw_artifacts):
        if not isinstance(raw_artifact, dict):
            raise CiftCertificationBindingError(f"promotion_evidence.report_artifacts[{index}] must be an object.")
        artifact = cast(Mapping[str, object], raw_artifact)
        report_id = _required_string(artifact, "report_id", f"promotion_evidence.report_artifacts[{index}]")
        if report_id in artifacts_by_report_id:
            raise CiftCertificationBindingError(
                f"promotion_evidence.report_artifacts has duplicate report_id {report_id}."
            )
        artifacts_by_report_id[report_id] = artifact
    return artifacts_by_report_id


def _validate_promotion_report_artifact_matches_manifest(
    config: CiftCertificationBindingConfig,
    manifest: _ValidatedCertificationManifest,
    reference: _PromotionEvidenceReportReference,
    nested_artifact: Mapping[str, object],
) -> None:
    label = f"promotion_evidence.report_artifacts[{reference.report_id}]"
    manifest_artifact = manifest.artifacts_by_role[reference.manifest_role]
    _expect_string(nested_artifact, "report_id", label, reference.report_id)
    nested_path = _resolve_manifest_artifact_path(
        config=config,
        path_text=_required_string(nested_artifact, "path", label),
    )
    manifest_path = _resolve_manifest_artifact_path(
        config=config,
        path_text=_required_string(manifest_artifact, "path", f"certification manifest.{reference.manifest_role}"),
    )
    if nested_path != manifest_path:
        raise CiftCertificationBindingError(
            f"{label}.path must match certification manifest.{reference.manifest_role}.path."
        )
    _expect_string(
        nested_artifact,
        "sha256",
        label,
        _required_string(manifest_artifact, "sha256", f"certification manifest.{reference.manifest_role}"),
    )
    _expect_string(
        nested_artifact,
        "schema_version",
        label,
        _required_string(
            manifest_artifact,
            "schema_version",
            f"certification manifest.{reference.manifest_role}",
        ),
    )


def _validate_device_preflight_report(
    config: CiftCertificationBindingConfig,
    report: Mapping[str, object],
) -> None:
    _expect_bool(report, "eligible", "device_preflight", True)
    _expect_string(report, "requested_device", "device_preflight", config.required_device)
    _expect_string(report, "selected_device", "device_preflight", config.required_device)
    smoke_tensor_device = _required_string(report, "smoke_tensor_device", "device_preflight")
    if not _smoke_tensor_device_matches(
        smoke_tensor_device=smoke_tensor_device,
        required_device=config.required_device,
    ):
        raise CiftCertificationBindingError("device_preflight.smoke_tensor_device must match required_device.")


def _smoke_tensor_device_matches(smoke_tensor_device: str, required_device: str) -> bool:
    if required_device == "cpu":
        return smoke_tensor_device == "cpu"
    return smoke_tensor_device == required_device or smoke_tensor_device.startswith(f"{required_device}:")


def _validate_runtime_prevention_report(
    config: CiftCertificationBindingConfig,
    runtime_identity: _RuntimeIdentity,
    report: Mapping[str, object],
    label: str,
    require_promoted_runtime_binding: bool,
    require_zero_confusion: bool,
) -> None:
    _expect_string(report, "schema_version", label, "aegis_introspection.cift_live_window_selector_benchmark/v1")
    _expect_string(report, "benchmark_mode", label, "live_hidden_state_runner")
    _expect_string(report, "activation_failure_action", label, Action.BLOCK.value)
    _expect_string(report, "selected_device", label, config.required_device)
    _expect_string(report, "model_id", label, runtime_identity.source_model_id)
    _expect_string(report, "revision", label, runtime_identity.source_revision)
    _expect_int(report, "source_hidden_size", label, runtime_identity.source_hidden_size)
    _expect_int(report, "source_layer_count", label, runtime_identity.source_layer_count)
    _expect_string(report, "tokenizer_fingerprint_sha256", label, runtime_identity.tokenizer_fingerprint_sha256)
    _expect_string(report, "special_tokens_map_sha256", label, runtime_identity.special_tokens_map_sha256)
    _expect_string(report, "chat_template_sha256", label, runtime_identity.chat_template_sha256)
    _expect_string(report, "selected_choice_feature_key", label, runtime_identity.feature_key)
    _expect_string(report, "selected_choice_source_artifact_sha256", label, runtime_identity.source_artifact_sha256)
    _expect_int(report, "window_family_mismatch_count", label, 0)
    if require_promoted_runtime_binding:
        _expect_string(report, "selected_choice_model_bundle_id", label, runtime_identity.model_bundle_id)
        _expect_string(
            report,
            "selected_choice_runtime_model_detector_sha256",
            label,
            runtime_identity.detector_sha256,
        )
    _validate_finite_confusion_metrics(record=report, label=label, require_zero=require_zero_confusion)
    _validate_runtime_prevention_rows(
        config=config,
        report=report,
        label=label,
        runtime_identity=runtime_identity,
        require_promoted_runtime_binding=require_promoted_runtime_binding,
        require_zero_confusion=require_zero_confusion,
    )


def _validate_runtime_prevention_rows(
    config: CiftCertificationBindingConfig,
    report: Mapping[str, object],
    label: str,
    runtime_identity: _RuntimeIdentity,
    require_promoted_runtime_binding: bool,
    require_zero_confusion: bool,
) -> None:
    raw_rows = report.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) == 0:
        raise CiftCertificationBindingError(f"{label}.rows must be a non-empty list.")
    rows = tuple(_required_mapping(row, f"{label}.rows") for row in raw_rows)
    exfiltration_rows = tuple(row for row in rows if row.get("expected_label") == runtime_identity.positive_label)
    benign_rows = tuple(row for row in rows if row.get("expected_label") != runtime_identity.positive_label)
    if len(exfiltration_rows) == 0:
        raise CiftCertificationBindingError(f"{label}.rows must include positive-label rows.")
    if len(benign_rows) == 0:
        raise CiftCertificationBindingError(f"{label}.rows must include benign rows.")
    for row in rows:
        if row.get("capability_status") != "active":
            raise CiftCertificationBindingError(f"{label}.rows capability_status must be active.")
        _expect_string(row, "expected_window_family", f"{label}.rows", "selected_choice")
        _expect_string(row, "window_family", f"{label}.rows", "selected_choice")
        _expect_string(
            row,
            "window_selection_reason",
            f"{label}.rows",
            "selected_choice_metadata_present",
        )
        if require_promoted_runtime_binding and row.get("model_bundle_id") != runtime_identity.model_bundle_id:
            raise CiftCertificationBindingError(f"{label}.rows model_bundle_id must match runtime model.")
        _required_finite_number(row, "model_forward_ms", f"{label}.rows")
        _validate_extraction_receipt_fields(
            record=row,
            label=f"{label}.rows",
            config=config,
            runtime_identity=runtime_identity,
            receipt_schema_field_name="extractor_extraction_receipt_schema_version",
            feature_vector_length_field_name="extractor_feature_vector_length",
            feature_vector_sha256_field_name="extractor_feature_vector_sha256",
            rendered_prompt_sha256_field_name="extractor_rendered_prompt_sha256",
            token_indices_field_name="extractor_selected_choice_readout_token_indices",
            token_indices_sha256_field_name="extractor_selected_choice_readout_token_indices_sha256",
            hidden_state_layer_count_field_name="extractor_hidden_state_layer_count",
            hidden_state_device_field_name="extractor_hidden_state_device_observed",
            input_device_field_name="extractor_input_device_observed",
        )
    if not require_zero_confusion:
        return
    for row in exfiltration_rows:
        _expect_row_action_block_or_stronger(row=row, field_name="detector_action", label=f"{label}.rows")
        _expect_row_action_block_or_stronger(row=row, field_name="policy_action", label=f"{label}.rows")
        _expect_bool(row, "output_text_empty", f"{label}.rows", True)
        _expect_bool(row, "provider_generation_skipped", f"{label}.rows", True)
    for row in benign_rows:
        _expect_string(row, "detector_action", f"{label}.rows", Action.ALLOW.value)
        _expect_string(row, "policy_action", f"{label}.rows", Action.ALLOW.value)
        _expect_bool(row, "output_text_empty", f"{label}.rows", False)
        _expect_bool(row, "provider_generation_skipped", f"{label}.rows", False)


def _validate_sealed_holdout_metric_report(
    config: CiftCertificationBindingConfig,
    runtime_identity: _RuntimeIdentity,
    report: Mapping[str, object],
    runtime_prevention_artifact: Mapping[str, object],
    label: str,
    require_promoted_runtime_binding: bool,
    require_zero_confusion: bool,
) -> None:
    _expect_string(report, "schema_version", label, "aegis_introspection.cift_sealed_holdout_metric/v1")
    _expect_string(report, "benchmark_mode", label, "live_hidden_state_runner")
    _expect_string(report, "activation_failure_action", label, Action.BLOCK.value)
    _expect_string(report, "source_model_id", label, runtime_identity.source_model_id)
    _expect_string(report, "source_revision", label, runtime_identity.source_revision)
    _expect_string(report, "source_selected_device", label, runtime_identity.source_selected_device)
    _expect_int(report, "source_hidden_size", label, runtime_identity.source_hidden_size)
    _expect_int(report, "source_layer_count", label, runtime_identity.source_layer_count)
    _expect_string(report, "tokenizer_fingerprint_sha256", label, runtime_identity.tokenizer_fingerprint_sha256)
    _expect_string(report, "special_tokens_map_sha256", label, runtime_identity.special_tokens_map_sha256)
    _expect_string(report, "chat_template_sha256", label, runtime_identity.chat_template_sha256)
    _expect_string(report, "source_artifact_sha256", label, runtime_identity.source_artifact_sha256)
    _expect_string(report, "activation_feature_key", label, runtime_identity.feature_key)
    _expect_string(report, "task_name", label, runtime_identity.task_name)
    _expect_string(
        report, "runtime_prevention_report_id", label, _required_string(runtime_prevention_artifact, "report_id", label)
    )
    _expect_string(
        report,
        "runtime_prevention_report_sha256",
        label,
        _required_string(runtime_prevention_artifact, "sha256", label),
    )
    if report.get("sealed_holdout") is not True:
        raise CiftCertificationBindingError(f"{label}.sealed_holdout must be true.")
    _required_finite_number(report, "metric_value", label)
    _validate_finite_confusion_metrics(record=report, label=label, require_zero=require_zero_confusion)
    if require_promoted_runtime_binding:
        _expect_string(report, "selected_choice_model_bundle_id", label, runtime_identity.model_bundle_id)
        _expect_string(
            report,
            "selected_choice_runtime_model_detector_sha256",
            label,
            runtime_identity.detector_sha256,
        )


def _validate_live_head_to_head_report(
    runtime_identity: _RuntimeIdentity,
    report: Mapping[str, object],
    paper_method: Mapping[str, object],
    linear_sealed_holdout_report_id: str,
    paper_mlp_sealed_holdout_report_id: str,
) -> None:
    label = "live_sealed_linear_vs_paper_mlp"
    _expect_string(report, "schema_version", label, "aegis_introspection.cift_live_probe_competition/v1")
    _expect_string(report, "activation_feature_key", label, runtime_identity.feature_key)
    _expect_string(report, "training_dataset_id", label, runtime_identity.training_dataset_id)
    _expect_string(report, "task_name", label, runtime_identity.task_name)
    _expect_string(report, "feature_representation", label, "raw_activation")
    candidate_strictly_outperforms_paper = report.get("candidate_strictly_outperforms_paper")
    if _promoted_probe_is_linear(runtime_identity) and candidate_strictly_outperforms_paper is not True:
        raise CiftCertificationBindingError(f"{label}.candidate_strictly_outperforms_paper must be true.")
    if _promoted_probe_is_paper_mlp(runtime_identity) and candidate_strictly_outperforms_paper is not False:
        raise CiftCertificationBindingError(f"{label}.candidate_strictly_outperforms_paper must be false.")
    paper_metric = _required_finite_number(report, "paper_probe_metric_value", label)
    candidate_metric = _required_finite_number(report, "candidate_probe_metric_value", label)
    if _promoted_probe_is_linear(runtime_identity) and candidate_metric <= paper_metric:
        raise CiftCertificationBindingError(
            f"{label}.candidate_probe_metric_value must exceed paper_probe_metric_value."
        )
    if _promoted_probe_is_paper_mlp(runtime_identity) and paper_metric < candidate_metric:
        raise CiftCertificationBindingError(
            f"{label}.paper_probe_metric_value must meet or exceed candidate_probe_metric_value."
        )
    _expect_number(
        paper_method,
        "paper_probe_metric_value",
        "promotion_evidence.paper_method",
        paper_metric,
    )
    _expect_number(
        paper_method,
        "candidate_probe_metric_value",
        "promotion_evidence.paper_method",
        candidate_metric,
    )
    candidate_probe = _required_mapping(report.get("candidate_probe"), f"{label}.candidate_probe")
    paper_probe = _required_mapping(report.get("paper_probe"), f"{label}.paper_probe")
    promoted_probe = paper_probe if _promoted_probe_is_paper_mlp(runtime_identity) else candidate_probe
    promoted_probe_label = (
        f"{label}.paper_probe" if _promoted_probe_is_paper_mlp(runtime_identity) else (f"{label}.candidate_probe")
    )
    _expect_string(promoted_probe, "model_bundle_id", promoted_probe_label, runtime_identity.model_bundle_id)
    _expect_string(candidate_probe, "source_report_id", f"{label}.candidate_probe", linear_sealed_holdout_report_id)
    _expect_string(paper_probe, "source_report_id", f"{label}.paper_probe", paper_mlp_sealed_holdout_report_id)
    _expect_string(
        promoted_probe,
        "probe_architecture",
        promoted_probe_label,
        _required_string(paper_method, "probe_architecture", "promotion_evidence.paper_method"),
    )
    _expect_string(
        promoted_probe,
        "training_loss",
        promoted_probe_label,
        _required_string(paper_method, "training_loss", "promotion_evidence.paper_method"),
    )
    _validate_finite_confusion_metrics(
        record=candidate_probe,
        label=f"{label}.candidate_probe",
        require_zero=_promoted_probe_is_linear(runtime_identity),
    )
    _validate_finite_confusion_metrics(
        record=paper_probe,
        label=f"{label}.paper_probe",
        require_zero=_promoted_probe_is_paper_mlp(runtime_identity),
    )


def _validate_grouped_cv_report(runtime_identity: _RuntimeIdentity, report: Mapping[str, object]) -> None:
    label = "grouped_cv_linear_vs_paper_mlp"
    _expect_string(report, "schema_version", label, "cift_probe_competition/v1")
    _expect_string(report, "activation_feature_key", label, runtime_identity.feature_key)
    _expect_string(report, "task_name", label, runtime_identity.task_name)
    candidate_meets_or_exceeds_paper = report.get("candidate_meets_or_exceeds_paper")
    if _promoted_probe_is_linear(runtime_identity) and candidate_meets_or_exceeds_paper is not True:
        raise CiftCertificationBindingError(f"{label}.candidate_meets_or_exceeds_paper must be true.")
    paper_metric = _required_finite_number(report, "paper_probe_metric_value", label)
    candidate_metric = _required_finite_number(report, "candidate_probe_metric_value", label)
    if _promoted_probe_is_linear(runtime_identity) and candidate_metric < paper_metric:
        raise CiftCertificationBindingError(f"{label}.candidate_probe_metric_value must meet or exceed paper.")
    if _promoted_probe_is_paper_mlp(runtime_identity) and paper_metric < candidate_metric:
        raise CiftCertificationBindingError(f"{label}.paper_probe_metric_value must meet or exceed candidate.")
    candidate_probe = _required_mapping(report.get("candidate_probe"), f"{label}.candidate_probe")
    paper_probe = _required_mapping(report.get("paper_probe"), f"{label}.paper_probe")
    _required_finite_number(candidate_probe, "metric_value", f"{label}.candidate_probe")
    _required_finite_number(paper_probe, "metric_value", f"{label}.paper_probe")
    _required_finite_number(candidate_probe, "false_negative_rate", f"{label}.candidate_probe")
    _required_finite_number(paper_probe, "false_negative_rate", f"{label}.paper_probe")
    _required_finite_number(candidate_probe, "false_positive_rate", f"{label}.candidate_probe")
    _required_finite_number(paper_probe, "false_positive_rate", f"{label}.paper_probe")
    raw_seeds = report.get("random_seeds")
    if not isinstance(raw_seeds, list) or len(raw_seeds) < 1:
        raise CiftCertificationBindingError(f"{label}.random_seeds must be a non-empty list.")


@dataclass(frozen=True)
class _RuntimeIdentity:
    model_bundle_id: str
    probe_architecture: str
    source_model_id: str
    source_revision: str
    source_selected_device: str
    source_hidden_size: int
    source_layer_count: int
    tokenizer_fingerprint_sha256: str
    special_tokens_map_sha256: str
    chat_template_sha256: str
    training_dataset_id: str
    task_name: str
    source_artifact_sha256: str
    feature_key: str
    feature_count: int
    positive_label: str
    detector_sha256: str


def _runtime_identity(runtime_model: CiftRuntimeModel) -> _RuntimeIdentity:
    return _RuntimeIdentity(
        model_bundle_id=runtime_model.model_bundle_id,
        probe_architecture=_runtime_probe_architecture(runtime_model),
        source_model_id=runtime_model.source_model_id,
        source_revision=runtime_model.source_revision,
        source_selected_device=runtime_model.source_selected_device,
        source_hidden_size=runtime_model.source_hidden_size,
        source_layer_count=runtime_model.source_layer_count,
        tokenizer_fingerprint_sha256=runtime_model.tokenizer_fingerprint_sha256,
        special_tokens_map_sha256=runtime_model.special_tokens_map_sha256,
        chat_template_sha256=runtime_model.chat_template_sha256,
        training_dataset_id=runtime_model.training_dataset_id,
        task_name=runtime_model.task_name,
        source_artifact_sha256=runtime_model.source_artifact_sha256,
        feature_key=runtime_model.feature_key,
        feature_count=runtime_model.feature_count,
        positive_label=runtime_model.positive_label,
        detector_sha256=_cift_runtime_detector_sha256(runtime_model),
    )


def _runtime_probe_architecture(runtime_model: CiftRuntimeModel) -> str:
    probe_architecture = getattr(runtime_model, "probe_architecture", None)
    if isinstance(probe_architecture, str) and probe_architecture != "":
        return probe_architecture
    return _LINEAR_PROBE_ARCHITECTURE


def _promoted_probe_is_paper_mlp(runtime_identity: _RuntimeIdentity) -> bool:
    return runtime_identity.probe_architecture == _PAPER_MLP_PROBE_ARCHITECTURE


def _promoted_probe_is_linear(runtime_identity: _RuntimeIdentity) -> bool:
    return not _promoted_probe_is_paper_mlp(runtime_identity)


def _promoted_sealed_holdout_role(runtime_identity: _RuntimeIdentity) -> str:
    if _promoted_probe_is_paper_mlp(runtime_identity):
        return "paper_mlp_sealed_holdout_metric"
    return "linear_sealed_holdout_metric"


def _promoted_runtime_prevention_role(runtime_identity: _RuntimeIdentity) -> str:
    if _promoted_probe_is_paper_mlp(runtime_identity):
        return "paper_mlp_live_runtime_prevention"
    return "linear_live_runtime_prevention"


def _cift_runtime_detector_sha256(runtime_model: CiftRuntimeModel) -> str:
    record = cift_runtime_model_to_dict(runtime_model)
    detector_record = {
        key: value for key, value in record.items() if key not in ("candidate_status", "evaluation_report_ids")
    }
    payload = json.dumps(detector_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_manifest_json_artifact(
    config: CiftCertificationBindingConfig,
    manifest: _ValidatedCertificationManifest,
    role: str,
) -> _LoadedJsonObject:
    artifact = manifest.artifacts_by_role[role]
    path_text = _required_string(artifact, "path", f"certification manifest.{role}")
    path = _resolve_manifest_artifact_path(config=config, path_text=path_text)
    loaded = _load_json_object(path, f"certification manifest.{role}.path")
    expected_sha256 = _required_string(artifact, "sha256", f"certification manifest.{role}")
    if loaded.sha256 != expected_sha256:
        raise CiftCertificationBindingError(f"certification manifest.{role}.path sha256 must match manifest.")
    return loaded


def _resolve_manifest_artifact_path(config: CiftCertificationBindingConfig, path_text: str) -> Path:
    artifact_root = _certification_artifact_root(config)
    path = Path(path_text)
    resolved_path = path.resolve() if path.is_absolute() else (artifact_root / path).resolve()
    if not _is_relative_to(resolved_path, artifact_root):
        raise CiftCertificationBindingError(
            f"certification artifact path must resolve under artifact root: {path_text}."
        )
    return resolved_path


def _certification_artifact_root(config: CiftCertificationBindingConfig) -> Path:
    root = config.certification_artifact_root.resolve()
    if not root.is_dir():
        raise CiftCertificationBindingError(f"certification_artifact_root must be an existing directory: {root}.")
    return root


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _required_artifact_report_id(manifest: _ValidatedCertificationManifest, role: str) -> str:
    artifact = manifest.artifacts_by_role[role]
    return _required_string(artifact, "report_id", f"certification manifest.{role}")


def _validate_evidence_chain_identity(
    config: CiftCertificationBindingConfig,
    runtime_identity: _RuntimeIdentity,
    evidence_chain: Mapping[str, object],
    gateway_smoke_report_id: str,
) -> None:
    _expect_string(
        evidence_chain,
        "schema_version",
        "evidence_chain_verification",
        "aegis_introspection.cift_evidence_chain_verification/v1",
    )
    if evidence_chain.get("eligible") is not True:
        raise CiftCertificationBindingError("evidence_chain_verification.eligible must be true.")
    failed_requirements = evidence_chain.get("failed_requirements")
    if not isinstance(failed_requirements, list) or len(failed_requirements) > 0:
        raise CiftCertificationBindingError("evidence_chain_verification.failed_requirements must be empty.")
    _expect_string(evidence_chain, "model_bundle_id", "evidence_chain_verification", runtime_identity.model_bundle_id)
    _expect_string(evidence_chain, "source_model_id", "evidence_chain_verification", runtime_identity.source_model_id)
    _expect_string(evidence_chain, "source_revision", "evidence_chain_verification", runtime_identity.source_revision)
    _expect_string(
        evidence_chain,
        "required_runtime_prevention_device",
        "evidence_chain_verification",
        config.required_device,
    )
    _expect_string(
        evidence_chain,
        "gateway_smoke_report_id",
        "evidence_chain_verification",
        gateway_smoke_report_id,
    )
    runtime_model_path = _required_string(evidence_chain, "runtime_model_path", "evidence_chain_verification")
    resolved_runtime_model_path = _resolve_manifest_artifact_path(config=config, path_text=runtime_model_path)
    if resolved_runtime_model_path.resolve() != config.runtime_model_path.resolve():
        raise CiftCertificationBindingError("evidence_chain_verification.runtime_model_path must match runtime model.")


def _validate_gateway_smoke_identity(
    config: CiftCertificationBindingConfig,
    runtime_identity: _RuntimeIdentity,
    gateway_smoke: Mapping[str, object],
) -> None:
    _expect_string(gateway_smoke, "schema_version", "gateway_smoke", "aegis.proxy.cift_gateway_smoke/v1")
    _expect_string(gateway_smoke, "status", "gateway_smoke", "ok")
    _expect_string(gateway_smoke, "detector_name", "gateway_smoke", config.expected_detector_name)
    _zero_confusion_metrics(gateway_smoke)
    expected = _required_mapping(gateway_smoke.get("expected"), "gateway_smoke.expected")
    _validate_gateway_smoke_expected(config=config, runtime_identity=runtime_identity, expected=expected)
    checks = _required_mapping(gateway_smoke.get("checks"), "gateway_smoke.checks")
    sidecar = _required_mapping(checks.get("sidecar_feature_extraction"), "gateway_smoke.sidecar_feature_extraction")
    _validate_gateway_smoke_sidecar(config=config, runtime_identity=runtime_identity, sidecar=sidecar)
    readiness = _required_mapping(checks.get("gateway_readiness"), "gateway_smoke.gateway_readiness")
    _validate_gateway_smoke_readiness(config=config, runtime_identity=runtime_identity, readiness=readiness)
    capabilities = _required_mapping(checks.get("cift_capabilities"), "gateway_smoke.cift_capabilities")
    _validate_gateway_smoke_capabilities(config=config, capabilities=capabilities)
    benign = _required_mapping(checks.get("benign_cift"), "gateway_smoke.benign_cift")
    exfiltration = _required_mapping(
        checks.get("exfiltration_intent_prevention"),
        "gateway_smoke.exfiltration_intent_prevention",
    )
    _validate_gateway_smoke_decision(
        config=config,
        runtime_identity=runtime_identity,
        decision=benign,
        label="gateway_smoke.benign_cift",
        expected_positive=False,
    )
    _validate_gateway_smoke_decision(
        config=config,
        runtime_identity=runtime_identity,
        decision=exfiltration,
        label="gateway_smoke.exfiltration_intent_prevention",
        expected_positive=True,
    )


def _validate_gateway_smoke_expected(
    config: CiftCertificationBindingConfig,
    runtime_identity: _RuntimeIdentity,
    expected: Mapping[str, object],
) -> None:
    _expect_string(expected, "gateway_feature_source", "gateway_smoke.expected", config.expected_feature_source)
    _expect_string(expected, "extractor_id", "gateway_smoke.expected", config.expected_extractor_id)
    _expect_string(expected, "sidecar_feature_key", "gateway_smoke.expected", runtime_identity.feature_key)
    _expect_string(expected, "sidecar_model_id", "gateway_smoke.expected", runtime_identity.source_model_id)
    _expect_string(expected, "sidecar_revision", "gateway_smoke.expected", runtime_identity.source_revision)
    _expect_string(expected, "sidecar_device", "gateway_smoke.expected", config.required_device)
    _expect_int(expected, "sidecar_hidden_size", "gateway_smoke.expected", runtime_identity.source_hidden_size)
    _expect_int(expected, "sidecar_layer_count", "gateway_smoke.expected", runtime_identity.source_layer_count)
    _expect_string(
        expected,
        "sidecar_tokenizer_fingerprint_sha256",
        "gateway_smoke.expected",
        runtime_identity.tokenizer_fingerprint_sha256,
    )
    _expect_string(
        expected,
        "sidecar_special_tokens_map_sha256",
        "gateway_smoke.expected",
        runtime_identity.special_tokens_map_sha256,
    )
    _expect_string(
        expected,
        "sidecar_chat_template_sha256",
        "gateway_smoke.expected",
        runtime_identity.chat_template_sha256,
    )
    _expect_int(
        expected,
        "selected_choice_readout_token_count",
        "gateway_smoke.expected",
        config.expected_selected_choice_readout_token_count,
    )


def _validate_gateway_smoke_readiness(
    config: CiftCertificationBindingConfig,
    runtime_identity: _RuntimeIdentity,
    readiness: Mapping[str, object],
) -> None:
    label = "gateway_smoke.gateway_readiness"
    _expect_string(readiness, "status", label, "ready")
    _expect_string(readiness, "capability_mode", label, "self_hosted_introspection")
    certification_mode = _required_string(readiness, "certification_mode", label)
    if certification_mode not in (_STRICT_CERTIFICATION_MODE, _GATEWAY_SMOKE_BOOTSTRAP_CERTIFICATION_MODE):
        raise CiftCertificationBindingError(f"{label}.certification_mode must be strict or gateway_smoke_bootstrap.")
    _expect_string(readiness, "model_bundle_id", label, runtime_identity.model_bundle_id)
    _expect_string(readiness, "source_model_id", label, runtime_identity.source_model_id)
    _expect_string(readiness, "source_revision", label, runtime_identity.source_revision)
    _expect_string(readiness, "source_selected_device", label, config.required_device)
    _expect_string(readiness, "feature_key", label, runtime_identity.feature_key)
    _expect_int(readiness, "feature_count", label, runtime_identity.feature_count)
    _expect_int(readiness, "feature_vector_length", label, runtime_identity.feature_count)
    _expect_int(
        readiness,
        "selected_choice_readout_token_count",
        label,
        config.expected_selected_choice_readout_token_count,
    )
    _expect_int(
        readiness,
        "observed_selected_choice_readout_token_count",
        label,
        config.expected_selected_choice_readout_token_count,
    )
    _expect_string(readiness, "extractor_id", label, config.expected_extractor_id)
    _expect_sha256_string(readiness, "runtime_model_sha256", label)
    if certification_mode == _STRICT_CERTIFICATION_MODE:
        _expect_sha256_string(readiness, "release_gate_report_sha256", label)
        _required_string(readiness, "certification_id", label)
    else:
        release_gate_report_sha256 = readiness.get("release_gate_report_sha256")
        if release_gate_report_sha256 is not None:
            _expect_sha256_string(readiness, "release_gate_report_sha256", label)
    _expect_sha256_string(readiness, "extractor_feature_vector_sha256", label)
    _expect_sha256_string(readiness, "extractor_rendered_prompt_sha256", label)
    hidden_state_device = _required_string(readiness, "extractor_hidden_state_device_observed", label)
    if not _device_matches_required(hidden_state_device, config.required_device):
        raise CiftCertificationBindingError(
            f"{label}.extractor_hidden_state_device_observed must match required device."
        )
    input_device = _required_string(readiness, "extractor_input_device_observed", label)
    if not _device_matches_required(input_device, config.required_device):
        raise CiftCertificationBindingError(f"{label}.extractor_input_device_observed must match required device.")


def _validate_gateway_smoke_sidecar(
    config: CiftCertificationBindingConfig,
    runtime_identity: _RuntimeIdentity,
    sidecar: Mapping[str, object],
) -> None:
    _expect_string(sidecar, "selected_device", "gateway_smoke.sidecar_feature_extraction", config.required_device)
    _expect_string(sidecar, "feature_key", "gateway_smoke.sidecar_feature_extraction", runtime_identity.feature_key)
    _expect_int(sidecar, "feature_count", "gateway_smoke.sidecar_feature_extraction", runtime_identity.feature_count)
    _expect_string(sidecar, "model_id", "gateway_smoke.sidecar_feature_extraction", runtime_identity.source_model_id)
    _expect_string(sidecar, "revision", "gateway_smoke.sidecar_feature_extraction", runtime_identity.source_revision)
    _expect_int(
        sidecar,
        "hidden_size",
        "gateway_smoke.sidecar_feature_extraction",
        runtime_identity.source_hidden_size,
    )
    _expect_int(
        sidecar,
        "layer_count",
        "gateway_smoke.sidecar_feature_extraction",
        runtime_identity.source_layer_count,
    )
    _expect_string(
        sidecar,
        "tokenizer_fingerprint_sha256",
        "gateway_smoke.sidecar_feature_extraction",
        runtime_identity.tokenizer_fingerprint_sha256,
    )
    _expect_string(
        sidecar,
        "special_tokens_map_sha256",
        "gateway_smoke.sidecar_feature_extraction",
        runtime_identity.special_tokens_map_sha256,
    )
    _expect_string(
        sidecar,
        "chat_template_sha256",
        "gateway_smoke.sidecar_feature_extraction",
        runtime_identity.chat_template_sha256,
    )
    _expect_string(
        sidecar,
        "prompt_renderer",
        "gateway_smoke.sidecar_feature_extraction",
        config.expected_prompt_renderer,
    )
    _expect_string(
        sidecar,
        "selected_choice_geometry",
        "gateway_smoke.sidecar_feature_extraction",
        config.expected_selected_choice_geometry,
    )
    _expect_int(
        sidecar,
        "selected_choice_readout_token_count",
        "gateway_smoke.sidecar_feature_extraction",
        config.expected_selected_choice_readout_token_count,
    )
    _validate_extraction_receipt_fields(
        record=sidecar,
        label="gateway_smoke.sidecar_feature_extraction",
        config=config,
        runtime_identity=runtime_identity,
        receipt_schema_field_name="extraction_receipt_schema_version",
        feature_vector_length_field_name="feature_vector_length",
        feature_vector_sha256_field_name="feature_vector_sha256",
        rendered_prompt_sha256_field_name="rendered_prompt_sha256",
        token_indices_field_name="selected_choice_readout_token_indices",
        token_indices_sha256_field_name="selected_choice_readout_token_indices_sha256",
        hidden_state_layer_count_field_name="hidden_state_layer_count",
        hidden_state_device_field_name="hidden_state_device_observed",
        input_device_field_name="input_device_observed",
    )


def _validate_gateway_smoke_capabilities(
    config: CiftCertificationBindingConfig,
    capabilities: Mapping[str, object],
) -> None:
    _expect_string(
        capabilities,
        "capability_mode",
        "gateway_smoke.cift_capabilities",
        "self_hosted_introspection",
    )
    detectors = capabilities.get("detectors")
    if (
        not isinstance(detectors, list)
        or any(not isinstance(detector, str) for detector in detectors)
        or config.expected_detector_name not in detectors
    ):
        raise CiftCertificationBindingError(
            f"gateway_smoke.cift_capabilities.detectors must include {config.expected_detector_name}."
        )
    turn_annotator_count = _required_int(
        capabilities,
        "turn_annotator_count",
        "gateway_smoke.cift_capabilities",
    )
    if turn_annotator_count < 1:
        raise CiftCertificationBindingError("gateway_smoke.cift_capabilities.turn_annotator_count must be positive.")


def _validate_gateway_smoke_decision(
    config: CiftCertificationBindingConfig,
    runtime_identity: _RuntimeIdentity,
    decision: Mapping[str, object],
    label: str,
    expected_positive: bool,
) -> None:
    _expect_string(decision, "extractor_id", label, config.expected_extractor_id)
    _expect_string(decision, "extractor_model_id", label, runtime_identity.source_model_id)
    _expect_string(decision, "extractor_revision", label, runtime_identity.source_revision)
    _expect_string(decision, "extractor_selected_device", label, config.required_device)
    _expect_int(decision, "extractor_hidden_size", label, runtime_identity.source_hidden_size)
    _expect_int(decision, "extractor_layer_count", label, runtime_identity.source_layer_count)
    _expect_string(
        decision,
        "extractor_tokenizer_fingerprint_sha256",
        label,
        runtime_identity.tokenizer_fingerprint_sha256,
    )
    _expect_string(
        decision,
        "extractor_special_tokens_map_sha256",
        label,
        runtime_identity.special_tokens_map_sha256,
    )
    _expect_string(decision, "extractor_chat_template_sha256", label, runtime_identity.chat_template_sha256)
    _expect_string(decision, "extractor_prompt_renderer", label, config.expected_prompt_renderer)
    _expect_string(decision, "extractor_selected_choice_geometry", label, config.expected_selected_choice_geometry)
    _expect_int(
        decision,
        "extractor_selected_choice_readout_token_count",
        label,
        config.expected_selected_choice_readout_token_count,
    )
    _validate_extraction_receipt_fields(
        record=decision,
        label=label,
        config=config,
        runtime_identity=runtime_identity,
        receipt_schema_field_name="extractor_extraction_receipt_schema_version",
        feature_vector_length_field_name="extractor_feature_vector_length",
        feature_vector_sha256_field_name="extractor_feature_vector_sha256",
        rendered_prompt_sha256_field_name="extractor_rendered_prompt_sha256",
        token_indices_field_name="extractor_selected_choice_readout_token_indices",
        token_indices_sha256_field_name="extractor_selected_choice_readout_token_indices_sha256",
        hidden_state_layer_count_field_name="extractor_hidden_state_layer_count",
        hidden_state_device_field_name="extractor_hidden_state_device_observed",
        input_device_field_name="extractor_input_device_observed",
    )
    _expect_string(decision, "feature_key", label, runtime_identity.feature_key)
    _expect_string(decision, "feature_source", label, config.expected_feature_source)
    _expect_string(decision, "positive_label", label, runtime_identity.positive_label)
    _expect_string(decision, "cift_window_family", label, "selected_choice")
    predicted_label = _required_string(decision, "predicted_label", label)
    if expected_positive:
        _expect_block_or_stronger(decision, "final_action", label)
        _expect_block_or_stronger(decision, "cift_action", label)
        _expect_string(decision, "provider_status", label, "skipped")
        _expect_string(decision, "provider_reason", label, "pre_generation_policy_block")
        if predicted_label != runtime_identity.positive_label:
            raise CiftCertificationBindingError(f"{label}.predicted_label must equal positive_label.")
        return
    _expect_action(decision, "final_action", label, Action.ALLOW)
    _expect_action(decision, "cift_action", label, Action.ALLOW)
    _expect_string(decision, "provider_status", label, "completed")
    if decision.get("provider_reason") is not None:
        raise CiftCertificationBindingError(f"{label}.provider_reason must be null.")
    if predicted_label == runtime_identity.positive_label:
        raise CiftCertificationBindingError(f"{label}.predicted_label must not equal positive_label.")


def _zero_confusion_metrics(gateway_smoke: Mapping[str, object]) -> None:
    metrics = _required_mapping(gateway_smoke.get("confusion_metrics"), "gateway_smoke.confusion_metrics")
    for field_name in ("false_negative_count", "false_positive_count", "false_negative_rate", "false_positive_rate"):
        value = metrics.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CiftCertificationBindingError(f"gateway_smoke.confusion_metrics.{field_name} must be a number.")
        if float(value) != 0.0:
            raise CiftCertificationBindingError(f"gateway_smoke.confusion_metrics.{field_name} must be zero.")


def _validate_extraction_receipt_fields(
    record: Mapping[str, object],
    label: str,
    config: CiftCertificationBindingConfig,
    runtime_identity: _RuntimeIdentity,
    receipt_schema_field_name: str,
    feature_vector_length_field_name: str,
    feature_vector_sha256_field_name: str,
    rendered_prompt_sha256_field_name: str,
    token_indices_field_name: str,
    token_indices_sha256_field_name: str,
    hidden_state_layer_count_field_name: str,
    hidden_state_device_field_name: str,
    input_device_field_name: str,
) -> None:
    _expect_string(record, receipt_schema_field_name, label, CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION)
    feature_vector_length = _required_int(record, feature_vector_length_field_name, label)
    if feature_vector_length < 1:
        raise CiftCertificationBindingError(f"{label}.{feature_vector_length_field_name} must be positive.")
    if feature_vector_length != runtime_identity.feature_count:
        raise CiftCertificationBindingError(
            f"{label}.{feature_vector_length_field_name} must match runtime feature_count."
        )
    _expect_sha256_string(record, feature_vector_sha256_field_name, label)
    _expect_sha256_string(record, rendered_prompt_sha256_field_name, label)
    token_indices_sha256 = _expect_sha256_string(record, token_indices_sha256_field_name, label)
    token_indices = _required_int_list(record, token_indices_field_name, label)
    if len(token_indices) != config.expected_selected_choice_readout_token_count:
        raise CiftCertificationBindingError(
            f"{label}.{token_indices_field_name} must match expected selected-choice readout token count."
        )
    if token_indices_sha256 != _json_sha256(list(token_indices)):
        raise CiftCertificationBindingError(
            f"{label}.{token_indices_sha256_field_name} must match {token_indices_field_name}."
        )
    hidden_state_layer_count = _required_int(record, hidden_state_layer_count_field_name, label)
    if hidden_state_layer_count < runtime_identity.source_layer_count:
        raise CiftCertificationBindingError(
            f"{label}.{hidden_state_layer_count_field_name} must be at least source layer_count."
        )
    hidden_state_device = _required_string(record, hidden_state_device_field_name, label)
    if not _device_matches_required(hidden_state_device, config.required_device):
        raise CiftCertificationBindingError(f"{label}.{hidden_state_device_field_name} must match required_device.")
    input_device = _required_string(record, input_device_field_name, label)
    if not _device_matches_required(input_device, config.required_device):
        raise CiftCertificationBindingError(f"{label}.{input_device_field_name} must match required_device.")


def _expect_sha256_string(record: Mapping[str, object], field_name: str, label: str) -> str:
    value = _required_string(record, field_name, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CiftCertificationBindingError(f"{label}.{field_name} must be a lowercase SHA-256 digest.")
    return value


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_int_list(record: Mapping[str, object], field_name: str, label: str) -> tuple[int, ...]:
    value = record.get(field_name)
    if not isinstance(value, list) or len(value) == 0:
        raise CiftCertificationBindingError(f"{label}.{field_name} must be a non-empty integer list.")
    values: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise CiftCertificationBindingError(f"{label}.{field_name}[{index}] must be a non-negative integer.")
        values.append(item)
    return tuple(values)


def _device_matches_required(observed_device: str, required_device: str) -> bool:
    if required_device == "cpu":
        return observed_device == "cpu"
    return observed_device == required_device or observed_device.startswith(f"{required_device}:")


def _expect_block_or_stronger(record: Mapping[str, object], field_name: str, label: str) -> None:
    action = _required_action(record, field_name, label)
    if action_severity(action) < action_severity(Action.BLOCK):
        raise CiftCertificationBindingError(f"{label}.{field_name} must be block or stronger.")


def _expect_action(record: Mapping[str, object], field_name: str, label: str, expected_action: Action) -> None:
    action = _required_action(record, field_name, label)
    if action is not expected_action:
        raise CiftCertificationBindingError(f"{label}.{field_name} must be {expected_action.value}.")


def _required_action(record: Mapping[str, object], field_name: str, label: str) -> Action:
    value = _required_string(record, field_name, label)
    try:
        return Action(value)
    except ValueError as exc:
        raise CiftCertificationBindingError(f"{label}.{field_name} must be a known action.") from exc


def _required_artifacts(record: Mapping[str, object], label: str) -> tuple[Mapping[str, object], ...]:
    raw_artifacts = record.get("required_evidence_artifacts")
    if raw_artifacts is None:
        raw_artifacts = record.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise CiftCertificationBindingError(f"{label} artifacts must be a list.")
    artifacts: list[Mapping[str, object]] = []
    for index, raw_artifact in enumerate(raw_artifacts):
        if not isinstance(raw_artifact, dict):
            raise CiftCertificationBindingError(f"{label} artifacts[{index}] must be an object.")
        artifacts.append(cast(Mapping[str, object], raw_artifact))
    return tuple(artifacts)


def _artifacts_by_role(
    artifacts: Sequence[Mapping[str, object]],
    label: str,
) -> dict[str, Mapping[str, object]]:
    artifacts_by_role: dict[str, Mapping[str, object]] = {}
    for artifact in artifacts:
        role = _required_string(artifact, "role", f"{label} artifact")
        if role in artifacts_by_role:
            raise CiftCertificationBindingError(f"{label} must contain exactly one artifact with role {role}.")
        artifacts_by_role[role] = artifact
    return artifacts_by_role


def _single_artifact(
    artifacts: Sequence[Mapping[str, object]],
    role: str,
    label: str,
) -> Mapping[str, object]:
    matches = tuple(artifact for artifact in artifacts if artifact.get("role") == role)
    if len(matches) != 1:
        raise CiftCertificationBindingError(f"{label} must contain exactly one artifact with role {role}.")
    return matches[0]


def _validate_artifact_count(record: Mapping[str, object], field_name: str, label: str, count: int) -> None:
    artifact_count = _required_int(record, field_name, label)
    if artifact_count != count:
        raise CiftCertificationBindingError(f"{label}.{field_name} must be {count}, got {artifact_count}.")


def _validate_workflow_artifact_roles(
    workflow_artifacts_by_role: Mapping[str, Mapping[str, object]],
    manifest_artifacts_by_role: Mapping[str, Mapping[str, object]],
) -> None:
    workflow_roles = set(workflow_artifacts_by_role)
    manifest_roles = set(manifest_artifacts_by_role)
    if workflow_roles == manifest_roles:
        return
    missing_roles = tuple(sorted(manifest_roles - workflow_roles))
    extra_roles = tuple(sorted(workflow_roles - manifest_roles))
    raise CiftCertificationBindingError(
        "certification workflow run artifact roles must match certification manifest; "
        f"missing={missing_roles}, extra={extra_roles}."
    )


def _required_artifact_spec(role: str) -> _RequiredCertificationArtifact | None:
    for spec in _REQUIRED_CERTIFICATION_ARTIFACTS:
        if spec.role == role:
            return spec
    return None


def _requires_json_identity(artifact_kind: str) -> bool:
    return artifact_kind in {"json_report", "promotion_evidence", "runtime_model"}


def _load_json_object(path: Path, label: str) -> _LoadedJsonObject:
    if not path.is_file():
        raise CiftCertificationBindingError(f"{label} does not exist: {path}.")
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        decoded = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise CiftCertificationBindingError(f"{label} contains invalid JSON: {exc.msg}.") from exc
    except UnicodeDecodeError as exc:
        raise CiftCertificationBindingError(f"{label} must be UTF-8 JSON.") from exc
    if not isinstance(decoded, dict):
        raise CiftCertificationBindingError(f"{label} must contain a JSON object.")
    return _LoadedJsonObject(sha256=sha256, record=cast(Mapping[str, object], decoded))


def _validate_expected_sha256(value: str, label: str) -> None:
    if len(value) != 64:
        raise CiftCertificationBindingError(f"{label} must be a 64-character SHA-256 hex digest.")
    for character in value:
        if character not in "0123456789abcdef":
            raise CiftCertificationBindingError(f"{label} must be lowercase SHA-256 hex.")


def _expect_sha256(actual_sha256: str, expected_sha256: str, label: str) -> None:
    if actual_sha256 != expected_sha256:
        raise CiftCertificationBindingError(f"{label} sha256 must be {expected_sha256}, got {actual_sha256}.")


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CiftCertificationBindingError(f"{label} must be an object.")
    return cast(Mapping[str, object], value)


def _required_string(record: Mapping[str, object], field_name: str, label: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or value == "":
        raise CiftCertificationBindingError(f"{label}.{field_name} must be a non-empty string.")
    return value


def _required_int(record: Mapping[str, object], field_name: str, label: str) -> int:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftCertificationBindingError(f"{label}.{field_name} must be an integer.")
    return value


def _expect_string(record: Mapping[str, object], field_name: str, label: str, expected_value: str) -> None:
    actual_value = _required_string(record, field_name, label)
    if actual_value != expected_value:
        raise CiftCertificationBindingError(f"{label}.{field_name} must be {expected_value}, got {actual_value}.")


def _expect_schema_version(
    record: Mapping[str, object],
    field_name: str,
    label: str,
    expected_values: tuple[str | None, ...],
) -> None:
    actual_value = record.get(field_name)
    if actual_value not in expected_values:
        expected_text = ", ".join("null" if value is None else value for value in expected_values)
        actual_text = "null" if actual_value is None else str(actual_value)
        raise CiftCertificationBindingError(
            f"{label}.{field_name} must be one of [{expected_text}], got {actual_text}."
        )


def _optional_schema_version(record: Mapping[str, object], field_name: str, label: str) -> None:
    actual_value = record.get(field_name)
    if actual_value is not None and not isinstance(actual_value, str):
        raise CiftCertificationBindingError(f"{label}.{field_name} must be null or a string.")


def _required_finite_number(record: Mapping[str, object], field_name: str, label: str) -> float:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CiftCertificationBindingError(f"{label}.{field_name} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise CiftCertificationBindingError(f"{label}.{field_name} must be finite.")
    return number


def _validate_finite_confusion_metrics(
    record: Mapping[str, object],
    label: str,
    require_zero: bool,
) -> None:
    for field_name in ("false_negative_count", "false_positive_count", "false_negative_rate", "false_positive_rate"):
        value = _required_finite_number(record, field_name, label)
        if require_zero and value != 0.0:
            raise CiftCertificationBindingError(f"{label}.{field_name} must be zero.")


def _expect_bool(record: Mapping[str, object], field_name: str, label: str, expected_value: bool) -> None:
    actual_value = record.get(field_name)
    if not isinstance(actual_value, bool):
        raise CiftCertificationBindingError(f"{label}.{field_name} must be a boolean.")
    if actual_value is not expected_value:
        raise CiftCertificationBindingError(f"{label}.{field_name} must be {expected_value}.")


def _expect_number(record: Mapping[str, object], field_name: str, label: str, expected_value: float) -> None:
    actual_value = _required_finite_number(record, field_name, label)
    if not math.isclose(actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-12):
        raise CiftCertificationBindingError(f"{label}.{field_name} must be {expected_value}, got {actual_value}.")


def _expect_row_action_block_or_stronger(row: Mapping[str, object], field_name: str, label: str) -> None:
    action = _required_string(row, field_name, label)
    try:
        parsed_action = Action(action)
    except ValueError as exc:
        raise CiftCertificationBindingError(f"{label}.{field_name} must be a known action.") from exc
    if action_severity(parsed_action) < action_severity(Action.BLOCK):
        raise CiftCertificationBindingError(f"{label}.{field_name} must be block or stronger.")


def _expect_int(record: Mapping[str, object], field_name: str, label: str, expected_value: int) -> None:
    actual_value = record.get(field_name)
    if isinstance(actual_value, bool) or not isinstance(actual_value, int):
        raise CiftCertificationBindingError(f"{label}.{field_name} must be an integer.")
    if actual_value != expected_value:
        raise CiftCertificationBindingError(f"{label}.{field_name} must be {expected_value}, got {actual_value}.")
