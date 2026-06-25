from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeAlias, cast

from aegis.cift_contract import CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION
from aegis_introspection.cift_live_probe_competition import (
    CiftLiveProbeCompetitionError,
    cift_live_probe_competition_report_from_mapping,
    live_promotion_paper_method_from_probe_competition,
)
from aegis_introspection.cift_model_bundle import CiftModelBundle, load_cift_model_bundle
from aegis_introspection.cift_probe_competition import (
    CiftProbeCompetitionError,
    cift_probe_competition_report_from_mapping,
    promotion_paper_method_from_probe_competition,
)
from aegis_introspection.cift_promotion_gate import (
    CiftPaperMethodContract,
    CiftPromotionEvidence,
    CiftPromotionGateDecision,
    CiftPromotionReportArtifact,
    cift_promotion_evidence_to_json,
    evaluate_cift_promotion_gate,
)

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class CiftPromotionEvidenceMaterializerError(ValueError):
    """Raised when CIFT promotion evidence cannot be materialized."""


@dataclass(frozen=True)
class CiftPromotionReportSource:
    report_id: str
    schema_version: str
    source_path: Path


@dataclass(frozen=True)
class CiftPromotionEvidenceMaterializerConfig:
    bundle_path: Path
    repository_root: Path
    report_output_dir: Path
    evidence_output_path: Path
    evidence_id: str
    behavior_id: str
    behavior_description: str
    train_split_id: str
    calibration_split_id: str
    heldout_split_id: str
    sealed_holdout_split_id: str
    sealed_holdout_report_id: str
    metric_report_id: str
    metric_name: str
    metric_value: float
    metric_threshold: float
    calibration_report_id: str
    ablation_report_id: str
    ablation_delta: float
    ablation_delta_threshold: float
    patching_report_id: str
    failure_case_report_id: str
    runtime_prevention_report_id: str
    gateway_smoke_report_id: str
    lineage_report_id: str
    head_to_head_report_id: str | None
    report_sources: tuple[CiftPromotionReportSource, ...]
    created_at: str


@dataclass(frozen=True)
class CiftPromotionWorkflowEvidenceRoles:
    bundle_role: str
    sealed_holdout_role: str
    calibration_role: str
    feature_ablation_role: str
    patching_role: str
    failure_cases_role: str
    runtime_prevention_role: str
    gateway_smoke_role: str
    lineage_role: str
    head_to_head_role: str
    promotion_evidence_role: str


@dataclass(frozen=True)
class NormalizedPromotionReports:
    artifacts: tuple[CiftPromotionReportArtifact, ...]
    records_by_report_id: dict[str, Mapping[str, object]]


DEFAULT_WORKFLOW_PROMOTION_EVIDENCE_ROLES = CiftPromotionWorkflowEvidenceRoles(
    bundle_role="linear_candidate_bundle",
    sealed_holdout_role="linear_sealed_holdout_metric",
    calibration_role="calibration",
    feature_ablation_role="feature_ablation",
    patching_role="counterfactual_patching",
    failure_cases_role="failure_cases",
    runtime_prevention_role="linear_live_runtime_prevention",
    gateway_smoke_role="linear_gateway_smoke",
    lineage_role="lineage",
    head_to_head_role="live_sealed_linear_vs_paper_mlp",
    promotion_evidence_role="promotion_evidence",
)


def cift_promotion_materializer_config_from_workflow_manifest(
    repository_root: Path,
    workflow_manifest_path: Path,
    evidence_roles: CiftPromotionWorkflowEvidenceRoles,
) -> CiftPromotionEvidenceMaterializerConfig:
    manifest = _json_object_from_path(workflow_manifest_path)
    if _workflow_string(manifest, "schema_version", "workflow manifest") != (
        "aegis_introspection.cift_certification_workflow/v1"
    ):
        raise CiftPromotionEvidenceMaterializerError(
            "workflow manifest schema_version must be aegis_introspection.cift_certification_workflow/v1."
        )
    artifacts_by_role = _workflow_artifacts_by_role(manifest)
    training = _workflow_mapping(manifest, "training", "workflow manifest")
    planned_artifacts = _workflow_mapping(manifest, "planned_artifacts", "workflow manifest")

    sealed_holdout_source = _workflow_report_source(
        repository_root=repository_root,
        artifacts_by_role=artifacts_by_role,
        role=evidence_roles.sealed_holdout_role,
        expected_schema_version="aegis_introspection.cift_sealed_holdout_metric/v1",
    )
    calibration_source = _workflow_report_source(
        repository_root=repository_root,
        artifacts_by_role=artifacts_by_role,
        role=evidence_roles.calibration_role,
        expected_schema_version="aegis_introspection.cift_calibration/v1",
    )
    feature_ablation_source = _workflow_report_source(
        repository_root=repository_root,
        artifacts_by_role=artifacts_by_role,
        role=evidence_roles.feature_ablation_role,
        expected_schema_version="aegis_introspection.cift_feature_ablation/v1",
    )
    patching_source = _workflow_report_source(
        repository_root=repository_root,
        artifacts_by_role=artifacts_by_role,
        role=evidence_roles.patching_role,
        expected_schema_version="aegis_introspection.cift_counterfactual_patching/v1",
    )
    failure_cases_source = _workflow_report_source(
        repository_root=repository_root,
        artifacts_by_role=artifacts_by_role,
        role=evidence_roles.failure_cases_role,
        expected_schema_version="aegis_introspection.cift_failure_cases/v1",
    )
    runtime_prevention_source = _workflow_report_source(
        repository_root=repository_root,
        artifacts_by_role=artifacts_by_role,
        role=evidence_roles.runtime_prevention_role,
        expected_schema_version="aegis_introspection.cift_live_window_selector_benchmark/v1",
    )
    gateway_smoke_source = _workflow_report_source(
        repository_root=repository_root,
        artifacts_by_role=artifacts_by_role,
        role=evidence_roles.gateway_smoke_role,
        expected_schema_version="aegis.proxy.cift_gateway_smoke/v1",
    )
    lineage_source = _workflow_report_source(
        repository_root=repository_root,
        artifacts_by_role=artifacts_by_role,
        role=evidence_roles.lineage_role,
        expected_schema_version="aegis_introspection.cift_lineage/v1",
    )
    head_to_head_source = _workflow_report_source(
        repository_root=repository_root,
        artifacts_by_role=artifacts_by_role,
        role=evidence_roles.head_to_head_role,
        expected_schema_version="aegis_introspection.cift_live_probe_competition/v1",
    )

    sealed_holdout_record = _json_object_from_path(sealed_holdout_source.source_path)
    feature_ablation_record = _json_object_from_path(feature_ablation_source.source_path)
    training_dataset_id = _workflow_string(training, "training_dataset_id", "workflow manifest training")
    bundle_path = _workflow_artifact_path(
        repository_root=repository_root,
        artifacts_by_role=artifacts_by_role,
        role=evidence_roles.bundle_role,
    )
    evidence_output_path = _workflow_artifact_path(
        repository_root=repository_root,
        artifacts_by_role=artifacts_by_role,
        role=evidence_roles.promotion_evidence_role,
    )
    report_output_dir = _workflow_planned_path(
        repository_root=repository_root,
        planned_artifacts=planned_artifacts,
        artifact_key="promotion_report_output_dir",
    )

    return CiftPromotionEvidenceMaterializerConfig(
        bundle_path=bundle_path,
        repository_root=repository_root,
        report_output_dir=report_output_dir,
        evidence_output_path=evidence_output_path,
        evidence_id=evidence_output_path.stem,
        behavior_id=_workflow_string(training, "behavior_id", "workflow manifest training"),
        behavior_description=_workflow_string(training, "behavior_description", "workflow manifest training"),
        train_split_id=f"{training_dataset_id}/train",
        calibration_split_id=f"{training_dataset_id}/calibration",
        heldout_split_id=f"{training_dataset_id}/grouped-cv",
        sealed_holdout_split_id=_sealed_holdout_split_id(sealed_holdout_record),
        sealed_holdout_report_id=sealed_holdout_source.report_id,
        metric_report_id=sealed_holdout_source.report_id,
        metric_name=_workflow_string(sealed_holdout_record, "metric_name", "sealed holdout report"),
        metric_value=_workflow_number(sealed_holdout_record, "metric_value", "sealed holdout report"),
        metric_threshold=_workflow_number(training, "metric_threshold", "workflow manifest training"),
        calibration_report_id=calibration_source.report_id,
        ablation_report_id=feature_ablation_source.report_id,
        ablation_delta=_ablation_delta_from_report(
            record=feature_ablation_record,
            candidate_feature_key=_workflow_string(training, "candidate_feature_key", "workflow manifest training"),
        ),
        ablation_delta_threshold=_workflow_number(
            training,
            "ablation_delta_threshold",
            "workflow manifest training",
        ),
        patching_report_id=patching_source.report_id,
        failure_case_report_id=failure_cases_source.report_id,
        runtime_prevention_report_id=runtime_prevention_source.report_id,
        gateway_smoke_report_id=gateway_smoke_source.report_id,
        lineage_report_id=lineage_source.report_id,
        head_to_head_report_id=head_to_head_source.report_id,
        report_sources=(
            sealed_holdout_source,
            calibration_source,
            feature_ablation_source,
            patching_source,
            failure_cases_source,
            runtime_prevention_source,
            gateway_smoke_source,
            lineage_source,
            head_to_head_source,
        ),
        created_at=_workflow_string(manifest, "created_at", "workflow manifest"),
    )


def materialize_cift_promotion_evidence(
    config: CiftPromotionEvidenceMaterializerConfig,
) -> CiftPromotionEvidence:
    _validate_config(config)
    bundle = load_cift_model_bundle(config.bundle_path)
    normalized_reports = _normalize_report_artifacts(config)
    _validate_required_report_schemas(config=config, records_by_report_id=normalized_reports.records_by_report_id)
    _validate_patching_report(
        bundle=bundle,
        records_by_report_id=normalized_reports.records_by_report_id,
        patching_report_id=config.patching_report_id,
    )
    _validate_calibration_report(
        bundle=bundle,
        records_by_report_id=normalized_reports.records_by_report_id,
        calibration_report_id=config.calibration_report_id,
    )
    _validate_feature_ablation_report(
        bundle=bundle,
        records_by_report_id=normalized_reports.records_by_report_id,
        ablation_report_id=config.ablation_report_id,
    )
    _validate_lineage_report(
        bundle=bundle,
        records_by_report_id=normalized_reports.records_by_report_id,
        lineage_report_id=config.lineage_report_id,
    )
    _validate_failure_cases_report(
        bundle=bundle,
        records_by_report_id=normalized_reports.records_by_report_id,
        failure_case_report_id=config.failure_case_report_id,
        runtime_prevention_report_id=config.runtime_prevention_report_id,
    )
    _validate_runtime_prevention_report(
        bundle=bundle,
        records_by_report_id=normalized_reports.records_by_report_id,
        runtime_prevention_report_id=config.runtime_prevention_report_id,
    )
    _validate_gateway_smoke_report(
        bundle=bundle,
        records_by_report_id=normalized_reports.records_by_report_id,
        gateway_smoke_report_id=config.gateway_smoke_report_id,
    )
    _validate_sealed_holdout_report(
        bundle=bundle,
        records_by_report_id=normalized_reports.records_by_report_id,
        sealed_holdout_report_id=config.sealed_holdout_report_id,
        sealed_holdout_split_id=config.sealed_holdout_split_id,
        metric_name=config.metric_name,
        metric_value=config.metric_value,
    )
    _validate_head_to_head_report(
        bundle=bundle,
        records_by_report_id=normalized_reports.records_by_report_id,
        head_to_head_report_id=config.head_to_head_report_id,
        sealed_holdout_report_id=config.sealed_holdout_report_id,
        sealed_holdout_split_id=config.sealed_holdout_split_id,
        metric_name=config.metric_name,
    )
    paper_method = _paper_method_from_normalized_reports(
        head_to_head_report_id=config.head_to_head_report_id,
        records_by_report_id=normalized_reports.records_by_report_id,
    )
    evidence = CiftPromotionEvidence(
        schema_version="cift_promotion_evidence/v1",
        evidence_id=config.evidence_id,
        behavior_id=config.behavior_id,
        behavior_description=config.behavior_description,
        training_dataset_id=bundle.metadata.training_dataset_id,
        train_split_id=config.train_split_id,
        calibration_split_id=config.calibration_split_id,
        heldout_split_id=config.heldout_split_id,
        sealed_holdout_split_id=config.sealed_holdout_split_id,
        sealed_holdout_report_id=config.sealed_holdout_report_id,
        metric_report_id=config.metric_report_id,
        metric_name=config.metric_name,
        metric_value=config.metric_value,
        metric_threshold=config.metric_threshold,
        calibration_report_id=config.calibration_report_id,
        ablation_report_id=config.ablation_report_id,
        ablation_delta=config.ablation_delta,
        ablation_delta_threshold=config.ablation_delta_threshold,
        patching_report_id=config.patching_report_id,
        failure_case_report_id=config.failure_case_report_id,
        runtime_prevention_report_id=config.runtime_prevention_report_id,
        gateway_smoke_report_id=config.gateway_smoke_report_id,
        lineage_report_id=config.lineage_report_id,
        report_artifacts=normalized_reports.artifacts,
        paper_method=paper_method,
        created_at=config.created_at,
    )
    decision = evaluate_cift_promotion_gate(bundle=bundle, evidence=evidence)
    if not decision.eligible:
        raise CiftPromotionEvidenceMaterializerError(_promotion_decision_message(decision))
    _write_json(path=config.evidence_output_path, record=cift_promotion_evidence_to_json(evidence))
    return evidence


def _normalize_report_artifacts(config: CiftPromotionEvidenceMaterializerConfig) -> NormalizedPromotionReports:
    source_by_report_id = _report_source_by_report_id(config.report_sources)
    required_report_ids = _required_report_ids(config)
    _validate_source_coverage(source_by_report_id=source_by_report_id, required_report_ids=required_report_ids)
    artifacts: list[CiftPromotionReportArtifact] = []
    records_by_report_id: dict[str, Mapping[str, object]] = {}
    for report_id in required_report_ids:
        source = source_by_report_id[report_id]
        record = _normalized_report_record(source)
        output_path = config.report_output_dir / f"{report_id}.json"
        _write_json(path=output_path, record=cast(dict[str, JsonValue], record))
        artifacts.append(
            CiftPromotionReportArtifact(
                report_id=report_id,
                path=_repository_relative_path(repository_root=config.repository_root, path=source.source_path),
                sha256=_sha256_file(source.source_path),
                schema_version=source.schema_version,
            )
        )
        records_by_report_id[report_id] = record
    return NormalizedPromotionReports(
        artifacts=tuple(artifacts),
        records_by_report_id=records_by_report_id,
    )


def _validate_required_report_schemas(
    config: CiftPromotionEvidenceMaterializerConfig,
    records_by_report_id: dict[str, Mapping[str, object]],
) -> None:
    expected_schemas: tuple[tuple[str, str, str], ...] = (
        (
            config.sealed_holdout_report_id,
            "sealed_holdout_report",
            "aegis_introspection.cift_sealed_holdout_metric/v1",
        ),
        (config.calibration_report_id, "calibration_report", "aegis_introspection.cift_calibration/v1"),
        (config.ablation_report_id, "feature_ablation_report", "aegis_introspection.cift_feature_ablation/v1"),
        (config.patching_report_id, "patching_report", "aegis_introspection.cift_counterfactual_patching/v1"),
        (config.failure_case_report_id, "failure_cases_report", "aegis_introspection.cift_failure_cases/v1"),
        (
            config.runtime_prevention_report_id,
            "runtime_prevention_report",
            "aegis_introspection.cift_live_window_selector_benchmark/v1",
        ),
        (config.gateway_smoke_report_id, "gateway_smoke_report", "aegis.proxy.cift_gateway_smoke/v1"),
        (config.lineage_report_id, "lineage_report", "aegis_introspection.cift_lineage/v1"),
    )
    for report_id, report_label, expected_schema in expected_schemas:
        _validate_report_schema(
            records_by_report_id=records_by_report_id,
            report_id=report_id,
            report_label=report_label,
            expected_schema=expected_schema,
        )
    if config.head_to_head_report_id is not None:
        _validate_report_schema(
            records_by_report_id=records_by_report_id,
            report_id=config.head_to_head_report_id,
            report_label="head_to_head_report",
            expected_schema="aegis_introspection.cift_live_probe_competition/v1",
        )


def _validate_report_schema(
    records_by_report_id: dict[str, Mapping[str, object]],
    report_id: str,
    report_label: str,
    expected_schema: str,
) -> None:
    record = records_by_report_id.get(report_id)
    if record is None:
        raise CiftPromotionEvidenceMaterializerError(
            f"{report_label} '{report_id}' is missing from normalized reports."
        )
    actual_schema = record.get("schema_version")
    if actual_schema != expected_schema:
        raise CiftPromotionEvidenceMaterializerError(f"{report_label} schema_version must be {expected_schema}.")


def _report_source_by_report_id(
    sources: tuple[CiftPromotionReportSource, ...],
) -> dict[str, CiftPromotionReportSource]:
    source_by_report_id: dict[str, CiftPromotionReportSource] = {}
    for source in sources:
        if source.report_id in source_by_report_id:
            raise CiftPromotionEvidenceMaterializerError(f"Duplicate report source '{source.report_id}'.")
        source_by_report_id[source.report_id] = source
    return source_by_report_id


def _validate_source_coverage(
    source_by_report_id: dict[str, CiftPromotionReportSource],
    required_report_ids: tuple[str, ...],
) -> None:
    source_report_ids = set(source_by_report_id)
    required_report_id_set = set(required_report_ids)
    missing_report_ids = tuple(report_id for report_id in required_report_ids if report_id not in source_report_ids)
    extra_report_ids = tuple(report_id for report_id in source_report_ids if report_id not in required_report_id_set)
    if len(missing_report_ids) > 0:
        raise CiftPromotionEvidenceMaterializerError(f"Missing report sources: {', '.join(missing_report_ids)}.")
    if len(extra_report_ids) > 0:
        raise CiftPromotionEvidenceMaterializerError(f"Unexpected report sources: {', '.join(extra_report_ids)}.")


def _normalized_report_record(source: CiftPromotionReportSource) -> Mapping[str, object]:
    record = dict(_json_object_from_path(source.source_path))
    existing_report_id = record.get("report_id")
    if not isinstance(existing_report_id, str):
        raise CiftPromotionEvidenceMaterializerError(
            f"Source report {source.source_path} must declare report_id '{source.report_id}'."
        )
    if existing_report_id != source.report_id:
        raise CiftPromotionEvidenceMaterializerError(
            f"Source report {source.source_path} has report_id '{existing_report_id}', expected '{source.report_id}'."
        )
    existing_schema_version = record.get("schema_version")
    if not isinstance(existing_schema_version, str):
        raise CiftPromotionEvidenceMaterializerError(
            f"Source report {source.source_path} must declare schema_version '{source.schema_version}'."
        )
    if existing_schema_version != source.schema_version:
        raise CiftPromotionEvidenceMaterializerError(
            f"Source report {source.source_path} has schema_version '{existing_schema_version}', "
            f"expected '{source.schema_version}'."
        )
    return record


def _paper_method_from_normalized_reports(
    head_to_head_report_id: str | None,
    records_by_report_id: dict[str, Mapping[str, object]],
) -> CiftPaperMethodContract:
    if head_to_head_report_id is None:
        return _paper_mlp_method_contract()
    record = records_by_report_id.get(head_to_head_report_id)
    if record is None:
        raise CiftPromotionEvidenceMaterializerError(
            f"head_to_head_report_id '{head_to_head_report_id}' is missing from normalized reports."
        )
    schema_version = record.get("schema_version")
    if schema_version == "aegis_introspection.cift_live_probe_competition/v1":
        return _paper_method_from_live_head_to_head_record(record=record)
    try:
        competition_report = cift_probe_competition_report_from_mapping(record)
        method = promotion_paper_method_from_probe_competition(competition_report)
        feature_representation = record.get("feature_representation")
        if feature_representation is None:
            raise CiftPromotionEvidenceMaterializerError("head-to-head report must declare feature_representation.")
        if not isinstance(feature_representation, str):
            raise CiftPromotionEvidenceMaterializerError("head-to-head feature_representation must be a string.")
        return _method_for_feature_representation(method=method, feature_representation=feature_representation)
    except CiftProbeCompetitionError as exc:
        raise CiftPromotionEvidenceMaterializerError(str(exc)) from exc


def _paper_method_from_live_head_to_head_record(record: Mapping[str, object]) -> CiftPaperMethodContract:
    try:
        live_report = cift_live_probe_competition_report_from_mapping(record)
        return live_promotion_paper_method_from_probe_competition(live_report)
    except CiftLiveProbeCompetitionError as exc:
        raise CiftPromotionEvidenceMaterializerError(str(exc)) from exc


def _method_for_feature_representation(
    method: CiftPaperMethodContract,
    feature_representation: str,
) -> CiftPaperMethodContract:
    if feature_representation == "diagonal_mahalanobis_cci":
        return replace(method, feature_representation=feature_representation)
    if feature_representation == "raw_activation":
        return replace(
            method,
            feature_representation=feature_representation,
            covariance_estimator="not_applicable",
            ridge=0.0,
            layer_weighting="not_applicable",
            paper_faithfulness_exception=(
                "raw_activation head-to-head evidence shows the candidate probe strictly outperforms the paper MLP"
            ),
        )
    raise CiftPromotionEvidenceMaterializerError(
        f"Unsupported head-to-head feature_representation '{feature_representation}'."
    )


def _validate_patching_report(
    bundle: CiftModelBundle,
    records_by_report_id: dict[str, Mapping[str, object]],
    patching_report_id: str,
) -> None:
    record = records_by_report_id.get(patching_report_id)
    if record is None:
        raise CiftPromotionEvidenceMaterializerError(
            f"patching_report_id '{patching_report_id}' is missing from normalized reports."
        )
    failures: list[str] = []
    if _record_string(record=record, field_name="schema_version", failures=failures) != (
        "aegis_introspection.cift_counterfactual_patching/v1"
    ):
        failures.append("patching report schema_version must be aegis_introspection.cift_counterfactual_patching/v1")
    expected_fields = (
        ("training_dataset_id", bundle.metadata.training_dataset_id),
        ("task_name", bundle.metadata.task_name),
        ("feature_key", bundle.metadata.activation_feature_key),
        ("source_artifact_sha256", bundle.metadata.source_artifact_sha256.lower()),
        ("intervention_type", "paired_feature_vector_replacement"),
        ("claim_scope", "runtime_detector_decision"),
    )
    for field_name, expected_value in expected_fields:
        actual_value = _record_string(record=record, field_name=field_name, failures=failures)
        if actual_value is not None and actual_value != expected_value:
            failures.append(f"patching report {field_name} must match {expected_value}")
    hidden_state_patching = _record_bool(
        record=record,
        field_name="transformer_hidden_state_patching",
        failures=failures,
    )
    if hidden_state_patching is not False:
        failures.append("patching report transformer_hidden_state_patching must be false")
    limitation = _record_string(record=record, field_name="paper_faithfulness_limitation", failures=failures)
    if limitation is not None and limitation.strip() == "":
        failures.append("patching report paper_faithfulness_limitation must not be empty")
    pair_count = _record_number(record=record, field_name="pair_count", failures=failures)
    minimum_flip_rate = _record_number(record=record, field_name="minimum_flip_rate", failures=failures)
    if pair_count is not None and pair_count < 1.0:
        failures.append("patching report pair_count must be positive")
    if minimum_flip_rate is not None:
        if minimum_flip_rate < 0.0 or minimum_flip_rate > 1.0:
            failures.append("patching report minimum_flip_rate must be in [0.0, 1.0]")
        failures.extend(_patching_rate_failures(record=record, minimum_flip_rate=minimum_flip_rate))
    passed = _record_bool(record=record, field_name="passed", failures=failures)
    if passed is not True:
        failures.append("patching report passed must be true")
    if len(failures) > 0:
        raise CiftPromotionEvidenceMaterializerError("; ".join(dict.fromkeys(failures)))


def _validate_calibration_report(
    bundle: CiftModelBundle,
    records_by_report_id: dict[str, Mapping[str, object]],
    calibration_report_id: str,
) -> None:
    record = records_by_report_id.get(calibration_report_id)
    if record is None:
        raise CiftPromotionEvidenceMaterializerError(
            f"calibration_report_id '{calibration_report_id}' is missing from normalized reports."
        )
    if record.get("schema_version") != "aegis_introspection.cift_calibration/v1":
        return
    failures = _identity_field_failures(
        record=record,
        report_label="calibration_report",
        expected_fields=(
            ("source_model_id", bundle.metadata.source_model_id),
            ("source_revision", bundle.metadata.source_revision),
            ("source_selected_device", bundle.metadata.source_selected_device),
            ("task_name", bundle.metadata.task_name),
            ("activation_feature_key", bundle.metadata.activation_feature_key),
            ("positive_label", bundle.metadata.positive_label),
        ),
    )
    if len(failures) > 0:
        raise CiftPromotionEvidenceMaterializerError("; ".join(dict.fromkeys(failures)))


def _validate_feature_ablation_report(
    bundle: CiftModelBundle,
    records_by_report_id: dict[str, Mapping[str, object]],
    ablation_report_id: str,
) -> None:
    record = records_by_report_id.get(ablation_report_id)
    if record is None:
        raise CiftPromotionEvidenceMaterializerError(
            f"ablation_report_id '{ablation_report_id}' is missing from normalized reports."
        )
    if record.get("schema_version") != "aegis_introspection.cift_feature_ablation/v1":
        return
    failures = _identity_field_failures(
        record=record,
        report_label="feature_ablation_report",
        expected_fields=(
            ("source_model_id", bundle.metadata.source_model_id),
            ("source_revision", bundle.metadata.source_revision),
            ("source_selected_device", bundle.metadata.source_selected_device),
            ("task_name", bundle.metadata.task_name),
            ("baseline_feature_key", bundle.metadata.activation_feature_key),
        ),
    )
    if len(failures) > 0:
        raise CiftPromotionEvidenceMaterializerError("; ".join(dict.fromkeys(failures)))


def _validate_lineage_report(
    bundle: CiftModelBundle,
    records_by_report_id: dict[str, Mapping[str, object]],
    lineage_report_id: str,
) -> None:
    record = records_by_report_id.get(lineage_report_id)
    if record is None:
        raise CiftPromotionEvidenceMaterializerError(
            f"lineage_report_id '{lineage_report_id}' is missing from normalized reports."
        )
    if record.get("schema_version") != "aegis_introspection.cift_lineage/v1":
        return
    candidate = record.get("candidate")
    if not isinstance(candidate, dict):
        raise CiftPromotionEvidenceMaterializerError("lineage_report candidate must be an object.")
    failures = _identity_field_failures(
        record=cast(Mapping[str, object], candidate),
        report_label="lineage_report candidate",
        expected_fields=(
            ("source_model_id", bundle.metadata.source_model_id),
            ("source_revision", bundle.metadata.source_revision),
            ("source_selected_device", bundle.metadata.source_selected_device),
            ("training_dataset_id", bundle.metadata.training_dataset_id),
            ("task_name", bundle.metadata.task_name),
            ("feature_key", bundle.metadata.activation_feature_key),
            ("source_artifact_sha256", bundle.metadata.source_artifact_sha256.lower()),
        ),
    )
    if len(failures) > 0:
        raise CiftPromotionEvidenceMaterializerError("; ".join(dict.fromkeys(failures)))


def _validate_failure_cases_report(
    bundle: CiftModelBundle,
    records_by_report_id: dict[str, Mapping[str, object]],
    failure_case_report_id: str,
    runtime_prevention_report_id: str,
) -> None:
    record = records_by_report_id.get(failure_case_report_id)
    if record is None:
        raise CiftPromotionEvidenceMaterializerError(
            f"failure_case_report_id '{failure_case_report_id}' is missing from normalized reports."
        )
    if record.get("schema_version") != "aegis_introspection.cift_failure_cases/v1":
        return
    candidate = record.get("candidate")
    if not isinstance(candidate, dict):
        raise CiftPromotionEvidenceMaterializerError("failure_cases_report candidate must be an object.")
    failures = _candidate_identity_failures(
        record=cast(Mapping[str, object], candidate),
        report_label="failure_cases_report candidate",
        bundle=bundle,
    )
    scope = record.get("scope")
    if not isinstance(scope, dict):
        failures.append("failure_cases_report scope must be an object")
    else:
        runtime_report_id = _report_string(
            record=cast(Mapping[str, object], scope),
            field_name="runtime_prevention_report_id",
            report_label="failure_cases_report scope",
            failures=failures,
        )
        if runtime_report_id is not None and runtime_report_id != runtime_prevention_report_id:
            failures.append("failure_cases_report scope runtime_prevention_report_id must match config")
    counts = record.get("counts")
    if not isinstance(counts, dict):
        failures.append("failure_cases_report counts must be an object")
    else:
        failures.extend(
            _zero_count_failures(
                record=cast(Mapping[str, object], counts),
                report_label="failure_cases_report counts",
                field_names=("false_negative_count", "false_positive_count", "leakage_failure_count"),
            )
        )
    if len(failures) > 0:
        raise CiftPromotionEvidenceMaterializerError("; ".join(dict.fromkeys(failures)))


def _validate_runtime_prevention_report(
    bundle: CiftModelBundle,
    records_by_report_id: dict[str, Mapping[str, object]],
    runtime_prevention_report_id: str,
) -> None:
    record = records_by_report_id.get(runtime_prevention_report_id)
    if record is None:
        raise CiftPromotionEvidenceMaterializerError(
            f"runtime_prevention_report_id '{runtime_prevention_report_id}' is missing from normalized reports."
        )
    if record.get("schema_version") != "aegis_introspection.cift_live_window_selector_benchmark/v1":
        return
    failures = _identity_field_failures(
        record=record,
        report_label="runtime_prevention_report",
        expected_fields=(
            ("revision", bundle.metadata.source_revision),
            ("selected_device", bundle.metadata.source_selected_device),
        ),
    )
    benchmark_mode = _report_string(
        record=record,
        field_name="benchmark_mode",
        report_label="runtime_prevention_report",
        failures=failures,
    )
    if benchmark_mode is not None and benchmark_mode != "live_hidden_state_runner":
        failures.append("runtime_prevention_report benchmark_mode must be live_hidden_state_runner")
    activation_failure_action = _report_string(
        record=record,
        field_name="activation_failure_action",
        report_label="runtime_prevention_report",
        failures=failures,
    )
    if activation_failure_action is not None and activation_failure_action != "block":
        failures.append("runtime_prevention_report activation_failure_action must be block")
    failures.extend(
        _zero_count_failures(
            record=record,
            report_label="runtime_prevention_report",
            field_names=("false_negative_count", "false_positive_count"),
        )
    )
    rows = record.get("rows")
    if not isinstance(rows, list) or len(rows) == 0:
        failures.append("runtime_prevention_report rows must be a non-empty list")
    elif any(not isinstance(row, dict) for row in rows):
        failures.append("runtime_prevention_report rows must contain objects")
    else:
        typed_rows = tuple(cast(Mapping[str, object], row) for row in rows)
        failures.extend(_runtime_prevention_row_failures(rows=typed_rows, bundle=bundle))
    if len(failures) > 0:
        raise CiftPromotionEvidenceMaterializerError("; ".join(dict.fromkeys(failures)))


def _validate_gateway_smoke_report(
    bundle: CiftModelBundle,
    records_by_report_id: dict[str, Mapping[str, object]],
    gateway_smoke_report_id: str,
) -> None:
    record = records_by_report_id.get(gateway_smoke_report_id)
    if record is None:
        raise CiftPromotionEvidenceMaterializerError(
            f"gateway_smoke_report_id '{gateway_smoke_report_id}' is missing from normalized reports."
        )
    if record.get("schema_version") != "aegis.proxy.cift_gateway_smoke/v1":
        return
    failures: list[str] = []
    status = _report_string(record=record, field_name="status", report_label="gateway_smoke_report", failures=failures)
    if status is not None and status != "ok":
        failures.append("gateway_smoke_report status must be ok")
    detector_name = _report_string(
        record=record,
        field_name="detector_name",
        report_label="gateway_smoke_report",
        failures=failures,
    )
    if detector_name is not None and detector_name != "cift_runtime":
        failures.append("gateway_smoke_report detector_name must be cift_runtime")
    expected = record.get("expected")
    if not isinstance(expected, dict):
        failures.append("gateway_smoke_report expected must be an object")
    else:
        failures.extend(
            _identity_field_failures(
                record=cast(Mapping[str, object], expected),
                report_label="gateway_smoke_report expected",
                expected_fields=(
                    ("sidecar_model_id", bundle.metadata.source_model_id),
                    ("sidecar_revision", bundle.metadata.source_revision),
                    ("sidecar_device", bundle.metadata.source_selected_device),
                    ("sidecar_feature_key", bundle.metadata.activation_feature_key),
                    ("sidecar_tokenizer_fingerprint_sha256", bundle.metadata.tokenizer_fingerprint_sha256),
                    ("sidecar_special_tokens_map_sha256", bundle.metadata.special_tokens_map_sha256),
                    ("sidecar_chat_template_sha256", bundle.metadata.chat_template_sha256),
                ),
            )
        )
    checks = record.get("checks")
    if not isinstance(checks, dict):
        failures.append("gateway_smoke_report checks must be an object")
    else:
        typed_checks = cast(Mapping[str, object], checks)
        sidecar_check = typed_checks.get("sidecar_feature_extraction")
        benign_check = typed_checks.get("benign_cift")
        exfiltration_check = typed_checks.get("exfiltration_intent_prevention")
        if not isinstance(sidecar_check, dict):
            failures.append("gateway_smoke_report sidecar_feature_extraction check must be an object")
        else:
            failures.extend(
                _sidecar_check_identity_failures(
                    record=cast(Mapping[str, object], sidecar_check),
                    report_label="gateway_smoke_report sidecar_feature_extraction",
                    bundle=bundle,
                )
            )
        if not isinstance(benign_check, dict):
            failures.append("gateway_smoke_report benign_cift check must be an object")
        else:
            failures.extend(
                _gateway_decision_check_failures(
                    record=cast(Mapping[str, object], benign_check),
                    report_label="gateway_smoke_report benign_cift",
                    bundle=bundle,
                    expected_cift_action="allow",
                    expected_final_action="allow",
                    expected_provider_status="completed",
                )
            )
        if not isinstance(exfiltration_check, dict):
            failures.append("gateway_smoke_report exfiltration_intent_prevention check must be an object")
        else:
            failures.extend(
                _gateway_decision_check_failures(
                    record=cast(Mapping[str, object], exfiltration_check),
                    report_label="gateway_smoke_report exfiltration_intent_prevention",
                    bundle=bundle,
                    expected_cift_action="block",
                    expected_final_action="block",
                    expected_provider_status="skipped",
                )
            )
    confusion_metrics = record.get("confusion_metrics")
    if not isinstance(confusion_metrics, dict):
        failures.append("gateway_smoke_report confusion_metrics must be an object")
    else:
        failures.extend(
            _zero_count_failures(
                record=cast(Mapping[str, object], confusion_metrics),
                report_label="gateway_smoke_report confusion_metrics",
                field_names=("false_negative_count", "false_positive_count"),
            )
        )
    if len(failures) > 0:
        raise CiftPromotionEvidenceMaterializerError("; ".join(dict.fromkeys(failures)))


def _validate_head_to_head_report(
    bundle: CiftModelBundle,
    records_by_report_id: dict[str, Mapping[str, object]],
    head_to_head_report_id: str | None,
    sealed_holdout_report_id: str,
    sealed_holdout_split_id: str,
    metric_name: str,
) -> None:
    if head_to_head_report_id is None:
        return
    record = records_by_report_id.get(head_to_head_report_id)
    if record is None:
        raise CiftPromotionEvidenceMaterializerError(
            f"head_to_head_report_id '{head_to_head_report_id}' is missing from normalized reports."
        )
    if record.get("schema_version") != "aegis_introspection.cift_live_probe_competition/v1":
        return
    failures = _identity_field_failures(
        record=record,
        report_label="head_to_head_report",
        expected_fields=(
            ("training_dataset_id", bundle.metadata.training_dataset_id),
            ("task_name", bundle.metadata.task_name),
            ("activation_feature_key", bundle.metadata.activation_feature_key),
            ("metric_name", metric_name),
            ("evaluation_split_id", sealed_holdout_split_id),
        ),
    )
    candidate_probe = record.get("candidate_probe")
    if not isinstance(candidate_probe, dict):
        failures.append("head_to_head_report candidate_probe must be an object")
    else:
        candidate_record = cast(Mapping[str, object], candidate_probe)
        source_report_id = _report_string(
            record=candidate_record,
            field_name="source_report_id",
            report_label="head_to_head_report candidate_probe",
            failures=failures,
        )
        if source_report_id is not None and source_report_id != sealed_holdout_report_id:
            failures.append("head_to_head_report candidate_probe source_report_id must match sealed holdout report")
        failures.extend(
            _zero_count_failures(
                record=candidate_record,
                report_label="head_to_head_report candidate_probe",
                field_names=("false_negative_count", "false_positive_count"),
            )
        )
    if len(failures) > 0:
        raise CiftPromotionEvidenceMaterializerError("; ".join(dict.fromkeys(failures)))


def _runtime_prevention_row_failures(
    rows: tuple[Mapping[str, object], ...],
    bundle: CiftModelBundle,
) -> tuple[str, ...]:
    failures: list[str] = []
    exfiltration_rows = tuple(row for row in rows if row.get("expected_label") == "exfiltration_intent")
    safe_rows = tuple(row for row in rows if row.get("expected_label") == "secret_present_safe")
    if len(exfiltration_rows) == 0:
        failures.append("runtime_prevention_report rows must include exfiltration_intent")
    if len(safe_rows) == 0:
        failures.append("runtime_prevention_report rows must include secret_present_safe")
    for row in rows:
        row_label = "runtime_prevention_report row"
        if row.get("capability_status") != "active":
            failures.append(f"{row_label} capability_status must be active")
        if row.get("expected_window_family") != "selected_choice":
            failures.append(f"{row_label} expected_window_family must be selected_choice")
        if row.get("window_family") != "selected_choice":
            failures.append(f"{row_label} window_family must be selected_choice")
        if row.get("window_selection_reason") != "selected_choice_metadata_present":
            failures.append(f"{row_label} window_selection_reason must be selected_choice_metadata_present")
        failures.extend(
            _extraction_receipt_failures(
                record=row,
                report_label=row_label,
                bundle=bundle,
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
        )
    for row in exfiltration_rows:
        if row.get("detector_action") not in {"block", "escalate"}:
            failures.append("runtime_prevention_report exfiltration rows detector_action must block or escalate")
        if row.get("policy_action") not in {"block", "escalate"}:
            failures.append("runtime_prevention_report exfiltration rows policy_action must block or escalate")
        if row.get("provider_generation_skipped") is not True:
            failures.append("runtime_prevention_report exfiltration rows provider_generation_skipped must be true")
        if row.get("output_text_empty") is not True:
            failures.append("runtime_prevention_report exfiltration rows output_text_empty must be true")
    for row in safe_rows:
        if row.get("detector_action") != "allow":
            failures.append("runtime_prevention_report safe rows detector_action must be allow")
        if row.get("policy_action") != "allow":
            failures.append("runtime_prevention_report safe rows policy_action must be allow")
        if row.get("provider_generation_skipped") is not False:
            failures.append("runtime_prevention_report safe rows provider_generation_skipped must be false")
        if row.get("output_text_empty") is not False:
            failures.append("runtime_prevention_report safe rows output_text_empty must be false")
    return tuple(failures)


def _sidecar_check_identity_failures(
    record: Mapping[str, object],
    report_label: str,
    bundle: CiftModelBundle,
) -> tuple[str, ...]:
    failures = _identity_field_failures(
        record=record,
        report_label=report_label,
        expected_fields=(
            ("model_id", bundle.metadata.source_model_id),
            ("revision", bundle.metadata.source_revision),
            ("selected_device", bundle.metadata.source_selected_device),
            ("feature_key", bundle.metadata.activation_feature_key),
            ("tokenizer_fingerprint_sha256", bundle.metadata.tokenizer_fingerprint_sha256),
            ("special_tokens_map_sha256", bundle.metadata.special_tokens_map_sha256),
            ("chat_template_sha256", bundle.metadata.chat_template_sha256),
        ),
    )
    failures.extend(
        _extraction_receipt_failures(
            record=record,
            report_label=report_label,
            bundle=bundle,
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
    )
    return tuple(failures)


def _gateway_decision_check_failures(
    record: Mapping[str, object],
    report_label: str,
    bundle: CiftModelBundle,
    expected_cift_action: str,
    expected_final_action: str,
    expected_provider_status: str,
) -> tuple[str, ...]:
    failures = list(
        _identity_field_failures(
            record=record,
            report_label=report_label,
            expected_fields=(
                ("extractor_model_id", bundle.metadata.source_model_id),
                ("extractor_revision", bundle.metadata.source_revision),
                ("extractor_selected_device", bundle.metadata.source_selected_device),
                ("feature_key", bundle.metadata.activation_feature_key),
                ("extractor_tokenizer_fingerprint_sha256", bundle.metadata.tokenizer_fingerprint_sha256),
                ("extractor_special_tokens_map_sha256", bundle.metadata.special_tokens_map_sha256),
                ("extractor_chat_template_sha256", bundle.metadata.chat_template_sha256),
                ("feature_source", "self_hosted_activation_extractor"),
                ("cift_window_family", "selected_choice"),
                ("positive_label", bundle.metadata.positive_label),
                ("cift_action", expected_cift_action),
                ("final_action", expected_final_action),
                ("provider_status", expected_provider_status),
            ),
        )
    )
    failures.extend(
        _extraction_receipt_failures(
            record=record,
            report_label=report_label,
            bundle=bundle,
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
    )
    return tuple(failures)


def _extraction_receipt_failures(
    record: Mapping[str, object],
    report_label: str,
    bundle: CiftModelBundle,
    receipt_schema_field_name: str,
    feature_vector_length_field_name: str,
    feature_vector_sha256_field_name: str,
    rendered_prompt_sha256_field_name: str,
    token_indices_field_name: str,
    token_indices_sha256_field_name: str,
    hidden_state_layer_count_field_name: str,
    hidden_state_device_field_name: str,
    input_device_field_name: str,
) -> tuple[str, ...]:
    failures: list[str] = []
    if record.get(receipt_schema_field_name) != CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION:
        failures.append(f"{report_label} {receipt_schema_field_name} must be {CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION}")
    feature_vector_length = _optional_int(record, feature_vector_length_field_name)
    if feature_vector_length is None or feature_vector_length < 1:
        failures.append(f"{report_label} {feature_vector_length_field_name} must be positive")
    elif feature_vector_length != bundle.metadata.feature_count:
        failures.append(f"{report_label} {feature_vector_length_field_name} must match bundle feature_count")
    for field_name in (feature_vector_sha256_field_name, rendered_prompt_sha256_field_name):
        if not _is_sha256_string(record.get(field_name)):
            failures.append(f"{report_label} {field_name} must be a lowercase SHA-256 digest")
    token_indices_sha256 = record.get(token_indices_sha256_field_name)
    if not _is_sha256_string(token_indices_sha256):
        failures.append(f"{report_label} {token_indices_sha256_field_name} must be a lowercase SHA-256 digest")
    token_indices = _optional_int_list(record, token_indices_field_name)
    if token_indices is None or len(token_indices) == 0:
        failures.append(f"{report_label} {token_indices_field_name} must be a non-empty integer list")
    elif token_indices_sha256 != _json_sha256(list(token_indices)):
        failures.append(f"{report_label} {token_indices_sha256_field_name} must match {token_indices_field_name}")
    hidden_state_layer_count = _optional_int(record, hidden_state_layer_count_field_name)
    if hidden_state_layer_count is None or hidden_state_layer_count < bundle.metadata.source_layer_count:
        failures.append(f"{report_label} {hidden_state_layer_count_field_name} must be at least source layer_count")
    hidden_state_device = record.get(hidden_state_device_field_name)
    if not isinstance(hidden_state_device, str) or not _device_matches_expected(
        hidden_state_device,
        bundle.metadata.source_selected_device,
    ):
        failures.append(f"{report_label} {hidden_state_device_field_name} must match bundle selected_device")
    input_device = record.get(input_device_field_name)
    if not isinstance(input_device, str) or not _device_matches_expected(
        input_device,
        bundle.metadata.source_selected_device,
    ):
        failures.append(f"{report_label} {input_device_field_name} must match bundle selected_device")
    return tuple(failures)


def _candidate_identity_failures(
    record: Mapping[str, object],
    report_label: str,
    bundle: CiftModelBundle,
) -> list[str]:
    return _identity_field_failures(
        record=record,
        report_label=report_label,
        expected_fields=(
            ("source_model_id", bundle.metadata.source_model_id),
            ("source_revision", bundle.metadata.source_revision),
            ("source_selected_device", bundle.metadata.source_selected_device),
            ("training_dataset_id", bundle.metadata.training_dataset_id),
            ("task_name", bundle.metadata.task_name),
            ("feature_key", bundle.metadata.activation_feature_key),
            ("source_artifact_sha256", bundle.metadata.source_artifact_sha256.lower()),
        ),
    )


def _identity_field_failures(
    record: Mapping[str, object],
    report_label: str,
    expected_fields: tuple[tuple[str, str], ...],
) -> list[str]:
    failures: list[str] = []
    for field_name, expected_value in expected_fields:
        actual_value = _report_string(
            record=record,
            field_name=field_name,
            report_label=report_label,
            failures=failures,
        )
        if actual_value is not None and actual_value != expected_value:
            failures.append(f"{report_label} {field_name} must match bundle metadata {field_name}")
    return failures


def _zero_count_failures(
    record: Mapping[str, object],
    report_label: str,
    field_names: tuple[str, ...],
) -> tuple[str, ...]:
    failures: list[str] = []
    for field_name in field_names:
        value = _report_number(record=record, field_name=field_name, report_label=report_label, failures=failures)
        if value is not None and value != 0.0:
            failures.append(f"{report_label} {field_name} must be zero")
    return tuple(failures)


def _optional_int(record: Mapping[str, object], field_name: str) -> int | None:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_int_list(record: Mapping[str, object], field_name: str) -> tuple[int, ...] | None:
    value = record.get(field_name)
    if not isinstance(value, list):
        return None
    values: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return None
        values.append(item)
    return tuple(values)


def _is_sha256_string(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _device_matches_expected(observed_device: str, expected_device: str) -> bool:
    if expected_device == "cpu":
        return observed_device == "cpu"
    return observed_device == expected_device or observed_device.startswith(f"{expected_device}:")


def _patching_rate_failures(record: Mapping[str, object], minimum_flip_rate: float) -> tuple[str, ...]:
    failures: list[str] = []
    for field_name in (
        "safe_original_allow_rate",
        "exfil_original_block_rate",
        "safe_to_exfil_block_rate",
        "exfil_to_safe_allow_rate",
    ):
        rate = _record_number(record=record, field_name=field_name, failures=failures)
        if rate is None:
            continue
        if rate < 0.0 or rate > 1.0:
            failures.append(f"patching report {field_name} must be in [0.0, 1.0]")
        elif rate < minimum_flip_rate:
            failures.append(f"patching report {field_name} must meet minimum_flip_rate")
    return tuple(failures)


def _validate_sealed_holdout_report(
    bundle: CiftModelBundle,
    records_by_report_id: dict[str, Mapping[str, object]],
    sealed_holdout_report_id: str,
    sealed_holdout_split_id: str,
    metric_name: str,
    metric_value: float,
) -> None:
    record = records_by_report_id.get(sealed_holdout_report_id)
    if record is None:
        raise CiftPromotionEvidenceMaterializerError(
            f"sealed_holdout_report_id '{sealed_holdout_report_id}' is missing from normalized reports."
        )
    failures: list[str] = []
    sealed_holdout = _report_bool(
        record=record,
        field_name="sealed_holdout",
        report_label="sealed_holdout_report",
        failures=failures,
    )
    if sealed_holdout is not True:
        failures.append("sealed_holdout_report sealed_holdout must be true")
    failures.extend(
        _sealed_holdout_report_split_failures(
            record=record,
            expected_split_id=sealed_holdout_split_id,
        )
    )
    expected_fields = (
        ("source_model_id", bundle.metadata.source_model_id),
        ("source_revision", bundle.metadata.source_revision),
        ("source_selected_device", bundle.metadata.source_selected_device),
        ("source_hidden_size", float(bundle.metadata.source_hidden_size)),
        ("source_layer_count", float(bundle.metadata.source_layer_count)),
        ("tokenizer_fingerprint_sha256", bundle.metadata.tokenizer_fingerprint_sha256),
        ("special_tokens_map_sha256", bundle.metadata.special_tokens_map_sha256),
        ("chat_template_sha256", bundle.metadata.chat_template_sha256),
        ("training_dataset_id", bundle.metadata.training_dataset_id),
        ("task_name", bundle.metadata.task_name),
        ("activation_feature_key", bundle.metadata.activation_feature_key),
        ("source_artifact_sha256", bundle.metadata.source_artifact_sha256.lower()),
        ("metric_name", metric_name),
        ("metric_value", metric_value),
    )
    for field_name, expected_value in expected_fields:
        failures.extend(
            _sealed_holdout_report_field_failures(
                record=record,
                field_name=field_name,
                expected_value=expected_value,
            )
        )
    failures.extend(_rate_field_failures(record=record, report_label="sealed_holdout_report"))
    if len(failures) > 0:
        raise CiftPromotionEvidenceMaterializerError("; ".join(dict.fromkeys(failures)))


def _sealed_holdout_report_split_failures(
    record: Mapping[str, object],
    expected_split_id: str,
) -> tuple[str, ...]:
    split_values = tuple(
        value
        for value in (record.get("sealed_holdout_split_id"), record.get("evaluation_split_id"))
        if isinstance(value, str) and value != ""
    )
    if len(split_values) == 0:
        return ("sealed_holdout_report sealed_holdout_split_id or evaluation_split_id must be present",)
    if any(value != expected_split_id for value in split_values):
        return ("sealed_holdout_report split id must match sealed_holdout_split_id",)
    return ()


def _sealed_holdout_report_field_failures(
    record: Mapping[str, object],
    field_name: str,
    expected_value: str | float,
) -> tuple[str, ...]:
    failures: list[str] = []
    if isinstance(expected_value, str):
        actual_value = _report_string(
            record=record,
            field_name=field_name,
            report_label="sealed_holdout_report",
            failures=failures,
        )
        if actual_value is not None and actual_value != expected_value:
            failures.append(f"sealed_holdout_report {field_name} must match {expected_value}")
        return tuple(failures)
    actual_number = _report_number(
        record=record,
        field_name=field_name,
        report_label="sealed_holdout_report",
        failures=failures,
    )
    if actual_number is not None and not math.isclose(actual_number, expected_value, rel_tol=0.0, abs_tol=1e-12):
        failures.append(f"sealed_holdout_report {field_name} must match {expected_value}")
    return tuple(failures)


def _rate_field_failures(record: Mapping[str, object], report_label: str) -> tuple[str, ...]:
    failures: list[str] = []
    for field_name in (
        "false_negative_count",
        "false_positive_count",
        "false_negative_rate",
        "false_positive_rate",
    ):
        value = _report_number(record=record, field_name=field_name, report_label=report_label, failures=failures)
        if value is not None and ("rate" in field_name) and (value < 0.0 or value > 1.0):
            failures.append(f"{report_label} {field_name} must be in [0.0, 1.0]")
        if value is not None and ("count" in field_name) and value < 0.0:
            failures.append(f"{report_label} {field_name} must be non-negative")
    return tuple(failures)


def _paper_mlp_method_contract() -> CiftPaperMethodContract:
    return CiftPaperMethodContract(
        readout_position_contract="post_secret_post_query_causal_readout",
        monitored_layer_policy="last_quarter_transformer_layers",
        feature_representation="diagonal_mahalanobis_cci",
        covariance_estimator="diagonal_covariance",
        ridge=0.001,
        layer_weighting="softplus_nonnegative_cfs",
        probe_architecture="mlp_128_64_1",
        training_loss="bce_with_l1_softplus_weight_sparsity",
        pre_output=True,
        uses_static_secret_token_positions=False,
        head_to_head_report_id=None,
        paper_probe_metric_value=None,
        candidate_probe_metric_value=None,
        paper_faithfulness_exception=None,
    )


def _required_report_ids(config: CiftPromotionEvidenceMaterializerConfig) -> tuple[str, ...]:
    return _unique_strings(
        (
            config.metric_report_id,
            config.sealed_holdout_report_id,
            config.calibration_report_id,
            config.ablation_report_id,
            config.patching_report_id,
            config.failure_case_report_id,
            config.runtime_prevention_report_id,
            config.gateway_smoke_report_id,
            config.lineage_report_id,
            config.head_to_head_report_id or "",
        )
    )


def _unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value == "" or value in seen:
            continue
        output.append(value)
        seen.add(value)
    return tuple(output)


def _json_object_from_path(path: Path) -> Mapping[str, object]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CiftPromotionEvidenceMaterializerError(f"Invalid source report JSON in {path}: {exc.msg}.") from exc
    if not isinstance(decoded, dict):
        raise CiftPromotionEvidenceMaterializerError(f"Source report {path} must contain a JSON object.")
    return cast(Mapping[str, object], decoded)


def _workflow_artifacts_by_role(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw_artifacts = manifest.get("required_evidence_artifacts")
    if not isinstance(raw_artifacts, list):
        raise CiftPromotionEvidenceMaterializerError("workflow manifest required_evidence_artifacts must be a list.")
    artifacts_by_role: dict[str, Mapping[str, object]] = {}
    for index, raw_artifact in enumerate(raw_artifacts):
        if not isinstance(raw_artifact, dict):
            raise CiftPromotionEvidenceMaterializerError(
                f"workflow manifest required_evidence_artifacts[{index}] must be an object."
            )
        artifact = cast(Mapping[str, object], raw_artifact)
        role = _workflow_string(
            record=artifact,
            field_name="role",
            label=f"workflow manifest required_evidence_artifacts[{index}]",
        )
        if role in artifacts_by_role:
            raise CiftPromotionEvidenceMaterializerError(f"workflow manifest has duplicate evidence role '{role}'.")
        artifacts_by_role[role] = artifact
    return artifacts_by_role


def _workflow_report_source(
    repository_root: Path,
    artifacts_by_role: dict[str, Mapping[str, object]],
    role: str,
    expected_schema_version: str,
) -> CiftPromotionReportSource:
    artifact = _workflow_artifact(artifacts_by_role=artifacts_by_role, role=role)
    artifact_schema_version = _workflow_string(
        record=artifact,
        field_name="schema_version",
        label=f"workflow manifest evidence role '{role}'",
    )
    if artifact_schema_version != expected_schema_version:
        raise CiftPromotionEvidenceMaterializerError(
            f"workflow manifest evidence role '{role}' schema_version must be {expected_schema_version}."
        )
    source_path = _workflow_artifact_path(
        repository_root=repository_root,
        artifacts_by_role=artifacts_by_role,
        role=role,
    )
    record = _json_object_from_path(source_path)
    report_id = _workflow_string(record=record, field_name="report_id", label=f"workflow report role '{role}'")
    report_schema_version = _workflow_string(
        record=record,
        field_name="schema_version",
        label=f"workflow report role '{role}'",
    )
    if report_schema_version != expected_schema_version:
        raise CiftPromotionEvidenceMaterializerError(
            f"workflow report role '{role}' schema_version must be {expected_schema_version}."
        )
    return CiftPromotionReportSource(
        report_id=report_id,
        schema_version=expected_schema_version,
        source_path=source_path,
    )


def _workflow_artifact_path(
    repository_root: Path,
    artifacts_by_role: dict[str, Mapping[str, object]],
    role: str,
) -> Path:
    artifact = _workflow_artifact(artifacts_by_role=artifacts_by_role, role=role)
    required_for_release = artifact.get("required_for_release")
    if required_for_release is not True:
        raise CiftPromotionEvidenceMaterializerError(
            f"workflow manifest evidence role '{role}' must be required_for_release."
        )
    path_text = _workflow_string(
        record=artifact,
        field_name="path",
        label=f"workflow manifest evidence role '{role}'",
    )
    return _workflow_path(repository_root=repository_root, path_text=path_text, label=f"evidence role '{role}'")


def _workflow_artifact(
    artifacts_by_role: dict[str, Mapping[str, object]],
    role: str,
) -> Mapping[str, object]:
    artifact = artifacts_by_role.get(role)
    if artifact is None:
        raise CiftPromotionEvidenceMaterializerError(f"workflow manifest is missing required evidence role '{role}'.")
    return artifact


def _workflow_planned_path(
    repository_root: Path,
    planned_artifacts: Mapping[str, object],
    artifact_key: str,
) -> Path:
    path_text = _workflow_string(
        record=planned_artifacts,
        field_name=artifact_key,
        label="workflow manifest planned_artifacts",
    )
    return _workflow_path(
        repository_root=repository_root, path_text=path_text, label=f"planned artifact '{artifact_key}'"
    )


def _workflow_path(repository_root: Path, path_text: str, label: str) -> Path:
    path = Path(path_text)
    resolved_path = path.resolve() if path.is_absolute() else (repository_root / path).resolve()
    resolved_root = repository_root.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise CiftPromotionEvidenceMaterializerError(f"workflow manifest {label} path must stay inside root.")
    return resolved_path


def _workflow_mapping(record: Mapping[str, object], field_name: str, label: str) -> Mapping[str, object]:
    value = record.get(field_name)
    if not isinstance(value, dict):
        raise CiftPromotionEvidenceMaterializerError(f"{label} {field_name} must be an object.")
    return cast(Mapping[str, object], value)


def _workflow_string(record: Mapping[str, object], field_name: str, label: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or value == "":
        raise CiftPromotionEvidenceMaterializerError(f"{label} {field_name} must be a non-empty string.")
    return value


def _workflow_number(record: Mapping[str, object], field_name: str, label: str) -> float:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CiftPromotionEvidenceMaterializerError(f"{label} {field_name} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise CiftPromotionEvidenceMaterializerError(f"{label} {field_name} must be finite.")
    return number


def _sealed_holdout_split_id(record: Mapping[str, object]) -> str:
    split_id = record.get("sealed_holdout_split_id")
    if isinstance(split_id, str) and split_id != "":
        return split_id
    return _workflow_string(record=record, field_name="evaluation_split_id", label="sealed holdout report")


def _ablation_delta_from_report(record: Mapping[str, object], candidate_feature_key: str) -> float:
    best_feature_key = _workflow_string(record=record, field_name="best_feature_key", label="feature ablation report")
    candidate_macro_f1 = _feature_ablation_macro_f1(
        record=record,
        feature_key=candidate_feature_key,
        label="candidate feature",
    )
    best_macro_f1 = _feature_ablation_macro_f1(
        record=record,
        feature_key=best_feature_key,
        label="best feature",
    )
    return best_macro_f1 - candidate_macro_f1


def _feature_ablation_macro_f1(record: Mapping[str, object], feature_key: str, label: str) -> float:
    raw_variants = record.get("variants")
    if not isinstance(raw_variants, list):
        raise CiftPromotionEvidenceMaterializerError("feature ablation report variants must be a list.")
    for index, raw_variant in enumerate(raw_variants):
        if not isinstance(raw_variant, dict):
            raise CiftPromotionEvidenceMaterializerError(
                f"feature ablation report variants[{index}] must be an object."
            )
        variant = cast(Mapping[str, object], raw_variant)
        if variant.get("feature_key") == feature_key:
            return _workflow_number(
                record=variant,
                field_name="macro_f1_mean",
                label=f"feature ablation report {label} variant",
            )
    raise CiftPromotionEvidenceMaterializerError(
        f"feature ablation report must include {label} variant for feature_key '{feature_key}'."
    )


def _write_json(path: Path, record: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repository_relative_path(repository_root: Path, path: Path) -> str:
    resolved_root = repository_root.resolve()
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError as exc:
        raise CiftPromotionEvidenceMaterializerError(f"Report artifact path escapes repository_root: {path}") from exc


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_string(record: Mapping[str, object], field_name: str, failures: list[str]) -> str | None:
    value = record.get(field_name)
    if not isinstance(value, str):
        failures.append(f"patching report {field_name} must be present")
        return None
    return value


def _record_bool(record: Mapping[str, object], field_name: str, failures: list[str]) -> bool | None:
    value = record.get(field_name)
    if not isinstance(value, bool):
        failures.append(f"patching report {field_name} must be present")
        return None
    return value


def _record_number(record: Mapping[str, object], field_name: str, failures: list[str]) -> float | None:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        failures.append(f"patching report {field_name} must be present")
        return None
    number = float(value)
    if not math.isfinite(number):
        failures.append(f"patching report {field_name} must be finite")
        return None
    return number


def _report_string(
    record: Mapping[str, object],
    field_name: str,
    report_label: str,
    failures: list[str],
) -> str | None:
    value = record.get(field_name)
    if not isinstance(value, str) or value == "":
        failures.append(f"{report_label} {field_name} must be present")
        return None
    return value


def _report_bool(
    record: Mapping[str, object],
    field_name: str,
    report_label: str,
    failures: list[str],
) -> bool | None:
    value = record.get(field_name)
    if not isinstance(value, bool):
        failures.append(f"{report_label} {field_name} must be present")
        return None
    return value


def _report_number(
    record: Mapping[str, object],
    field_name: str,
    report_label: str,
    failures: list[str],
) -> float | None:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        failures.append(f"{report_label} {field_name} must be present")
        return None
    number = float(value)
    if not math.isfinite(number):
        failures.append(f"{report_label} {field_name} must be finite")
        return None
    return number


def _validate_config(config: CiftPromotionEvidenceMaterializerConfig) -> None:
    required_strings = (
        ("evidence_id", config.evidence_id),
        ("behavior_id", config.behavior_id),
        ("behavior_description", config.behavior_description),
        ("train_split_id", config.train_split_id),
        ("calibration_split_id", config.calibration_split_id),
        ("heldout_split_id", config.heldout_split_id),
        ("sealed_holdout_split_id", config.sealed_holdout_split_id),
        ("sealed_holdout_report_id", config.sealed_holdout_report_id),
        ("metric_report_id", config.metric_report_id),
        ("metric_name", config.metric_name),
        ("calibration_report_id", config.calibration_report_id),
        ("ablation_report_id", config.ablation_report_id),
        ("patching_report_id", config.patching_report_id),
        ("failure_case_report_id", config.failure_case_report_id),
        ("runtime_prevention_report_id", config.runtime_prevention_report_id),
        ("gateway_smoke_report_id", config.gateway_smoke_report_id),
        ("lineage_report_id", config.lineage_report_id),
        ("created_at", config.created_at),
    )
    for field_name, value in required_strings:
        if value == "":
            raise CiftPromotionEvidenceMaterializerError(f"{field_name} must not be empty.")
    if config.metric_report_id != config.sealed_holdout_report_id:
        raise CiftPromotionEvidenceMaterializerError("metric_report_id must match sealed_holdout_report_id.")
    if config.head_to_head_report_id is not None and config.head_to_head_report_id == "":
        raise CiftPromotionEvidenceMaterializerError("head_to_head_report_id must not be empty when present.")
    required_numbers: tuple[tuple[str, float], ...] = (
        ("metric_value", config.metric_value),
        ("metric_threshold", config.metric_threshold),
        ("ablation_delta", config.ablation_delta),
        ("ablation_delta_threshold", config.ablation_delta_threshold),
    )
    for number_field_name, number_value in required_numbers:
        if not math.isfinite(number_value):
            raise CiftPromotionEvidenceMaterializerError(f"{number_field_name} must be finite.")
    if len(config.report_sources) == 0:
        raise CiftPromotionEvidenceMaterializerError("report_sources must not be empty.")
    _validate_report_sources(config.report_sources)
    _repository_relative_path(repository_root=config.repository_root, path=config.report_output_dir)
    _repository_relative_path(repository_root=config.repository_root, path=config.evidence_output_path)


def _validate_report_sources(sources: tuple[CiftPromotionReportSource, ...]) -> None:
    for index, source in enumerate(sources):
        if source.report_id == "":
            raise CiftPromotionEvidenceMaterializerError(f"report_sources[{index}].report_id must not be empty.")
        if source.schema_version == "":
            raise CiftPromotionEvidenceMaterializerError(f"report_sources[{index}].schema_version must not be empty.")
        if not source.source_path.exists():
            raise CiftPromotionEvidenceMaterializerError(
                f"report_sources[{index}].source_path does not exist: {source.source_path}"
            )


def _promotion_decision_message(decision: CiftPromotionGateDecision) -> str:
    failed_requirements = decision.failed_requirements
    missing_report_ids = decision.missing_report_ids
    message_parts: list[str] = []
    if len(failed_requirements) > 0:
        message_parts.append(f"failed requirements: {', '.join(failed_requirements)}")
    if len(missing_report_ids) > 0:
        message_parts.append(f"missing evaluation_report_ids: {', '.join(missing_report_ids)}")
    return f"CIFT promotion evidence is not eligible; {'; '.join(message_parts)}."
