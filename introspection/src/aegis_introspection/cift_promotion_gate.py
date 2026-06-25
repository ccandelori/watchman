from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

from aegis_introspection.cift_model_bundle import CandidateStatus, CiftModelBundle, validate_cift_model_bundle
from aegis_introspection.cift_paper_mlp import CiftPaperMlpClassifier

_PROMOTION_EVIDENCE_SCHEMA_VERSION = "cift_promotion_evidence/v1"
_PROMOTION_GATE_RESULT_SCHEMA_VERSION = "cift_promotion_gate_result/v1"
_PROMOTION_GATES_SCHEMA_VERSION = "cift_promotion_gates/v1"
_PAPER_MLP_PROBE_ARCHITECTURE = "mlp_128_64_1"

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class CiftPromotionGateError(ValueError):
    """Raised when a CIFT bundle is not eligible for runtime promotion."""


@dataclass(frozen=True)
class CiftPaperMethodContract:
    readout_position_contract: str
    monitored_layer_policy: str
    feature_representation: str
    covariance_estimator: str
    ridge: float
    layer_weighting: str
    probe_architecture: str
    training_loss: str
    pre_output: bool
    uses_static_secret_token_positions: bool
    head_to_head_report_id: str | None
    paper_probe_metric_value: float | None
    candidate_probe_metric_value: float | None
    paper_faithfulness_exception: str | None


@dataclass(frozen=True)
class CiftPromotionEvidence:
    schema_version: str
    evidence_id: str
    behavior_id: str
    behavior_description: str
    training_dataset_id: str
    train_split_id: str
    calibration_split_id: str
    heldout_split_id: str
    sealed_holdout_split_id: str | None
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
    report_artifacts: tuple[CiftPromotionReportArtifact, ...]
    paper_method: CiftPaperMethodContract
    created_at: str


@dataclass(frozen=True)
class CiftPromotionReportArtifact:
    report_id: str
    path: str
    sha256: str
    schema_version: str


@dataclass(frozen=True)
class CiftPromotionGateDecision:
    schema_version: str
    candidate_status: CandidateStatus
    eligible: bool
    failed_requirements: tuple[str, ...]
    required_report_ids: tuple[str, ...]
    missing_report_ids: tuple[str, ...]


def evaluate_cift_promotion_gate(
    bundle: CiftModelBundle,
    evidence: CiftPromotionEvidence,
) -> CiftPromotionGateDecision:
    validate_cift_model_bundle(bundle)
    failed_requirements = _failed_requirements(bundle=bundle, evidence=evidence)
    required_report_ids = _required_report_ids(evidence)
    available_report_ids = set(bundle.metadata.evaluation_report_ids)
    missing_report_ids = tuple(report_id for report_id in required_report_ids if report_id not in available_report_ids)
    return CiftPromotionGateDecision(
        schema_version=_PROMOTION_GATE_RESULT_SCHEMA_VERSION,
        candidate_status=bundle.metadata.candidate_status,
        eligible=len(failed_requirements) == 0 and len(missing_report_ids) == 0,
        failed_requirements=failed_requirements,
        required_report_ids=required_report_ids,
        missing_report_ids=missing_report_ids,
    )


def assert_cift_runtime_promotion_eligible(
    bundle: CiftModelBundle,
    evidence: CiftPromotionEvidence,
) -> None:
    decision = evaluate_cift_promotion_gate(bundle=bundle, evidence=evidence)
    if decision.eligible:
        return
    message_parts: list[str] = []
    if len(decision.failed_requirements) > 0:
        message_parts.append(f"failed requirements: {', '.join(decision.failed_requirements)}")
    if len(decision.missing_report_ids) > 0:
        message_parts.append(f"missing evaluation_report_ids: {', '.join(decision.missing_report_ids)}")
    raise CiftPromotionGateError(f"CIFT runtime_candidate promotion is blocked; {'; '.join(message_parts)}.")


def cift_promotion_evidence_to_json(evidence: CiftPromotionEvidence) -> dict[str, JsonValue]:
    return {
        "schema_version": evidence.schema_version,
        "evidence_id": evidence.evidence_id,
        "behavior_id": evidence.behavior_id,
        "behavior_description": evidence.behavior_description,
        "training_dataset_id": evidence.training_dataset_id,
        "train_split_id": evidence.train_split_id,
        "calibration_split_id": evidence.calibration_split_id,
        "heldout_split_id": evidence.heldout_split_id,
        "sealed_holdout_split_id": evidence.sealed_holdout_split_id,
        "sealed_holdout_report_id": evidence.sealed_holdout_report_id,
        "metric_report_id": evidence.metric_report_id,
        "metric_name": evidence.metric_name,
        "metric_value": evidence.metric_value,
        "metric_threshold": evidence.metric_threshold,
        "calibration_report_id": evidence.calibration_report_id,
        "ablation_report_id": evidence.ablation_report_id,
        "ablation_delta": evidence.ablation_delta,
        "ablation_delta_threshold": evidence.ablation_delta_threshold,
        "patching_report_id": evidence.patching_report_id,
        "failure_case_report_id": evidence.failure_case_report_id,
        "runtime_prevention_report_id": evidence.runtime_prevention_report_id,
        "gateway_smoke_report_id": evidence.gateway_smoke_report_id,
        "lineage_report_id": evidence.lineage_report_id,
        "report_artifacts": [
            cift_promotion_report_artifact_to_json(artifact) for artifact in evidence.report_artifacts
        ],
        "paper_method": cift_paper_method_contract_to_json(evidence.paper_method),
        "created_at": evidence.created_at,
    }


def cift_paper_method_contract_to_json(method: CiftPaperMethodContract) -> dict[str, JsonValue]:
    return {
        "readout_position_contract": method.readout_position_contract,
        "monitored_layer_policy": method.monitored_layer_policy,
        "feature_representation": method.feature_representation,
        "covariance_estimator": method.covariance_estimator,
        "ridge": method.ridge,
        "layer_weighting": method.layer_weighting,
        "probe_architecture": method.probe_architecture,
        "training_loss": method.training_loss,
        "pre_output": method.pre_output,
        "uses_static_secret_token_positions": method.uses_static_secret_token_positions,
        "head_to_head_report_id": method.head_to_head_report_id,
        "paper_probe_metric_value": method.paper_probe_metric_value,
        "candidate_probe_metric_value": method.candidate_probe_metric_value,
        "paper_faithfulness_exception": method.paper_faithfulness_exception,
    }


def cift_promotion_report_artifact_to_json(artifact: CiftPromotionReportArtifact) -> dict[str, JsonValue]:
    return {
        "report_id": artifact.report_id,
        "path": artifact.path,
        "sha256": artifact.sha256,
        "schema_version": artifact.schema_version,
    }


def cift_promotion_evidence_from_mapping(record: Mapping[str, object]) -> CiftPromotionEvidence:
    return CiftPromotionEvidence(
        schema_version=_required_string(record=record, field_name="schema_version"),
        evidence_id=_required_string(record=record, field_name="evidence_id"),
        behavior_id=_required_string(record=record, field_name="behavior_id"),
        behavior_description=_required_string(record=record, field_name="behavior_description"),
        training_dataset_id=_required_string(record=record, field_name="training_dataset_id"),
        train_split_id=_required_string(record=record, field_name="train_split_id"),
        calibration_split_id=_required_string(record=record, field_name="calibration_split_id"),
        heldout_split_id=_required_string(record=record, field_name="heldout_split_id"),
        sealed_holdout_split_id=_optional_string(record=record, field_name="sealed_holdout_split_id"),
        sealed_holdout_report_id=_required_string(record=record, field_name="sealed_holdout_report_id"),
        metric_report_id=_required_string(record=record, field_name="metric_report_id"),
        metric_name=_required_string(record=record, field_name="metric_name"),
        metric_value=_required_float(record=record, field_name="metric_value"),
        metric_threshold=_required_float(record=record, field_name="metric_threshold"),
        calibration_report_id=_required_string(record=record, field_name="calibration_report_id"),
        ablation_report_id=_required_string(record=record, field_name="ablation_report_id"),
        ablation_delta=_required_float(record=record, field_name="ablation_delta"),
        ablation_delta_threshold=_required_float(record=record, field_name="ablation_delta_threshold"),
        patching_report_id=_required_string(record=record, field_name="patching_report_id"),
        failure_case_report_id=_required_string(record=record, field_name="failure_case_report_id"),
        runtime_prevention_report_id=_required_string(record=record, field_name="runtime_prevention_report_id"),
        gateway_smoke_report_id=_required_string(record=record, field_name="gateway_smoke_report_id"),
        lineage_report_id=_required_string(record=record, field_name="lineage_report_id"),
        report_artifacts=_report_artifacts_from_value(value=record.get("report_artifacts")),
        paper_method=cift_paper_method_contract_from_mapping(
            _required_mapping(value=record.get("paper_method"), field_name="paper_method")
        ),
        created_at=_required_string(record=record, field_name="created_at"),
    )


def cift_paper_method_contract_from_mapping(record: Mapping[str, object]) -> CiftPaperMethodContract:
    return CiftPaperMethodContract(
        readout_position_contract=_required_string(record=record, field_name="readout_position_contract"),
        monitored_layer_policy=_required_string(record=record, field_name="monitored_layer_policy"),
        feature_representation=_required_string(record=record, field_name="feature_representation"),
        covariance_estimator=_required_string(record=record, field_name="covariance_estimator"),
        ridge=_required_float(record=record, field_name="ridge"),
        layer_weighting=_required_string(record=record, field_name="layer_weighting"),
        probe_architecture=_required_string(record=record, field_name="probe_architecture"),
        training_loss=_required_string(record=record, field_name="training_loss"),
        pre_output=_required_bool(record=record, field_name="pre_output"),
        uses_static_secret_token_positions=_required_bool(
            record=record,
            field_name="uses_static_secret_token_positions",
        ),
        head_to_head_report_id=_optional_string(record=record, field_name="head_to_head_report_id"),
        paper_probe_metric_value=_optional_float(record=record, field_name="paper_probe_metric_value"),
        candidate_probe_metric_value=_optional_float(record=record, field_name="candidate_probe_metric_value"),
        paper_faithfulness_exception=_optional_string(record=record, field_name="paper_faithfulness_exception"),
    )


def load_cift_promotion_evidence(path: Path) -> CiftPromotionEvidence:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CiftPromotionGateError(f"Invalid CIFT promotion evidence JSON in {path}: {exc.msg}.") from exc
    return cift_promotion_evidence_from_mapping(_required_mapping(value=decoded, field_name=str(path)))


def cift_promotion_gate_result_to_json(
    evidence: CiftPromotionEvidence,
    decision: CiftPromotionGateDecision,
) -> dict[str, JsonValue]:
    return {
        "schema_version": decision.schema_version,
        "evidence_id": evidence.evidence_id,
        "candidate_status": decision.candidate_status,
        "eligible": decision.eligible,
        "eligibility_scope": "runtime_candidate_promotion_only",
        "production_release_eligible": False,
        "requires_certification_binding": True,
        "behavior_id": evidence.behavior_id,
        "behavior_description": evidence.behavior_description,
        "training_dataset_id": evidence.training_dataset_id,
        "splits": {
            "train": evidence.train_split_id,
            "calibration": evidence.calibration_split_id,
            "heldout": evidence.heldout_split_id,
            "sealed_holdout": evidence.sealed_holdout_split_id,
        },
        "metric": {
            "report_id": evidence.metric_report_id,
            "name": evidence.metric_name,
            "value": evidence.metric_value,
            "threshold": evidence.metric_threshold,
        },
        "ablation": {
            "report_id": evidence.ablation_report_id,
            "delta": evidence.ablation_delta,
            "delta_threshold": evidence.ablation_delta_threshold,
        },
        "reports": {
            "sealed_holdout": evidence.sealed_holdout_report_id,
            "metric": evidence.metric_report_id,
            "calibration": evidence.calibration_report_id,
            "ablation": evidence.ablation_report_id,
            "patching": evidence.patching_report_id,
            "failure_cases": evidence.failure_case_report_id,
            "runtime_prevention": evidence.runtime_prevention_report_id,
            "gateway_smoke": evidence.gateway_smoke_report_id,
            "lineage": evidence.lineage_report_id,
            "head_to_head": evidence.paper_method.head_to_head_report_id,
        },
        "paper_method": cift_paper_method_contract_to_json(evidence.paper_method),
        "report_artifacts": [
            cift_promotion_report_artifact_to_json(artifact) for artifact in evidence.report_artifacts
        ],
        "required_report_ids": list(decision.required_report_ids),
        "missing_report_ids": list(decision.missing_report_ids),
        "failed_requirements": list(decision.failed_requirements),
        "created_at": evidence.created_at,
    }


def cift_promotion_gates_to_json(
    evidence: CiftPromotionEvidence,
    decision: CiftPromotionGateDecision,
) -> dict[str, JsonValue]:
    return {
        "schema_version": _PROMOTION_GATES_SCHEMA_VERSION,
        "runtime_candidate": cift_promotion_gate_result_to_json(evidence=evidence, decision=decision),
    }


def _failed_requirements(bundle: CiftModelBundle, evidence: CiftPromotionEvidence) -> tuple[str, ...]:
    failures: list[str] = []
    if bundle.metadata.candidate_status != "runtime_candidate":
        failures.append("candidate_status must be runtime_candidate")
    if evidence.schema_version != _PROMOTION_EVIDENCE_SCHEMA_VERSION:
        failures.append(f"schema_version must be {_PROMOTION_EVIDENCE_SCHEMA_VERSION}")
    failures.extend(_empty_string_failures(evidence))
    if evidence.training_dataset_id != "" and evidence.training_dataset_id != bundle.metadata.training_dataset_id:
        failures.append("training_dataset_id must match bundle metadata")
    failures.extend(_paper_method_failures(evidence.paper_method))
    failures.extend(_classifier_probe_failures(bundle=bundle, evidence=evidence))
    failures.extend(_split_failures(evidence))
    failures.extend(_metric_failures(evidence))
    failures.extend(_report_artifact_failures(evidence))
    return tuple(failures)


def _empty_string_failures(evidence: CiftPromotionEvidence) -> tuple[str, ...]:
    field_values = (
        ("evidence_id", evidence.evidence_id),
        ("behavior_id", evidence.behavior_id),
        ("behavior_description", evidence.behavior_description),
        ("training_dataset_id", evidence.training_dataset_id),
        ("train_split_id", evidence.train_split_id),
        ("calibration_split_id", evidence.calibration_split_id),
        ("heldout_split_id", evidence.heldout_split_id),
        ("sealed_holdout_report_id", evidence.sealed_holdout_report_id),
        ("metric_report_id", evidence.metric_report_id),
        ("metric_name", evidence.metric_name),
        ("calibration_report_id", evidence.calibration_report_id),
        ("ablation_report_id", evidence.ablation_report_id),
        ("patching_report_id", evidence.patching_report_id),
        ("failure_case_report_id", evidence.failure_case_report_id),
        ("runtime_prevention_report_id", evidence.runtime_prevention_report_id),
        ("gateway_smoke_report_id", evidence.gateway_smoke_report_id),
        ("lineage_report_id", evidence.lineage_report_id),
        ("created_at", evidence.created_at),
    )
    failures = tuple(f"{field_name} must not be empty" for field_name, value in field_values if value == "")
    if evidence.sealed_holdout_split_id is None or evidence.sealed_holdout_split_id == "":
        return (*failures, "sealed_holdout_split_id must not be empty")
    return failures


def _split_failures(evidence: CiftPromotionEvidence) -> tuple[str, ...]:
    if evidence.sealed_holdout_split_id is None:
        return ()
    split_ids = (
        evidence.train_split_id,
        evidence.calibration_split_id,
        evidence.heldout_split_id,
        evidence.sealed_holdout_split_id,
    )
    if "" not in split_ids and len(set(split_ids)) != len(split_ids):
        return ("promotion split ids must be distinct",)
    return ()


def _metric_failures(evidence: CiftPromotionEvidence) -> tuple[str, ...]:
    failures: list[str] = []
    if not math.isfinite(evidence.metric_value):
        failures.append("metric_value must be finite")
    if not math.isfinite(evidence.metric_threshold):
        failures.append("metric_threshold must be finite")
    if (
        math.isfinite(evidence.metric_value)
        and math.isfinite(evidence.metric_threshold)
        and evidence.metric_value < evidence.metric_threshold
    ):
        failures.append("metric_value must meet or exceed metric_threshold")
    if not math.isfinite(evidence.ablation_delta):
        failures.append("ablation_delta must be finite")
    if not math.isfinite(evidence.ablation_delta_threshold):
        failures.append("ablation_delta_threshold must be finite")
    if (
        math.isfinite(evidence.ablation_delta)
        and math.isfinite(evidence.ablation_delta_threshold)
        and evidence.ablation_delta < evidence.ablation_delta_threshold
    ):
        failures.append("ablation_delta must meet or exceed ablation_delta_threshold")
    return tuple(failures)


def _required_report_ids(evidence: CiftPromotionEvidence) -> tuple[str, ...]:
    return _unique_nonempty_strings(
        (
            evidence.metric_report_id,
            evidence.sealed_holdout_report_id,
            evidence.calibration_report_id,
            evidence.ablation_report_id,
            evidence.patching_report_id,
            evidence.failure_case_report_id,
            evidence.runtime_prevention_report_id,
            evidence.gateway_smoke_report_id,
            evidence.lineage_report_id,
            evidence.paper_method.head_to_head_report_id or "",
        )
    )


def _report_artifact_failures(evidence: CiftPromotionEvidence) -> tuple[str, ...]:
    failures: list[str] = []
    required_report_ids = set(_required_report_ids(evidence))
    artifact_report_ids = tuple(artifact.report_id for artifact in evidence.report_artifacts)
    if len(evidence.report_artifacts) == 0:
        failures.append("report_artifacts must cover required_report_ids")
    if len(set(artifact_report_ids)) != len(artifact_report_ids):
        failures.append("report_artifacts.report_id values must be unique")
    if set(artifact_report_ids) != required_report_ids:
        failures.append("report_artifacts must cover required_report_ids")
    for index, artifact in enumerate(evidence.report_artifacts):
        failures.extend(_single_report_artifact_failures(index=index, artifact=artifact))
    return tuple(dict.fromkeys(failures))


def _single_report_artifact_failures(index: int, artifact: CiftPromotionReportArtifact) -> tuple[str, ...]:
    failures: list[str] = []
    field_values = (
        ("report_id", artifact.report_id),
        ("path", artifact.path),
        ("schema_version", artifact.schema_version),
    )
    failures.extend(
        f"report_artifacts[{index}].{field_name} must not be empty" for field_name, value in field_values if value == ""
    )
    if not _is_lowercase_sha256(artifact.sha256):
        failures.append(f"report_artifacts[{index}].sha256 must contain 64 lowercase hexadecimal characters")
    return tuple(failures)


def _paper_method_failures(method: CiftPaperMethodContract) -> tuple[str, ...]:
    failures: list[str] = []
    expected_values = (
        (
            "paper_method.readout_position_contract",
            method.readout_position_contract,
            "post_secret_post_query_causal_readout",
        ),
        ("paper_method.monitored_layer_policy", method.monitored_layer_policy, "last_quarter_transformer_layers"),
    )
    for field_name, actual_value, expected_value in expected_values:
        if actual_value != expected_value:
            failures.append(f"{field_name} must be {expected_value}")
    failures.extend(_feature_representation_failures(method))
    if method.pre_output is not True:
        failures.append("paper_method.pre_output must be true")
    if method.uses_static_secret_token_positions is not False:
        failures.append("paper_method.uses_static_secret_token_positions must be false")
    if method.training_loss == "":
        failures.append("paper_method.training_loss must not be empty")
    if method.probe_architecture == _PAPER_MLP_PROBE_ARCHITECTURE:
        if method.training_loss != "bce_with_l1_softplus_weight_sparsity":
            failures.append("paper_method.training_loss must be bce_with_l1_softplus_weight_sparsity")
        failures.extend(_paper_mlp_head_to_head_failures(method))
        return tuple(failures)
    failures.extend(_alternative_probe_failures(method))
    return tuple(failures)


def _feature_representation_failures(method: CiftPaperMethodContract) -> tuple[str, ...]:
    if method.feature_representation == "diagonal_mahalanobis_cci":
        return _paper_faithful_cci_failures(method)
    if method.feature_representation == "raw_activation":
        return _raw_activation_exception_failures(method)
    return ("paper_method.feature_representation must be diagonal_mahalanobis_cci or raw_activation",)


def _paper_faithful_cci_failures(method: CiftPaperMethodContract) -> tuple[str, ...]:
    failures: list[str] = []
    if method.covariance_estimator != "diagonal_covariance":
        failures.append("paper_method.covariance_estimator must be diagonal_covariance")
    if not math.isfinite(method.ridge) or method.ridge != 0.001:
        failures.append("paper_method.ridge must be 0.001")
    if method.layer_weighting != "softplus_nonnegative_cfs":
        failures.append("paper_method.layer_weighting must be softplus_nonnegative_cfs")
    return tuple(failures)


def _raw_activation_exception_failures(method: CiftPaperMethodContract) -> tuple[str, ...]:
    failures: list[str] = []
    if method.covariance_estimator != "not_applicable":
        failures.append("paper_method.covariance_estimator must be not_applicable for raw_activation")
    if not math.isfinite(method.ridge) or method.ridge != 0.0:
        failures.append("paper_method.ridge must be 0.0 for raw_activation")
    if method.layer_weighting != "not_applicable":
        failures.append("paper_method.layer_weighting must be not_applicable for raw_activation")
    if method.probe_architecture == _PAPER_MLP_PROBE_ARCHITECTURE:
        return tuple(failures)
    if method.paper_faithfulness_exception is None or method.paper_faithfulness_exception == "":
        failures.append("paper_method.paper_faithfulness_exception must explain non-paper feature_representation")
    return tuple(failures)


def _paper_mlp_head_to_head_failures(method: CiftPaperMethodContract) -> tuple[str, ...]:
    if method.head_to_head_report_id is None and method.paper_probe_metric_value is None:
        return ()
    failures: list[str] = []
    if method.head_to_head_report_id is None or method.head_to_head_report_id == "":
        failures.append("paper MLP promotion comparison requires head_to_head_report_id")
    if method.paper_probe_metric_value is None:
        failures.append("paper MLP promotion comparison requires paper_probe_metric_value")
    if method.candidate_probe_metric_value is None:
        failures.append("paper MLP promotion comparison requires candidate_probe_metric_value")
    if method.paper_probe_metric_value is not None and not math.isfinite(method.paper_probe_metric_value):
        failures.append("paper_probe_metric_value must be finite")
    if method.candidate_probe_metric_value is not None and not math.isfinite(method.candidate_probe_metric_value):
        failures.append("candidate_probe_metric_value must be finite")
    if (
        method.paper_probe_metric_value is not None
        and method.candidate_probe_metric_value is not None
        and math.isfinite(method.paper_probe_metric_value)
        and math.isfinite(method.candidate_probe_metric_value)
        and method.paper_probe_metric_value < method.candidate_probe_metric_value
    ):
        failures.append("paper_probe_metric_value must meet or exceed candidate_probe_metric_value for paper MLP")
    return tuple(failures)


def _classifier_probe_failures(bundle: CiftModelBundle, evidence: CiftPromotionEvidence) -> tuple[str, ...]:
    method = evidence.paper_method
    is_paper_mlp_classifier = type(bundle.classifier).__name__ == CiftPaperMlpClassifier.__name__
    if method.probe_architecture == _PAPER_MLP_PROBE_ARCHITECTURE:
        if not is_paper_mlp_classifier:
            return ("paper_method.probe_architecture mlp_128_64_1 requires CiftPaperMlpClassifier",)
        return ()
    if is_paper_mlp_classifier:
        return ("CiftPaperMlpClassifier requires paper_method.probe_architecture mlp_128_64_1",)
    return ()


def _alternative_probe_failures(method: CiftPaperMethodContract) -> tuple[str, ...]:
    failures: list[str] = []
    if method.probe_architecture == "":
        failures.append("paper_method.probe_architecture must not be empty")
    if method.head_to_head_report_id is None or method.head_to_head_report_id == "":
        failures.append("alternative probe architecture requires head_to_head_report_id")
    if method.paper_probe_metric_value is None:
        failures.append("alternative probe architecture requires paper_probe_metric_value")
    if method.candidate_probe_metric_value is None:
        failures.append("alternative probe architecture requires candidate_probe_metric_value")
    if method.paper_probe_metric_value is not None and not math.isfinite(method.paper_probe_metric_value):
        failures.append("paper_probe_metric_value must be finite")
    if method.candidate_probe_metric_value is not None and not math.isfinite(method.candidate_probe_metric_value):
        failures.append("candidate_probe_metric_value must be finite")
    if (
        method.paper_probe_metric_value is not None
        and method.candidate_probe_metric_value is not None
        and math.isfinite(method.paper_probe_metric_value)
        and math.isfinite(method.candidate_probe_metric_value)
    ):
        if method.feature_representation == "raw_activation":
            if method.candidate_probe_metric_value <= method.paper_probe_metric_value:
                failures.append("raw_activation candidate_probe_metric_value must exceed paper_probe_metric_value")
        elif method.candidate_probe_metric_value < method.paper_probe_metric_value:
            failures.append("candidate_probe_metric_value must meet or exceed paper_probe_metric_value")
    return tuple(failures)


def _unique_nonempty_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    unique_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value == "" or value in seen:
            continue
        unique_values.append(value)
        seen.add(value)
    return tuple(unique_values)


def _required_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CiftPromotionGateError(f"{field_name} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _report_artifacts_from_value(value: object) -> tuple[CiftPromotionReportArtifact, ...]:
    if not isinstance(value, list):
        raise CiftPromotionGateError("report_artifacts must be a list.")
    return tuple(
        cift_promotion_report_artifact_from_mapping(
            _required_mapping(value=item, field_name=f"report_artifacts[{index}]")
        )
        for index, item in enumerate(value)
    )


def cift_promotion_report_artifact_from_mapping(record: Mapping[str, object]) -> CiftPromotionReportArtifact:
    return CiftPromotionReportArtifact(
        report_id=_required_string(record=record, field_name="report_id"),
        path=_required_string(record=record, field_name="path"),
        sha256=_required_string(record=record, field_name="sha256"),
        schema_version=_required_string(record=record, field_name="schema_version"),
    )


def _required_string(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise CiftPromotionGateError(f"{field_name} must be a string.")
    return value


def _optional_string(record: Mapping[str, object], field_name: str) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CiftPromotionGateError(f"{field_name} must be a string when present.")
    return value


def _required_bool(record: Mapping[str, object], field_name: str) -> bool:
    value = record.get(field_name)
    if not isinstance(value, bool):
        raise CiftPromotionGateError(f"{field_name} must be a boolean.")
    return value


def _required_float(record: Mapping[str, object], field_name: str) -> float:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CiftPromotionGateError(f"{field_name} must be a number.")
    return float(value)


def _optional_float(record: Mapping[str, object], field_name: str) -> float | None:
    value = record.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CiftPromotionGateError(f"{field_name} must be a number when present.")
    return float(value)


def _is_lowercase_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
