from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

from aegis_introspection.cift_promotion_gate import CiftPaperMethodContract

_SCHEMA_VERSION = "aegis_introspection.cift_live_probe_competition/v1"
_PAPER_PROBE_ARCHITECTURE = "mlp_128_64_1"
_PAPER_TRAINING_LOSS = "bce_with_l1_softplus_weight_sparsity"
_RAW_ACTIVATION_FEATURE_KEY_PREFIXES = (
    "final_token_layer_",
    "mean_pool_layer_",
    "readout_window_layer_",
    "query_tail_window_layer_",
    "selected_choice_window_layer_",
    "combined_readout_window_layer_",
)

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class CiftLiveProbeCompetitionError(ValueError):
    """Raised when live sealed CIFT probe competition evidence is invalid."""


@dataclass(frozen=True)
class CiftLiveProbeRun:
    source_report_id: str
    probe_architecture: str
    training_loss: str
    model_bundle_id: str
    metric_value: float
    false_negative_count: int
    false_positive_count: int
    false_negative_rate: float
    false_positive_rate: float
    operating_threshold: float


@dataclass(frozen=True)
class CiftLiveProbeCompetitionConfig:
    report_id: str
    training_dataset_id: str
    task_name: str
    evaluation_split_id: str
    evaluation_split_manifest_id: str
    evaluation_split_sha256: str
    feature_representation: str
    activation_feature_key: str
    metric_name: str
    paper_probe: CiftLiveProbeRun
    candidate_probe: CiftLiveProbeRun
    higher_is_better: bool
    created_at: str


@dataclass(frozen=True)
class CiftLiveProbeCompetitionReport:
    schema_version: str
    report_id: str
    training_dataset_id: str
    task_name: str
    evaluation_split_id: str
    evaluation_split_manifest_id: str
    evaluation_split_sha256: str
    feature_representation: str
    activation_feature_key: str
    metric_name: str
    higher_is_better: bool
    paper_probe: CiftLiveProbeRun
    candidate_probe: CiftLiveProbeRun
    paper_probe_metric_value: float
    candidate_probe_metric_value: float
    candidate_delta: float
    candidate_strictly_outperforms_paper: bool
    winner_probe_architecture: str
    created_at: str


def compare_cift_live_probe_candidates(config: CiftLiveProbeCompetitionConfig) -> CiftLiveProbeCompetitionReport:
    _validate_config(config)
    paper_value = config.paper_probe.metric_value
    candidate_value = config.candidate_probe.metric_value
    candidate_delta = candidate_value - paper_value if config.higher_is_better else paper_value - candidate_value
    candidate_strictly_outperforms_paper = candidate_delta > 0.0
    return CiftLiveProbeCompetitionReport(
        schema_version=_SCHEMA_VERSION,
        report_id=config.report_id,
        training_dataset_id=config.training_dataset_id,
        task_name=config.task_name,
        evaluation_split_id=config.evaluation_split_id,
        evaluation_split_manifest_id=config.evaluation_split_manifest_id,
        evaluation_split_sha256=config.evaluation_split_sha256,
        feature_representation=config.feature_representation,
        activation_feature_key=config.activation_feature_key,
        metric_name=config.metric_name,
        higher_is_better=config.higher_is_better,
        paper_probe=config.paper_probe,
        candidate_probe=config.candidate_probe,
        paper_probe_metric_value=paper_value,
        candidate_probe_metric_value=candidate_value,
        candidate_delta=candidate_delta,
        candidate_strictly_outperforms_paper=candidate_strictly_outperforms_paper,
        winner_probe_architecture=_winner_probe_architecture(
            paper_probe=config.paper_probe,
            candidate_probe=config.candidate_probe,
            candidate_delta=candidate_delta,
        ),
        created_at=config.created_at,
    )


def cift_live_probe_competition_report_to_json(
    report: CiftLiveProbeCompetitionReport,
) -> dict[str, JsonValue]:
    _validate_report(report)
    return {
        "schema_version": report.schema_version,
        "report_id": report.report_id,
        "training_dataset_id": report.training_dataset_id,
        "task_name": report.task_name,
        "evaluation_split_id": report.evaluation_split_id,
        "evaluation_split_manifest_id": report.evaluation_split_manifest_id,
        "evaluation_split_sha256": report.evaluation_split_sha256,
        "feature_representation": report.feature_representation,
        "activation_feature_key": report.activation_feature_key,
        "metric_name": report.metric_name,
        "higher_is_better": report.higher_is_better,
        "paper_probe": cift_live_probe_run_to_json(report.paper_probe),
        "candidate_probe": cift_live_probe_run_to_json(report.candidate_probe),
        "paper_probe_metric_value": report.paper_probe_metric_value,
        "candidate_probe_metric_value": report.candidate_probe_metric_value,
        "candidate_delta": report.candidate_delta,
        "candidate_strictly_outperforms_paper": report.candidate_strictly_outperforms_paper,
        "winner_probe_architecture": report.winner_probe_architecture,
        "created_at": report.created_at,
    }


def cift_live_probe_competition_report_from_mapping(
    record: Mapping[str, object],
) -> CiftLiveProbeCompetitionReport:
    report = CiftLiveProbeCompetitionReport(
        schema_version=_required_string(record=record, field_name="schema_version"),
        report_id=_required_string(record=record, field_name="report_id"),
        training_dataset_id=_required_string(record=record, field_name="training_dataset_id"),
        task_name=_required_string(record=record, field_name="task_name"),
        evaluation_split_id=_required_string(record=record, field_name="evaluation_split_id"),
        evaluation_split_manifest_id=_required_string(record=record, field_name="evaluation_split_manifest_id"),
        evaluation_split_sha256=_required_string(record=record, field_name="evaluation_split_sha256"),
        feature_representation=_required_string(record=record, field_name="feature_representation"),
        activation_feature_key=_required_string(record=record, field_name="activation_feature_key"),
        metric_name=_required_string(record=record, field_name="metric_name"),
        higher_is_better=_required_bool(record=record, field_name="higher_is_better"),
        paper_probe=cift_live_probe_run_from_mapping(
            _required_mapping(value=record.get("paper_probe"), field_name="paper_probe")
        ),
        candidate_probe=cift_live_probe_run_from_mapping(
            _required_mapping(value=record.get("candidate_probe"), field_name="candidate_probe")
        ),
        paper_probe_metric_value=_required_float(record=record, field_name="paper_probe_metric_value"),
        candidate_probe_metric_value=_required_float(record=record, field_name="candidate_probe_metric_value"),
        candidate_delta=_required_float(record=record, field_name="candidate_delta"),
        candidate_strictly_outperforms_paper=_required_bool(
            record=record,
            field_name="candidate_strictly_outperforms_paper",
        ),
        winner_probe_architecture=_required_string(record=record, field_name="winner_probe_architecture"),
        created_at=_required_string(record=record, field_name="created_at"),
    )
    _validate_report(report)
    return report


def cift_live_probe_run_to_json(run: CiftLiveProbeRun) -> dict[str, JsonValue]:
    _validate_probe_run(run=run, field_name="probe")
    return {
        "source_report_id": run.source_report_id,
        "probe_architecture": run.probe_architecture,
        "training_loss": run.training_loss,
        "model_bundle_id": run.model_bundle_id,
        "metric_value": run.metric_value,
        "false_negative_count": run.false_negative_count,
        "false_positive_count": run.false_positive_count,
        "false_negative_rate": run.false_negative_rate,
        "false_positive_rate": run.false_positive_rate,
        "operating_threshold": run.operating_threshold,
    }


def cift_live_probe_run_from_mapping(record: Mapping[str, object]) -> CiftLiveProbeRun:
    run = CiftLiveProbeRun(
        source_report_id=_required_string(record=record, field_name="source_report_id"),
        probe_architecture=_required_string(record=record, field_name="probe_architecture"),
        training_loss=_required_string(record=record, field_name="training_loss"),
        model_bundle_id=_required_string(record=record, field_name="model_bundle_id"),
        metric_value=_required_float(record=record, field_name="metric_value"),
        false_negative_count=_required_int(record=record, field_name="false_negative_count"),
        false_positive_count=_required_int(record=record, field_name="false_positive_count"),
        false_negative_rate=_required_float(record=record, field_name="false_negative_rate"),
        false_positive_rate=_required_float(record=record, field_name="false_positive_rate"),
        operating_threshold=_required_float(record=record, field_name="operating_threshold"),
    )
    _validate_probe_run(run=run, field_name="probe")
    return run


def live_promotion_paper_method_from_probe_competition(
    report: CiftLiveProbeCompetitionReport,
) -> CiftPaperMethodContract:
    _validate_report(report)
    feature_representation = promotion_feature_representation_family(report.feature_representation)
    if not report.candidate_strictly_outperforms_paper:
        return _paper_mlp_promotion_method_from_probe_competition(report)
    if feature_representation == "raw_activation":
        return CiftPaperMethodContract(
            readout_position_contract="post_secret_post_query_causal_readout",
            monitored_layer_policy="last_quarter_transformer_layers",
            feature_representation="raw_activation",
            covariance_estimator="not_applicable",
            ridge=0.0,
            layer_weighting="not_applicable",
            probe_architecture=report.candidate_probe.probe_architecture,
            training_loss=report.candidate_probe.training_loss,
            pre_output=True,
            uses_static_secret_token_positions=False,
            head_to_head_report_id=report.report_id,
            paper_probe_metric_value=report.paper_probe_metric_value,
            candidate_probe_metric_value=report.candidate_probe_metric_value,
            paper_faithfulness_exception=(
                f"raw_activation live sealed head-to-head evidence for readout '{report.feature_representation}' "
                "shows the candidate probe strictly outperforms the paper MLP"
            ),
        )
    if feature_representation == "diagonal_mahalanobis_cci":
        return CiftPaperMethodContract(
            readout_position_contract="post_secret_post_query_causal_readout",
            monitored_layer_policy="last_quarter_transformer_layers",
            feature_representation="diagonal_mahalanobis_cci",
            covariance_estimator="diagonal_covariance",
            ridge=0.001,
            layer_weighting="softplus_nonnegative_cfs",
            probe_architecture=report.candidate_probe.probe_architecture,
            training_loss=report.candidate_probe.training_loss,
            pre_output=True,
            uses_static_secret_token_positions=False,
            head_to_head_report_id=report.report_id,
            paper_probe_metric_value=report.paper_probe_metric_value,
            candidate_probe_metric_value=report.candidate_probe_metric_value,
            paper_faithfulness_exception=None,
        )
    raise CiftLiveProbeCompetitionError(f"Unsupported feature_representation '{report.feature_representation}'.")


def _paper_mlp_promotion_method_from_probe_competition(
    report: CiftLiveProbeCompetitionReport,
) -> CiftPaperMethodContract:
    feature_representation = promotion_feature_representation_family(report.feature_representation)
    if feature_representation == "raw_activation":
        return CiftPaperMethodContract(
            readout_position_contract="post_secret_post_query_causal_readout",
            monitored_layer_policy="last_quarter_transformer_layers",
            feature_representation="raw_activation",
            covariance_estimator="not_applicable",
            ridge=0.0,
            layer_weighting="not_applicable",
            probe_architecture=report.paper_probe.probe_architecture,
            training_loss=report.paper_probe.training_loss,
            pre_output=True,
            uses_static_secret_token_positions=False,
            head_to_head_report_id=report.report_id,
            paper_probe_metric_value=report.paper_probe_metric_value,
            candidate_probe_metric_value=report.candidate_probe_metric_value,
            paper_faithfulness_exception=None,
        )
    if feature_representation == "diagonal_mahalanobis_cci":
        return CiftPaperMethodContract(
            readout_position_contract="post_secret_post_query_causal_readout",
            monitored_layer_policy="last_quarter_transformer_layers",
            feature_representation="diagonal_mahalanobis_cci",
            covariance_estimator="diagonal_covariance",
            ridge=0.001,
            layer_weighting="softplus_nonnegative_cfs",
            probe_architecture=report.paper_probe.probe_architecture,
            training_loss=report.paper_probe.training_loss,
            pre_output=True,
            uses_static_secret_token_positions=False,
            head_to_head_report_id=report.report_id,
            paper_probe_metric_value=report.paper_probe_metric_value,
            candidate_probe_metric_value=report.candidate_probe_metric_value,
            paper_faithfulness_exception=None,
        )
    raise CiftLiveProbeCompetitionError(f"Unsupported feature_representation '{report.feature_representation}'.")


def promotion_feature_representation_family(feature_representation: str) -> str:
    if feature_representation in ("raw_activation", "diagonal_mahalanobis_cci"):
        return feature_representation
    if _is_raw_activation_feature_key(feature_representation):
        return "raw_activation"
    raise CiftLiveProbeCompetitionError(f"Unsupported feature_representation '{feature_representation}'.")


def _is_raw_activation_feature_key(feature_representation: str) -> bool:
    if feature_representation.startswith("concat("):
        if not feature_representation.endswith(")"):
            return False
        source_feature_keys = tuple(
            source_feature_key.strip() for source_feature_key in feature_representation[len("concat(") : -1].split(",")
        )
        return len(source_feature_keys) >= 2 and all(
            _is_single_raw_activation_feature_key(source_feature_key) for source_feature_key in source_feature_keys
        )
    return _is_single_raw_activation_feature_key(feature_representation)


def _is_single_raw_activation_feature_key(feature_key: str) -> bool:
    if not feature_key.startswith(_RAW_ACTIVATION_FEATURE_KEY_PREFIXES):
        return False
    layer_text = feature_key.rsplit("_layer_", maxsplit=1)[1]
    return layer_text.isdecimal()


def materialize_cift_live_probe_competition(
    config: CiftLiveProbeCompetitionConfig,
    output_path: Path,
) -> CiftLiveProbeCompetitionReport:
    report = compare_cift_live_probe_candidates(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cift_live_probe_competition_report_to_json(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _validate_config(config: CiftLiveProbeCompetitionConfig) -> None:
    _validate_required_string(value=config.report_id, field_name="report_id")
    _validate_required_string(value=config.training_dataset_id, field_name="training_dataset_id")
    _validate_required_string(value=config.task_name, field_name="task_name")
    _validate_required_string(value=config.evaluation_split_id, field_name="evaluation_split_id")
    _validate_required_string(value=config.evaluation_split_manifest_id, field_name="evaluation_split_manifest_id")
    _validate_sha256(value=config.evaluation_split_sha256, field_name="evaluation_split_sha256")
    _validate_required_string(value=config.feature_representation, field_name="feature_representation")
    _validate_required_string(value=config.activation_feature_key, field_name="activation_feature_key")
    _validate_required_string(value=config.metric_name, field_name="metric_name")
    _validate_required_string(value=config.created_at, field_name="created_at")
    _validate_probe_run(run=config.paper_probe, field_name="paper_probe")
    _validate_probe_run(run=config.candidate_probe, field_name="candidate_probe")
    if config.paper_probe.probe_architecture != _PAPER_PROBE_ARCHITECTURE:
        raise CiftLiveProbeCompetitionError(f"paper_probe.probe_architecture must be {_PAPER_PROBE_ARCHITECTURE}.")
    if config.paper_probe.training_loss != _PAPER_TRAINING_LOSS:
        raise CiftLiveProbeCompetitionError(f"paper_probe.training_loss must be {_PAPER_TRAINING_LOSS}.")
    if config.paper_probe.source_report_id == config.candidate_probe.source_report_id:
        raise CiftLiveProbeCompetitionError("paper_probe and candidate_probe must come from distinct reports.")


def _validate_report(report: CiftLiveProbeCompetitionReport) -> None:
    if report.schema_version != _SCHEMA_VERSION:
        raise CiftLiveProbeCompetitionError(f"schema_version must be {_SCHEMA_VERSION}.")
    expected_report = compare_cift_live_probe_candidates(
        CiftLiveProbeCompetitionConfig(
            report_id=report.report_id,
            training_dataset_id=report.training_dataset_id,
            task_name=report.task_name,
            evaluation_split_id=report.evaluation_split_id,
            evaluation_split_manifest_id=report.evaluation_split_manifest_id,
            evaluation_split_sha256=report.evaluation_split_sha256,
            feature_representation=report.feature_representation,
            activation_feature_key=report.activation_feature_key,
            metric_name=report.metric_name,
            paper_probe=report.paper_probe,
            candidate_probe=report.candidate_probe,
            higher_is_better=report.higher_is_better,
            created_at=report.created_at,
        )
    )
    mismatched_fields = tuple(
        field_name
        for field_name, actual_value, expected_value in (
            ("paper_probe_metric_value", report.paper_probe_metric_value, expected_report.paper_probe_metric_value),
            (
                "candidate_probe_metric_value",
                report.candidate_probe_metric_value,
                expected_report.candidate_probe_metric_value,
            ),
            ("candidate_delta", report.candidate_delta, expected_report.candidate_delta),
            (
                "candidate_strictly_outperforms_paper",
                report.candidate_strictly_outperforms_paper,
                expected_report.candidate_strictly_outperforms_paper,
            ),
            ("winner_probe_architecture", report.winner_probe_architecture, expected_report.winner_probe_architecture),
        )
        if actual_value != expected_value
    )
    if len(mismatched_fields) > 0:
        raise CiftLiveProbeCompetitionError(f"report fields are inconsistent: {', '.join(mismatched_fields)}.")


def _validate_probe_run(run: CiftLiveProbeRun, field_name: str) -> None:
    _validate_required_string(value=run.source_report_id, field_name=f"{field_name}.source_report_id")
    _validate_required_string(value=run.probe_architecture, field_name=f"{field_name}.probe_architecture")
    _validate_required_string(value=run.training_loss, field_name=f"{field_name}.training_loss")
    _validate_required_string(value=run.model_bundle_id, field_name=f"{field_name}.model_bundle_id")
    _validate_probability(value=run.metric_value, field_name=f"{field_name}.metric_value")
    _validate_nonnegative_int(value=run.false_negative_count, field_name=f"{field_name}.false_negative_count")
    _validate_nonnegative_int(value=run.false_positive_count, field_name=f"{field_name}.false_positive_count")
    _validate_probability(value=run.false_negative_rate, field_name=f"{field_name}.false_negative_rate")
    _validate_probability(value=run.false_positive_rate, field_name=f"{field_name}.false_positive_rate")
    _validate_probability(value=run.operating_threshold, field_name=f"{field_name}.operating_threshold")


def _winner_probe_architecture(
    paper_probe: CiftLiveProbeRun,
    candidate_probe: CiftLiveProbeRun,
    candidate_delta: float,
) -> str:
    if candidate_delta > 0.0:
        return candidate_probe.probe_architecture
    if candidate_delta < 0.0:
        return paper_probe.probe_architecture
    return "tie"


def _required_string(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise CiftLiveProbeCompetitionError(f"{field_name} must be a string.")
    _validate_required_string(value=value, field_name=field_name)
    return value


def _validate_required_string(value: str, field_name: str) -> None:
    if value == "":
        raise CiftLiveProbeCompetitionError(f"{field_name} must not be empty.")


def _validate_sha256(value: str, field_name: str) -> None:
    _validate_required_string(value=value, field_name=field_name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CiftLiveProbeCompetitionError(f"{field_name} must be a lowercase SHA-256 hex digest.")


def _validate_probability(value: float, field_name: str) -> None:
    if not math.isfinite(value):
        raise CiftLiveProbeCompetitionError(f"{field_name} must be finite.")
    if value < 0.0 or value > 1.0:
        raise CiftLiveProbeCompetitionError(f"{field_name} must be in [0.0, 1.0].")


def _validate_nonnegative_int(value: int, field_name: str) -> None:
    if value < 0:
        raise CiftLiveProbeCompetitionError(f"{field_name} must be non-negative.")


def _required_bool(record: Mapping[str, object], field_name: str) -> bool:
    value = record.get(field_name)
    if not isinstance(value, bool):
        raise CiftLiveProbeCompetitionError(f"{field_name} must be a boolean.")
    return value


def _required_int(record: Mapping[str, object], field_name: str) -> int:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftLiveProbeCompetitionError(f"{field_name} must be an integer.")
    return value


def _required_float(record: Mapping[str, object], field_name: str) -> float:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CiftLiveProbeCompetitionError(f"{field_name} must be a number.")
    float_value = float(value)
    if not math.isfinite(float_value):
        raise CiftLiveProbeCompetitionError(f"{field_name} must be finite.")
    return float_value


def _required_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CiftLiveProbeCompetitionError(f"{field_name} must be an object.")
    return cast(Mapping[str, object], value)
