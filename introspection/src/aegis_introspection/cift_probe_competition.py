from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from aegis_introspection.cift_promotion_gate import CiftPaperMethodContract

_SCHEMA_VERSION = "cift_probe_competition/v1"
_PAPER_PROBE_ARCHITECTURE = "mlp_128_64_1"
_PAPER_TRAINING_LOSS = "bce_with_l1_softplus_weight_sparsity"

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class CiftProbeCompetitionError(ValueError):
    """Raised when CIFT paper/challenger probe comparison evidence is invalid."""


@dataclass(frozen=True)
class CiftProbeRun:
    source_report_id: str
    probe_architecture: str
    training_loss: str
    training_dataset_id: str
    training_dataset_sha256: str
    task_name: str
    evaluation_split_id: str
    evaluation_split_manifest_id: str
    evaluation_split_sha256: str
    metric_name: str
    metric_value: float
    metric_confidence_interval_low: float
    metric_confidence_interval_high: float
    random_seeds: tuple[int, ...]
    hyperparameter_search_trials: int
    operating_threshold: float
    false_positive_rate: float
    true_positive_rate: float


@dataclass(frozen=True)
class CiftProbeCompetitionConfig:
    report_id: str
    paper_probe: CiftProbeRun
    candidate_probe: CiftProbeRun
    higher_is_better: bool
    created_at: str


@dataclass(frozen=True)
class CiftProbeCompetitionReport:
    schema_version: str
    report_id: str
    training_dataset_id: str
    training_dataset_sha256: str
    task_name: str
    evaluation_split_id: str
    evaluation_split_manifest_id: str
    evaluation_split_sha256: str
    metric_name: str
    higher_is_better: bool
    random_seeds: tuple[int, ...]
    paper_hyperparameter_search_trials: int
    candidate_hyperparameter_search_trials: int
    paper_probe: CiftProbeRun
    candidate_probe: CiftProbeRun
    paper_probe_metric_value: float
    candidate_probe_metric_value: float
    candidate_delta: float
    candidate_confidence_delta: float
    candidate_meets_or_exceeds_paper: bool
    winner_probe_architecture: str
    created_at: str


def compare_cift_probe_candidates(config: CiftProbeCompetitionConfig) -> CiftProbeCompetitionReport:
    _validate_config(config)
    paper_value = config.paper_probe.metric_value
    candidate_value = config.candidate_probe.metric_value
    candidate_delta = candidate_value - paper_value if config.higher_is_better else paper_value - candidate_value
    candidate_confidence_delta = _candidate_confidence_delta(
        paper_probe=config.paper_probe,
        candidate_probe=config.candidate_probe,
        higher_is_better=config.higher_is_better,
    )
    candidate_meets_or_exceeds_paper = candidate_delta >= 0.0 and candidate_confidence_delta >= 0.0
    winner_probe_architecture = _winner_probe_architecture(
        paper_probe=config.paper_probe,
        candidate_probe=config.candidate_probe,
        candidate_delta=candidate_delta,
    )
    return CiftProbeCompetitionReport(
        schema_version=_SCHEMA_VERSION,
        report_id=config.report_id,
        training_dataset_id=config.paper_probe.training_dataset_id,
        training_dataset_sha256=config.paper_probe.training_dataset_sha256,
        task_name=config.paper_probe.task_name,
        evaluation_split_id=config.paper_probe.evaluation_split_id,
        evaluation_split_manifest_id=config.paper_probe.evaluation_split_manifest_id,
        evaluation_split_sha256=config.paper_probe.evaluation_split_sha256,
        metric_name=config.paper_probe.metric_name,
        higher_is_better=config.higher_is_better,
        random_seeds=config.paper_probe.random_seeds,
        paper_hyperparameter_search_trials=config.paper_probe.hyperparameter_search_trials,
        candidate_hyperparameter_search_trials=config.candidate_probe.hyperparameter_search_trials,
        paper_probe=config.paper_probe,
        candidate_probe=config.candidate_probe,
        paper_probe_metric_value=paper_value,
        candidate_probe_metric_value=candidate_value,
        candidate_delta=candidate_delta,
        candidate_confidence_delta=candidate_confidence_delta,
        candidate_meets_or_exceeds_paper=candidate_meets_or_exceeds_paper,
        winner_probe_architecture=winner_probe_architecture,
        created_at=config.created_at,
    )


def cift_probe_competition_report_to_json(report: CiftProbeCompetitionReport) -> dict[str, JsonValue]:
    _validate_report(report)
    return {
        "schema_version": report.schema_version,
        "report_id": report.report_id,
        "training_dataset_id": report.training_dataset_id,
        "training_dataset_sha256": report.training_dataset_sha256,
        "task_name": report.task_name,
        "evaluation_split_id": report.evaluation_split_id,
        "evaluation_split_manifest_id": report.evaluation_split_manifest_id,
        "evaluation_split_sha256": report.evaluation_split_sha256,
        "metric_name": report.metric_name,
        "higher_is_better": report.higher_is_better,
        "random_seeds": list(report.random_seeds),
        "paper_hyperparameter_search_trials": report.paper_hyperparameter_search_trials,
        "candidate_hyperparameter_search_trials": report.candidate_hyperparameter_search_trials,
        "paper_probe": cift_probe_run_to_json(report.paper_probe),
        "candidate_probe": cift_probe_run_to_json(report.candidate_probe),
        "paper_probe_metric_value": report.paper_probe_metric_value,
        "candidate_probe_metric_value": report.candidate_probe_metric_value,
        "candidate_delta": report.candidate_delta,
        "candidate_confidence_delta": report.candidate_confidence_delta,
        "candidate_meets_or_exceeds_paper": report.candidate_meets_or_exceeds_paper,
        "winner_probe_architecture": report.winner_probe_architecture,
        "created_at": report.created_at,
    }


def cift_probe_competition_report_from_mapping(record: Mapping[str, object]) -> CiftProbeCompetitionReport:
    report = CiftProbeCompetitionReport(
        schema_version=_required_string(record=record, field_name="schema_version"),
        report_id=_required_string(record=record, field_name="report_id"),
        training_dataset_id=_required_string(record=record, field_name="training_dataset_id"),
        training_dataset_sha256=_required_string(record=record, field_name="training_dataset_sha256"),
        task_name=_required_string(record=record, field_name="task_name"),
        evaluation_split_id=_required_string(record=record, field_name="evaluation_split_id"),
        evaluation_split_manifest_id=_required_string(record=record, field_name="evaluation_split_manifest_id"),
        evaluation_split_sha256=_required_string(record=record, field_name="evaluation_split_sha256"),
        metric_name=_required_string(record=record, field_name="metric_name"),
        higher_is_better=_required_bool(record=record, field_name="higher_is_better"),
        random_seeds=_required_int_tuple(record=record, field_name="random_seeds"),
        paper_hyperparameter_search_trials=_required_int(
            record=record,
            field_name="paper_hyperparameter_search_trials",
        ),
        candidate_hyperparameter_search_trials=_required_int(
            record=record,
            field_name="candidate_hyperparameter_search_trials",
        ),
        paper_probe=cift_probe_run_from_mapping(
            _required_mapping(value=record.get("paper_probe"), field_name="paper_probe")
        ),
        candidate_probe=cift_probe_run_from_mapping(
            _required_mapping(value=record.get("candidate_probe"), field_name="candidate_probe")
        ),
        paper_probe_metric_value=_required_float(record=record, field_name="paper_probe_metric_value"),
        candidate_probe_metric_value=_required_float(record=record, field_name="candidate_probe_metric_value"),
        candidate_delta=_required_float(record=record, field_name="candidate_delta"),
        candidate_confidence_delta=_required_float(record=record, field_name="candidate_confidence_delta"),
        candidate_meets_or_exceeds_paper=_required_bool(
            record=record,
            field_name="candidate_meets_or_exceeds_paper",
        ),
        winner_probe_architecture=_required_string(record=record, field_name="winner_probe_architecture"),
        created_at=_required_string(record=record, field_name="created_at"),
    )
    _validate_report(report)
    return report


def cift_probe_run_to_json(run: CiftProbeRun) -> dict[str, JsonValue]:
    _validate_probe_run(run=run, field_name="probe")
    return {
        "source_report_id": run.source_report_id,
        "probe_architecture": run.probe_architecture,
        "training_loss": run.training_loss,
        "training_dataset_id": run.training_dataset_id,
        "training_dataset_sha256": run.training_dataset_sha256,
        "task_name": run.task_name,
        "evaluation_split_id": run.evaluation_split_id,
        "evaluation_split_manifest_id": run.evaluation_split_manifest_id,
        "evaluation_split_sha256": run.evaluation_split_sha256,
        "metric_name": run.metric_name,
        "metric_value": run.metric_value,
        "metric_confidence_interval_low": run.metric_confidence_interval_low,
        "metric_confidence_interval_high": run.metric_confidence_interval_high,
        "random_seeds": list(run.random_seeds),
        "hyperparameter_search_trials": run.hyperparameter_search_trials,
        "operating_threshold": run.operating_threshold,
        "false_negative_rate": _false_negative_rate(run.true_positive_rate),
        "false_positive_rate": run.false_positive_rate,
        "true_positive_rate": run.true_positive_rate,
    }


def cift_probe_run_from_mapping(record: Mapping[str, object]) -> CiftProbeRun:
    run = CiftProbeRun(
        source_report_id=_required_string(record=record, field_name="source_report_id"),
        probe_architecture=_required_string(record=record, field_name="probe_architecture"),
        training_loss=_required_string(record=record, field_name="training_loss"),
        training_dataset_id=_required_string(record=record, field_name="training_dataset_id"),
        training_dataset_sha256=_required_string(record=record, field_name="training_dataset_sha256"),
        task_name=_required_string(record=record, field_name="task_name"),
        evaluation_split_id=_required_string(record=record, field_name="evaluation_split_id"),
        evaluation_split_manifest_id=_required_string(record=record, field_name="evaluation_split_manifest_id"),
        evaluation_split_sha256=_required_string(record=record, field_name="evaluation_split_sha256"),
        metric_name=_required_string(record=record, field_name="metric_name"),
        metric_value=_required_float(record=record, field_name="metric_value"),
        metric_confidence_interval_low=_required_float(record=record, field_name="metric_confidence_interval_low"),
        metric_confidence_interval_high=_required_float(record=record, field_name="metric_confidence_interval_high"),
        random_seeds=_required_int_tuple(record=record, field_name="random_seeds"),
        hyperparameter_search_trials=_required_int(record=record, field_name="hyperparameter_search_trials"),
        operating_threshold=_required_float(record=record, field_name="operating_threshold"),
        false_positive_rate=_required_float(record=record, field_name="false_positive_rate"),
        true_positive_rate=_required_float(record=record, field_name="true_positive_rate"),
    )
    _validate_probe_run(run=run, field_name="probe")
    _validate_optional_false_negative_rate(record=record, true_positive_rate=run.true_positive_rate)
    return run


def promotion_paper_method_from_probe_competition(report: CiftProbeCompetitionReport) -> CiftPaperMethodContract:
    _validate_report(report)
    if not report.candidate_meets_or_exceeds_paper:
        if report.candidate_delta >= 0.0:
            raise CiftProbeCompetitionError(
                "candidate confidence interval must support meeting or exceeding the paper metric for promotion."
            )
        raise CiftProbeCompetitionError("candidate metric must meet or exceed paper metric for promotion.")
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


def _validate_config(config: CiftProbeCompetitionConfig) -> None:
    _validate_required_string(value=config.report_id, field_name="report_id")
    _validate_required_string(value=config.created_at, field_name="created_at")
    _validate_probe_run(run=config.paper_probe, field_name="paper_probe")
    _validate_probe_run(run=config.candidate_probe, field_name="candidate_probe")
    if config.paper_probe.probe_architecture != _PAPER_PROBE_ARCHITECTURE:
        raise CiftProbeCompetitionError(f"paper_probe.probe_architecture must be {_PAPER_PROBE_ARCHITECTURE}.")
    if config.paper_probe.training_loss != _PAPER_TRAINING_LOSS:
        raise CiftProbeCompetitionError(f"paper_probe.training_loss must be {_PAPER_TRAINING_LOSS}.")
    if config.paper_probe.source_report_id == config.candidate_probe.source_report_id:
        raise CiftProbeCompetitionError("paper_probe and candidate_probe must come from distinct source reports.")
    _validate_same_field(
        paper_value=config.paper_probe.training_dataset_id,
        candidate_value=config.candidate_probe.training_dataset_id,
        field_name="training_dataset_id",
    )
    _validate_same_field(
        paper_value=config.paper_probe.training_dataset_sha256,
        candidate_value=config.candidate_probe.training_dataset_sha256,
        field_name="training_dataset_sha256",
    )
    _validate_same_field(
        paper_value=config.paper_probe.task_name,
        candidate_value=config.candidate_probe.task_name,
        field_name="task_name",
    )
    _validate_same_field(
        paper_value=config.paper_probe.evaluation_split_id,
        candidate_value=config.candidate_probe.evaluation_split_id,
        field_name="evaluation_split_id",
    )
    _validate_same_field(
        paper_value=config.paper_probe.evaluation_split_manifest_id,
        candidate_value=config.candidate_probe.evaluation_split_manifest_id,
        field_name="evaluation_split_manifest_id",
    )
    _validate_same_field(
        paper_value=config.paper_probe.evaluation_split_sha256,
        candidate_value=config.candidate_probe.evaluation_split_sha256,
        field_name="evaluation_split_sha256",
    )
    _validate_same_field(
        paper_value=config.paper_probe.metric_name,
        candidate_value=config.candidate_probe.metric_name,
        field_name="metric_name",
    )
    if config.paper_probe.random_seeds != config.candidate_probe.random_seeds:
        raise CiftProbeCompetitionError("paper_probe and candidate_probe random_seeds must match.")
    if config.candidate_probe.hyperparameter_search_trials > config.paper_probe.hyperparameter_search_trials:
        raise CiftProbeCompetitionError(
            "candidate_probe.hyperparameter_search_trials must not exceed paper_probe.hyperparameter_search_trials."
        )


def _validate_report(report: CiftProbeCompetitionReport) -> None:
    if report.schema_version != _SCHEMA_VERSION:
        raise CiftProbeCompetitionError(f"schema_version must be {_SCHEMA_VERSION}.")
    expected_report = compare_cift_probe_candidates(
        CiftProbeCompetitionConfig(
            report_id=report.report_id,
            paper_probe=report.paper_probe,
            candidate_probe=report.candidate_probe,
            higher_is_better=report.higher_is_better,
            created_at=report.created_at,
        )
    )
    mismatched_fields = tuple(
        field_name
        for field_name, actual_value, expected_value in (
            ("training_dataset_id", report.training_dataset_id, expected_report.training_dataset_id),
            ("training_dataset_sha256", report.training_dataset_sha256, expected_report.training_dataset_sha256),
            ("task_name", report.task_name, expected_report.task_name),
            ("evaluation_split_id", report.evaluation_split_id, expected_report.evaluation_split_id),
            (
                "evaluation_split_manifest_id",
                report.evaluation_split_manifest_id,
                expected_report.evaluation_split_manifest_id,
            ),
            ("evaluation_split_sha256", report.evaluation_split_sha256, expected_report.evaluation_split_sha256),
            ("metric_name", report.metric_name, expected_report.metric_name),
            ("random_seeds", report.random_seeds, expected_report.random_seeds),
            (
                "paper_hyperparameter_search_trials",
                report.paper_hyperparameter_search_trials,
                expected_report.paper_hyperparameter_search_trials,
            ),
            (
                "candidate_hyperparameter_search_trials",
                report.candidate_hyperparameter_search_trials,
                expected_report.candidate_hyperparameter_search_trials,
            ),
            ("paper_probe_metric_value", report.paper_probe_metric_value, expected_report.paper_probe_metric_value),
            (
                "candidate_probe_metric_value",
                report.candidate_probe_metric_value,
                expected_report.candidate_probe_metric_value,
            ),
            ("candidate_delta", report.candidate_delta, expected_report.candidate_delta),
            (
                "candidate_confidence_delta",
                report.candidate_confidence_delta,
                expected_report.candidate_confidence_delta,
            ),
            (
                "candidate_meets_or_exceeds_paper",
                report.candidate_meets_or_exceeds_paper,
                expected_report.candidate_meets_or_exceeds_paper,
            ),
            ("winner_probe_architecture", report.winner_probe_architecture, expected_report.winner_probe_architecture),
        )
        if actual_value != expected_value
    )
    if len(mismatched_fields) > 0:
        raise CiftProbeCompetitionError(f"report fields are inconsistent: {', '.join(mismatched_fields)}.")


def _validate_probe_run(run: CiftProbeRun, field_name: str) -> None:
    _validate_required_string(value=run.source_report_id, field_name=f"{field_name}.source_report_id")
    _validate_required_string(value=run.probe_architecture, field_name=f"{field_name}.probe_architecture")
    _validate_required_string(value=run.training_loss, field_name=f"{field_name}.training_loss")
    _validate_required_string(value=run.training_dataset_id, field_name=f"{field_name}.training_dataset_id")
    _validate_sha256(value=run.training_dataset_sha256, field_name=f"{field_name}.training_dataset_sha256")
    _validate_required_string(value=run.task_name, field_name=f"{field_name}.task_name")
    _validate_required_string(value=run.evaluation_split_id, field_name=f"{field_name}.evaluation_split_id")
    _validate_required_string(
        value=run.evaluation_split_manifest_id,
        field_name=f"{field_name}.evaluation_split_manifest_id",
    )
    _validate_sha256(value=run.evaluation_split_sha256, field_name=f"{field_name}.evaluation_split_sha256")
    _validate_required_string(value=run.metric_name, field_name=f"{field_name}.metric_name")
    _validate_probability(value=run.metric_value, field_name=f"{field_name}.metric_value")
    _validate_probability(
        value=run.metric_confidence_interval_low,
        field_name=f"{field_name}.metric_confidence_interval_low",
    )
    _validate_probability(
        value=run.metric_confidence_interval_high,
        field_name=f"{field_name}.metric_confidence_interval_high",
    )
    if not run.metric_confidence_interval_low <= run.metric_value <= run.metric_confidence_interval_high:
        raise CiftProbeCompetitionError(f"{field_name}.metric confidence interval must contain metric_value.")
    _validate_random_seeds(value=run.random_seeds, field_name=f"{field_name}.random_seeds")
    if run.hyperparameter_search_trials < 1:
        raise CiftProbeCompetitionError(f"{field_name}.hyperparameter_search_trials must be at least 1.")
    _validate_probability(value=run.operating_threshold, field_name=f"{field_name}.operating_threshold")
    _validate_probability(value=run.false_positive_rate, field_name=f"{field_name}.false_positive_rate")
    _validate_probability(value=run.true_positive_rate, field_name=f"{field_name}.true_positive_rate")


def _false_negative_rate(true_positive_rate: float) -> float:
    return 1.0 - true_positive_rate


def _validate_optional_false_negative_rate(record: Mapping[str, object], true_positive_rate: float) -> None:
    if "false_negative_rate" not in record:
        return
    false_negative_rate = _required_float(record=record, field_name="false_negative_rate")
    _validate_probability(value=false_negative_rate, field_name="false_negative_rate")
    if not math.isclose(false_negative_rate, _false_negative_rate(true_positive_rate), abs_tol=1e-12):
        raise CiftProbeCompetitionError("false_negative_rate must equal 1 - true_positive_rate.")


def _candidate_confidence_delta(
    paper_probe: CiftProbeRun,
    candidate_probe: CiftProbeRun,
    higher_is_better: bool,
) -> float:
    if higher_is_better:
        return candidate_probe.metric_confidence_interval_low - paper_probe.metric_value
    return paper_probe.metric_value - candidate_probe.metric_confidence_interval_high


def _winner_probe_architecture(
    paper_probe: CiftProbeRun,
    candidate_probe: CiftProbeRun,
    candidate_delta: float,
) -> str:
    if candidate_delta > 0.0:
        return candidate_probe.probe_architecture
    if candidate_delta < 0.0:
        return paper_probe.probe_architecture
    return "tie"


def _validate_same_field(paper_value: str, candidate_value: str, field_name: str) -> None:
    if paper_value != candidate_value:
        raise CiftProbeCompetitionError(f"paper_probe and candidate_probe {field_name} must match.")


def _required_string(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise CiftProbeCompetitionError(f"{field_name} must be a string.")
    _validate_required_string(value=value, field_name=field_name)
    return value


def _validate_required_string(value: str, field_name: str) -> None:
    if value == "":
        raise CiftProbeCompetitionError(f"{field_name} must not be empty.")


def _validate_sha256(value: str, field_name: str) -> None:
    _validate_required_string(value=value, field_name=field_name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CiftProbeCompetitionError(f"{field_name} must be a lowercase SHA-256 hex digest.")


def _validate_probability(value: float, field_name: str) -> None:
    if not math.isfinite(value):
        raise CiftProbeCompetitionError(f"{field_name} must be finite.")
    if value < 0.0 or value > 1.0:
        raise CiftProbeCompetitionError(f"{field_name} must be in [0.0, 1.0].")


def _validate_random_seeds(value: tuple[int, ...], field_name: str) -> None:
    if len(value) < 3:
        raise CiftProbeCompetitionError(f"{field_name} must include at least three repeated-evaluation seeds.")
    if len(set(value)) != len(value):
        raise CiftProbeCompetitionError(f"{field_name} must not contain duplicate seeds.")


def _required_bool(record: Mapping[str, object], field_name: str) -> bool:
    value = record.get(field_name)
    if not isinstance(value, bool):
        raise CiftProbeCompetitionError(f"{field_name} must be a boolean.")
    return value


def _required_int(record: Mapping[str, object], field_name: str) -> int:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftProbeCompetitionError(f"{field_name} must be an integer.")
    return value


def _required_int_tuple(record: Mapping[str, object], field_name: str) -> tuple[int, ...]:
    value = record.get(field_name)
    if not isinstance(value, list | tuple):
        raise CiftProbeCompetitionError(f"{field_name} must be a list of integers.")
    parsed: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise CiftProbeCompetitionError(f"{field_name}[{index}] must be an integer.")
        parsed.append(item)
    return tuple(parsed)


def _required_float(record: Mapping[str, object], field_name: str) -> float:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CiftProbeCompetitionError(f"{field_name} must be a number.")
    float_value = float(value)
    if not math.isfinite(float_value):
        raise CiftProbeCompetitionError(f"{field_name} must be finite.")
    return float_value


def _required_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CiftProbeCompetitionError(f"{field_name} must be an object.")
    return value
