from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
INTROSPECTION_SRC_PATH = INTROSPECTION_ROOT / "src"
RUNTIME_SRC_PATH = WORKSPACE_ROOT / "src"
for source_path in (INTROSPECTION_SRC_PATH, RUNTIME_SRC_PATH):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from aegis_introspection.cift_model_training import (  # noqa: E402
    CiftTrainingArtifact,
    load_cift_training_artifact_with_unseal_policy,
)
from aegis_introspection.cift_probe_head_to_head import (  # noqa: E402
    CiftFeatureRepresentation,
    CiftProbeHeadToHeadConfig,
    evaluate_cift_probe_head_to_head,
    write_cift_probe_head_to_head_json,
)
from aegis_introspection.sealed_holdout_policy import add_unseal_flag  # noqa: E402


@dataclass(frozen=True)
class CompareCiftProbeHeadToHeadCliConfig:
    artifact_path: Path
    output_json_path: Path
    report_id: str
    training_dataset_id: str
    task_name: str
    positive_label: str
    feature_representation: CiftFeatureRepresentation
    activation_feature_key: str
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    ridge: float
    fold_count: int
    random_seeds: tuple[int, ...]
    decision_threshold: float
    linear_max_epochs: int
    linear_regularization_c: float
    paper_mlp_max_epochs: int
    paper_mlp_learning_rate: float
    paper_mlp_l1_softplus_weight: float
    paper_mlp_batch_size: int
    paper_hyperparameter_search_trials: int
    candidate_hyperparameter_search_trials: int
    evaluation_split_id: str
    evaluation_split_manifest_id: str
    metric_name: str
    created_at: str
    allow_sealed_holdout: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare paper CIFT MLP and linear challenger probes head-to-head.")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--training-dataset-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--positive-label", required=True)
    parser.add_argument(
        "--feature-representation",
        required=True,
        choices=("raw_activation", "diagonal_mahalanobis_cci"),
    )
    parser.add_argument("--activation-feature", required=True)
    parser.add_argument("--source-feature", action="append")
    parser.add_argument("--calibration-source-label", action="append")
    parser.add_argument("--ridge", required=True, type=float)
    parser.add_argument("--fold-count", required=True, type=int)
    parser.add_argument("--random-seeds", required=True)
    parser.add_argument("--decision-threshold", required=True, type=float)
    parser.add_argument("--linear-max-epochs", required=True, type=int)
    parser.add_argument("--linear-regularization-c", required=True, type=float)
    parser.add_argument("--paper-mlp-max-epochs", required=True, type=int)
    parser.add_argument("--paper-mlp-learning-rate", required=True, type=float)
    parser.add_argument("--paper-mlp-l1-softplus-weight", required=True, type=float)
    parser.add_argument("--paper-mlp-batch-size", required=True, type=int)
    parser.add_argument("--paper-hyperparameter-search-trials", required=True, type=int)
    parser.add_argument("--candidate-hyperparameter-search-trials", required=True, type=int)
    parser.add_argument("--evaluation-split-id", required=True)
    parser.add_argument("--evaluation-split-manifest-id", required=True)
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--created-at", required=True)
    add_unseal_flag(parser)
    return parser


def _parse_args(argv: Sequence[str]) -> CompareCiftProbeHeadToHeadCliConfig:
    namespace = _build_parser().parse_args(argv)
    return CompareCiftProbeHeadToHeadCliConfig(
        artifact_path=Path(str(namespace.artifact)),
        output_json_path=Path(str(namespace.output_json)),
        report_id=str(namespace.report_id),
        training_dataset_id=str(namespace.training_dataset_id),
        task_name=str(namespace.task),
        positive_label=str(namespace.positive_label),
        feature_representation=cast(CiftFeatureRepresentation, str(namespace.feature_representation)),
        activation_feature_key=str(namespace.activation_feature),
        source_feature_keys=_parse_repeated_values(namespace.source_feature),
        calibration_source_labels=_parse_repeated_values(namespace.calibration_source_label),
        ridge=float(namespace.ridge),
        fold_count=int(namespace.fold_count),
        random_seeds=_parse_random_seeds(str(namespace.random_seeds)),
        decision_threshold=float(namespace.decision_threshold),
        linear_max_epochs=int(namespace.linear_max_epochs),
        linear_regularization_c=float(namespace.linear_regularization_c),
        paper_mlp_max_epochs=int(namespace.paper_mlp_max_epochs),
        paper_mlp_learning_rate=float(namespace.paper_mlp_learning_rate),
        paper_mlp_l1_softplus_weight=float(namespace.paper_mlp_l1_softplus_weight),
        paper_mlp_batch_size=int(namespace.paper_mlp_batch_size),
        paper_hyperparameter_search_trials=int(namespace.paper_hyperparameter_search_trials),
        candidate_hyperparameter_search_trials=int(namespace.candidate_hyperparameter_search_trials),
        evaluation_split_id=str(namespace.evaluation_split_id),
        evaluation_split_manifest_id=str(namespace.evaluation_split_manifest_id),
        metric_name=str(namespace.metric_name),
        created_at=str(namespace.created_at),
        allow_sealed_holdout=bool(namespace.allow_sealed_holdout),
    )


def _parse_random_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip() != "")
    if len(seeds) == 0:
        raise ValueError("random-seeds must contain at least one integer.")
    return seeds


def _parse_repeated_values(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value) for value in values)


def _head_to_head_config(
    cli_config: CompareCiftProbeHeadToHeadCliConfig,
    artifact: CiftTrainingArtifact,
) -> CiftProbeHeadToHeadConfig:
    return CiftProbeHeadToHeadConfig(
        report_id=cli_config.report_id,
        artifact=artifact,
        training_dataset_id=cli_config.training_dataset_id,
        task_name=cli_config.task_name,
        positive_label=cli_config.positive_label,
        feature_representation=cli_config.feature_representation,
        activation_feature_key=cli_config.activation_feature_key,
        source_feature_keys=cli_config.source_feature_keys,
        calibration_source_labels=cli_config.calibration_source_labels,
        ridge=cli_config.ridge,
        fold_count=cli_config.fold_count,
        random_seeds=cli_config.random_seeds,
        decision_threshold=cli_config.decision_threshold,
        linear_max_epochs=cli_config.linear_max_epochs,
        linear_regularization_c=cli_config.linear_regularization_c,
        paper_mlp_max_epochs=cli_config.paper_mlp_max_epochs,
        paper_mlp_learning_rate=cli_config.paper_mlp_learning_rate,
        paper_mlp_l1_softplus_weight=cli_config.paper_mlp_l1_softplus_weight,
        paper_mlp_batch_size=cli_config.paper_mlp_batch_size,
        paper_hyperparameter_search_trials=cli_config.paper_hyperparameter_search_trials,
        candidate_hyperparameter_search_trials=cli_config.candidate_hyperparameter_search_trials,
        evaluation_split_id=cli_config.evaluation_split_id,
        evaluation_split_manifest_id=cli_config.evaluation_split_manifest_id,
        metric_name=cli_config.metric_name,
        created_at=cli_config.created_at,
    )


def run_cli(cli_config: CompareCiftProbeHeadToHeadCliConfig) -> None:
    artifact = load_cift_training_artifact_with_unseal_policy(
        path=cli_config.artifact_path,
        allow_sealed_holdout=cli_config.allow_sealed_holdout,
        context="CIFT probe head-to-head comparison",
    )
    report = evaluate_cift_probe_head_to_head(_head_to_head_config(cli_config=cli_config, artifact=artifact))
    write_cift_probe_head_to_head_json(path=cli_config.output_json_path, report=report)
    print(f"Wrote CIFT probe head-to-head JSON to {cli_config.output_json_path}")
    print(
        f"Paper MLP {report.competition_report.metric_name}: {report.competition_report.paper_probe_metric_value:.6f}"
    )
    print(
        "Linear challenger "
        f"{report.competition_report.metric_name}: {report.competition_report.candidate_probe_metric_value:.6f}"
    )
    print(f"Winner: {report.competition_report.winner_probe_architecture}")


def main(argv: Sequence[str]) -> None:
    run_cli(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
