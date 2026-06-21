from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.cift_model_bundle import CandidateStatus
from aegis_introspection.cift_model_training import CiftModelTrainingConfig, train_cift_model_bundle


@dataclass(frozen=True)
class TrainCiftModelBundleCliConfig:
    artifact_path: Path
    output_bundle_path: Path
    training_dataset_id: str
    task_name: str
    positive_label: str
    activation_feature_key: str
    decision_threshold: float
    random_seed: int
    max_iter: int
    regularization_c: float
    evaluation_report_ids: tuple[str, ...]
    score_semantics: str
    candidate_status: CandidateStatus
    created_at: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and freeze a full-train CIFT detector model bundle.")
    parser.add_argument(
        "--artifact",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "activations"
            / "qwen3_0_6b_dp_honey_lite_v4_1_selector_windows.pt"
        ),
    )
    parser.add_argument(
        "--output-bundle",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "models"
            / "cift_qwen3_0_6b_dp_honey_lite_v4_1_selector_window_layer_15_v1.pkl"
        ),
    )
    parser.add_argument("--training-dataset-id", required=False, default="dp_honey_lite_v4_1_selector_windows")
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument("--positive-label", required=False, default="exfiltration_intent")
    parser.add_argument("--activation-feature", required=False, default="readout_window_layer_15")
    parser.add_argument("--decision-threshold", required=False, type=float, default=0.5)
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument("--max-iter", required=False, type=int, default=1000)
    parser.add_argument("--regularization-c", required=False, type=float, default=1.0)
    parser.add_argument(
        "--evaluation-report-ids",
        required=False,
        default=(
            "dp_honey_lite_v4_1_selector_window_grouped_binary_tasks_readout_window_layer_15_v1,"
            "dp_honey_lite_v4_1_selector_window_error_analysis_readout_window_layer_15_v1"
        ),
    )
    parser.add_argument("--score-semantics", required=False, default="full_train_classifier_probability")
    parser.add_argument(
        "--candidate-status",
        required=False,
        choices=("offline_research_candidate", "runtime_candidate"),
        default="offline_research_candidate",
    )
    parser.add_argument("--created-at", required=False, default="2026-06-21T00:00:00Z")
    return parser


def _parse_args(argv: Sequence[str]) -> TrainCiftModelBundleCliConfig:
    namespace = _build_parser().parse_args(argv)
    return TrainCiftModelBundleCliConfig(
        artifact_path=Path(namespace.artifact),
        output_bundle_path=Path(namespace.output_bundle),
        training_dataset_id=str(namespace.training_dataset_id),
        task_name=str(namespace.task),
        positive_label=str(namespace.positive_label),
        activation_feature_key=str(namespace.activation_feature),
        decision_threshold=float(namespace.decision_threshold),
        random_seed=int(namespace.seed),
        max_iter=int(namespace.max_iter),
        regularization_c=float(namespace.regularization_c),
        evaluation_report_ids=_parse_ids(str(namespace.evaluation_report_ids)),
        score_semantics=str(namespace.score_semantics),
        candidate_status=_candidate_status(str(namespace.candidate_status)),
        created_at=str(namespace.created_at),
    )


def _parse_ids(value: str) -> tuple[str, ...]:
    ids = tuple(item.strip() for item in value.split(",") if item.strip() != "")
    if len(ids) == 0:
        raise ValueError("evaluation-report-ids must contain at least one report id.")
    return ids


def _candidate_status(value: str) -> CandidateStatus:
    if value == "offline_research_candidate":
        return "offline_research_candidate"
    if value == "runtime_candidate":
        return "runtime_candidate"
    raise ValueError(f"Unsupported candidate-status '{value}'.")


def _training_config(config: TrainCiftModelBundleCliConfig) -> CiftModelTrainingConfig:
    return CiftModelTrainingConfig(
        artifact_path=config.artifact_path,
        output_bundle_path=config.output_bundle_path,
        training_dataset_id=config.training_dataset_id,
        task_name=config.task_name,
        positive_label=config.positive_label,
        activation_feature_key=config.activation_feature_key,
        decision_threshold=config.decision_threshold,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        evaluation_report_ids=config.evaluation_report_ids,
        score_semantics=config.score_semantics,
        candidate_status=config.candidate_status,
        created_at=config.created_at,
    )


def run_training(config: TrainCiftModelBundleCliConfig) -> None:
    report = train_cift_model_bundle(_training_config(config))
    print(f"Wrote CIFT model bundle to {report.output_bundle_path}")
    print(f"Task: {report.task_name}")
    print(f"Positive label: {report.positive_label}")
    print(f"Activation feature: {report.activation_feature_key}")
    print(f"Examples: {report.example_count}")
    print(f"Features: {report.feature_count}")
    print(f"Labels: {', '.join(report.label_names)}")
    print(f"Source artifact SHA-256: {report.source_artifact_sha256}")


def main(argv: Sequence[str]) -> None:
    run_training(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
