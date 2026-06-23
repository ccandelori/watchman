from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.artifacts import load_activation_artifact  # noqa: E402
from aegis_introspection.feature_transfer_error_analysis import (  # noqa: E402
    TransferErrorAnalysisConfig,
    TransferErrorDataset,
    analyze_feature_transfer_errors,
    load_structured_prompt_records,
    write_feature_transfer_error_analysis_json,
    write_feature_transfer_error_analysis_markdown,
)


@dataclass(frozen=True)
class DatasetArtifactSpec:
    dataset_id: str
    artifact_path: Path


@dataclass(frozen=True)
class FeatureTransferErrorScriptConfig:
    train_dataset_artifacts: tuple[DatasetArtifactSpec, ...]
    test_dataset_artifact: DatasetArtifactSpec
    test_prompts_path: Path
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    activation_feature_key: str
    positive_label: str
    decision_threshold: float
    random_seed: int
    max_iter: int
    regularization_c: float
    max_error_examples: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze errors from a cross-profile activation transfer run.")
    parser.add_argument(
        "--train-dataset-artifact",
        required=True,
        action="append",
        help="Training dataset/artifact pair in the form dataset_id:path. May be provided multiple times.",
    )
    parser.add_argument(
        "--test-dataset-artifact",
        required=True,
        help="Held-out test dataset/artifact pair in the form dataset_id:path.",
    )
    parser.add_argument("--test-prompts", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--positive-label", required=False, default="exfiltration_intent")
    parser.add_argument("--threshold", required=False, type=float, default=0.5)
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument("--max-iter", required=False, type=int, default=1000)
    parser.add_argument("--regularization-c", required=False, type=float, default=1.0)
    parser.add_argument("--max-error-examples", required=False, type=int, default=20)
    return parser


def _parse_dataset_artifact(value: str) -> DatasetArtifactSpec:
    parts = value.split(":", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Dataset artifact spec '{value}' must use the form dataset_id:path.")
    dataset_id, artifact_path = parts
    if dataset_id == "":
        raise ValueError(f"Dataset artifact spec '{value}' has an empty dataset id.")
    if artifact_path == "":
        raise ValueError(f"Dataset artifact spec '{value}' has an empty artifact path.")
    return DatasetArtifactSpec(dataset_id=dataset_id, artifact_path=Path(artifact_path))


def _parse_train_dataset_artifacts(values: Sequence[str]) -> tuple[DatasetArtifactSpec, ...]:
    return tuple(_parse_dataset_artifact(value) for value in values)


def _parse_args(argv: Sequence[str]) -> FeatureTransferErrorScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return FeatureTransferErrorScriptConfig(
        train_dataset_artifacts=_parse_train_dataset_artifacts(tuple(namespace.train_dataset_artifact)),
        test_dataset_artifact=_parse_dataset_artifact(str(namespace.test_dataset_artifact)),
        test_prompts_path=Path(namespace.test_prompts),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        activation_feature_key=str(namespace.feature),
        positive_label=str(namespace.positive_label),
        decision_threshold=float(namespace.threshold),
        random_seed=int(namespace.seed),
        max_iter=int(namespace.max_iter),
        regularization_c=float(namespace.regularization_c),
        max_error_examples=int(namespace.max_error_examples),
    )


def _load_dataset(spec: DatasetArtifactSpec) -> TransferErrorDataset:
    return TransferErrorDataset(
        dataset_id=spec.dataset_id,
        artifact=load_activation_artifact(spec.artifact_path),
    )


def _analysis_config(config: FeatureTransferErrorScriptConfig) -> TransferErrorAnalysisConfig:
    return TransferErrorAnalysisConfig(
        task_name=config.task_name,
        activation_feature_key=config.activation_feature_key,
        positive_label=config.positive_label,
        decision_threshold=config.decision_threshold,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        max_error_examples=config.max_error_examples,
    )


def run_analysis(config: FeatureTransferErrorScriptConfig) -> None:
    report = analyze_feature_transfer_errors(
        train_datasets=tuple(_load_dataset(spec) for spec in config.train_dataset_artifacts),
        test_dataset=_load_dataset(config.test_dataset_artifact),
        structured_prompt_records=load_structured_prompt_records(config.test_prompts_path),
        config=_analysis_config(config),
    )
    write_feature_transfer_error_analysis_json(config.output_json_path, report)
    write_feature_transfer_error_analysis_markdown(config.output_markdown_path, report)

    print(f"Wrote feature transfer error analysis to {config.output_json_path}")
    print(f"Wrote feature transfer error summary to {config.output_markdown_path}")
    print(
        f"errors={report.error_count} false_positives={report.false_positive_count} "
        f"false_negatives={report.false_negative_count} macro_f1={report.macro_f1:.4f}"
    )


def main(argv: Sequence[str]) -> None:
    run_analysis(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
