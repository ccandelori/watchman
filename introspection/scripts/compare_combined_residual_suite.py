from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.artifacts import ActivationArtifact, load_activation_artifact
from aegis_introspection.binary_tasks import BinaryTaskConfig
from aegis_introspection.error_analysis import BinaryErrorAnalysisReport, evaluate_grouped_binary_error_analysis
from aegis_introspection.residual_error_comparison import (
    ResidualErrorSuiteInput,
    compare_binary_error_residual_suite,
    write_residual_error_suite_json,
    write_residual_error_suite_markdown,
)


@dataclass(frozen=True)
class DatasetArtifactSpec:
    dataset_id: str
    artifact_path: Path


@dataclass(frozen=True)
class CombinedResidualSuiteScriptConfig:
    dataset_artifacts: tuple[DatasetArtifactSpec, ...]
    reference_feature_keys: tuple[str, ...]
    candidate_feature_key: str
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    fold_count: int
    random_seed: int
    max_iter: int
    regularization_c: float
    word_ngram_range: tuple[int, int]
    char_ngram_range: tuple[int, int]


def _default_dataset_artifacts() -> tuple[str, ...]:
    return (
        f"baseline_prompts_v1:{INTROSPECTION_ROOT / 'data' / 'activations' / 'qwen3_0_6b_all_layers.pt'}",
        f"hard_prompts_v1:{INTROSPECTION_ROOT / 'data' / 'activations' / 'qwen3_0_6b_hard_all_layers.pt'}",
        f"hard_prompts_v2:{INTROSPECTION_ROOT / 'data' / 'activations' / 'qwen3_0_6b_hard_v2_all_layers.pt'}",
        f"hard_prompts_v3:{INTROSPECTION_ROOT / 'data' / 'activations' / 'qwen3_0_6b_hard_v3_all_layers.pt'}",
    )


def _default_reference_feature_keys() -> tuple[str, ...]:
    return (
        "mean_pool_layer_18",
        "final_token_layer_11",
        "final_token_layer_16",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare combined-feature residuals against reference features.")
    parser.add_argument(
        "--dataset-artifact",
        required=False,
        action="append",
        help="Dataset/artifact pair in the form dataset_id:path. May be provided multiple times.",
    )
    parser.add_argument(
        "--reference-feature",
        required=False,
        action="append",
        help="Reference activation feature key. May be provided multiple times.",
    )
    parser.add_argument(
        "--candidate-feature",
        required=False,
        default="concat(final_token_layer_11,final_token_layer_16)",
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "combined_feature_residual_suite.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "combined_feature_residual_suite_summary.md"),
    )
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument("--folds", required=False, type=int, default=5)
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument("--max-iter", required=False, type=int, default=1000)
    parser.add_argument("--regularization-c", required=False, type=float, default=1.0)
    parser.add_argument("--word-ngram-min", required=False, type=int, default=1)
    parser.add_argument("--word-ngram-max", required=False, type=int, default=2)
    parser.add_argument("--char-ngram-min", required=False, type=int, default=3)
    parser.add_argument("--char-ngram-max", required=False, type=int, default=5)
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


def _parse_dataset_artifacts(values: Sequence[str] | None) -> tuple[DatasetArtifactSpec, ...]:
    raw_values = tuple(values) if values is not None else _default_dataset_artifacts()
    return tuple(_parse_dataset_artifact(value) for value in raw_values)


def _parse_reference_feature_keys(values: Sequence[str] | None) -> tuple[str, ...]:
    raw_values = tuple(values) if values is not None else _default_reference_feature_keys()
    if len(raw_values) == 0:
        raise ValueError("At least one reference feature key is required.")
    if len(set(raw_values)) != len(raw_values):
        raise ValueError("Reference feature keys must be unique.")
    return tuple(raw_values)


def _parse_args(argv: Sequence[str]) -> CombinedResidualSuiteScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return CombinedResidualSuiteScriptConfig(
        dataset_artifacts=_parse_dataset_artifacts(namespace.dataset_artifact),
        reference_feature_keys=_parse_reference_feature_keys(namespace.reference_feature),
        candidate_feature_key=str(namespace.candidate_feature),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        fold_count=int(namespace.folds),
        random_seed=int(namespace.seed),
        max_iter=int(namespace.max_iter),
        regularization_c=float(namespace.regularization_c),
        word_ngram_range=(int(namespace.word_ngram_min), int(namespace.word_ngram_max)),
        char_ngram_range=(int(namespace.char_ngram_min), int(namespace.char_ngram_max)),
    )


def _binary_task_config(config: CombinedResidualSuiteScriptConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.candidate_feature_key,
        word_ngram_range=config.word_ngram_range,
        char_ngram_range=config.char_ngram_range,
    )


def _feature_error_report(
    artifact: ActivationArtifact,
    config: BinaryTaskConfig,
    feature_key: str,
) -> BinaryErrorAnalysisReport:
    return evaluate_grouped_binary_error_analysis(
        artifact=artifact,
        config=replace(config, activation_feature_key=feature_key),
    )


def _suite_inputs(config: CombinedResidualSuiteScriptConfig) -> tuple[ResidualErrorSuiteInput, ...]:
    base_config = _binary_task_config(config)
    inputs: list[ResidualErrorSuiteInput] = []
    for spec in config.dataset_artifacts:
        artifact = load_activation_artifact(spec.artifact_path)
        candidate_report = _feature_error_report(
            artifact=artifact,
            config=base_config,
            feature_key=config.candidate_feature_key,
        )
        for reference_feature_key in config.reference_feature_keys:
            reference_report = _feature_error_report(
                artifact=artifact,
                config=base_config,
                feature_key=reference_feature_key,
            )
            inputs.append(
                ResidualErrorSuiteInput(
                    dataset_id=spec.dataset_id,
                    reference_report=reference_report,
                    candidate_report=candidate_report,
                )
            )
    return tuple(inputs)


def run_comparison(config: CombinedResidualSuiteScriptConfig) -> None:
    report = compare_binary_error_residual_suite(
        inputs=_suite_inputs(config),
        task_name=config.task_name,
        method_name="activation_probe",
    )
    write_residual_error_suite_json(config.output_json_path, report)
    write_residual_error_suite_markdown(config.output_markdown_path, report)

    print(f"Wrote residual suite to {config.output_json_path}")
    print(f"Wrote residual suite summary to {config.output_markdown_path}")
    for summary in report.feature_summaries:
        print(
            f"reference={summary.reference_feature_key} "
            f"comparisons={summary.comparison_count} "
            f"fixed={summary.fixed_error_count} "
            f"persistent={summary.persistent_error_count} "
            f"introduced={summary.introduced_error_count} "
            f"net_delta={summary.net_error_delta}"
        )


def main(argv: Sequence[str]) -> None:
    run_comparison(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
