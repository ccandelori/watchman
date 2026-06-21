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

from aegis_introspection.artifacts import load_activation_artifact
from aegis_introspection.binary_feature_crosscheck import FeatureCrosscheckDataset
from aegis_introspection.binary_feature_stability import (
    compare_grouped_binary_feature_stability,
    write_feature_stability_json,
    write_feature_stability_markdown,
)
from aegis_introspection.binary_tasks import BinaryTaskConfig


@dataclass(frozen=True)
class DatasetArtifactSpec:
    dataset_id: str
    artifact_path: Path


@dataclass(frozen=True)
class CompareFeatureStabilityScriptConfig:
    dataset_artifacts: tuple[DatasetArtifactSpec, ...]
    feature_keys: tuple[str, ...]
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


def _default_feature_keys() -> tuple[str, ...]:
    return (
        "mean_pool_layer_18",
        "final_token_layer_11",
        "final_token_layer_16",
        "concat(final_token_layer_11,final_token_layer_16)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multiple activation features across grouped datasets.")
    parser.add_argument(
        "--dataset-artifact",
        required=False,
        action="append",
        help="Dataset/artifact pair in the form dataset_id:path. May be provided multiple times.",
    )
    parser.add_argument(
        "--feature",
        required=False,
        action="append",
        help="Activation feature key to compare. May be provided multiple times.",
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "feature_stability_combined_l11_l16.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "feature_stability_combined_l11_l16_summary.md"),
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


def _parse_feature_keys(values: Sequence[str] | None) -> tuple[str, ...]:
    raw_values = tuple(values) if values is not None else _default_feature_keys()
    if len(raw_values) == 0:
        raise ValueError("At least one feature key is required.")
    return tuple(raw_values)


def _parse_args(argv: Sequence[str]) -> CompareFeatureStabilityScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return CompareFeatureStabilityScriptConfig(
        dataset_artifacts=_parse_dataset_artifacts(namespace.dataset_artifact),
        feature_keys=_parse_feature_keys(namespace.feature),
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


def _binary_task_config(config: CompareFeatureStabilityScriptConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.feature_keys[0],
        word_ngram_range=config.word_ngram_range,
        char_ngram_range=config.char_ngram_range,
    )


def _load_datasets(config: CompareFeatureStabilityScriptConfig) -> tuple[FeatureCrosscheckDataset, ...]:
    return tuple(
        FeatureCrosscheckDataset(
            dataset_id=spec.dataset_id,
            artifact=load_activation_artifact(spec.artifact_path),
        )
        for spec in config.dataset_artifacts
    )


def run_comparison(config: CompareFeatureStabilityScriptConfig) -> None:
    report = compare_grouped_binary_feature_stability(
        datasets=_load_datasets(config),
        task_name=config.task_name,
        feature_keys=config.feature_keys,
        config=_binary_task_config(config),
    )
    write_feature_stability_json(config.output_json_path, report)
    write_feature_stability_markdown(config.output_markdown_path, report)

    print(f"Wrote feature stability report to {config.output_json_path}")
    print(f"Wrote feature stability summary to {config.output_markdown_path}")
    for summary in report.feature_summaries:
        print(
            f"rank={summary.rank} feature={summary.feature_key} wins={summary.win_count} "
            f"mean_macro_f1={summary.mean_macro_f1:.4f} "
            f"min_macro_f1={summary.min_macro_f1:.4f} "
            f"range={summary.macro_f1_range:.4f}"
        )


def main(argv: Sequence[str]) -> None:
    run_comparison(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
