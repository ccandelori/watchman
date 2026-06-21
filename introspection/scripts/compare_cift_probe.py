from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.artifacts import load_activation_artifact
from aegis_introspection.binary_tasks import BinaryTaskConfig
from aegis_introspection.cift import (
    CiftComparisonDataset,
    CiftProbeConfig,
    compare_grouped_cift_probe,
    last_quarter_readout_feature_keys,
    write_cift_probe_comparison_json,
    write_cift_probe_comparison_markdown,
)
from aegis_introspection.features import PoolingMethod


@dataclass(frozen=True)
class DatasetArtifactSpec:
    dataset_id: str
    artifact_path: Path


@dataclass(frozen=True)
class CompareCiftProbeScriptConfig:
    dataset_artifacts: tuple[DatasetArtifactSpec, ...]
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    baseline_feature_key: str
    source_feature_keys: tuple[str, ...] | None
    pooling_method: PoolingMethod
    calibration_source_labels: tuple[str, ...]
    cift_feature_key: str
    ridge: float
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a CIFT-like calibrated probe against a static feature.")
    parser.add_argument(
        "--dataset-artifact",
        required=False,
        action="append",
        help="Dataset/artifact pair in the form dataset_id:path. May be provided multiple times.",
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "cift_like_probe_comparison.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "cift_like_probe_comparison_summary.md"),
    )
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument(
        "--baseline-feature",
        required=False,
        default="concat(final_token_layer_11,final_token_layer_16)",
    )
    parser.add_argument(
        "--source-feature",
        required=False,
        action="append",
        help="CIFT source feature key. Defaults to final-token last-quarter layers from the first artifact.",
    )
    parser.add_argument("--pooling-method", required=False, choices=("final_token", "mean_pool"), default="final_token")
    parser.add_argument(
        "--calibration-source-label",
        required=False,
        action="append",
        help="Source label used as benign calibration rows. Defaults to secret_present_safe.",
    )
    parser.add_argument(
        "--cift-feature",
        required=False,
        default="cift_diag_final_token_last_quarter",
    )
    parser.add_argument("--ridge", required=False, type=float, default=0.001)
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


def _parse_optional_tuple(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    parsed_values = tuple(value for value in values if value != "")
    if len(parsed_values) == 0:
        raise ValueError("At least one non-empty value is required.")
    if len(set(parsed_values)) != len(parsed_values):
        raise ValueError("Values must be unique.")
    return parsed_values


def _parse_calibration_labels(values: Sequence[str] | None) -> tuple[str, ...]:
    parsed_values = _parse_optional_tuple(values)
    if parsed_values is None:
        return ("secret_present_safe",)
    return parsed_values


def _parse_args(argv: Sequence[str]) -> CompareCiftProbeScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return CompareCiftProbeScriptConfig(
        dataset_artifacts=_parse_dataset_artifacts(namespace.dataset_artifact),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        baseline_feature_key=str(namespace.baseline_feature),
        source_feature_keys=_parse_optional_tuple(namespace.source_feature),
        pooling_method=cast(PoolingMethod, namespace.pooling_method),
        calibration_source_labels=_parse_calibration_labels(namespace.calibration_source_label),
        cift_feature_key=str(namespace.cift_feature),
        ridge=float(namespace.ridge),
        fold_count=int(namespace.folds),
        random_seed=int(namespace.seed),
        max_iter=int(namespace.max_iter),
        regularization_c=float(namespace.regularization_c),
        word_ngram_range=(int(namespace.word_ngram_min), int(namespace.word_ngram_max)),
        char_ngram_range=(int(namespace.char_ngram_min), int(namespace.char_ngram_max)),
    )


def _binary_task_config(config: CompareCiftProbeScriptConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.baseline_feature_key,
        word_ngram_range=config.word_ngram_range,
        char_ngram_range=config.char_ngram_range,
    )


def _load_datasets(config: CompareCiftProbeScriptConfig) -> tuple[CiftComparisonDataset, ...]:
    return tuple(
        CiftComparisonDataset(
            dataset_id=spec.dataset_id,
            artifact=load_activation_artifact(spec.artifact_path),
        )
        for spec in config.dataset_artifacts
    )


def _source_feature_keys(
    datasets: tuple[CiftComparisonDataset, ...],
    config: CompareCiftProbeScriptConfig,
) -> tuple[str, ...]:
    if len(datasets) == 0:
        raise ValueError("At least one dataset is required to resolve CIFT source features.")
    if config.source_feature_keys is not None:
        return config.source_feature_keys
    return last_quarter_readout_feature_keys(datasets[0].artifact, config.pooling_method)


def _cift_config(
    datasets: tuple[CiftComparisonDataset, ...],
    config: CompareCiftProbeScriptConfig,
) -> CiftProbeConfig:
    return CiftProbeConfig(
        source_feature_keys=_source_feature_keys(datasets, config),
        calibration_source_labels=config.calibration_source_labels,
        ridge=config.ridge,
        output_feature_key=config.cift_feature_key,
    )


def run_comparison(config: CompareCiftProbeScriptConfig) -> None:
    datasets = _load_datasets(config)
    cift_config = _cift_config(datasets, config)
    report = compare_grouped_cift_probe(
        datasets=datasets,
        task_name=config.task_name,
        baseline_feature_key=config.baseline_feature_key,
        cift_config=cift_config,
        binary_config=_binary_task_config(config),
    )
    write_cift_probe_comparison_json(config.output_json_path, report)
    write_cift_probe_comparison_markdown(config.output_markdown_path, report)

    print(f"Wrote CIFT-like probe comparison to {config.output_json_path}")
    print(f"Wrote CIFT-like probe summary to {config.output_markdown_path}")
    print(
        f"cift_wins={report.cift_win_count} "
        f"baseline_wins={report.baseline_win_count} ties={report.tie_count}"
    )
    for dataset in report.datasets:
        print(
            f"{dataset.dataset_id}: winner={dataset.winning_feature_key} "
            f"delta_macro_f1={dataset.macro_f1_delta:+.4f} "
            f"delta_accuracy={dataset.accuracy_delta:+.4f}"
        )


def main(argv: Sequence[str]) -> None:
    run_comparison(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
