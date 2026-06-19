from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, TypeAlias, cast

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.artifacts import load_activation_artifact
from aegis_introspection.binary_tasks import BinaryTaskConfig
from aegis_introspection.cift import last_quarter_readout_feature_keys
from aegis_introspection.cift_meta_head import (
    CiftMetaHeadComparisonDataset,
    CiftMetaHeadVariant,
    compare_grouped_cift_meta_head,
    write_cift_meta_head_json,
    write_cift_meta_head_markdown,
)


ReadoutVariant: TypeAlias = Literal["final_token", "mean_pool", "final_token_plus_mean_pool"]


@dataclass(frozen=True)
class DatasetArtifactSpec:
    dataset_id: str
    artifact_path: Path


@dataclass(frozen=True)
class CompareCiftMetaHeadScriptConfig:
    dataset_artifacts: tuple[DatasetArtifactSpec, ...]
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    baseline_feature_key: str
    readout_variants: tuple[ReadoutVariant, ...]
    calibration_source_labels: tuple[str, ...]
    risk_label: str
    ridge: float
    fold_count: int
    inner_fold_count: int
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
    parser = argparse.ArgumentParser(description="Compare an out-of-fold CIFT-like meta-head.")
    parser.add_argument(
        "--dataset-artifact",
        required=False,
        action="append",
        help="Dataset/artifact pair in the form dataset_id:path. May be provided multiple times.",
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "cift_meta_head_v1.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "cift_meta_head_v1_summary.md"),
    )
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument(
        "--baseline-feature",
        required=False,
        default="concat(final_token_layer_11,final_token_layer_16)",
    )
    parser.add_argument(
        "--readout-variant",
        required=False,
        action="append",
        choices=("final_token", "mean_pool", "final_token_plus_mean_pool"),
        help="Readout source set. Defaults to final_token and final_token_plus_mean_pool.",
    )
    parser.add_argument(
        "--calibration-source-label",
        required=False,
        action="append",
        help="Source label used as calibration rows. Defaults to secret_present_safe.",
    )
    parser.add_argument("--risk-label", required=False, default="exfiltration_intent")
    parser.add_argument("--ridge", required=False, type=float, default=0.001)
    parser.add_argument("--folds", required=False, type=int, default=5)
    parser.add_argument("--inner-folds", required=False, type=int, default=3)
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


def _parse_readout_variants(values: Sequence[str] | None) -> tuple[ReadoutVariant, ...]:
    raw_values = tuple(values) if values is not None else ("final_token", "final_token_plus_mean_pool")
    if len(raw_values) == 0:
        raise ValueError("At least one readout variant is required.")
    if len(set(raw_values)) != len(raw_values):
        raise ValueError("Readout variants must be unique.")
    parsed_values: list[ReadoutVariant] = []
    for value in raw_values:
        if value not in ("final_token", "mean_pool", "final_token_plus_mean_pool"):
            raise ValueError(f"Unsupported readout variant '{value}'.")
        parsed_values.append(cast(ReadoutVariant, value))
    return tuple(parsed_values)


def _parse_calibration_source_labels(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ("secret_present_safe",)
    parsed_values = tuple(value for value in values if value != "")
    if len(parsed_values) == 0:
        raise ValueError("At least one non-empty calibration source label is required.")
    if len(set(parsed_values)) != len(parsed_values):
        raise ValueError("Calibration source labels must be unique.")
    return parsed_values


def _parse_args(argv: Sequence[str]) -> CompareCiftMetaHeadScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return CompareCiftMetaHeadScriptConfig(
        dataset_artifacts=_parse_dataset_artifacts(namespace.dataset_artifact),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        baseline_feature_key=str(namespace.baseline_feature),
        readout_variants=_parse_readout_variants(namespace.readout_variant),
        calibration_source_labels=_parse_calibration_source_labels(namespace.calibration_source_label),
        risk_label=str(namespace.risk_label),
        ridge=float(namespace.ridge),
        fold_count=int(namespace.folds),
        inner_fold_count=int(namespace.inner_folds),
        random_seed=int(namespace.seed),
        max_iter=int(namespace.max_iter),
        regularization_c=float(namespace.regularization_c),
        word_ngram_range=(int(namespace.word_ngram_min), int(namespace.word_ngram_max)),
        char_ngram_range=(int(namespace.char_ngram_min), int(namespace.char_ngram_max)),
    )


def _binary_task_config(config: CompareCiftMetaHeadScriptConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.baseline_feature_key,
        word_ngram_range=config.word_ngram_range,
        char_ngram_range=config.char_ngram_range,
    )


def _load_datasets(config: CompareCiftMetaHeadScriptConfig) -> tuple[CiftMetaHeadComparisonDataset, ...]:
    return tuple(
        CiftMetaHeadComparisonDataset(
            dataset_id=spec.dataset_id,
            artifact=load_activation_artifact(spec.artifact_path),
        )
        for spec in config.dataset_artifacts
    )


def _variant_source_keys(
    dataset: CiftMetaHeadComparisonDataset,
    readout_variant: ReadoutVariant,
) -> tuple[str, ...]:
    final_token_keys = last_quarter_readout_feature_keys(dataset.artifact, "final_token")
    mean_pool_keys = last_quarter_readout_feature_keys(dataset.artifact, "mean_pool")
    if readout_variant == "final_token":
        return final_token_keys
    if readout_variant == "mean_pool":
        return mean_pool_keys
    if readout_variant == "final_token_plus_mean_pool":
        return final_token_keys + mean_pool_keys
    raise ValueError(f"Unsupported readout variant '{readout_variant}'.")


def _variant_feature_name(readout_variant: ReadoutVariant) -> str:
    if readout_variant == "final_token":
        return "cift_meta_oof_final_token_signed_residual"
    if readout_variant == "mean_pool":
        return "cift_meta_oof_mean_pool_signed_residual"
    if readout_variant == "final_token_plus_mean_pool":
        return "cift_meta_oof_final_token_mean_pool_signed_residual"
    raise ValueError(f"Unsupported readout variant '{readout_variant}'.")


def _variants(
    datasets: tuple[CiftMetaHeadComparisonDataset, ...],
    config: CompareCiftMetaHeadScriptConfig,
) -> tuple[CiftMetaHeadVariant, ...]:
    if len(datasets) == 0:
        raise ValueError("At least one dataset is required to resolve CIFT meta-head source features.")
    reference_dataset = datasets[0]
    return tuple(
        CiftMetaHeadVariant(
            variant_id=readout_variant,
            feature_name=_variant_feature_name(readout_variant),
            source_feature_keys=_variant_source_keys(reference_dataset, readout_variant),
            calibration_source_labels=config.calibration_source_labels,
            ridge=config.ridge,
            risk_label=config.risk_label,
            inner_fold_count=config.inner_fold_count,
            decision_rule="logistic_default",
        )
        for readout_variant in config.readout_variants
    )


def run_comparison(config: CompareCiftMetaHeadScriptConfig) -> None:
    datasets = _load_datasets(config)
    variants = _variants(datasets, config)
    report = compare_grouped_cift_meta_head(
        datasets=datasets,
        task_name=config.task_name,
        baseline_feature_key=config.baseline_feature_key,
        variants=variants,
        binary_config=_binary_task_config(config),
    )
    write_cift_meta_head_json(config.output_json_path, report)
    write_cift_meta_head_markdown(config.output_markdown_path, report)

    print(f"Wrote CIFT meta-head report to {config.output_json_path}")
    print(f"Wrote CIFT meta-head summary to {config.output_markdown_path}")
    print(
        f"meta_head_wins={report.meta_head_win_count} "
        f"baseline_wins={report.baseline_win_count} ties={report.tie_count}"
    )
    for dataset in report.datasets:
        print(
            f"{dataset.dataset_id}: best={dataset.best_variant.feature_name} "
            f"winner={dataset.winning_feature_key} "
            f"delta_macro_f1={dataset.macro_f1_delta:+.4f} "
            f"delta_accuracy={dataset.accuracy_delta:+.4f}"
        )


def main(argv: Sequence[str]) -> None:
    run_comparison(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
