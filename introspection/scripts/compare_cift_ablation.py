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
from aegis_introspection.cift import last_quarter_readout_feature_keys
from aegis_introspection.cift_ablation import (
    CiftAblationClassifierMode,
    CiftAblationDataset,
    CiftAblationRepresentation,
    CiftAblationVariant,
    compare_grouped_cift_ablation,
    write_cift_ablation_json,
    write_cift_ablation_markdown,
)
from aegis_introspection.features import PoolingMethod


@dataclass(frozen=True)
class DatasetArtifactSpec:
    dataset_id: str
    artifact_path: Path


@dataclass(frozen=True)
class CompareCiftAblationScriptConfig:
    dataset_artifacts: tuple[DatasetArtifactSpec, ...]
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    baseline_feature_key: str
    pooling_methods: tuple[PoolingMethod, ...]
    representations: tuple[CiftAblationRepresentation, ...]
    classifier_modes: tuple[CiftAblationClassifierMode, ...]
    calibration_sets: tuple[str, ...]
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
    parser = argparse.ArgumentParser(description="Run a CIFT-like ablation sweep against a fixed baseline feature.")
    parser.add_argument(
        "--dataset-artifact",
        required=False,
        action="append",
        help="Dataset/artifact pair in the form dataset_id:path. May be provided multiple times.",
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "cift_like_ablation_v3.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "cift_like_ablation_v3_summary.md"),
    )
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument(
        "--baseline-feature",
        required=False,
        default="concat(final_token_layer_11,final_token_layer_16)",
    )
    parser.add_argument(
        "--pooling-method",
        required=False,
        action="append",
        choices=("final_token", "mean_pool"),
        help="Pooling/readout proxy to ablate. Defaults to final_token and mean_pool.",
    )
    parser.add_argument(
        "--representation",
        required=False,
        action="append",
        choices=("diagonal_distance", "standardized_residual_concat", "absolute_standardized_residual_concat"),
        help="CIFT-like row representation. Defaults to all supported representations.",
    )
    parser.add_argument(
        "--classifier-mode",
        required=False,
        action="append",
        choices=("standard_scaled_logreg", "raw_logreg"),
        help="Classifier mode for CIFT-like features. Defaults to both supported modes.",
    )
    parser.add_argument(
        "--calibration-set",
        required=False,
        action="append",
        choices=("safe_secret", "nonleaking"),
        help="Calibration label set. Defaults to safe_secret and nonleaking.",
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


def _unique_pooling_methods(values: Sequence[str] | None) -> tuple[PoolingMethod, ...]:
    raw_values = tuple(values) if values is not None else ("final_token", "mean_pool")
    if len(raw_values) == 0:
        raise ValueError("At least one pooling method is required.")
    if len(set(raw_values)) != len(raw_values):
        raise ValueError("Pooling methods must be unique.")
    return tuple(cast(PoolingMethod, value) for value in raw_values)


def _unique_representations(values: Sequence[str] | None) -> tuple[CiftAblationRepresentation, ...]:
    raw_values = (
        tuple(values)
        if values is not None
        else ("diagonal_distance", "standardized_residual_concat", "absolute_standardized_residual_concat")
    )
    if len(raw_values) == 0:
        raise ValueError("At least one representation is required.")
    if len(set(raw_values)) != len(raw_values):
        raise ValueError("Representations must be unique.")
    return tuple(cast(CiftAblationRepresentation, value) for value in raw_values)


def _unique_classifier_modes(values: Sequence[str] | None) -> tuple[CiftAblationClassifierMode, ...]:
    raw_values = tuple(values) if values is not None else ("standard_scaled_logreg", "raw_logreg")
    if len(raw_values) == 0:
        raise ValueError("At least one classifier mode is required.")
    if len(set(raw_values)) != len(raw_values):
        raise ValueError("Classifier modes must be unique.")
    return tuple(cast(CiftAblationClassifierMode, value) for value in raw_values)


def _unique_calibration_sets(values: Sequence[str] | None) -> tuple[str, ...]:
    raw_values = tuple(values) if values is not None else ("safe_secret", "nonleaking")
    if len(raw_values) == 0:
        raise ValueError("At least one calibration set is required.")
    if len(set(raw_values)) != len(raw_values):
        raise ValueError("Calibration sets must be unique.")
    return raw_values


def _parse_args(argv: Sequence[str]) -> CompareCiftAblationScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return CompareCiftAblationScriptConfig(
        dataset_artifacts=_parse_dataset_artifacts(namespace.dataset_artifact),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        baseline_feature_key=str(namespace.baseline_feature),
        pooling_methods=_unique_pooling_methods(namespace.pooling_method),
        representations=_unique_representations(namespace.representation),
        classifier_modes=_unique_classifier_modes(namespace.classifier_mode),
        calibration_sets=_unique_calibration_sets(namespace.calibration_set),
        ridge=float(namespace.ridge),
        fold_count=int(namespace.folds),
        random_seed=int(namespace.seed),
        max_iter=int(namespace.max_iter),
        regularization_c=float(namespace.regularization_c),
        word_ngram_range=(int(namespace.word_ngram_min), int(namespace.word_ngram_max)),
        char_ngram_range=(int(namespace.char_ngram_min), int(namespace.char_ngram_max)),
    )


def _binary_task_config(config: CompareCiftAblationScriptConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.baseline_feature_key,
        word_ngram_range=config.word_ngram_range,
        char_ngram_range=config.char_ngram_range,
    )


def _load_datasets(config: CompareCiftAblationScriptConfig) -> tuple[CiftAblationDataset, ...]:
    return tuple(
        CiftAblationDataset(
            dataset_id=spec.dataset_id,
            artifact=load_activation_artifact(spec.artifact_path),
        )
        for spec in config.dataset_artifacts
    )


def _calibration_source_labels(calibration_set: str) -> tuple[str, ...]:
    if calibration_set == "safe_secret":
        return ("secret_present_safe",)
    if calibration_set == "nonleaking":
        return ("benign", "secret_present_safe")
    raise ValueError(f"Unsupported calibration set '{calibration_set}'.")


def _representation_name(representation: CiftAblationRepresentation) -> str:
    if representation == "diagonal_distance":
        return "diag"
    if representation == "standardized_residual_concat":
        return "residual"
    if representation == "absolute_standardized_residual_concat":
        return "abs_residual"
    raise ValueError(f"Unsupported representation '{representation}'.")


def _classifier_mode_name(classifier_mode: CiftAblationClassifierMode) -> str:
    if classifier_mode == "standard_scaled_logreg":
        return "scaled"
    if classifier_mode == "raw_logreg":
        return "raw"
    raise ValueError(f"Unsupported classifier mode '{classifier_mode}'.")


def _variants(
    datasets: tuple[CiftAblationDataset, ...],
    config: CompareCiftAblationScriptConfig,
) -> tuple[CiftAblationVariant, ...]:
    if len(datasets) == 0:
        raise ValueError("At least one dataset is required to resolve default CIFT ablation variants.")

    variants: list[CiftAblationVariant] = []
    for pooling_method in config.pooling_methods:
        source_feature_keys = last_quarter_readout_feature_keys(datasets[0].artifact, pooling_method)
        for representation in config.representations:
            representation_name = _representation_name(representation)
            for classifier_mode in config.classifier_modes:
                classifier_mode_name = _classifier_mode_name(classifier_mode)
                for calibration_set in config.calibration_sets:
                    variant_id = (
                        f"{representation_name}_{classifier_mode_name}_{calibration_set}_"
                        f"{pooling_method}_last_quarter"
                    )
                    variants.append(
                        CiftAblationVariant(
                            variant_id=variant_id,
                            feature_name=f"cift_{variant_id}",
                            source_feature_keys=source_feature_keys,
                            calibration_source_labels=_calibration_source_labels(calibration_set),
                            representation=representation,
                            classifier_mode=classifier_mode,
                            ridge=config.ridge,
                        )
                    )
    return tuple(variants)


def run_comparison(config: CompareCiftAblationScriptConfig) -> None:
    datasets = _load_datasets(config)
    variants = _variants(datasets, config)
    report = compare_grouped_cift_ablation(
        datasets=datasets,
        task_name=config.task_name,
        baseline_feature_key=config.baseline_feature_key,
        variants=variants,
        binary_config=_binary_task_config(config),
    )
    write_cift_ablation_json(config.output_json_path, report)
    write_cift_ablation_markdown(config.output_markdown_path, report)

    print(f"Wrote CIFT-like ablation report to {config.output_json_path}")
    print(f"Wrote CIFT-like ablation summary to {config.output_markdown_path}")
    print(
        f"ablation_wins={report.ablation_win_count} "
        f"baseline_wins={report.baseline_win_count} ties={report.tie_count}"
    )
    for dataset in report.datasets:
        print(
            f"{dataset.dataset_id}: best={dataset.best_variant.feature_name} "
            f"winner={dataset.winning_feature_key} "
            f"delta_macro_f1={dataset.macro_f1_delta:+.4f}"
        )


def main(argv: Sequence[str]) -> None:
    run_comparison(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
