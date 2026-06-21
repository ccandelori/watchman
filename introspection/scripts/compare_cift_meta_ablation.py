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
from aegis_introspection.cift_meta_ablation import (
    CiftMetaAblationDataset,
    CiftMetaAblationVariant,
    compare_cift_meta_ablation,
    write_cift_meta_ablation_json,
    write_cift_meta_ablation_markdown,
)
from aegis_introspection.cift_meta_head import CiftMetaDecisionRule


CiftMetaSourceSet: TypeAlias = Literal["full_dual_readout", "early_dual_readout", "early_final_token"]
CiftMetaCalibrationSet: TypeAlias = Literal["safe_secret", "nonleaking"]


@dataclass(frozen=True)
class DatasetArtifactSpec:
    dataset_id: str
    artifact_path: Path


@dataclass(frozen=True)
class CompareCiftMetaAblationScriptConfig:
    dataset_artifacts: tuple[DatasetArtifactSpec, ...]
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    baseline_feature_key: str
    source_sets: tuple[CiftMetaSourceSet, ...]
    calibration_sets: tuple[CiftMetaCalibrationSet, ...]
    decision_rules: tuple[CiftMetaDecisionRule, ...]
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
        f"hard_prompts_v2:{INTROSPECTION_ROOT / 'data' / 'activations' / 'qwen3_0_6b_hard_v2_all_layers.pt'}",
        f"hard_prompts_v3:{INTROSPECTION_ROOT / 'data' / 'activations' / 'qwen3_0_6b_hard_v3_all_layers.pt'}",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact CIFT OOF meta-head ablation grid.")
    parser.add_argument(
        "--dataset-artifact",
        required=False,
        action="append",
        help="Dataset/artifact pair in the form dataset_id:path. May be provided multiple times.",
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "cift_meta_ablation_v1.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "cift_meta_ablation_v1_summary.md"),
    )
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument(
        "--baseline-feature",
        required=False,
        default="concat(final_token_layer_11,final_token_layer_16)",
    )
    parser.add_argument(
        "--source-set",
        required=False,
        action="append",
        choices=("full_dual_readout", "early_dual_readout", "early_final_token"),
        help="Source subset. Defaults to all supported source sets.",
    )
    parser.add_argument(
        "--calibration-set",
        required=False,
        action="append",
        choices=("safe_secret", "nonleaking"),
        help="Calibration label set. Defaults to safe_secret and nonleaking.",
    )
    parser.add_argument(
        "--decision-rule",
        required=False,
        action="append",
        choices=("logistic_default", "train_f1_threshold"),
        help="Meta-head decision rule. Defaults to both supported rules.",
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


def _parse_source_sets(values: Sequence[str] | None) -> tuple[CiftMetaSourceSet, ...]:
    raw_values = tuple(values) if values is not None else ("full_dual_readout", "early_dual_readout", "early_final_token")
    if len(raw_values) == 0:
        raise ValueError("At least one source set is required.")
    if len(set(raw_values)) != len(raw_values):
        raise ValueError("Source sets must be unique.")
    return tuple(cast(CiftMetaSourceSet, value) for value in raw_values)


def _parse_calibration_sets(values: Sequence[str] | None) -> tuple[CiftMetaCalibrationSet, ...]:
    raw_values = tuple(values) if values is not None else ("safe_secret", "nonleaking")
    if len(raw_values) == 0:
        raise ValueError("At least one calibration set is required.")
    if len(set(raw_values)) != len(raw_values):
        raise ValueError("Calibration sets must be unique.")
    return tuple(cast(CiftMetaCalibrationSet, value) for value in raw_values)


def _parse_decision_rules(values: Sequence[str] | None) -> tuple[CiftMetaDecisionRule, ...]:
    raw_values = tuple(values) if values is not None else ("logistic_default", "train_f1_threshold")
    if len(raw_values) == 0:
        raise ValueError("At least one decision rule is required.")
    if len(set(raw_values)) != len(raw_values):
        raise ValueError("Decision rules must be unique.")
    return tuple(cast(CiftMetaDecisionRule, value) for value in raw_values)


def _parse_args(argv: Sequence[str]) -> CompareCiftMetaAblationScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return CompareCiftMetaAblationScriptConfig(
        dataset_artifacts=_parse_dataset_artifacts(namespace.dataset_artifact),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        baseline_feature_key=str(namespace.baseline_feature),
        source_sets=_parse_source_sets(namespace.source_set),
        calibration_sets=_parse_calibration_sets(namespace.calibration_set),
        decision_rules=_parse_decision_rules(namespace.decision_rule),
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


def _binary_task_config(config: CompareCiftMetaAblationScriptConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.baseline_feature_key,
        word_ngram_range=config.word_ngram_range,
        char_ngram_range=config.char_ngram_range,
    )


def _load_datasets(config: CompareCiftMetaAblationScriptConfig) -> tuple[CiftMetaAblationDataset, ...]:
    return tuple(
        CiftMetaAblationDataset(
            dataset_id=spec.dataset_id,
            artifact=load_activation_artifact(spec.artifact_path),
        )
        for spec in config.dataset_artifacts
    )


def _source_feature_keys(
    dataset: CiftMetaAblationDataset,
    source_set: CiftMetaSourceSet,
) -> tuple[str, ...]:
    final_token_keys = last_quarter_readout_feature_keys(dataset.artifact, "final_token")
    mean_pool_keys = last_quarter_readout_feature_keys(dataset.artifact, "mean_pool")
    early_count = min(4, len(final_token_keys), len(mean_pool_keys))
    if source_set == "full_dual_readout":
        return final_token_keys + mean_pool_keys
    if source_set == "early_dual_readout":
        return final_token_keys[:early_count] + mean_pool_keys[:early_count]
    if source_set == "early_final_token":
        return final_token_keys[:early_count]
    raise ValueError(f"Unsupported source set '{source_set}'.")


def _calibration_source_labels(calibration_set: CiftMetaCalibrationSet) -> tuple[str, ...]:
    if calibration_set == "safe_secret":
        return ("secret_present_safe",)
    if calibration_set == "nonleaking":
        return ("benign", "secret_present_safe")
    raise ValueError(f"Unsupported calibration set '{calibration_set}'.")


def _variants(
    datasets: tuple[CiftMetaAblationDataset, ...],
    config: CompareCiftMetaAblationScriptConfig,
) -> tuple[CiftMetaAblationVariant, ...]:
    if len(datasets) == 0:
        raise ValueError("At least one dataset is required to resolve CIFT meta ablation source features.")
    reference_dataset = datasets[0]
    variants: list[CiftMetaAblationVariant] = []
    for source_set in config.source_sets:
        source_feature_keys = _source_feature_keys(reference_dataset, source_set)
        for calibration_set in config.calibration_sets:
            calibration_source_labels = _calibration_source_labels(calibration_set)
            for decision_rule in config.decision_rules:
                variant_id = f"{source_set}_{calibration_set}_{decision_rule}"
                variants.append(
                    CiftMetaAblationVariant(
                        variant_id=variant_id,
                        feature_name=f"cift_meta_ablation_{variant_id}",
                        source_feature_keys=source_feature_keys,
                        calibration_source_labels=calibration_source_labels,
                        ridge=config.ridge,
                        risk_label=config.risk_label,
                        inner_fold_count=config.inner_fold_count,
                        decision_rule=decision_rule,
                    )
                )
    return tuple(variants)


def run_comparison(config: CompareCiftMetaAblationScriptConfig) -> None:
    datasets = _load_datasets(config)
    variants = _variants(datasets, config)
    report = compare_cift_meta_ablation(
        datasets=datasets,
        task_name=config.task_name,
        baseline_feature_key=config.baseline_feature_key,
        variants=variants,
        binary_config=_binary_task_config(config),
    )
    write_cift_meta_ablation_json(config.output_json_path, report)
    write_cift_meta_ablation_markdown(config.output_markdown_path, report)

    print(f"Wrote CIFT meta-head ablation report to {config.output_json_path}")
    print(f"Wrote CIFT meta-head ablation summary to {config.output_markdown_path}")
    print(
        f"variants={report.variant_count} datasets={report.dataset_count} "
        f"best={report.best_variant_summary.variant_id} "
        f"best_candidate_errors={report.best_variant_summary.candidate_error_count} "
        f"best_fixed={report.best_variant_summary.fixed_error_count} "
        f"best_introduced={report.best_variant_summary.introduced_error_count} "
        f"best_net_error_delta={report.best_variant_summary.net_error_delta}"
    )


def main(argv: Sequence[str]) -> None:
    run_comparison(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
