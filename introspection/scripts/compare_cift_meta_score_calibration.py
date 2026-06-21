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
from aegis_introspection.cift_meta_score_calibration import (
    CiftMetaScoreCalibrationDataset,
    CiftMetaScoreCalibrationRule,
    CiftMetaScoreCalibrationVariant,
    compare_cift_meta_score_calibration,
    write_cift_meta_score_calibration_json,
    write_cift_meta_score_calibration_markdown,
)


@dataclass(frozen=True)
class DatasetArtifactSpec:
    dataset_id: str
    artifact_path: Path


@dataclass(frozen=True)
class CompareCiftMetaScoreCalibrationScriptConfig:
    dataset_artifacts: tuple[DatasetArtifactSpec, ...]
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    baseline_feature_key: str
    calibration_source_labels: tuple[str, ...]
    score_calibration_rules: tuple[CiftMetaScoreCalibrationRule, ...]
    meta_regularization_c: float
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
    parser = argparse.ArgumentParser(description="Run CIFT meta-head source-score calibration comparisons.")
    parser.add_argument(
        "--dataset-artifact",
        required=False,
        action="append",
        help="Dataset/artifact pair in the form dataset_id:path. May be provided multiple times.",
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "cift_meta_score_calibration_v1.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "cift_meta_score_calibration_v1_summary.md"),
    )
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument(
        "--baseline-feature",
        required=False,
        default="concat(final_token_layer_11,final_token_layer_16)",
    )
    parser.add_argument(
        "--calibration-source-label",
        required=False,
        action="append",
        help="Source label used as calibration rows. Defaults to secret_present_safe.",
    )
    parser.add_argument(
        "--score-calibration-rule",
        required=False,
        action="append",
        choices=("raw_probability", "clipped_logit", "platt_probability"),
        help="Source-score calibration rule. Defaults to all supported rules.",
    )
    parser.add_argument("--meta-regularization-c", required=False, type=float, default=10.0)
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


def _parse_calibration_source_labels(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ("secret_present_safe",)
    parsed_values = tuple(value for value in values if value != "")
    if len(parsed_values) == 0:
        raise ValueError("At least one non-empty calibration source label is required.")
    if len(set(parsed_values)) != len(parsed_values):
        raise ValueError("Calibration source labels must be unique.")
    return parsed_values


def _parse_score_calibration_rules(values: Sequence[str] | None) -> tuple[CiftMetaScoreCalibrationRule, ...]:
    raw_values = tuple(values) if values is not None else (
        "raw_probability",
        "clipped_logit",
        "platt_probability",
    )
    if len(raw_values) == 0:
        raise ValueError("At least one score calibration rule is required.")
    if len(set(raw_values)) != len(raw_values):
        raise ValueError("Score calibration rules must be unique.")
    return tuple(cast(CiftMetaScoreCalibrationRule, value) for value in raw_values)


def _parse_args(argv: Sequence[str]) -> CompareCiftMetaScoreCalibrationScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return CompareCiftMetaScoreCalibrationScriptConfig(
        dataset_artifacts=_parse_dataset_artifacts(namespace.dataset_artifact),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        baseline_feature_key=str(namespace.baseline_feature),
        calibration_source_labels=_parse_calibration_source_labels(namespace.calibration_source_label),
        score_calibration_rules=_parse_score_calibration_rules(namespace.score_calibration_rule),
        meta_regularization_c=float(namespace.meta_regularization_c),
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


def _binary_task_config(config: CompareCiftMetaScoreCalibrationScriptConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.baseline_feature_key,
        word_ngram_range=config.word_ngram_range,
        char_ngram_range=config.char_ngram_range,
    )


def _load_datasets(config: CompareCiftMetaScoreCalibrationScriptConfig) -> tuple[CiftMetaScoreCalibrationDataset, ...]:
    return tuple(
        CiftMetaScoreCalibrationDataset(
            dataset_id=spec.dataset_id,
            artifact=load_activation_artifact(spec.artifact_path),
        )
        for spec in config.dataset_artifacts
    )


def _source_feature_keys(dataset: CiftMetaScoreCalibrationDataset) -> tuple[str, ...]:
    return (
        last_quarter_readout_feature_keys(dataset.artifact, "final_token")
        + last_quarter_readout_feature_keys(dataset.artifact, "mean_pool")
    )


def _variants(
    datasets: tuple[CiftMetaScoreCalibrationDataset, ...],
    config: CompareCiftMetaScoreCalibrationScriptConfig,
) -> tuple[CiftMetaScoreCalibrationVariant, ...]:
    if len(datasets) == 0:
        raise ValueError("At least one dataset is required to resolve CIFT score calibration source features.")
    source_feature_keys = _source_feature_keys(datasets[0])
    return tuple(
        CiftMetaScoreCalibrationVariant(
            variant_id=score_calibration_rule,
            feature_name=f"cift_meta_score_calibration_{score_calibration_rule}",
            source_feature_keys=source_feature_keys,
            calibration_source_labels=config.calibration_source_labels,
            ridge=config.ridge,
            risk_label=config.risk_label,
            inner_fold_count=config.inner_fold_count,
            meta_regularization_c=config.meta_regularization_c,
            score_calibration_rule=score_calibration_rule,
        )
        for score_calibration_rule in config.score_calibration_rules
    )


def run_comparison(config: CompareCiftMetaScoreCalibrationScriptConfig) -> None:
    datasets = _load_datasets(config)
    variants = _variants(datasets=datasets, config=config)
    report = compare_cift_meta_score_calibration(
        datasets=datasets,
        task_name=config.task_name,
        baseline_feature_key=config.baseline_feature_key,
        variants=variants,
        binary_config=_binary_task_config(config),
    )
    write_cift_meta_score_calibration_json(config.output_json_path, report)
    write_cift_meta_score_calibration_markdown(config.output_markdown_path, report)

    print(f"Wrote CIFT meta-head source-score calibration report to {config.output_json_path}")
    print(f"Wrote CIFT meta-head source-score calibration summary to {config.output_markdown_path}")
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
