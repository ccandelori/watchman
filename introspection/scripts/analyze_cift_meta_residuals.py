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
from aegis_introspection.binary_tasks import BinaryTaskConfig
from aegis_introspection.cift import last_quarter_readout_feature_keys
from aegis_introspection.cift_meta_head import CiftMetaHeadVariant
from aegis_introspection.cift_meta_residuals import (
    CiftMetaResidualDataset,
    compare_cift_meta_residual_suite,
    write_cift_meta_residual_suite_json,
    write_cift_meta_residual_suite_markdown,
)


@dataclass(frozen=True)
class DatasetArtifactSpec:
    dataset_id: str
    artifact_path: Path


@dataclass(frozen=True)
class AnalyzeCiftMetaResidualsScriptConfig:
    dataset_artifacts: tuple[DatasetArtifactSpec, ...]
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    baseline_feature_key: str
    calibration_source_labels: tuple[str, ...]
    variant_id: str
    feature_name: str
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
    parser = argparse.ArgumentParser(description="Analyze residual errors for the CIFT OOF meta-head.")
    parser.add_argument(
        "--dataset-artifact",
        required=False,
        action="append",
        help="Dataset/artifact pair in the form dataset_id:path. May be provided multiple times.",
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "cift_meta_head_residual_suite_v1.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "cift_meta_head_residual_suite_v1_summary.md"),
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
    parser.add_argument("--variant-id", required=False, default="final_token_plus_mean_pool")
    parser.add_argument(
        "--feature-name",
        required=False,
        default="cift_meta_oof_final_token_mean_pool_signed_residual",
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


def _parse_calibration_source_labels(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ("secret_present_safe",)
    parsed_values = tuple(value for value in values if value != "")
    if len(parsed_values) == 0:
        raise ValueError("At least one non-empty calibration source label is required.")
    if len(set(parsed_values)) != len(parsed_values):
        raise ValueError("Calibration source labels must be unique.")
    return parsed_values


def _parse_args(argv: Sequence[str]) -> AnalyzeCiftMetaResidualsScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return AnalyzeCiftMetaResidualsScriptConfig(
        dataset_artifacts=_parse_dataset_artifacts(namespace.dataset_artifact),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        baseline_feature_key=str(namespace.baseline_feature),
        calibration_source_labels=_parse_calibration_source_labels(namespace.calibration_source_label),
        variant_id=str(namespace.variant_id),
        feature_name=str(namespace.feature_name),
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


def _binary_task_config(config: AnalyzeCiftMetaResidualsScriptConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.baseline_feature_key,
        word_ngram_range=config.word_ngram_range,
        char_ngram_range=config.char_ngram_range,
    )


def _load_datasets(config: AnalyzeCiftMetaResidualsScriptConfig) -> tuple[CiftMetaResidualDataset, ...]:
    return tuple(
        CiftMetaResidualDataset(
            dataset_id=spec.dataset_id,
            artifact=load_activation_artifact(spec.artifact_path),
        )
        for spec in config.dataset_artifacts
    )


def _variant(
    datasets: tuple[CiftMetaResidualDataset, ...],
    config: AnalyzeCiftMetaResidualsScriptConfig,
) -> CiftMetaHeadVariant:
    if len(datasets) == 0:
        raise ValueError("At least one dataset is required to resolve CIFT meta-head source features.")
    reference_artifact = datasets[0].artifact
    final_token_keys = last_quarter_readout_feature_keys(reference_artifact, "final_token")
    mean_pool_keys = last_quarter_readout_feature_keys(reference_artifact, "mean_pool")
    return CiftMetaHeadVariant(
        variant_id=config.variant_id,
        feature_name=config.feature_name,
        source_feature_keys=final_token_keys + mean_pool_keys,
        calibration_source_labels=config.calibration_source_labels,
        ridge=config.ridge,
        risk_label=config.risk_label,
        inner_fold_count=config.inner_fold_count,
    )


def run_analysis(config: AnalyzeCiftMetaResidualsScriptConfig) -> None:
    datasets = _load_datasets(config)
    variant = _variant(datasets, config)
    report = compare_cift_meta_residual_suite(
        datasets=datasets,
        task_name=config.task_name,
        baseline_feature_key=config.baseline_feature_key,
        variant=variant,
        binary_config=_binary_task_config(config),
    )
    write_cift_meta_residual_suite_json(config.output_json_path, report)
    write_cift_meta_residual_suite_markdown(config.output_markdown_path, report)

    print(f"Wrote CIFT meta-head residual report to {config.output_json_path}")
    print(f"Wrote CIFT meta-head residual summary to {config.output_markdown_path}")
    print(
        f"reference_errors={report.reference_error_count} "
        f"candidate_errors={report.candidate_error_count} "
        f"fixed={report.fixed_error_count} persistent={report.persistent_error_count} "
        f"introduced={report.introduced_error_count} net_error_delta={report.net_error_delta}"
    )
    for item in report.comparisons:
        comparison = item.comparison
        print(
            f"{item.dataset_id}: reference_errors={comparison.reference_error_count} "
            f"candidate_errors={comparison.candidate_error_count} fixed={comparison.fixed_error_count} "
            f"persistent={comparison.persistent_error_count} introduced={comparison.introduced_error_count}"
        )


def main(argv: Sequence[str]) -> None:
    run_analysis(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
