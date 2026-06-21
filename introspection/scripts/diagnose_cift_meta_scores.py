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

from aegis_introspection.artifacts import ActivationArtifact, load_activation_artifact
from aegis_introspection.binary_tasks import BinaryTaskConfig
from aegis_introspection.cift import last_quarter_readout_feature_keys
from aegis_introspection.cift_meta_head import CiftMetaHeadVariant
from aegis_introspection.cift_meta_score_diagnostics import (
    CiftMetaScoreDiagnosticDataset,
    diagnose_cift_meta_introduced_errors,
    write_cift_meta_score_diagnostics_json,
    write_cift_meta_score_diagnostics_markdown,
)


@dataclass(frozen=True)
class DiagnoseCiftMetaScoresScriptConfig:
    dataset_id: str
    artifact_path: Path
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    baseline_feature_key: str
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose CIFT meta-head introduced errors with source scores.")
    parser.add_argument("--dataset-id", required=False, default="hard_prompts_v2")
    parser.add_argument(
        "--artifact",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "activations" / "qwen3_0_6b_hard_v2_all_layers.pt"),
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "cift_meta_score_diagnostics_hard_v2_v1.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "cift_meta_score_diagnostics_hard_v2_v1_summary.md"),
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


def _parse_calibration_source_labels(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ("secret_present_safe",)
    parsed_values = tuple(value for value in values if value != "")
    if len(parsed_values) == 0:
        raise ValueError("At least one non-empty calibration source label is required.")
    if len(set(parsed_values)) != len(parsed_values):
        raise ValueError("Calibration source labels must be unique.")
    return parsed_values


def _parse_args(argv: Sequence[str]) -> DiagnoseCiftMetaScoresScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return DiagnoseCiftMetaScoresScriptConfig(
        dataset_id=str(namespace.dataset_id),
        artifact_path=Path(namespace.artifact),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        baseline_feature_key=str(namespace.baseline_feature),
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


def _binary_task_config(config: DiagnoseCiftMetaScoresScriptConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.baseline_feature_key,
        word_ngram_range=config.word_ngram_range,
        char_ngram_range=config.char_ngram_range,
    )


def _variant(artifact: ActivationArtifact, config: DiagnoseCiftMetaScoresScriptConfig) -> CiftMetaHeadVariant:
    source_feature_keys = (
        last_quarter_readout_feature_keys(artifact, "final_token")
        + last_quarter_readout_feature_keys(artifact, "mean_pool")
    )
    return CiftMetaHeadVariant(
        variant_id="final_token_plus_mean_pool",
        feature_name="cift_meta_oof_final_token_mean_pool_signed_residual",
        source_feature_keys=source_feature_keys,
        calibration_source_labels=config.calibration_source_labels,
        ridge=config.ridge,
        risk_label=config.risk_label,
        inner_fold_count=config.inner_fold_count,
        decision_rule="logistic_default",
    )


def run_diagnosis(config: DiagnoseCiftMetaScoresScriptConfig) -> None:
    artifact = load_activation_artifact(config.artifact_path)
    variant = _variant(artifact, config)
    report = diagnose_cift_meta_introduced_errors(
        dataset=CiftMetaScoreDiagnosticDataset(dataset_id=config.dataset_id, artifact=artifact),
        task_name=config.task_name,
        baseline_feature_key=config.baseline_feature_key,
        variant=variant,
        binary_config=_binary_task_config(config),
    )
    write_cift_meta_score_diagnostics_json(config.output_json_path, report)
    write_cift_meta_score_diagnostics_markdown(config.output_markdown_path, report)

    print(f"Wrote CIFT meta-head score diagnostics to {config.output_json_path}")
    print(f"Wrote CIFT meta-head score diagnostics summary to {config.output_markdown_path}")
    print(
        f"reference_errors={report.reference_error_count} "
        f"candidate_errors={report.candidate_error_count} "
        f"introduced_errors={report.introduced_error_count}"
    )
    for error in report.introduced_errors:
        print(
            f"{error.example_id}: true={error.true_label} "
            f"reference={error.reference_predicted_label} "
            f"candidate={error.candidate_predicted_label} "
            f"meta_risk={error.meta_risk_score:.4f} "
            f"threshold={error.decision_threshold:.4f}"
        )


def main(argv: Sequence[str]) -> None:
    run_diagnosis(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
