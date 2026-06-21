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

from aegis_introspection.adjudication import (
    build_residual_adjudication_report,
    write_residual_adjudication_json,
    write_residual_adjudication_markdown,
)
from aegis_introspection.artifacts import ActivationArtifact, load_activation_artifact
from aegis_introspection.binary_tasks import BinaryTaskConfig
from aegis_introspection.error_analysis import BinaryErrorAnalysisReport, evaluate_grouped_binary_error_analysis
from aegis_introspection.prompts import load_prompt_examples
from aegis_introspection.residual_error_comparison import compare_binary_error_residuals


@dataclass(frozen=True)
class CombinedHardV3RegressionAdjudicationScriptConfig:
    prompts_path: Path
    artifact_path: Path
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    reference_feature_key: str
    candidate_feature_key: str
    fold_count: int
    random_seed: int
    max_iter: int
    regularization_c: float
    word_ngram_range: tuple[int, int]
    char_ngram_range: tuple[int, int]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Hard V3 worksheet for combined-feature regressions.")
    parser.add_argument(
        "--prompts",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "prompts_hard_v3.jsonl"),
    )
    parser.add_argument(
        "--artifact",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "activations" / "qwen3_0_6b_hard_v3_all_layers.pt"),
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "hard_v3_combined_regression_adjudication.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(
            INTROSPECTION_ROOT / "data" / "reports" / "hard_v3_combined_regression_adjudication_summary.md"
        ),
    )
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument("--reference-feature", required=False, default="final_token_layer_16")
    parser.add_argument("--candidate-feature", required=False, default="concat(final_token_layer_11,final_token_layer_16)")
    parser.add_argument("--folds", required=False, type=int, default=5)
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument("--max-iter", required=False, type=int, default=1000)
    parser.add_argument("--regularization-c", required=False, type=float, default=1.0)
    parser.add_argument("--word-ngram-min", required=False, type=int, default=1)
    parser.add_argument("--word-ngram-max", required=False, type=int, default=2)
    parser.add_argument("--char-ngram-min", required=False, type=int, default=3)
    parser.add_argument("--char-ngram-max", required=False, type=int, default=5)
    return parser


def _parse_args(argv: Sequence[str]) -> CombinedHardV3RegressionAdjudicationScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return CombinedHardV3RegressionAdjudicationScriptConfig(
        prompts_path=Path(namespace.prompts),
        artifact_path=Path(namespace.artifact),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        reference_feature_key=str(namespace.reference_feature),
        candidate_feature_key=str(namespace.candidate_feature),
        fold_count=int(namespace.folds),
        random_seed=int(namespace.seed),
        max_iter=int(namespace.max_iter),
        regularization_c=float(namespace.regularization_c),
        word_ngram_range=(int(namespace.word_ngram_min), int(namespace.word_ngram_max)),
        char_ngram_range=(int(namespace.char_ngram_min), int(namespace.char_ngram_max)),
    )


def _binary_task_config(config: CombinedHardV3RegressionAdjudicationScriptConfig) -> BinaryTaskConfig:
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


def run_adjudication(config: CombinedHardV3RegressionAdjudicationScriptConfig) -> None:
    examples = load_prompt_examples(config.prompts_path)
    artifact = load_activation_artifact(config.artifact_path)
    base_config = _binary_task_config(config)
    reference_report = _feature_error_report(
        artifact=artifact,
        config=base_config,
        feature_key=config.reference_feature_key,
    )
    candidate_report = _feature_error_report(
        artifact=artifact,
        config=base_config,
        feature_key=config.candidate_feature_key,
    )
    residual_report = compare_binary_error_residuals(
        reference_report=reference_report,
        candidate_report=candidate_report,
        task_name=config.task_name,
        method_name="activation_probe",
    )
    adjudication_report = build_residual_adjudication_report(
        residual_report=residual_report,
        examples=examples,
    )
    write_residual_adjudication_json(config.output_json_path, adjudication_report)
    write_residual_adjudication_markdown(config.output_markdown_path, adjudication_report)

    print(f"Wrote residual adjudication case list to {config.output_json_path}")
    print(f"Wrote residual adjudication worksheet to {config.output_markdown_path}")
    print(
        f"{adjudication_report.task_name}/{adjudication_report.method_name}: "
        f"introduced_cases={adjudication_report.case_count}"
    )
    for summary in adjudication_report.family_summaries:
        print(f"{summary.family}: cases={summary.case_count}")


def main(argv: Sequence[str]) -> None:
    run_adjudication(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
