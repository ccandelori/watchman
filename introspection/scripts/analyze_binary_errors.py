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

from aegis_introspection.binary_tasks import BinaryTaskConfig
from aegis_introspection.error_analysis import (
    evaluate_selected_grouped_binary_error_analysis,
    evaluate_grouped_binary_error_analysis,
    write_binary_error_analysis_json,
    write_binary_error_analysis_markdown,
)
from aegis_introspection.sealed_holdout import (
    add_unseal_flag,
    assert_unsealed_paths,
    load_activation_artifact_with_unseal_policy,
)


@dataclass(frozen=True)
class AnalyzeBinaryErrorsScriptConfig:
    artifact_path: Path
    output_json_path: Path
    output_markdown_path: Path
    fold_count: int
    random_seed: int
    max_iter: int
    regularization_c: float
    activation_feature_key: str
    word_ngram_range: tuple[int, int]
    char_ngram_range: tuple[int, int]
    task_names: tuple[str, ...]
    allow_sealed_holdout: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze grouped binary task errors by prompt family.")
    parser.add_argument(
        "--artifact",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "activations" / "qwen3_0_6b_hard_all_layers.pt"),
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "binary_error_analysis_hard_grouped.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "binary_error_analysis_hard_grouped_summary.md"),
    )
    parser.add_argument("--activation-feature", required=False, default="mean_pool_layer_18")
    parser.add_argument("--folds", required=False, type=int, default=5)
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument("--max-iter", required=False, type=int, default=1000)
    parser.add_argument("--regularization-c", required=False, type=float, default=1.0)
    parser.add_argument("--word-ngram-min", required=False, type=int, default=1)
    parser.add_argument("--word-ngram-max", required=False, type=int, default=2)
    parser.add_argument("--char-ngram-min", required=False, type=int, default=3)
    parser.add_argument("--char-ngram-max", required=False, type=int, default=5)
    parser.add_argument(
        "--task",
        required=False,
        action="append",
        help="Binary task to analyze. May be provided multiple times. Defaults to all binary tasks.",
    )
    add_unseal_flag(parser)
    return parser


def _parse_args(argv: Sequence[str]) -> AnalyzeBinaryErrorsScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return AnalyzeBinaryErrorsScriptConfig(
        artifact_path=Path(namespace.artifact),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        fold_count=int(namespace.folds),
        random_seed=int(namespace.seed),
        max_iter=int(namespace.max_iter),
        regularization_c=float(namespace.regularization_c),
        activation_feature_key=str(namespace.activation_feature),
        word_ngram_range=(int(namespace.word_ngram_min), int(namespace.word_ngram_max)),
        char_ngram_range=(int(namespace.char_ngram_min), int(namespace.char_ngram_max)),
        task_names=_parse_task_names(namespace.task),
        allow_sealed_holdout=bool(namespace.allow_sealed_holdout),
    )


def _parse_task_names(values: list[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    task_names = tuple(value.strip() for value in values)
    if "" in task_names:
        raise ValueError("task names must not be empty.")
    if len(set(task_names)) != len(task_names):
        raise ValueError("task names must be unique.")
    return task_names


def _binary_task_config(config: AnalyzeBinaryErrorsScriptConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.activation_feature_key,
        word_ngram_range=config.word_ngram_range,
        char_ngram_range=config.char_ngram_range,
    )


def run_analysis(config: AnalyzeBinaryErrorsScriptConfig) -> None:
    assert_unsealed_paths(
        paths=(config.artifact_path, config.output_json_path, config.output_markdown_path),
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="binary error analysis",
    )
    artifact = load_activation_artifact_with_unseal_policy(
        path=config.artifact_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="binary error analysis",
    )
    if len(config.task_names) == 0:
        report = evaluate_grouped_binary_error_analysis(artifact, _binary_task_config(config))
    else:
        report = evaluate_selected_grouped_binary_error_analysis(
            artifact=artifact,
            config=_binary_task_config(config),
            task_names=config.task_names,
        )
    write_binary_error_analysis_json(config.output_json_path, report)
    write_binary_error_analysis_markdown(config.output_markdown_path, report)

    print(f"Wrote grouped binary error analysis to {config.output_json_path}")
    print(f"Wrote grouped binary error summary to {config.output_markdown_path}")
    for task in report.tasks:
        for method in task.methods:
            print(
                f"{task.task_name}/{method.method_name}: "
                f"errors={method.error_count} accuracy={method.accuracy:.4f}"
            )


def main(argv: Sequence[str]) -> None:
    run_analysis(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
