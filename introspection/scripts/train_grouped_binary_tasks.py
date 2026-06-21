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
from aegis_introspection.binary_tasks import (
    BinaryTaskConfig,
    evaluate_grouped_binary_tasks,
    write_binary_tasks_markdown,
    write_binary_tasks_report_json,
)


@dataclass(frozen=True)
class TrainGroupedBinaryTasksScriptConfig:
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate grouped binary Aegis detector tasks.")
    parser.add_argument(
        "--artifact",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "activations" / "qwen3_0_6b_all_layers.pt"),
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "binary_tasks_grouped.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "binary_tasks_grouped_summary.md"),
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
    return parser


def _parse_args(argv: Sequence[str]) -> TrainGroupedBinaryTasksScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return TrainGroupedBinaryTasksScriptConfig(
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
    )


def _binary_task_config(config: TrainGroupedBinaryTasksScriptConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.activation_feature_key,
        word_ngram_range=config.word_ngram_range,
        char_ngram_range=config.char_ngram_range,
    )


def run_training(config: TrainGroupedBinaryTasksScriptConfig) -> None:
    artifact = load_activation_artifact(config.artifact_path)
    report = evaluate_grouped_binary_tasks(artifact, _binary_task_config(config))
    write_binary_tasks_report_json(config.output_json_path, report)
    write_binary_tasks_markdown(config.output_markdown_path, report)

    print(f"Wrote grouped binary task report to {config.output_json_path}")
    print(f"Wrote grouped binary task summary to {config.output_markdown_path}")
    for task in report.tasks:
        best_method = max(task.methods, key=lambda method: (method.macro_f1_mean, method.accuracy_mean))
        print(
            f"{task.task_name}: best={best_method.method_name} "
            f"macro_f1={best_method.macro_f1_mean:.4f} accuracy={best_method.accuracy_mean:.4f}"
        )


def main(argv: Sequence[str]) -> None:
    run_training(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
