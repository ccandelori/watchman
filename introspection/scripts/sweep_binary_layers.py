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
from aegis_introspection.binary_layer_sweep import (
    evaluate_grouped_binary_layer_sweep,
    write_binary_layer_sweep_json,
    write_binary_layer_sweep_markdown,
)
from aegis_introspection.binary_tasks import BinaryTaskConfig


@dataclass(frozen=True)
class SweepBinaryLayersScriptConfig:
    artifact_path: Path
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    reference_feature_key: str
    fold_count: int
    random_seed: int
    max_iter: int
    regularization_c: float
    word_ngram_range: tuple[int, int]
    char_ngram_range: tuple[int, int]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a grouped activation-feature sweep for a binary task.")
    parser.add_argument(
        "--artifact",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "activations" / "qwen3_0_6b_hard_v2_all_layers.pt"),
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "binary_layer_sweep_hard_v2_grouped.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "binary_layer_sweep_hard_v2_grouped_summary.md"),
    )
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument("--reference-feature", required=False, default="mean_pool_layer_18")
    parser.add_argument("--folds", required=False, type=int, default=5)
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument("--max-iter", required=False, type=int, default=1000)
    parser.add_argument("--regularization-c", required=False, type=float, default=1.0)
    parser.add_argument("--word-ngram-min", required=False, type=int, default=1)
    parser.add_argument("--word-ngram-max", required=False, type=int, default=2)
    parser.add_argument("--char-ngram-min", required=False, type=int, default=3)
    parser.add_argument("--char-ngram-max", required=False, type=int, default=5)
    return parser


def _parse_args(argv: Sequence[str]) -> SweepBinaryLayersScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return SweepBinaryLayersScriptConfig(
        artifact_path=Path(namespace.artifact),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        reference_feature_key=str(namespace.reference_feature),
        fold_count=int(namespace.folds),
        random_seed=int(namespace.seed),
        max_iter=int(namespace.max_iter),
        regularization_c=float(namespace.regularization_c),
        word_ngram_range=(int(namespace.word_ngram_min), int(namespace.word_ngram_max)),
        char_ngram_range=(int(namespace.char_ngram_min), int(namespace.char_ngram_max)),
    )


def _binary_task_config(config: SweepBinaryLayersScriptConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.reference_feature_key,
        word_ngram_range=config.word_ngram_range,
        char_ngram_range=config.char_ngram_range,
    )


def run_sweep(config: SweepBinaryLayersScriptConfig) -> None:
    artifact = load_activation_artifact(config.artifact_path)
    report = evaluate_grouped_binary_layer_sweep(
        artifact=artifact,
        task_name=config.task_name,
        config=_binary_task_config(config),
    )
    write_binary_layer_sweep_json(config.output_json_path, report)
    write_binary_layer_sweep_markdown(config.output_markdown_path, report)

    reference_feature = next(
        feature
        for feature in report.features
        if feature.feature_name == report.reference_feature_key
    )
    best_feature = report.features[0]
    print(f"Wrote binary layer sweep report to {config.output_json_path}")
    print(f"Wrote binary layer sweep summary to {config.output_markdown_path}")
    print(
        f"best={best_feature.feature_name} rank={best_feature.rank} "
        f"macro_f1={best_feature.macro_f1_mean:.4f} accuracy={best_feature.accuracy_mean:.4f}"
    )
    print(
        f"reference={reference_feature.feature_name} rank={reference_feature.rank} "
        f"macro_f1={reference_feature.macro_f1_mean:.4f} accuracy={reference_feature.accuracy_mean:.4f}"
    )


def main(argv: Sequence[str]) -> None:
    run_sweep(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
