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
from aegis_introspection.text_baseline import (
    TextBaselineTrainingConfig,
    train_text_baseline_report,
    write_text_baseline_report_json,
)


@dataclass(frozen=True)
class TrainTextBaselineScriptConfig:
    artifact_path: Path
    output_path: Path
    fold_count: int
    random_seed: int
    max_iter: int
    regularization_c: float
    lowercase: bool
    min_df: int
    ngram_range: tuple[int, int]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a TF-IDF text baseline on Aegis prompt text.")
    parser.add_argument(
        "--artifact",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "activations" / "qwen3_0_6b_features.pt"),
    )
    parser.add_argument(
        "--output",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "text_baseline.json"),
    )
    parser.add_argument("--folds", required=False, type=int, default=5)
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument("--max-iter", required=False, type=int, default=1000)
    parser.add_argument("--regularization-c", required=False, type=float, default=1.0)
    parser.add_argument("--min-df", required=False, type=int, default=1)
    parser.add_argument("--ngram-min", required=False, type=int, default=1)
    parser.add_argument("--ngram-max", required=False, type=int, default=2)
    parser.add_argument("--preserve-case", action="store_true")
    return parser


def _parse_args(argv: Sequence[str]) -> TrainTextBaselineScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return TrainTextBaselineScriptConfig(
        artifact_path=Path(namespace.artifact),
        output_path=Path(namespace.output),
        fold_count=int(namespace.folds),
        random_seed=int(namespace.seed),
        max_iter=int(namespace.max_iter),
        regularization_c=float(namespace.regularization_c),
        lowercase=not bool(namespace.preserve_case),
        min_df=int(namespace.min_df),
        ngram_range=(int(namespace.ngram_min), int(namespace.ngram_max)),
    )


def _text_baseline_config(config: TrainTextBaselineScriptConfig) -> TextBaselineTrainingConfig:
    return TextBaselineTrainingConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        lowercase=config.lowercase,
        min_df=config.min_df,
        ngram_range=config.ngram_range,
    )


def run_training(config: TrainTextBaselineScriptConfig) -> None:
    artifact = load_activation_artifact(config.artifact_path)
    report = train_text_baseline_report(artifact, _text_baseline_config(config))
    write_text_baseline_report_json(config.output_path, report)

    print(f"Wrote text baseline report to {config.output_path}")
    print(f"baseline_name: {report.baseline_name}")
    print(f"macro_f1_mean: {report.macro_f1_mean:.4f}")
    print(f"accuracy_mean: {report.accuracy_mean:.4f}")


def main(argv: Sequence[str]) -> None:
    run_training(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
