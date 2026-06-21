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
from aegis_introspection.v3_policy_diagnostics import (
    evaluate_v3_policy_diagnostics,
    write_v3_policy_diagnostics_json,
    write_v3_policy_diagnostics_markdown,
)


@dataclass(frozen=True)
class DiagnoseV3PolicyScriptConfig:
    artifact_path: Path
    output_json_path: Path
    output_markdown_path: Path
    activation_feature_key: str
    fold_count: int
    random_seed: int
    max_iter: int
    regularization_c: float
    word_ngram_range: tuple[int, int]
    char_ngram_range: tuple[int, int]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V3 policy-parser diagnostics against grouped detector metrics.")
    parser.add_argument(
        "--artifact",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "activations"
            / "qwen3_0_6b_dp_honey_lite_v3_all_pooling.pt"
        ),
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "dp_honey_lite_v3_policy_diagnostics_v1.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(
            INTROSPECTION_ROOT / "data" / "reports" / "dp_honey_lite_v3_policy_diagnostics_v1_summary.md"
        ),
    )
    parser.add_argument("--activation-feature", required=False, default="readout_window_layer_21")
    parser.add_argument("--folds", required=False, type=int, default=5)
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument("--max-iter", required=False, type=int, default=1000)
    parser.add_argument("--regularization-c", required=False, type=float, default=1.0)
    parser.add_argument("--word-ngram-min", required=False, type=int, default=1)
    parser.add_argument("--word-ngram-max", required=False, type=int, default=2)
    parser.add_argument("--char-ngram-min", required=False, type=int, default=3)
    parser.add_argument("--char-ngram-max", required=False, type=int, default=5)
    return parser


def _parse_args(argv: Sequence[str]) -> DiagnoseV3PolicyScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return DiagnoseV3PolicyScriptConfig(
        artifact_path=Path(namespace.artifact),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        activation_feature_key=str(namespace.activation_feature),
        fold_count=int(namespace.folds),
        random_seed=int(namespace.seed),
        max_iter=int(namespace.max_iter),
        regularization_c=float(namespace.regularization_c),
        word_ngram_range=(int(namespace.word_ngram_min), int(namespace.word_ngram_max)),
        char_ngram_range=(int(namespace.char_ngram_min), int(namespace.char_ngram_max)),
    )


def _binary_task_config(config: DiagnoseV3PolicyScriptConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.activation_feature_key,
        word_ngram_range=config.word_ngram_range,
        char_ngram_range=config.char_ngram_range,
    )


def run_diagnostics(config: DiagnoseV3PolicyScriptConfig) -> None:
    artifact = load_activation_artifact(config.artifact_path)
    report = evaluate_v3_policy_diagnostics(
        artifact=artifact,
        config=_binary_task_config(config),
    )
    write_v3_policy_diagnostics_json(config.output_json_path, report)
    write_v3_policy_diagnostics_markdown(config.output_markdown_path, report)

    print(f"Wrote V3 policy diagnostic report to {config.output_json_path}")
    print(f"Wrote V3 policy diagnostic summary to {config.output_markdown_path}")
    for slice_report in report.slices:
        activation_metric = next(metric for metric in slice_report.metrics if metric.method_name == "activation_probe")
        print(
            f"{slice_report.slice_name}: parser_macro_f1={slice_report.parser.macro_f1:.4f} "
            f"activation_macro_f1={activation_metric.macro_f1:.4f}"
        )


def main(argv: Sequence[str]) -> None:
    run_diagnostics(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
