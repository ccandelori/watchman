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
from aegis_introspection.cift_calibration import (
    CiftCalibrationConfig,
    collect_grouped_calibrated_cift_predictions,
    write_cift_calibration_json,
    write_cift_calibration_markdown,
)


@dataclass(frozen=True)
class CalibrateCiftDetectorCliConfig:
    artifact_path: Path
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    positive_label: str
    activation_feature_key: str
    fold_count: int
    inner_fold_count: int
    random_seed: int
    max_iter: int
    regularization_c: float
    decision_threshold: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit grouped, out-of-fold calibrated scores for a CIFT detector.")
    parser.add_argument(
        "--artifact",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "activations" / "qwen3_0_6b_dp_honey_lite_v3_selector_windows.pt"),
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "dp_honey_lite_v3_selector_window_cift_calibration_readout_window_layer_15_v1.json"
        ),
    )
    parser.add_argument(
        "--output-markdown",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "dp_honey_lite_v3_selector_window_cift_calibration_readout_window_layer_15_v1_summary.md"
        ),
    )
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument("--positive-label", required=False, default="exfiltration_intent")
    parser.add_argument("--activation-feature", required=False, default="readout_window_layer_15")
    parser.add_argument("--folds", required=False, type=int, default=5)
    parser.add_argument("--inner-folds", required=False, type=int, default=3)
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument("--max-iter", required=False, type=int, default=1000)
    parser.add_argument("--regularization-c", required=False, type=float, default=1.0)
    parser.add_argument("--decision-threshold", required=False, type=float, default=0.5)
    return parser


def _parse_args(argv: Sequence[str]) -> CalibrateCiftDetectorCliConfig:
    namespace = _build_parser().parse_args(argv)
    return CalibrateCiftDetectorCliConfig(
        artifact_path=Path(namespace.artifact),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_markdown),
        task_name=str(namespace.task),
        positive_label=str(namespace.positive_label),
        activation_feature_key=str(namespace.activation_feature),
        fold_count=int(namespace.folds),
        inner_fold_count=int(namespace.inner_folds),
        random_seed=int(namespace.seed),
        max_iter=int(namespace.max_iter),
        regularization_c=float(namespace.regularization_c),
        decision_threshold=float(namespace.decision_threshold),
    )


def _calibration_config(config: CalibrateCiftDetectorCliConfig) -> CiftCalibrationConfig:
    return CiftCalibrationConfig(
        task_name=config.task_name,
        positive_label=config.positive_label,
        activation_feature_key=config.activation_feature_key,
        fold_count=config.fold_count,
        inner_fold_count=config.inner_fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        decision_threshold=config.decision_threshold,
    )


def run_calibration(config: CalibrateCiftDetectorCliConfig) -> None:
    artifact = load_activation_artifact(config.artifact_path)
    report = collect_grouped_calibrated_cift_predictions(
        artifact=artifact,
        config=_calibration_config(config),
    )
    write_cift_calibration_json(config.output_json_path, report)
    write_cift_calibration_markdown(config.output_markdown_path, report)
    print(f"Wrote CIFT calibration JSON to {config.output_json_path}")
    print(f"Wrote CIFT calibration summary to {config.output_markdown_path}")
    print(f"Accuracy: {report.accuracy:.4f}")
    print(f"Macro F1: {report.macro_f1:.4f}")
    print(f"Brier score: {report.brier_score:.4f}")
    print(f"Expected calibration error: {report.expected_calibration_error:.4f}")


def main(argv: Sequence[str]) -> None:
    run_calibration(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
