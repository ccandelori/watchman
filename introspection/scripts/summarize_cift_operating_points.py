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

from aegis_introspection.cift_calibration import load_cift_calibration_report_json
from aegis_introspection.cift_operating_points import (
    CiftOperatingPointConfig,
    build_cift_operating_point_report,
    write_cift_operating_point_json,
    write_cift_operating_point_markdown,
)


@dataclass(frozen=True)
class SummarizeCiftOperatingPointsCliConfig:
    calibration_report_path: Path
    output_json_path: Path
    output_markdown_path: Path
    thresholds: tuple[float, ...]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize calibrated CIFT detector operating points.")
    parser.add_argument(
        "--calibration-report",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "dp_honey_lite_v3_selector_window_cift_calibration_readout_window_layer_15_v1.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "dp_honey_lite_v3_selector_window_cift_operating_points_readout_window_layer_15_v1.json"
        ),
    )
    parser.add_argument(
        "--output-markdown",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "dp_honey_lite_v3_selector_window_cift_operating_points_readout_window_layer_15_v1_summary.md"
        ),
    )
    parser.add_argument("--thresholds", required=False, default="0.05:0.95:0.05")
    return parser


def _parse_args(argv: Sequence[str]) -> SummarizeCiftOperatingPointsCliConfig:
    namespace = _build_parser().parse_args(argv)
    return SummarizeCiftOperatingPointsCliConfig(
        calibration_report_path=Path(namespace.calibration_report),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_markdown),
        thresholds=_parse_thresholds(str(namespace.thresholds)),
    )


def _parse_thresholds(value: str) -> tuple[float, ...]:
    if ":" in value:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError("Threshold range must use start:end:step format.")
        start = float(parts[0])
        end = float(parts[1])
        step = float(parts[2])
        if step <= 0.0:
            raise ValueError("Threshold range step must be greater than 0.")
        thresholds: list[float] = []
        current = start
        while current <= end + (step / 10.0):
            thresholds.append(round(current, 10))
            current += step
        return tuple(thresholds)
    return tuple(float(part.strip()) for part in value.split(",") if part.strip() != "")


def run_summary(config: SummarizeCiftOperatingPointsCliConfig) -> None:
    calibration_report = load_cift_calibration_report_json(config.calibration_report_path)
    report = build_cift_operating_point_report(
        report=calibration_report,
        config=CiftOperatingPointConfig(thresholds=config.thresholds),
    )
    write_cift_operating_point_json(config.output_json_path, report)
    write_cift_operating_point_markdown(config.output_markdown_path, report)
    print(f"Wrote CIFT operating-point JSON to {config.output_json_path}")
    print(f"Wrote CIFT operating-point summary to {config.output_markdown_path}")
    print(f"Best macro-F1 threshold: {report.best_macro_f1_threshold:.4f}")
    print(f"High-recall threshold: {report.high_recall_threshold:.4f}")


def main(argv: Sequence[str]) -> None:
    run_summary(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
