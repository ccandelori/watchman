from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.calibrated_detector_export import (
    CalibratedDetectorExportConfig,
    export_calibrated_cift_detector_results,
)
from aegis_introspection.detector_result_bridge import RecommendedAction


@dataclass(frozen=True)
class ExportCalibratedCiftDetectorResultsCliConfig:
    runtime_turns_path: Path
    calibration_report_path: Path
    output_path: Path
    detector_name: str
    probe_version: str
    capability_required: str
    positive_action: RecommendedAction
    negative_action: RecommendedAction
    confidence: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export calibrated CIFT predictions as Aegis DetectorResult JSONL.")
    parser.add_argument(
        "--runtime-turns",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "runtime_turns_dp_honey_lite_v3_selector_windows.jsonl"),
    )
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
        "--output",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "detector_results_cift_v3_selector_window_layer_15_calibrated_v1.jsonl"
        ),
    )
    parser.add_argument("--detector-name", required=False, default="cift_selector_probe")
    parser.add_argument(
        "--probe-version",
        required=False,
        default="dp_honey_lite_v3_selector_window_layer_15_calibrated_v1",
    )
    parser.add_argument("--capability-required", required=False, default="self_hosted_introspection")
    parser.add_argument(
        "--positive-action",
        required=False,
        choices=("allow", "warn", "sanitize", "block", "escalate"),
        default="warn",
    )
    parser.add_argument(
        "--negative-action",
        required=False,
        choices=("allow", "warn", "sanitize", "block", "escalate"),
        default="allow",
    )
    parser.add_argument("--confidence", required=False, type=float, default=0.7736)
    return parser


def _parse_args(argv: Sequence[str]) -> ExportCalibratedCiftDetectorResultsCliConfig:
    namespace = _build_parser().parse_args(argv)
    return ExportCalibratedCiftDetectorResultsCliConfig(
        runtime_turns_path=Path(namespace.runtime_turns),
        calibration_report_path=Path(namespace.calibration_report),
        output_path=Path(namespace.output),
        detector_name=str(namespace.detector_name),
        probe_version=str(namespace.probe_version),
        capability_required=str(namespace.capability_required),
        positive_action=cast(RecommendedAction, namespace.positive_action),
        negative_action=cast(RecommendedAction, namespace.negative_action),
        confidence=float(namespace.confidence),
    )


def _export_config(config: ExportCalibratedCiftDetectorResultsCliConfig) -> CalibratedDetectorExportConfig:
    return CalibratedDetectorExportConfig(
        runtime_turns_path=config.runtime_turns_path,
        calibration_report_path=config.calibration_report_path,
        output_path=config.output_path,
        detector_name=config.detector_name,
        probe_version=config.probe_version,
        capability_required=config.capability_required,
        positive_action=config.positive_action,
        negative_action=config.negative_action,
        confidence=config.confidence,
    )


def run_export(config: ExportCalibratedCiftDetectorResultsCliConfig) -> None:
    row_count = export_calibrated_cift_detector_results(_export_config(config))
    print(f"Wrote {row_count} calibrated CIFT detector-result rows to {config.output_path}")


def main(argv: Sequence[str]) -> None:
    run_export(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
