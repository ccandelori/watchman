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

from aegis_introspection.cift_calibration import load_cift_calibration_report_json
from aegis_introspection.cift_selector_diagnostics import (
    SelectorDiagnosticConfig,
    build_selector_diagnostic_report,
    write_selector_diagnostic_json,
    write_selector_diagnostic_markdown,
)
from aegis_introspection.policy_window_error_slices import ErrorSliceDimension, load_prompt_policy_metadata


_DEFAULT_DIMENSIONS: tuple[ErrorSliceDimension, ...] = (
    "family",
    "source_label",
    "credential_type",
    "payload_condition",
    "selected_field",
    "selected_mode",
    "selected_action",
)

_DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
)


@dataclass(frozen=True)
class DiagnoseSelectorScoresConfig:
    prompts_path: Path
    calibration_report_path: Path
    output_json_path: Path
    output_markdown_path: Path
    dimensions: tuple[ErrorSliceDimension, ...]
    thresholds: tuple[float, ...]
    near_threshold_radius: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose calibrated CIFT selector-window scores by prompt slice.")
    parser.add_argument(
        "--prompts",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "prompts_dp_honey_lite_v3_selector_windows.jsonl"),
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
        "--output-json",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "dp_honey_lite_v3_selector_window_score_diagnostics_readout_window_layer_15_v1.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "dp_honey_lite_v3_selector_window_score_diagnostics_readout_window_layer_15_v1_summary.md"
        ),
    )
    parser.add_argument(
        "--dimension",
        required=False,
        action="append",
        choices=_DEFAULT_DIMENSIONS,
    )
    parser.add_argument(
        "--threshold",
        required=False,
        action="append",
        type=float,
    )
    parser.add_argument("--near-threshold-radius", required=False, type=float, default=0.10)
    return parser


def _parse_args(argv: Sequence[str]) -> DiagnoseSelectorScoresConfig:
    namespace = _build_parser().parse_args(argv)
    dimensions = tuple(namespace.dimension) if namespace.dimension is not None else _DEFAULT_DIMENSIONS
    thresholds = tuple(namespace.threshold) if namespace.threshold is not None else _DEFAULT_THRESHOLDS
    return DiagnoseSelectorScoresConfig(
        prompts_path=Path(namespace.prompts),
        calibration_report_path=Path(namespace.calibration_report),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        dimensions=cast(tuple[ErrorSliceDimension, ...], dimensions),
        thresholds=thresholds,
        near_threshold_radius=float(namespace.near_threshold_radius),
    )


def run_diagnostics(config: DiagnoseSelectorScoresConfig) -> None:
    calibration_report = load_cift_calibration_report_json(config.calibration_report_path)
    metadata_by_id = load_prompt_policy_metadata(config.prompts_path)
    diagnostic_report = build_selector_diagnostic_report(
        report=calibration_report,
        metadata_by_id=metadata_by_id,
        config=SelectorDiagnosticConfig(
            dimensions=config.dimensions,
            thresholds=config.thresholds,
            near_threshold_radius=config.near_threshold_radius,
        ),
    )
    write_selector_diagnostic_json(config.output_json_path, diagnostic_report)
    write_selector_diagnostic_markdown(config.output_markdown_path, diagnostic_report)
    print(f"Wrote selector score diagnostics to {config.output_json_path}")
    print(f"Wrote selector score diagnostic summary to {config.output_markdown_path}")
    print(f"Errors: {len(diagnostic_report.error_examples)}")
    if len(diagnostic_report.error_examples) > 0:
        near_errors = sum(
            1
            for example in diagnostic_report.error_examples
            if example.absolute_margin <= diagnostic_report.near_threshold_radius
        )
        print(f"Near-threshold errors: {near_errors}")
        print(f"Confident errors: {len(diagnostic_report.error_examples) - near_errors}")


def main(argv: Sequence[str]) -> None:
    run_diagnostics(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
