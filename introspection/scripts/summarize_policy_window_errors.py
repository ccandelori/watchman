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

from aegis_introspection.adjudication import load_binary_error_analysis_report_json
from aegis_introspection.binary_tasks import BinaryMethodName
from aegis_introspection.policy_window_error_slices import (
    ErrorSliceDimension,
    build_policy_window_error_slice_report,
    load_prompt_policy_metadata,
    write_policy_window_error_slice_json,
    write_policy_window_error_slice_markdown,
)


_DEFAULT_DIMENSIONS: tuple[ErrorSliceDimension, ...] = (
    "family",
    "source_label",
    "credential_type",
    "payload_condition",
    "selected_field",
    "selected_mode",
    "selected_action",
)


@dataclass(frozen=True)
class SummarizePolicyWindowErrorsConfig:
    prompts_path: Path
    error_report_path: Path
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    method_name: BinaryMethodName
    dimensions: tuple[ErrorSliceDimension, ...]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize policy-window prediction errors by prompt metadata.")
    parser.add_argument(
        "--prompts",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "prompts_dp_honey_lite_v3_selector_windows.jsonl"),
    )
    parser.add_argument(
        "--error-report",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "dp_honey_lite_v3_selector_window_error_analysis_readout_window_layer_15_v1.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "dp_honey_lite_v3_selector_window_error_slices_readout_window_layer_15_v1.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "dp_honey_lite_v3_selector_window_error_slices_readout_window_layer_15_v1_summary.md"
        ),
    )
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument(
        "--method",
        required=False,
        choices=("activation_probe", "word_tfidf", "char_tfidf"),
        default="activation_probe",
    )
    parser.add_argument(
        "--dimension",
        required=False,
        action="append",
        choices=_DEFAULT_DIMENSIONS,
    )
    return parser


def _parse_args(argv: Sequence[str]) -> SummarizePolicyWindowErrorsConfig:
    namespace = _build_parser().parse_args(argv)
    dimensions = tuple(namespace.dimension) if namespace.dimension is not None else _DEFAULT_DIMENSIONS
    return SummarizePolicyWindowErrorsConfig(
        prompts_path=Path(namespace.prompts),
        error_report_path=Path(namespace.error_report),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        method_name=cast(BinaryMethodName, namespace.method),
        dimensions=cast(tuple[ErrorSliceDimension, ...], dimensions),
    )


def run_summary(config: SummarizePolicyWindowErrorsConfig) -> None:
    error_report = load_binary_error_analysis_report_json(config.error_report_path)
    metadata_by_id = load_prompt_policy_metadata(config.prompts_path)
    slice_report = build_policy_window_error_slice_report(
        report=error_report,
        metadata_by_id=metadata_by_id,
        task_name=config.task_name,
        method_name=config.method_name,
        dimensions=config.dimensions,
    )
    write_policy_window_error_slice_json(config.output_json_path, slice_report)
    write_policy_window_error_slice_markdown(config.output_markdown_path, slice_report)
    print(f"Wrote policy-window error slices to {config.output_json_path}")
    print(f"Wrote policy-window error slice summary to {config.output_markdown_path}")
    for summary in slice_report.summaries:
        if summary.error_count > 0:
            print(
                f"{summary.dimension}/{summary.value}/{summary.true_label}: "
                f"errors={summary.error_count} accuracy={summary.accuracy:.4f}"
            )


def main(argv: Sequence[str]) -> None:
    run_summary(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
