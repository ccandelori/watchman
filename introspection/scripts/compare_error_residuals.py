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
from aegis_introspection.residual_error_comparison import (
    compare_binary_error_residuals,
    write_residual_error_comparison_json,
    write_residual_error_comparison_markdown,
)


_VALID_METHOD_NAMES: frozenset[str] = frozenset(("activation_probe", "word_tfidf", "char_tfidf"))


@dataclass(frozen=True)
class CompareErrorResidualsScriptConfig:
    reference_report_path: Path
    candidate_report_path: Path
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    method_name: BinaryMethodName


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare residual errors between two grouped error-analysis reports.")
    parser.add_argument(
        "--reference-report",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "binary_error_analysis_hard_v2_grouped.json"),
    )
    parser.add_argument(
        "--candidate-report",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "binary_error_analysis_hard_v2_candidate_final_token_layer_11_grouped.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "hard_v2_candidate_residual_error_comparison.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "hard_v2_candidate_residual_error_comparison_summary.md"
        ),
    )
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument("--method", required=False, default="activation_probe")
    return parser


def _parse_method_name(value: str) -> BinaryMethodName:
    if value not in _VALID_METHOD_NAMES:
        raise ValueError(f"Unsupported method name '{value}'.")
    return cast(BinaryMethodName, value)


def _parse_args(argv: Sequence[str]) -> CompareErrorResidualsScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return CompareErrorResidualsScriptConfig(
        reference_report_path=Path(namespace.reference_report),
        candidate_report_path=Path(namespace.candidate_report),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        method_name=_parse_method_name(str(namespace.method)),
    )


def run_comparison(config: CompareErrorResidualsScriptConfig) -> None:
    reference_report = load_binary_error_analysis_report_json(config.reference_report_path)
    candidate_report = load_binary_error_analysis_report_json(config.candidate_report_path)
    report = compare_binary_error_residuals(
        reference_report=reference_report,
        candidate_report=candidate_report,
        task_name=config.task_name,
        method_name=config.method_name,
    )
    write_residual_error_comparison_json(config.output_json_path, report)
    write_residual_error_comparison_markdown(config.output_markdown_path, report)

    print(f"Wrote residual comparison to {config.output_json_path}")
    print(f"Wrote residual comparison summary to {config.output_markdown_path}")
    print(
        f"{report.task_name}/{report.method_name}: "
        f"reference_errors={report.reference_error_count} "
        f"candidate_errors={report.candidate_error_count} "
        f"fixed={report.fixed_error_count} "
        f"persistent={report.persistent_error_count} "
        f"introduced={report.introduced_error_count}"
    )


def main(argv: Sequence[str]) -> None:
    run_comparison(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
