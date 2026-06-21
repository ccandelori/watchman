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

from aegis_introspection.adjudication import (
    build_adjudication_report,
    load_binary_error_analysis_report_json,
    write_adjudication_json,
    write_adjudication_markdown,
)
from aegis_introspection.binary_tasks import BinaryMethodName
from aegis_introspection.prompts import load_prompt_examples


_VALID_METHOD_NAMES: frozenset[str] = frozenset(("activation_probe", "word_tfidf", "char_tfidf"))


@dataclass(frozen=True)
class AdjudicateV2ErrorsScriptConfig:
    prompts_path: Path
    error_report_path: Path
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    subject_method_name: BinaryMethodName
    context_method_names: tuple[BinaryMethodName, ...]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a human adjudication worksheet for V2 prediction errors.")
    parser.add_argument(
        "--prompts",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "prompts_hard_v2.jsonl"),
    )
    parser.add_argument(
        "--error-report",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "binary_error_analysis_hard_v2_grouped.json"),
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "hard_v2_error_adjudication.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "hard_v2_error_adjudication_summary.md"),
    )
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument("--subject-method", required=False, default="activation_probe")
    parser.add_argument("--context-methods", required=False, default="word_tfidf,char_tfidf")
    return parser


def _parse_context_methods(value: str) -> tuple[BinaryMethodName, ...]:
    methods = tuple(item.strip() for item in value.split(",") if item.strip() != "")
    if len(methods) == 0:
        raise ValueError("context-methods must contain at least one method name.")
    for method in methods:
        if method not in _VALID_METHOD_NAMES:
            raise ValueError(f"Unsupported method name '{method}'.")
    return tuple(cast(BinaryMethodName, method) for method in methods)


def _parse_method_name(value: str) -> BinaryMethodName:
    if value not in _VALID_METHOD_NAMES:
        raise ValueError(f"Unsupported method name '{value}'.")
    return cast(BinaryMethodName, value)


def _parse_args(argv: Sequence[str]) -> AdjudicateV2ErrorsScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return AdjudicateV2ErrorsScriptConfig(
        prompts_path=Path(namespace.prompts),
        error_report_path=Path(namespace.error_report),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        subject_method_name=_parse_method_name(str(namespace.subject_method)),
        context_method_names=_parse_context_methods(str(namespace.context_methods)),
    )


def run_adjudication(config: AdjudicateV2ErrorsScriptConfig) -> None:
    examples = load_prompt_examples(config.prompts_path)
    error_report = load_binary_error_analysis_report_json(config.error_report_path)
    report = build_adjudication_report(
        error_report=error_report,
        examples=examples,
        task_name=config.task_name,
        subject_method_name=config.subject_method_name,
        context_method_names=config.context_method_names,
    )
    write_adjudication_json(config.output_json_path, report)
    write_adjudication_markdown(config.output_markdown_path, report)

    print(f"Wrote adjudication case list to {config.output_json_path}")
    print(f"Wrote adjudication worksheet to {config.output_markdown_path}")
    print(f"{report.task_name}/{report.subject_method_name}: cases={report.case_count}")
    for summary in report.family_summaries:
        print(f"{summary.family}: cases={summary.case_count}")


def main(argv: Sequence[str]) -> None:
    run_adjudication(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
