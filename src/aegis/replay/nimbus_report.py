from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from aegis.replay.nimbus_redteam import (
    NimbusRedteamParseError,
    load_nimbus_redteam_metrics_jsonl,
    render_nimbus_redteam_markdown,
    summaries_to_json,
    summarize_nimbus_redteam_metrics,
)


class NimbusReportFormat(StrEnum):
    MARKDOWN = "markdown"
    JSON = "json"


@dataclass(frozen=True)
class NimbusReportConfig:
    input_path: Path
    output_format: NimbusReportFormat


def parse_args(argv: Sequence[str]) -> NimbusReportConfig:
    parser = argparse.ArgumentParser(description="Summarize NIMBUS behavior from redteam JSONL results.")
    parser.add_argument("--input", required=True, type=Path, help="Path to a redteam JSONL result file.")
    parser.add_argument(
        "--format",
        choices=tuple(item.value for item in NimbusReportFormat),
        default=NimbusReportFormat.MARKDOWN.value,
        help="Output format for the summary.",
    )
    args = parser.parse_args(argv)
    return NimbusReportConfig(input_path=args.input, output_format=NimbusReportFormat(args.format))


def render_report(config: NimbusReportConfig) -> str:
    metrics = load_nimbus_redteam_metrics_jsonl(config.input_path)
    summaries = summarize_nimbus_redteam_metrics(metrics)
    if config.output_format == NimbusReportFormat.MARKDOWN:
        return render_nimbus_redteam_markdown(summaries)
    if config.output_format == NimbusReportFormat.JSON:
        return summaries_to_json(summaries)
    raise NimbusRedteamParseError(f"Unsupported output format '{config.output_format}'.")


def main() -> None:
    try:
        sys.stdout.write(render_report(parse_args(tuple(sys.argv[1:]))))
    except NimbusRedteamParseError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
