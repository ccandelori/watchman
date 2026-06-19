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

from aegis_introspection.probe_report_summary import load_probe_report_summary, write_probe_report_markdown


@dataclass(frozen=True)
class SummarizeProbeReportConfig:
    report_path: Path
    output_path: Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize an activation probe JSON report as Markdown.")
    parser.add_argument(
        "--report",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "probe_baseline.json"),
    )
    parser.add_argument(
        "--output",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "probe_baseline_summary.md"),
    )
    return parser


def _parse_args(argv: Sequence[str]) -> SummarizeProbeReportConfig:
    namespace = _build_parser().parse_args(argv)
    return SummarizeProbeReportConfig(
        report_path=Path(namespace.report),
        output_path=Path(namespace.output),
    )


def run_summary(config: SummarizeProbeReportConfig) -> None:
    summary = load_probe_report_summary(config.report_path)
    write_probe_report_markdown(config.output_path, summary)
    print(f"Wrote probe report summary to {config.output_path}")


def main(argv: Sequence[str]) -> None:
    run_summary(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
