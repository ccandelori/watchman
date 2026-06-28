from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
INTROSPECTION_SRC_PATH = INTROSPECTION_ROOT / "src"
RUNTIME_SRC_PATH = WORKSPACE_ROOT / "src"
for source_path in (INTROSPECTION_SRC_PATH, RUNTIME_SRC_PATH):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from aegis_introspection.cift_support_reports import (  # noqa: E402
    CiftFailureCasesReportConfig,
    CiftSupportReportError,
    materialize_cift_failure_cases_report,
)


@dataclass(frozen=True)
class MaterializeCiftFailureCasesReportCliConfig:
    model_bundle_path: Path
    runtime_prevention_report_path: Path
    output_path: Path
    report_id: str
    created_at: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize aegis_introspection.cift_failure_cases/v1 from live CIFT runtime evidence."
    )
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--runtime-prevention-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--created-at", required=True)
    return parser


def _parse_args(argv: Sequence[str]) -> MaterializeCiftFailureCasesReportCliConfig:
    namespace = _build_parser().parse_args(argv)
    return MaterializeCiftFailureCasesReportCliConfig(
        model_bundle_path=Path(str(namespace.model_bundle)),
        runtime_prevention_report_path=Path(str(namespace.runtime_prevention_report)),
        output_path=Path(str(namespace.output)),
        report_id=str(namespace.report_id),
        created_at=str(namespace.created_at),
    )


def _report_config(config: MaterializeCiftFailureCasesReportCliConfig) -> CiftFailureCasesReportConfig:
    return CiftFailureCasesReportConfig(
        model_bundle_path=config.model_bundle_path,
        runtime_prevention_report_path=config.runtime_prevention_report_path,
        output_path=config.output_path,
        report_id=config.report_id,
        created_at=config.created_at,
    )


def run_materializer(config: MaterializeCiftFailureCasesReportCliConfig) -> None:
    record = materialize_cift_failure_cases_report(_report_config(config))
    counts = record["counts"]
    if not isinstance(counts, dict):
        raise CiftSupportReportError("failure cases report counts must be an object.")
    print(f"Wrote CIFT failure cases report to {config.output_path}")
    print(f"False negatives: {counts['false_negative_count']}")
    print(f"False positives: {counts['false_positive_count']}")
    print(f"Leakage failures: {counts['leakage_failure_count']}")


def main(argv: Sequence[str]) -> int:
    try:
        run_materializer(_parse_args(argv))
    except CiftSupportReportError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
