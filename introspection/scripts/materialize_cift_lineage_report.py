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
    CiftLineageReportConfig,
    CiftSupportReportError,
    materialize_cift_lineage_report,
)


@dataclass(frozen=True)
class MaterializeCiftLineageReportCliConfig:
    model_bundle_path: Path
    output_path: Path
    report_id: str
    created_at: str
    artifact_paths: tuple[Path, ...]
    report_paths: tuple[Path, ...]
    reproduction_commands: tuple[str, ...]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize aegis_introspection.cift_lineage/v1 for a CIFT candidate."
    )
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--artifact", action="append", required=False)
    parser.add_argument("--report", action="append", required=False)
    parser.add_argument("--reproduction-command", action="append", required=False)
    return parser


def _parse_args(argv: Sequence[str]) -> MaterializeCiftLineageReportCliConfig:
    namespace = _build_parser().parse_args(argv)
    return MaterializeCiftLineageReportCliConfig(
        model_bundle_path=Path(str(namespace.model_bundle)),
        output_path=Path(str(namespace.output)),
        report_id=str(namespace.report_id),
        created_at=str(namespace.created_at),
        artifact_paths=_paths_from_optional(namespace.artifact),
        report_paths=_paths_from_optional(namespace.report),
        reproduction_commands=_strings_from_optional(namespace.reproduction_command),
    )


def _paths_from_optional(value: object) -> tuple[Path, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise CiftSupportReportError("Repeated path arguments must parse as a list.")
    return tuple(Path(str(item)) for item in value)


def _strings_from_optional(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise CiftSupportReportError("Repeated string arguments must parse as a list.")
    return tuple(str(item) for item in value)


def _report_config(config: MaterializeCiftLineageReportCliConfig) -> CiftLineageReportConfig:
    return CiftLineageReportConfig(
        model_bundle_path=config.model_bundle_path,
        output_path=config.output_path,
        report_id=config.report_id,
        created_at=config.created_at,
        artifact_paths=config.artifact_paths,
        report_paths=config.report_paths,
        reproduction_commands=config.reproduction_commands,
    )


def run_materializer(config: MaterializeCiftLineageReportCliConfig) -> None:
    record = materialize_cift_lineage_report(_report_config(config))
    print(f"Wrote CIFT lineage report to {config.output_path}")
    print(f"Report ID: {record['report_id']}")
    print(f"Artifacts: {len(config.artifact_paths)}")
    print(f"Reports: {len(config.report_paths)}")


def main(argv: Sequence[str]) -> int:
    try:
        run_materializer(_parse_args(argv))
    except CiftSupportReportError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
