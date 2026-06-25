from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
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

from aegis_introspection.cift_certification_workflow_runner import (  # noqa: E402
    CiftCertificationWorkflowRunnerConfig,
    CiftCertificationWorkflowRunnerError,
    run_cift_certification_workflow,
)


@dataclass(frozen=True)
class RunCiftCertificationWorkflowCliConfig:
    repository_root: Path
    workflow_manifest_path: Path
    output_path: Path
    execute: bool
    allow_sealed_holdout_execution: bool
    overwrite_existing_outputs: bool
    template_values: Mapping[str, str]
    command_timeout_seconds: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or dry-run a CIFT certification workflow command plan.")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--workflow-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-sealed-holdout-execution", action="store_true")
    parser.add_argument("--overwrite-existing-outputs", action="store_true")
    parser.add_argument("--command-timeout-seconds", required=True, type=float)
    parser.add_argument(
        "--template-value",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Resolve an operator-supplied workflow template placeholder. May be repeated.",
    )
    return parser


def _parse_args(argv: Sequence[str]) -> RunCiftCertificationWorkflowCliConfig:
    namespace = _build_parser().parse_args(argv)
    return RunCiftCertificationWorkflowCliConfig(
        repository_root=Path(str(namespace.repository_root)),
        workflow_manifest_path=Path(str(namespace.workflow_manifest)),
        output_path=Path(str(namespace.output)),
        execute=bool(namespace.execute),
        allow_sealed_holdout_execution=bool(namespace.allow_sealed_holdout_execution),
        overwrite_existing_outputs=bool(namespace.overwrite_existing_outputs),
        template_values=_parse_template_values(tuple(str(value) for value in namespace.template_value)),
        command_timeout_seconds=_required_positive_float(
            namespace.command_timeout_seconds,
            "--command-timeout-seconds",
        ),
    )


def _parse_template_values(raw_values: tuple[str, ...]) -> Mapping[str, str]:
    parsed_values: dict[str, str] = {}
    for raw_value in raw_values:
        if "=" not in raw_value:
            raise CiftCertificationWorkflowRunnerError("--template-value must use NAME=VALUE.")
        name, value = raw_value.split("=", maxsplit=1)
        if name == "" or value == "":
            raise CiftCertificationWorkflowRunnerError("--template-value must include non-empty NAME and VALUE.")
        if name in parsed_values:
            raise CiftCertificationWorkflowRunnerError(f"duplicate --template-value for {name}.")
        parsed_values[name] = value
    return parsed_values


def _required_positive_float(raw_value: object, field_name: str) -> float:
    if raw_value is None:
        raise CiftCertificationWorkflowRunnerError(f"{field_name} is required.")
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        raise CiftCertificationWorkflowRunnerError(f"{field_name} must be numeric.")
    value = float(raw_value)
    if not math.isfinite(value) or value <= 0.0:
        raise CiftCertificationWorkflowRunnerError(f"{field_name} must be a finite positive number.")
    return value


def _runner_config(config: RunCiftCertificationWorkflowCliConfig) -> CiftCertificationWorkflowRunnerConfig:
    return CiftCertificationWorkflowRunnerConfig(
        repository_root=config.repository_root,
        workflow_manifest_path=config.workflow_manifest_path,
        output_path=config.output_path,
        execute=config.execute,
        allow_sealed_holdout_execution=config.allow_sealed_holdout_execution,
        overwrite_existing_outputs=config.overwrite_existing_outputs,
        template_values=config.template_values,
        command_timeout_seconds=config.command_timeout_seconds,
    )


def run_cli(config: RunCiftCertificationWorkflowCliConfig) -> int:
    report = run_cift_certification_workflow(_runner_config(config))
    mode = "execute" if config.execute else "dry-run"
    if report.eligible:
        print(f"CIFT certification workflow {mode} passed: {config.workflow_manifest_path}")
        print(f"Steps: {report.step_count}")
        return 0
    if not config.execute and report.plan_eligible:
        print(f"CIFT certification workflow {mode} plan passed: {config.workflow_manifest_path}")
        print(f"Steps: {report.step_count}")
        print("Evidence certification: not eligible")
        for failed_requirement in report.failed_requirements:
            print(f"- {failed_requirement}")
        return 0
    print(f"CIFT certification workflow {mode} failed: {config.workflow_manifest_path}")
    for failed_requirement in report.failed_requirements:
        print(f"- {failed_requirement}")
    return 1


def main(argv: Sequence[str]) -> int:
    try:
        return run_cli(_parse_args(argv))
    except CiftCertificationWorkflowRunnerError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
