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

from aegis_introspection.cift_runtime_eval import CiftRuntimeEvalConfig, run_cift_runtime_eval  # noqa: E402
from aegis_introspection.sealed_holdout import add_unseal_flag  # noqa: E402


@dataclass(frozen=True)
class RunCiftRuntimeEvalCliConfig:
    runtime_turns_path: Path
    activation_artifact_path: Path
    runtime_model_path: Path
    output_path: Path
    detector_name: str
    feature_source: str
    mock_response: str
    allow_sealed_holdout: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an exported CIFT runtime model through the Aegis runtime spine.")
    parser.add_argument("--runtime-turns", required=True)
    parser.add_argument("--activation-artifact", required=True)
    parser.add_argument("--runtime-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detector-name", required=True)
    parser.add_argument("--feature-source", required=True)
    parser.add_argument("--mock-response", required=True)
    add_unseal_flag(parser)
    return parser


def _parse_args(argv: Sequence[str]) -> RunCiftRuntimeEvalCliConfig:
    namespace = _build_parser().parse_args(argv)
    return RunCiftRuntimeEvalCliConfig(
        runtime_turns_path=Path(namespace.runtime_turns),
        activation_artifact_path=Path(namespace.activation_artifact),
        runtime_model_path=Path(namespace.runtime_model),
        output_path=Path(namespace.output),
        detector_name=str(namespace.detector_name),
        feature_source=str(namespace.feature_source),
        mock_response=str(namespace.mock_response),
        allow_sealed_holdout=bool(namespace.allow_sealed_holdout),
    )


def _eval_config(config: RunCiftRuntimeEvalCliConfig) -> CiftRuntimeEvalConfig:
    return CiftRuntimeEvalConfig(
        runtime_turns_path=config.runtime_turns_path,
        activation_artifact_path=config.activation_artifact_path,
        runtime_model_path=config.runtime_model_path,
        output_path=config.output_path,
        detector_name=config.detector_name,
        feature_source=config.feature_source,
        mock_response=config.mock_response,
        allow_sealed_holdout=config.allow_sealed_holdout,
    )


def run_cli(config: RunCiftRuntimeEvalCliConfig) -> None:
    summary = run_cift_runtime_eval(_eval_config(config))
    print(f"Wrote {summary.request_count} CIFT runtime eval rows to {summary.output_path}")
    print(f"Detector actions: {summary.detector_action_counts}")
    print(f"Policy actions: {summary.policy_action_counts}")
    print(f"Capability statuses: {summary.capability_status_counts}")


def main(argv: Sequence[str]) -> None:
    run_cli(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
