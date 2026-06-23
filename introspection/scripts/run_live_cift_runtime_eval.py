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

from aegis_introspection.cift_live_runtime_eval import (  # noqa: E402
    CiftLiveRuntimeEvalConfig,
    run_cift_live_runtime_eval,
)
from aegis_introspection.model_loader import ModelDTypeName, parse_model_dtype  # noqa: E402
from aegis_introspection.sealed_holdout import add_unseal_flag  # noqa: E402


@dataclass(frozen=True)
class RunLiveCiftRuntimeEvalCliConfig:
    runtime_turns_path: Path
    runtime_model_path: Path
    output_path: Path
    detector_name: str
    feature_source: str
    mock_response: str
    model_id: str
    revision: str
    requested_device: str
    local_files_only: bool
    dtype_name: ModelDTypeName
    trust_remote_code: bool
    allow_sealed_holdout: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live model-host CIFT extraction through the Aegis runtime spine.")
    parser.add_argument("--runtime-turns", required=True)
    parser.add_argument("--runtime-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detector-name", required=True)
    parser.add_argument("--feature-source", required=True)
    parser.add_argument("--mock-response", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    add_unseal_flag(parser)
    return parser


def _parse_args(argv: Sequence[str]) -> RunLiveCiftRuntimeEvalCliConfig:
    namespace = _build_parser().parse_args(argv)
    return RunLiveCiftRuntimeEvalCliConfig(
        runtime_turns_path=Path(namespace.runtime_turns),
        runtime_model_path=Path(namespace.runtime_model),
        output_path=Path(namespace.output),
        detector_name=str(namespace.detector_name),
        feature_source=str(namespace.feature_source),
        mock_response=str(namespace.mock_response),
        model_id=str(namespace.model_id),
        revision=str(namespace.revision),
        requested_device=str(namespace.device),
        local_files_only=not bool(namespace.allow_download),
        dtype_name=parse_model_dtype(str(namespace.dtype)),
        trust_remote_code=bool(namespace.trust_remote_code),
        allow_sealed_holdout=bool(namespace.allow_sealed_holdout),
    )


def _eval_config(config: RunLiveCiftRuntimeEvalCliConfig) -> CiftLiveRuntimeEvalConfig:
    return CiftLiveRuntimeEvalConfig(
        runtime_turns_path=config.runtime_turns_path,
        runtime_model_path=config.runtime_model_path,
        output_path=config.output_path,
        detector_name=config.detector_name,
        feature_source=config.feature_source,
        mock_response=config.mock_response,
        model_id=config.model_id,
        revision=config.revision,
        requested_device=config.requested_device,
        local_files_only=config.local_files_only,
        dtype_name=config.dtype_name,
        trust_remote_code=config.trust_remote_code,
        allow_sealed_holdout=config.allow_sealed_holdout,
    )


def run_cli(config: RunLiveCiftRuntimeEvalCliConfig) -> None:
    summary = run_cift_live_runtime_eval(_eval_config(config))
    print(f"Wrote {summary.request_count} live CIFT runtime eval rows to {summary.output_path}")
    print(f"Detector actions: {summary.detector_action_counts}")
    print(f"Policy actions: {summary.policy_action_counts}")
    print(f"Capability statuses: {summary.capability_status_counts}")


def main(argv: Sequence[str]) -> None:
    run_cli(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
