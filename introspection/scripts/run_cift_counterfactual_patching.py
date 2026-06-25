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

from aegis_introspection.cift_causal_patching import (  # noqa: E402
    CiftCounterfactualPatchingConfig,
    run_cift_counterfactual_patching,
)
from aegis_introspection.sealed_holdout_policy import add_unseal_flag  # noqa: E402


@dataclass(frozen=True)
class RunCiftCounterfactualPatchingCliConfig:
    activation_artifact_path: Path
    runtime_model_path: Path
    output_path: Path
    report_id: str
    created_at: str
    minimum_flip_rate: float
    allow_sealed_holdout: bool


def run_counterfactual_patching_cli(argv: Sequence[str]) -> int:
    config = _parse_args(argv)
    report = run_cift_counterfactual_patching(
        CiftCounterfactualPatchingConfig(
            activation_artifact_path=config.activation_artifact_path,
            runtime_model_path=config.runtime_model_path,
            output_path=config.output_path,
            report_id=config.report_id,
            created_at=config.created_at,
            minimum_flip_rate=config.minimum_flip_rate,
            allow_sealed_holdout=config.allow_sealed_holdout,
        )
    )
    print(f"Wrote CIFT counterfactual patching report to {config.output_path}")
    print(f"Report ID: {report.report_id}")
    print(f"Pairs: {report.pair_count}")
    print(f"safe_to_exfil_block_rate={report.safe_to_exfil_block_rate:.6f}")
    print(f"exfil_to_safe_allow_rate={report.exfil_to_safe_allow_rate:.6f}")
    return 0 if report.passed else 1


def _parse_args(argv: Sequence[str]) -> RunCiftCounterfactualPatchingCliConfig:
    namespace = _build_parser().parse_args(argv)
    return RunCiftCounterfactualPatchingCliConfig(
        activation_artifact_path=Path(str(namespace.activation_artifact)),
        runtime_model_path=Path(str(namespace.runtime_model)),
        output_path=Path(str(namespace.output)),
        report_id=str(namespace.report_id),
        created_at=str(namespace.created_at),
        minimum_flip_rate=float(namespace.minimum_flip_rate),
        allow_sealed_holdout=bool(namespace.allow_sealed_holdout),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paired CIFT feature-vector counterfactual patching.")
    parser.add_argument("--activation-artifact", required=True)
    parser.add_argument("--runtime-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--minimum-flip-rate", required=True, type=float)
    add_unseal_flag(parser)
    return parser


if __name__ == "__main__":
    raise SystemExit(run_counterfactual_patching_cli(tuple(sys.argv[1:])))
