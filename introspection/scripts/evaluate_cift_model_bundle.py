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

from aegis_introspection.cift_model_bundle_eval import (  # noqa: E402
    CiftModelBundleEvalConfig,
    CiftModelBundleEvalError,
    materialize_cift_model_bundle_eval,
)
from aegis_introspection.sealed_holdout_policy import add_unseal_flag  # noqa: E402


@dataclass(frozen=True)
class EvaluateCiftModelBundleCliConfig:
    activation_artifact_path: Path
    model_bundle_path: Path
    output_path: Path
    report_id: str
    evaluation_split_id: str
    metric_name: str
    created_at: str
    task_name: str
    allow_sealed_holdout: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a frozen CIFT model bundle on an activation artifact.")
    parser.add_argument("--activation-artifact", required=True)
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--evaluation-split-id", required=True)
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--task", required=True)
    add_unseal_flag(parser)
    return parser


def _parse_args(argv: Sequence[str]) -> EvaluateCiftModelBundleCliConfig:
    namespace = _build_parser().parse_args(argv)
    return EvaluateCiftModelBundleCliConfig(
        activation_artifact_path=Path(str(namespace.activation_artifact)),
        model_bundle_path=Path(str(namespace.model_bundle)),
        output_path=Path(str(namespace.output)),
        report_id=str(namespace.report_id),
        evaluation_split_id=str(namespace.evaluation_split_id),
        metric_name=str(namespace.metric_name),
        created_at=str(namespace.created_at),
        task_name=str(namespace.task),
        allow_sealed_holdout=bool(namespace.allow_sealed_holdout),
    )


def _eval_config(config: EvaluateCiftModelBundleCliConfig) -> CiftModelBundleEvalConfig:
    return CiftModelBundleEvalConfig(
        activation_artifact_path=config.activation_artifact_path,
        model_bundle_path=config.model_bundle_path,
        output_path=config.output_path,
        report_id=config.report_id,
        evaluation_split_id=config.evaluation_split_id,
        metric_name=config.metric_name,
        created_at=config.created_at,
        task_name=config.task_name,
        allow_sealed_holdout=config.allow_sealed_holdout,
    )


def run_eval(config: EvaluateCiftModelBundleCliConfig) -> None:
    record = materialize_cift_model_bundle_eval(_eval_config(config))
    print(f"Wrote CIFT model bundle eval to {config.output_path}")
    print(f"Metric: {record['metric_name']}={float(record['metric_value']):.6f}")
    print(f"False negative rate: {float(record['false_negative_rate']):.6f}")
    print(f"False positive rate: {float(record['false_positive_rate']):.6f}")


def main(argv: Sequence[str]) -> int:
    try:
        run_eval(_parse_args(argv))
    except CiftModelBundleEvalError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))

