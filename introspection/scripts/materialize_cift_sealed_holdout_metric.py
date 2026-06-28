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

from aegis_introspection.cift_sealed_holdout_metric import (  # noqa: E402
    CiftSealedHoldoutMetricConfig,
    CiftSealedHoldoutMetricError,
    materialize_cift_sealed_holdout_metric,
)
from aegis_introspection.sealed_holdout_policy import add_unseal_flag  # noqa: E402


@dataclass(frozen=True)
class MaterializeCiftSealedHoldoutMetricCliConfig:
    runtime_report_path: Path
    runtime_turns_path: Path
    selected_choice_runtime_model_path: Path
    output_path: Path
    report_id: str
    sealed_holdout_split_id: str
    metric_name: str
    created_at: str
    allow_sealed_holdout: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize aegis_introspection.cift_sealed_holdout_metric/v1 from live CIFT evidence."
    )
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--runtime-turns", required=True)
    parser.add_argument(
        "--runtime-model",
        "--selected-choice-runtime-model",
        dest="selected_choice_runtime_model",
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--sealed-holdout-split-id", required=True)
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--created-at", required=True)
    add_unseal_flag(parser)
    return parser


def _parse_args(argv: Sequence[str]) -> MaterializeCiftSealedHoldoutMetricCliConfig:
    namespace = _build_parser().parse_args(argv)
    return MaterializeCiftSealedHoldoutMetricCliConfig(
        runtime_report_path=Path(str(namespace.runtime_report)),
        runtime_turns_path=Path(str(namespace.runtime_turns)),
        selected_choice_runtime_model_path=Path(str(namespace.selected_choice_runtime_model)),
        output_path=Path(str(namespace.output)),
        report_id=str(namespace.report_id),
        sealed_holdout_split_id=str(namespace.sealed_holdout_split_id),
        metric_name=str(namespace.metric_name),
        created_at=str(namespace.created_at),
        allow_sealed_holdout=bool(namespace.allow_sealed_holdout),
    )


def _metric_config(config: MaterializeCiftSealedHoldoutMetricCliConfig) -> CiftSealedHoldoutMetricConfig:
    return CiftSealedHoldoutMetricConfig(
        runtime_report_path=config.runtime_report_path,
        runtime_turns_path=config.runtime_turns_path,
        selected_choice_runtime_model_path=config.selected_choice_runtime_model_path,
        output_path=config.output_path,
        report_id=config.report_id,
        sealed_holdout_split_id=config.sealed_holdout_split_id,
        metric_name=config.metric_name,
        created_at=config.created_at,
        allow_sealed_holdout=config.allow_sealed_holdout,
    )


def run_materializer(config: MaterializeCiftSealedHoldoutMetricCliConfig) -> None:
    record = materialize_cift_sealed_holdout_metric(_metric_config(config))
    metric_value = record["metric_value"]
    false_negative_rate = record["false_negative_rate"]
    false_positive_rate = record["false_positive_rate"]
    print(f"Wrote CIFT sealed holdout metric to {config.output_path}")
    print(f"Metric: {config.metric_name}={float(metric_value):.6f}")
    print(f"False negative rate: {float(false_negative_rate):.6f}")
    print(f"False positive rate: {float(false_positive_rate):.6f}")


def main(argv: Sequence[str]) -> int:
    try:
        run_materializer(_parse_args(argv))
    except CiftSealedHoldoutMetricError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
