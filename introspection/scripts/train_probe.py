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

from aegis_introspection.artifacts import load_activation_artifact
from aegis_introspection.probe import ProbeTrainingConfig, train_probe_report, write_probe_report_json


@dataclass(frozen=True)
class TrainProbeScriptConfig:
    artifact_path: Path
    output_path: Path
    fold_count: int
    random_seed: int
    max_iter: int
    regularization_c: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline linear probes on saved Aegis activation features.")
    parser.add_argument(
        "--artifact",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "activations" / "qwen3_0_6b_features.pt"),
    )
    parser.add_argument(
        "--output",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "probe_baseline.json"),
    )
    parser.add_argument("--folds", required=False, type=int, default=5)
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument("--max-iter", required=False, type=int, default=1000)
    parser.add_argument("--regularization-c", required=False, type=float, default=1.0)
    return parser


def _parse_args(argv: Sequence[str]) -> TrainProbeScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return TrainProbeScriptConfig(
        artifact_path=Path(namespace.artifact),
        output_path=Path(namespace.output),
        fold_count=int(namespace.folds),
        random_seed=int(namespace.seed),
        max_iter=int(namespace.max_iter),
        regularization_c=float(namespace.regularization_c),
    )


def _probe_config(config: TrainProbeScriptConfig) -> ProbeTrainingConfig:
    return ProbeTrainingConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
    )


def run_training(config: TrainProbeScriptConfig) -> None:
    artifact = load_activation_artifact(config.artifact_path)
    report = train_probe_report(artifact, _probe_config(config))
    write_probe_report_json(config.output_path, report)

    best_feature = next(feature for feature in report.features if feature.feature_key == report.best_feature_key)
    print(f"Wrote probe report to {config.output_path}")
    print(f"best_feature: {best_feature.feature_key}")
    print(f"macro_f1_mean: {best_feature.macro_f1_mean:.4f}")
    print(f"accuracy_mean: {best_feature.accuracy_mean:.4f}")


def main(argv: Sequence[str]) -> None:
    run_training(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
