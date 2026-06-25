from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.cift_training_artifact_conversion import (  # noqa: E402
    CiftTrainingArtifactConversionConfig,
    convert_cift_training_artifact,
)
from aegis_introspection.sealed_holdout_policy import add_unseal_flag  # noqa: E402


@dataclass(frozen=True)
class ConvertCiftTrainingArtifactCliConfig:
    source_path: Path
    output_path: Path
    allow_sealed_holdout: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a CIFT activation artifact to the dependency-clean training artifact pickle format."
    )
    parser.add_argument(
        "--source",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "activations" / "qwen3_0_6b_dp_honey_lite_v4_1_selector_windows.pt"),
    )
    parser.add_argument(
        "--output",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "activations" / "qwen3_0_6b_dp_honey_lite_v4_1_selector_windows.pkl"),
    )
    add_unseal_flag(parser)
    return parser


def _parse_args(argv: Sequence[str]) -> ConvertCiftTrainingArtifactCliConfig:
    namespace = _build_parser().parse_args(argv)
    return ConvertCiftTrainingArtifactCliConfig(
        source_path=Path(namespace.source),
        output_path=Path(namespace.output),
        allow_sealed_holdout=bool(namespace.allow_sealed_holdout),
    )


def _conversion_config(config: ConvertCiftTrainingArtifactCliConfig) -> CiftTrainingArtifactConversionConfig:
    return CiftTrainingArtifactConversionConfig(
        source_path=config.source_path,
        output_path=config.output_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
    )


def run_conversion(config: ConvertCiftTrainingArtifactCliConfig) -> None:
    report = convert_cift_training_artifact(_conversion_config(config))
    print(f"Wrote CIFT training artifact to {report.output_path}")
    print(f"Source artifact SHA-256: {report.source_artifact_sha256}")
    print(f"Examples: {report.example_count}")
    print(f"Features: {', '.join(report.feature_keys)}")


def main(argv: Sequence[str]) -> None:
    run_conversion(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
