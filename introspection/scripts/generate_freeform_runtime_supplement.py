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

from aegis_introspection.freeform_supplement import (  # noqa: E402
    FreeformSupplementConfig,
    FreeformSupplementError,
    generate_freeform_runtime_supplement,
)


@dataclass(frozen=True)
class GenerateFreeformRuntimeSupplementCliConfig:
    normalized_output_path: Path
    calibration_output_path: Path
    sealed_holdout_output_path: Path
    manifest_output_path: Path
    corpus_id: str
    calibration_split_id: str
    sealed_holdout_split_id: str
    created_at: str
    records_per_family_label: int
    sealed_every: int
    overwrite: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate focused freeform CIFT runtime supplement records.")
    parser.add_argument("--normalized-output", required=True)
    parser.add_argument("--calibration-output", required=True)
    parser.add_argument("--sealed-holdout-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--calibration-split-id", required=True)
    parser.add_argument("--sealed-holdout-split-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--records-per-family-label", required=True, type=int)
    parser.add_argument("--sealed-every", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_args(argv: Sequence[str]) -> GenerateFreeformRuntimeSupplementCliConfig:
    namespace = _build_parser().parse_args(argv)
    return GenerateFreeformRuntimeSupplementCliConfig(
        normalized_output_path=Path(str(namespace.normalized_output)),
        calibration_output_path=Path(str(namespace.calibration_output)),
        sealed_holdout_output_path=Path(str(namespace.sealed_holdout_output)),
        manifest_output_path=Path(str(namespace.manifest_output)),
        corpus_id=str(namespace.corpus_id),
        calibration_split_id=str(namespace.calibration_split_id),
        sealed_holdout_split_id=str(namespace.sealed_holdout_split_id),
        created_at=str(namespace.created_at),
        records_per_family_label=int(namespace.records_per_family_label),
        sealed_every=int(namespace.sealed_every),
        overwrite=bool(namespace.overwrite),
    )


def _supplement_config(config: GenerateFreeformRuntimeSupplementCliConfig) -> FreeformSupplementConfig:
    return FreeformSupplementConfig(
        normalized_output_path=config.normalized_output_path,
        calibration_output_path=config.calibration_output_path,
        sealed_holdout_output_path=config.sealed_holdout_output_path,
        manifest_output_path=config.manifest_output_path,
        corpus_id=config.corpus_id,
        calibration_split_id=config.calibration_split_id,
        sealed_holdout_split_id=config.sealed_holdout_split_id,
        created_at=config.created_at,
        records_per_family_label=config.records_per_family_label,
        sealed_every=config.sealed_every,
        overwrite=config.overwrite,
    )


def main(argv: Sequence[str]) -> int:
    try:
        result = generate_freeform_runtime_supplement(_supplement_config(_parse_args(argv)))
    except FreeformSupplementError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    print(f"Wrote normalized supplement records: {result.normalized_count}")
    print(f"Wrote calibration supplement records: {result.calibration_count}")
    print(f"Wrote sealed holdout supplement records: {result.sealed_holdout_count}")
    print(f"Training eligible: {result.manifest['training_eligible']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
