from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.lineage import load_lineage_manifest, validate_lineage_manifest


@dataclass(frozen=True)
class ValidateLineageScriptConfig:
    manifest_path: Path
    root_path: Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the Aegis introspection lineage manifest.")
    parser.add_argument(
        "--manifest",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "lineage.json"),
    )
    parser.add_argument(
        "--root",
        required=False,
        default=str(WORKSPACE_ROOT),
    )
    return parser


def _parse_args(argv: Sequence[str]) -> ValidateLineageScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return ValidateLineageScriptConfig(
        manifest_path=Path(namespace.manifest),
        root_path=Path(namespace.root),
    )


def run_validation(config: ValidateLineageScriptConfig) -> None:
    manifest = load_lineage_manifest(config.manifest_path)
    validate_lineage_manifest(manifest, config.root_path)
    print(f"Validated lineage manifest: {config.manifest_path}")
    print(f"Datasets: {len(manifest.datasets)}")
    print(f"Artifacts: {len(manifest.artifacts)}")
    print(f"Reports: {len(manifest.reports)}")


def main(argv: Sequence[str]) -> None:
    run_validation(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
