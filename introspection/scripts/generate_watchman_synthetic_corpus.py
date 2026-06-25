from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
INTROSPECTION_SRC_PATH = INTROSPECTION_ROOT / "src"
RUNTIME_SRC_PATH = WORKSPACE_ROOT / "src"
for source_path in (INTROSPECTION_SRC_PATH, RUNTIME_SRC_PATH):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from aegis_introspection.watchman_synthetic_corpus import (  # noqa: E402
    WatchmanSyntheticCorpusConfig,
    WatchmanSyntheticCorpusError,
    materialize_watchman_synthetic_corpus,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate crossed Watchman synthetic redteam shards from a seed family inventory."
    )
    parser.add_argument("--seed-source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--records-per-shard", required=True, type=int)
    parser.add_argument("--family-record-count", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_args(argv: Sequence[str]) -> WatchmanSyntheticCorpusConfig:
    namespace = _build_parser().parse_args(argv)
    return WatchmanSyntheticCorpusConfig(
        seed_source_dir=Path(str(namespace.seed_source_dir)),
        output_dir=Path(str(namespace.output_dir)),
        manifest_output_path=Path(str(namespace.manifest_output)),
        corpus_id=str(namespace.corpus_id),
        created_at=str(namespace.created_at),
        shard_count=int(namespace.shard_count),
        records_per_shard=int(namespace.records_per_shard),
        family_record_count=int(namespace.family_record_count),
        overwrite=bool(namespace.overwrite),
    )


def main(argv: Sequence[str]) -> int:
    try:
        result = materialize_watchman_synthetic_corpus(_parse_args(argv))
    except WatchmanSyntheticCorpusError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    print(f"Wrote source records: {result.record_count}")
    print(f"Seed source SHA-256: {result.source_corpus_sha256}")
    print(f"Manifest status: {result.manifest['schema_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
