from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
INTROSPECTION_SRC_PATH = INTROSPECTION_ROOT / "src"
RUNTIME_SRC_PATH = WORKSPACE_ROOT / "src"
for source_path in (INTROSPECTION_SRC_PATH, RUNTIME_SRC_PATH):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from aegis_introspection.grok_redteam_corpus import (  # noqa: E402
    GrokRedteamCorpusConfig,
    GrokRedteamCorpusError,
    ingest_grok_redteam_corpus,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and normalize Grok-generated Watchman redteam JSONL shards."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--normalized-output", required=True)
    parser.add_argument("--calibration-output", required=True)
    parser.add_argument("--sealed-holdout-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--calibration-split-id", required=True)
    parser.add_argument("--sealed-holdout-split-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--expected-shard-count", required=True, type=int)
    parser.add_argument("--expected-records-per-shard", required=True, type=int)
    parser.add_argument("--expected-family-record-count", required=True, type=int)
    parser.add_argument("--expected-label-count", required=True, action="append", metavar="LABEL=COUNT")
    parser.add_argument("--hard-near-neighbor-min-rate", required=True, type=float)
    parser.add_argument("--tool-payload-min-rate", required=True, type=float)
    parser.add_argument("--multi-turn-min-rate", required=True, type=float)
    parser.add_argument("--min-unique-message-ratio", required=True, type=float)
    parser.add_argument("--sealed-fraction", required=True, type=float)
    parser.add_argument("--require-family-label-crossing", action="store_true")
    parser.add_argument("--allow-quarantine-output", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_args(argv: Sequence[str]) -> GrokRedteamCorpusConfig:
    namespace = _build_parser().parse_args(argv)
    return GrokRedteamCorpusConfig(
        source_dir=Path(str(namespace.source_dir)),
        normalized_output_path=Path(str(namespace.normalized_output)),
        calibration_output_path=Path(str(namespace.calibration_output)),
        sealed_holdout_output_path=Path(str(namespace.sealed_holdout_output)),
        manifest_output_path=Path(str(namespace.manifest_output)),
        corpus_id=str(namespace.corpus_id),
        calibration_split_id=str(namespace.calibration_split_id),
        sealed_holdout_split_id=str(namespace.sealed_holdout_split_id),
        created_at=str(namespace.created_at),
        expected_shard_count=int(namespace.expected_shard_count),
        expected_records_per_shard=int(namespace.expected_records_per_shard),
        expected_label_counts=_parse_expected_label_counts(
            tuple(str(value) for value in namespace.expected_label_count)
        ),
        expected_family_record_count=int(namespace.expected_family_record_count),
        hard_near_neighbor_min_rate=float(namespace.hard_near_neighbor_min_rate),
        tool_payload_min_rate=float(namespace.tool_payload_min_rate),
        multi_turn_min_rate=float(namespace.multi_turn_min_rate),
        min_unique_message_ratio=float(namespace.min_unique_message_ratio),
        sealed_fraction=float(namespace.sealed_fraction),
        require_family_label_crossing=bool(namespace.require_family_label_crossing),
        allow_quarantine_output=bool(namespace.allow_quarantine_output),
        overwrite=bool(namespace.overwrite),
    )


def _parse_expected_label_counts(raw_values: tuple[str, ...]) -> Mapping[str, int]:
    parsed: dict[str, int] = {}
    for raw_value in raw_values:
        if "=" not in raw_value:
            raise GrokRedteamCorpusError("--expected-label-count must use LABEL=COUNT.")
        label, raw_count = raw_value.split("=", maxsplit=1)
        if label == "":
            raise GrokRedteamCorpusError("--expected-label-count label must not be empty.")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise GrokRedteamCorpusError("--expected-label-count count must be an integer.") from exc
        if count < 1:
            raise GrokRedteamCorpusError("--expected-label-count count must be positive.")
        if label in parsed:
            raise GrokRedteamCorpusError(f"duplicate --expected-label-count for {label}.")
        parsed[label] = count
    return parsed


def main(argv: Sequence[str]) -> int:
    try:
        result = ingest_grok_redteam_corpus(_parse_args(argv))
    except GrokRedteamCorpusError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    print(f"Wrote normalized records: {result.normalized_count}")
    print(f"Wrote calibration split records: {result.calibration_count}")
    print(f"Wrote sealed holdout split records: {result.sealed_holdout_count}")
    print(f"Training eligible: {result.manifest['training_eligible']}")
    print(f"Manifest status: {result.manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
