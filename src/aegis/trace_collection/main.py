from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from aegis.trace_collection.harness import (
    build_trace_collection_assignments,
    write_trace_collection_assignments_jsonl,
)
from aegis.trace_collection.tasks import default_trace_collection_tasks


@dataclass(frozen=True)
class _AssignmentCliArgs:
    participant_ids: tuple[str, ...]
    output_path: Path


def run_assignment_cli(argv: tuple[str, ...]) -> None:
    args = _parse_assignment_args(argv)
    assignments = build_trace_collection_assignments(
        participant_ids=args.participant_ids,
        tasks=default_trace_collection_tasks(),
    )
    write_trace_collection_assignments_jsonl(path=args.output_path, assignments=assignments)


def _parse_assignment_args(argv: tuple[str, ...]) -> _AssignmentCliArgs:
    parser = argparse.ArgumentParser(
        prog="aegis-trace-assignments",
        description="Write controlled trace-collection assignment packets as JSONL.",
    )
    parser.add_argument(
        "--participant",
        action="append",
        dest="participants",
        required=True,
        help="Participant identifier. Repeat once per human operator.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for assignment packets.",
    )
    namespace = parser.parse_args(list(argv))
    participants_value: object = namespace.participants
    output_value: object = namespace.output
    if not isinstance(participants_value, list) or not all(isinstance(item, str) for item in participants_value):
        raise TypeError("--participant values must parse as strings.")
    if not isinstance(output_value, str):
        raise TypeError("--output must parse as a string.")
    return _AssignmentCliArgs(
        participant_ids=tuple(participants_value),
        output_path=Path(output_value),
    )
