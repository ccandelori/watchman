from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from aegis.core.contracts import CapabilityMode, ModelInfo
from aegis.trace_collection.harness import (
    build_seed_trace_collection_submissions,
    build_trace_collection_assignments,
    build_trace_collection_records_from_submissions,
    read_trace_collection_assignments_jsonl,
    read_trace_collection_submissions_jsonl,
    write_trace_collection_assignments_jsonl,
    write_trace_collection_jsonl,
    write_trace_collection_submissions_jsonl,
)
from aegis.trace_collection.tasks import default_trace_collection_tasks


@dataclass(frozen=True)
class _AssignmentCliArgs:
    participant_ids: tuple[str, ...]
    output_path: Path


@dataclass(frozen=True)
class _RecordBuilderCliArgs:
    assignments_path: Path
    inputs_path: Path
    output_path: Path
    model: ModelInfo
    capability_mode: CapabilityMode


@dataclass(frozen=True)
class _SeedInputCliArgs:
    assignments_path: Path
    output_path: Path


def run_assignment_cli(argv: tuple[str, ...]) -> None:
    args = _parse_assignment_args(argv)
    assignments = build_trace_collection_assignments(
        participant_ids=args.participant_ids,
        tasks=default_trace_collection_tasks(),
    )
    write_trace_collection_assignments_jsonl(path=args.output_path, assignments=assignments)


def run_record_builder_cli(argv: tuple[str, ...]) -> None:
    args = _parse_record_builder_args(argv)
    assignments = read_trace_collection_assignments_jsonl(path=args.assignments_path)
    submissions = read_trace_collection_submissions_jsonl(path=args.inputs_path)
    records = build_trace_collection_records_from_submissions(
        assignments=assignments,
        submissions=submissions,
        tasks=default_trace_collection_tasks(),
        model=args.model,
        capability_mode=args.capability_mode,
    )
    write_trace_collection_jsonl(path=args.output_path, records=records)


def run_seed_input_cli(argv: tuple[str, ...]) -> None:
    args = _parse_seed_input_args(argv)
    assignments = read_trace_collection_assignments_jsonl(path=args.assignments_path)
    submissions = build_seed_trace_collection_submissions(
        assignments=assignments,
        tasks=default_trace_collection_tasks(),
    )
    write_trace_collection_submissions_jsonl(path=args.output_path, submissions=submissions)


def record_builder_main() -> None:
    run_record_builder_cli(argv=tuple(sys.argv[1:]))


def seed_input_main() -> None:
    run_seed_input_cli(argv=tuple(sys.argv[1:]))


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


def _parse_record_builder_args(argv: tuple[str, ...]) -> _RecordBuilderCliArgs:
    parser = argparse.ArgumentParser(
        prog="aegis-trace-build-records",
        description="Build normalized trace records from assignments and human collection inputs.",
    )
    parser.add_argument("--assignments", required=True, help="Assignment JSONL path.")
    parser.add_argument("--inputs", required=True, help="Human collection input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path for normalized trace records.")
    parser.add_argument("--model-provider", required=True, help="Provider name recorded in ModelInfo.")
    parser.add_argument("--model-id", required=True, help="Model identifier recorded in ModelInfo.")
    parser.add_argument(
        "--capability-mode",
        required=True,
        choices=[mode.value for mode in CapabilityMode],
        help="Runtime capability mode recorded on each NormalizedTurn.",
    )
    namespace = parser.parse_args(list(argv))
    assignments_value: object = namespace.assignments
    inputs_value: object = namespace.inputs
    output_value: object = namespace.output
    model_provider_value: object = namespace.model_provider
    model_id_value: object = namespace.model_id
    capability_mode_value: object = namespace.capability_mode
    if not isinstance(assignments_value, str):
        raise TypeError("--assignments must parse as a string.")
    if not isinstance(inputs_value, str):
        raise TypeError("--inputs must parse as a string.")
    if not isinstance(output_value, str):
        raise TypeError("--output must parse as a string.")
    if not isinstance(model_provider_value, str):
        raise TypeError("--model-provider must parse as a string.")
    if not isinstance(model_id_value, str):
        raise TypeError("--model-id must parse as a string.")
    if not isinstance(capability_mode_value, str):
        raise TypeError("--capability-mode must parse as a string.")
    return _RecordBuilderCliArgs(
        assignments_path=Path(assignments_value),
        inputs_path=Path(inputs_value),
        output_path=Path(output_value),
        model=ModelInfo(provider=model_provider_value, model_id=model_id_value, revision=None, selected_device=None),
        capability_mode=CapabilityMode(capability_mode_value),
    )


def _parse_seed_input_args(argv: tuple[str, ...]) -> _SeedInputCliArgs:
    parser = argparse.ArgumentParser(
        prog="aegis-trace-seed-inputs",
        description="Write deterministic seed collection inputs from assignment packets.",
    )
    parser.add_argument("--assignments", required=True, help="Assignment JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path for seeded collection inputs.")
    namespace = parser.parse_args(list(argv))
    assignments_value: object = namespace.assignments
    output_value: object = namespace.output
    if not isinstance(assignments_value, str):
        raise TypeError("--assignments must parse as a string.")
    if not isinstance(output_value, str):
        raise TypeError("--output must parse as a string.")
    return _SeedInputCliArgs(
        assignments_path=Path(assignments_value),
        output_path=Path(output_value),
    )
