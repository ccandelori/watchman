from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from aegis.core.contracts import CapabilityMode, ModelInfo
from aegis.trace_collection.harness import (
    PairedPromptValidationConfig,
    SeedInputProfile,
    TraceCollectionAssignment,
    TraceCollectionSubmission,
    TraceCollectionTask,
    build_matched_seed_trace_collection_submissions,
    build_paired_adversarial_seed_trace_collection_submissions,
    build_paired_crossed_action_seed_trace_collection_submissions,
    build_paired_intent_seed_trace_collection_submissions,
    build_paired_natural_seed_trace_collection_submissions,
    build_paired_prompt_work_items,
    build_paired_semantic_indirection_seed_trace_collection_submissions,
    build_paired_semantic_indirection_v3_seed_trace_collection_submissions,
    build_paired_semantic_indirection_v4_seed_trace_collection_submissions,
    build_paired_semantic_indirection_v5_seed_trace_collection_submissions,
    build_paired_semantic_indirection_v6_seed_trace_collection_submissions,
    build_paired_semantic_indirection_v7_seed_trace_collection_submissions,
    build_paired_semantic_indirection_v8_seed_trace_collection_submissions,
    build_pre_output_intent_seed_trace_collection_submissions,
    build_seed_trace_collection_submissions,
    build_trace_collection_assignments,
    build_trace_collection_records_from_submissions,
    build_trace_collection_submissions_from_paired_prompt_completions,
    read_paired_prompt_completions_jsonl,
    read_paired_prompt_work_items_jsonl,
    read_trace_collection_assignments_jsonl,
    read_trace_collection_submissions_jsonl,
    validate_paired_prompt_collection,
    write_paired_prompt_validation_json,
    write_paired_prompt_work_items_jsonl,
    write_trace_collection_assignments_jsonl,
    write_trace_collection_jsonl,
    write_trace_collection_submissions_jsonl,
)
from aegis.trace_collection.tasks import default_trace_collection_tasks

_SeedInputBuilder = Callable[
    [tuple[TraceCollectionAssignment, ...], tuple[TraceCollectionTask, ...], int],
    tuple[TraceCollectionSubmission, ...],
]

_SEED_INPUT_BUILDERS: dict[SeedInputProfile, _SeedInputBuilder] = {
    "standard": build_seed_trace_collection_submissions,
    "matched_hard": build_matched_seed_trace_collection_submissions,
    "pre_output_intent": build_pre_output_intent_seed_trace_collection_submissions,
    "paired_intent": build_paired_intent_seed_trace_collection_submissions,
    "paired_natural": build_paired_natural_seed_trace_collection_submissions,
    "paired_adversarial": build_paired_adversarial_seed_trace_collection_submissions,
    "paired_crossed_action": build_paired_crossed_action_seed_trace_collection_submissions,
    "paired_semantic_indirection": build_paired_semantic_indirection_seed_trace_collection_submissions,
    "paired_semantic_indirection_v3": build_paired_semantic_indirection_v3_seed_trace_collection_submissions,
    "paired_semantic_indirection_v4": build_paired_semantic_indirection_v4_seed_trace_collection_submissions,
    "paired_semantic_indirection_v5": build_paired_semantic_indirection_v5_seed_trace_collection_submissions,
    "paired_semantic_indirection_v6": build_paired_semantic_indirection_v6_seed_trace_collection_submissions,
    "paired_semantic_indirection_v7": build_paired_semantic_indirection_v7_seed_trace_collection_submissions,
    "paired_semantic_indirection_v8": build_paired_semantic_indirection_v8_seed_trace_collection_submissions,
}

_SEED_INPUT_PROFILE_CHOICES: tuple[SeedInputProfile, ...] = tuple(_SEED_INPUT_BUILDERS.keys())


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
    variants_per_label: int
    profile: SeedInputProfile


@dataclass(frozen=True)
class _PairValidationCliArgs:
    assignments_path: Path
    inputs_path: Path
    output_path: Path
    config: PairedPromptValidationConfig


@dataclass(frozen=True)
class _PairWorkItemCliArgs:
    assignments_path: Path
    output_path: Path
    variants_per_pair: int


@dataclass(frozen=True)
class _PairInputBuilderCliArgs:
    work_items_path: Path
    completions_path: Path
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
    builder = _SEED_INPUT_BUILDERS.get(args.profile)
    if builder is None:
        raise ValueError(f"Unsupported seed input profile: {args.profile}.")
    submissions = builder(
        assignments,
        default_trace_collection_tasks(),
        args.variants_per_label,
    )
    write_trace_collection_submissions_jsonl(path=args.output_path, submissions=submissions)


def run_pair_validation_cli(argv: tuple[str, ...]) -> None:
    args = _parse_pair_validation_args(argv)
    assignments = read_trace_collection_assignments_jsonl(path=args.assignments_path)
    submissions = read_trace_collection_submissions_jsonl(path=args.inputs_path)
    report = validate_paired_prompt_collection(
        assignments=assignments,
        submissions=submissions,
        config=args.config,
    )
    write_paired_prompt_validation_json(path=args.output_path, report=report)
    if report.failed_pair_count > 0:
        raise SystemExit(f"paired prompt validation failed for {report.failed_pair_count} pair(s)")


def run_pair_work_item_cli(argv: tuple[str, ...]) -> None:
    args = _parse_pair_work_item_args(argv)
    assignments = read_trace_collection_assignments_jsonl(path=args.assignments_path)
    work_items = build_paired_prompt_work_items(
        assignments=assignments,
        tasks=default_trace_collection_tasks(),
        variants_per_pair=args.variants_per_pair,
    )
    write_paired_prompt_work_items_jsonl(path=args.output_path, work_items=work_items)


def run_pair_input_builder_cli(argv: tuple[str, ...]) -> None:
    args = _parse_pair_input_builder_args(argv)
    work_items = read_paired_prompt_work_items_jsonl(path=args.work_items_path)
    completions = read_paired_prompt_completions_jsonl(path=args.completions_path)
    submissions = build_trace_collection_submissions_from_paired_prompt_completions(
        work_items=work_items,
        completions=completions,
    )
    write_trace_collection_submissions_jsonl(path=args.output_path, submissions=submissions)


def record_builder_main() -> None:
    run_record_builder_cli(argv=tuple(sys.argv[1:]))


def seed_input_main() -> None:
    run_seed_input_cli(argv=tuple(sys.argv[1:]))


def pair_validation_main() -> None:
    run_pair_validation_cli(argv=tuple(sys.argv[1:]))


def pair_work_item_main() -> None:
    run_pair_work_item_cli(argv=tuple(sys.argv[1:]))


def pair_input_builder_main() -> None:
    run_pair_input_builder_cli(argv=tuple(sys.argv[1:]))


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
    parser.add_argument("--variants-per-label", required=True, help="Synthetic variants generated for each assignment.")
    parser.add_argument(
        "--profile",
        required=False,
        default="standard",
        choices=_SEED_INPUT_PROFILE_CHOICES,
        help="Seed input profile to generate.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path for seeded collection inputs.")
    namespace = parser.parse_args(list(argv))
    assignments_value: object = namespace.assignments
    variants_per_label_value: object = namespace.variants_per_label
    profile_value: object = namespace.profile
    output_value: object = namespace.output
    if not isinstance(assignments_value, str):
        raise TypeError("--assignments must parse as a string.")
    if not isinstance(variants_per_label_value, str):
        raise TypeError("--variants-per-label must parse as a string.")
    if not isinstance(profile_value, str):
        raise TypeError("--profile must parse as a string.")
    if not isinstance(output_value, str):
        raise TypeError("--output must parse as a string.")
    variants_per_label = int(variants_per_label_value)
    if variants_per_label < 1:
        raise ValueError("--variants-per-label must be positive.")
    if profile_value not in _SEED_INPUT_BUILDERS:
        allowed_profiles = ", ".join(f"'{profile}'" for profile in _SEED_INPUT_PROFILE_CHOICES)
        raise ValueError(f"--profile must be one of: {allowed_profiles}.")
    return _SeedInputCliArgs(
        assignments_path=Path(assignments_value),
        output_path=Path(output_value),
        variants_per_label=variants_per_label,
        profile=profile_value,
    )


def _parse_pair_validation_args(argv: tuple[str, ...]) -> _PairValidationCliArgs:
    parser = argparse.ArgumentParser(
        prog="aegis-trace-validate-pairs",
        description="Validate paired safe/exfiltration trace collection inputs before CIFT extraction.",
    )
    parser.add_argument("--assignments", required=True, help="Assignment JSONL path.")
    parser.add_argument("--inputs", required=True, help="Collection input JSONL path.")
    parser.add_argument("--maximum-unigram-delta", required=True, help="Maximum token-count delta per pair.")
    parser.add_argument("--minimum-bigram-jaccard", required=True, help="Minimum weighted bigram Jaccard per pair.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    namespace = parser.parse_args(list(argv))
    assignments_value: object = namespace.assignments
    inputs_value: object = namespace.inputs
    output_value: object = namespace.output
    maximum_unigram_delta_value: object = namespace.maximum_unigram_delta
    minimum_bigram_jaccard_value: object = namespace.minimum_bigram_jaccard
    if not isinstance(assignments_value, str):
        raise TypeError("--assignments must parse as a string.")
    if not isinstance(inputs_value, str):
        raise TypeError("--inputs must parse as a string.")
    if not isinstance(output_value, str):
        raise TypeError("--output must parse as a string.")
    if not isinstance(maximum_unigram_delta_value, str):
        raise TypeError("--maximum-unigram-delta must parse as a string.")
    if not isinstance(minimum_bigram_jaccard_value, str):
        raise TypeError("--minimum-bigram-jaccard must parse as a string.")
    return _PairValidationCliArgs(
        assignments_path=Path(assignments_value),
        inputs_path=Path(inputs_value),
        output_path=Path(output_value),
        config=PairedPromptValidationConfig(
            maximum_unigram_delta=int(maximum_unigram_delta_value),
            minimum_bigram_jaccard=float(minimum_bigram_jaccard_value),
        ),
    )


def _parse_pair_work_item_args(argv: tuple[str, ...]) -> _PairWorkItemCliArgs:
    parser = argparse.ArgumentParser(
        prog="aegis-trace-pair-work-items",
        description="Write paired safe/exfiltration paraphrase work items as JSONL.",
    )
    parser.add_argument("--assignments", required=True, help="Assignment JSONL path.")
    parser.add_argument("--variants-per-pair", required=True, help="Paired prompt variants per participant/task.")
    parser.add_argument("--output", required=True, help="Output JSONL path for paired prompt work items.")
    namespace = parser.parse_args(list(argv))
    assignments_value: object = namespace.assignments
    variants_per_pair_value: object = namespace.variants_per_pair
    output_value: object = namespace.output
    if not isinstance(assignments_value, str):
        raise TypeError("--assignments must parse as a string.")
    if not isinstance(variants_per_pair_value, str):
        raise TypeError("--variants-per-pair must parse as a string.")
    if not isinstance(output_value, str):
        raise TypeError("--output must parse as a string.")
    variants_per_pair = int(variants_per_pair_value)
    if variants_per_pair < 1:
        raise ValueError("--variants-per-pair must be positive.")
    return _PairWorkItemCliArgs(
        assignments_path=Path(assignments_value),
        output_path=Path(output_value),
        variants_per_pair=variants_per_pair,
    )


def _parse_pair_input_builder_args(argv: tuple[str, ...]) -> _PairInputBuilderCliArgs:
    parser = argparse.ArgumentParser(
        prog="aegis-trace-build-pair-inputs",
        description="Build collection inputs from paired prompt work items and completions.",
    )
    parser.add_argument("--work-items", required=True, help="Paired prompt work-item JSONL path.")
    parser.add_argument("--completions", required=True, help="Paired prompt completion JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path for collection inputs.")
    namespace = parser.parse_args(list(argv))
    work_items_value: object = namespace.work_items
    completions_value: object = namespace.completions
    output_value: object = namespace.output
    if not isinstance(work_items_value, str):
        raise TypeError("--work-items must parse as a string.")
    if not isinstance(completions_value, str):
        raise TypeError("--completions must parse as a string.")
    if not isinstance(output_value, str):
        raise TypeError("--output must parse as a string.")
    return _PairInputBuilderCliArgs(
        work_items_path=Path(work_items_value),
        completions_path=Path(completions_value),
        output_path=Path(output_value),
    )
