from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias, cast

from aegis.core.contracts import CapabilityMode, JsonValue, ModelInfo
from aegis.trace_collection import (
    TraceCollectionAssignment,
    TraceCollectionSubmission,
    TraceCollectionTask,
    build_paired_semantic_indirection_v5_seed_trace_collection_submissions,
    build_paired_semantic_indirection_v8_seed_trace_collection_submissions,
    build_trace_collection_assignments,
    build_trace_collection_records_from_submissions,
    default_trace_collection_tasks,
    write_trace_collection_jsonl,
)
from aegis_introspection.prompts import load_structured_prompt_examples
from aegis_introspection.runtime_bridge import RuntimeBridgeConfig, structured_prompt_to_normalized_turn
from aegis_introspection.runtime_requests import load_runtime_requests_jsonl
from aegis_introspection.sealed_holdout_policy import SEALED_HOLDOUT_TAG
from aegis_introspection.trace_record_adapter import (
    StructuredTracePromptRecord,
    TokenOffsetEncoder,
    TracePromptConversionConfig,
    structured_prompt_records_from_trace_records,
    write_structured_prompt_jsonl,
)

JsonObject = dict[str, JsonValue]
SelectedChoiceSealedHoldoutSourceProfile: TypeAlias = Literal[
    "paired_semantic_indirection_v5",
    "paired_semantic_indirection_v8",
]
_SourceProfileBuilder: TypeAlias = Callable[
    [tuple[TraceCollectionAssignment, ...], tuple[TraceCollectionTask, ...], int],
    tuple[TraceCollectionSubmission, ...],
]

_SCHEMA_VERSION = "aegis_introspection.selected_choice_sealed_holdout_manifest/v1"
_SELECTED_CHOICE = "selected_choice"
_SAFE_LABEL = "secret_present_safe"
_EXFIL_LABEL = "exfiltration_intent"
_SOURCE_PROFILE_BUILDERS: dict[SelectedChoiceSealedHoldoutSourceProfile, _SourceProfileBuilder] = {
    "paired_semantic_indirection_v5": build_paired_semantic_indirection_v5_seed_trace_collection_submissions,
    "paired_semantic_indirection_v8": build_paired_semantic_indirection_v8_seed_trace_collection_submissions,
}
_SEALED_PROFILE_BY_SOURCE_PROFILE: dict[SelectedChoiceSealedHoldoutSourceProfile, str] = {
    "paired_semantic_indirection_v5": "paired_semantic_indirection_v5_sealed",
    "paired_semantic_indirection_v8": "paired_semantic_indirection_v8_sealed",
}


class SelectedChoiceSealedHoldoutError(ValueError):
    """Raised when selected-choice sealed holdout materialization cannot proceed."""


@dataclass(frozen=True)
class SelectedChoiceSealedHoldoutConfig:
    trace_records_path: Path
    structured_prompts_path: Path
    runtime_turns_path: Path
    manifest_path: Path
    corpus_id: str
    sealed_holdout_split_id: str
    source_profile: SelectedChoiceSealedHoldoutSourceProfile
    participant_ids: tuple[str, ...]
    variants_per_label: int
    model_provider: str
    model_id: str
    revision: str
    selected_device: str
    session_id: str
    sensitive_source: str
    readout_token_count: int
    capability_mode: CapabilityMode
    created_at: str
    overwrite: bool


@dataclass(frozen=True)
class SelectedChoiceSealedHoldoutResult:
    manifest: JsonObject
    source_trace_record_count: int
    selected_choice_record_count: int
    runtime_turn_count: int


def supported_selected_choice_sealed_holdout_source_profiles() -> tuple[str, ...]:
    return tuple(_SOURCE_PROFILE_BUILDERS.keys())


def materialize_selected_choice_sealed_holdout(
    config: SelectedChoiceSealedHoldoutConfig,
    encoder: TokenOffsetEncoder,
) -> SelectedChoiceSealedHoldoutResult:
    _validate_config(config=config)
    _ensure_output_paths(config=config)

    tasks = default_trace_collection_tasks()
    assignments = build_trace_collection_assignments(participant_ids=config.participant_ids, tasks=tasks)
    source_profile_builder = _source_profile_builder(source_profile=config.source_profile)
    submissions = source_profile_builder(
        assignments=assignments,
        tasks=tasks,
        variants_per_label=config.variants_per_label,
    )
    trace_records = build_trace_collection_records_from_submissions(
        assignments=assignments,
        submissions=submissions,
        tasks=tasks,
        model=ModelInfo(
            provider=config.model_provider,
            model_id=config.model_id,
            revision=config.revision,
            selected_device=config.selected_device,
        ),
        capability_mode=config.capability_mode,
    )
    conversion = structured_prompt_records_from_trace_records(
        records=tuple(record.to_dict() for record in trace_records),
        encoder=encoder,
        config=TracePromptConversionConfig(readout_token_count=config.readout_token_count),
    )
    if len(conversion.skipped_records) > 0:
        skipped_ids = ", ".join(skipped.record_id for skipped in conversion.skipped_records)
        raise SelectedChoiceSealedHoldoutError(f"Trace conversion skipped records: {skipped_ids}.")

    selected_records = _sealed_selected_choice_records(
        records=conversion.records,
        config=config,
        expected_count=len(tasks) * len(config.participant_ids) * 2 * config.variants_per_label,
    )
    runtime_turns = _runtime_turns_from_records(records=selected_records, config=config)

    write_trace_collection_jsonl(path=config.trace_records_path, records=trace_records)
    write_structured_prompt_jsonl(path=config.structured_prompts_path, records=selected_records)
    _write_runtime_turns(path=config.runtime_turns_path, turns=runtime_turns)

    _validate_written_structured_prompts(
        path=config.structured_prompts_path,
        expected_count=len(selected_records),
        config=config,
    )
    _validate_written_runtime_turns(path=config.runtime_turns_path, expected_count=len(runtime_turns), config=config)

    manifest = _manifest(
        config=config,
        source_trace_record_count=len(trace_records),
        selected_records=selected_records,
        runtime_turns=runtime_turns,
    )
    _write_json(path=config.manifest_path, record=manifest)
    return SelectedChoiceSealedHoldoutResult(
        manifest=manifest,
        source_trace_record_count=len(trace_records),
        selected_choice_record_count=len(selected_records),
        runtime_turn_count=len(runtime_turns),
    )


def _sealed_selected_choice_records(
    records: tuple[StructuredTracePromptRecord, ...],
    config: SelectedChoiceSealedHoldoutConfig,
    expected_count: int,
) -> tuple[StructuredTracePromptRecord, ...]:
    selected: list[StructuredTracePromptRecord] = []
    for record in records:
        if record.label == "benign":
            continue
        if record.label not in (_SAFE_LABEL, _EXFIL_LABEL):
            raise SelectedChoiceSealedHoldoutError(f"Unsupported selected-choice label: {record.label}.")
        if record.selected_choice_char_span is None:
            raise SelectedChoiceSealedHoldoutError(f"{record.id} missing selected_choice_char_span.")
        if record.selected_choice_token_span is None:
            raise SelectedChoiceSealedHoldoutError(f"{record.id} missing selected_choice_token_span.")
        if record.selected_choice_readout_token_indices is None:
            raise SelectedChoiceSealedHoldoutError(f"{record.id} missing selected_choice_readout_token_indices.")
        if len(record.selected_choice_readout_token_indices) == 0:
            raise SelectedChoiceSealedHoldoutError(f"{record.id} selected_choice_readout_token_indices is empty.")
        if record.fallback_reason is not None:
            raise SelectedChoiceSealedHoldoutError(f"{record.id} has fallback_reason: {record.fallback_reason}.")
        selected.append(replace(record, tags=_sealed_tags(record=record, config=config)))

    sealed_records = tuple(selected)
    if len(sealed_records) != expected_count:
        raise SelectedChoiceSealedHoldoutError(
            f"Expected {expected_count} selected-choice rows, found {len(sealed_records)}."
        )
    _validate_balanced_rows(records=sealed_records)
    if len({record.text for record in sealed_records}) != len(sealed_records):
        raise SelectedChoiceSealedHoldoutError("Selected-choice sealed holdout contains duplicate prompt text.")
    return sealed_records


def _sealed_tags(record: StructuredTracePromptRecord, config: SelectedChoiceSealedHoldoutConfig) -> tuple[str, ...]:
    tags = list(record.tags)
    sealed_profile = _sealed_profile_for_source_profile(source_profile=config.source_profile)
    for tag in (
        SEALED_HOLDOUT_TAG,
        f"corpus:{config.corpus_id}",
        f"split:{config.sealed_holdout_split_id}",
        f"profile:{sealed_profile}",
        f"source_profile:{config.source_profile}",
        f"model:{config.model_id}",
    ):
        if tag not in tags:
            tags.append(tag)
    return tuple(tags)


def _runtime_turns_from_records(
    records: tuple[StructuredTracePromptRecord, ...],
    config: SelectedChoiceSealedHoldoutConfig,
) -> tuple[JsonObject, ...]:
    turns: list[JsonObject] = []
    for turn_index, record in enumerate(records, start=1):
        turn = structured_prompt_to_normalized_turn(
            record=record.to_dict(),
            config=RuntimeBridgeConfig(
                trace_id=record.source_trace_id,
                session_id=config.session_id,
                turn_index=turn_index,
                capability_mode=config.capability_mode.value,
                model_provider=config.model_provider,
                model_id=config.model_id,
                revision=config.revision,
                selected_device=config.selected_device,
                sensitive_source=config.sensitive_source,
            ),
        )
        metadata = _required_json_object(turn, "metadata", f"runtime turn {turn_index}")
        eval_metadata = _required_json_object(metadata, "eval", f"runtime turn {turn_index}.metadata")
        eval_metadata["expected_cift_window_family"] = _SELECTED_CHOICE
        eval_metadata["sealed_holdout_split_id"] = config.sealed_holdout_split_id
        eval_metadata["corpus_id"] = config.corpus_id
        turns.append(turn)
    return tuple(turns)


def _manifest(
    config: SelectedChoiceSealedHoldoutConfig,
    source_trace_record_count: int,
    selected_records: tuple[StructuredTracePromptRecord, ...],
    runtime_turns: tuple[JsonObject, ...],
) -> JsonObject:
    return {
        "schema_version": _SCHEMA_VERSION,
        "corpus_id": config.corpus_id,
        "sealed_holdout_split_id": config.sealed_holdout_split_id,
        "sealed_holdout": True,
        "profile": _sealed_profile_for_source_profile(source_profile=config.source_profile),
        "source_profile": config.source_profile,
        "model_provider": config.model_provider,
        "model_id": config.model_id,
        "revision": config.revision,
        "selected_device": config.selected_device,
        "participant_ids": list(config.participant_ids),
        "variants_per_label": config.variants_per_label,
        "readout_token_count": config.readout_token_count,
        "capability_mode": config.capability_mode.value,
        "session_id": config.session_id,
        "sensitive_source": config.sensitive_source,
        "source_trace_record_count": source_trace_record_count,
        "selected_choice_record_count": len(selected_records),
        "runtime_turn_count": len(runtime_turns),
        "label_counts": _label_counts(records=selected_records),
        "family_counts": _family_counts(records=selected_records),
        "structured_prompts_path": str(config.structured_prompts_path),
        "structured_prompts_sha256": _sha256_file(config.structured_prompts_path),
        "runtime_turns_path": str(config.runtime_turns_path),
        "runtime_turns_sha256": _sha256_file(config.runtime_turns_path),
        "trace_records_path": str(config.trace_records_path),
        "trace_records_sha256": _sha256_file(config.trace_records_path),
        "created_at": config.created_at,
    }


def _validate_config(config: SelectedChoiceSealedHoldoutConfig) -> None:
    for field_name, value in (
        ("corpus_id", config.corpus_id),
        ("sealed_holdout_split_id", config.sealed_holdout_split_id),
        ("source_profile", config.source_profile),
        ("model_provider", config.model_provider),
        ("model_id", config.model_id),
        ("revision", config.revision),
        ("selected_device", config.selected_device),
        ("session_id", config.session_id),
        ("sensitive_source", config.sensitive_source),
        ("created_at", config.created_at),
    ):
        if value == "":
            raise SelectedChoiceSealedHoldoutError(f"{field_name} must not be empty.")
    if config.source_profile not in _SOURCE_PROFILE_BUILDERS:
        supported = ", ".join(supported_selected_choice_sealed_holdout_source_profiles())
        raise SelectedChoiceSealedHoldoutError(f"source_profile must be one of: {supported}.")
    if SEALED_HOLDOUT_TAG not in config.sealed_holdout_split_id and "sealed" not in config.sealed_holdout_split_id:
        raise SelectedChoiceSealedHoldoutError("sealed_holdout_split_id must explicitly identify a sealed split.")
    if len(config.participant_ids) == 0:
        raise SelectedChoiceSealedHoldoutError("participant_ids must not be empty.")
    for participant_id in config.participant_ids:
        if participant_id == "":
            raise SelectedChoiceSealedHoldoutError("participant_ids must not contain empty values.")
    if config.variants_per_label < 1:
        raise SelectedChoiceSealedHoldoutError("variants_per_label must be positive.")
    if config.readout_token_count < 1:
        raise SelectedChoiceSealedHoldoutError("readout_token_count must be positive.")


def _ensure_output_paths(config: SelectedChoiceSealedHoldoutConfig) -> None:
    if config.overwrite:
        return
    for field_name, path in (
        ("trace_records_path", config.trace_records_path),
        ("structured_prompts_path", config.structured_prompts_path),
        ("runtime_turns_path", config.runtime_turns_path),
        ("manifest_path", config.manifest_path),
    ):
        if path.exists():
            raise SelectedChoiceSealedHoldoutError(f"{field_name} already exists: {path}.")


def _validate_balanced_rows(records: tuple[StructuredTracePromptRecord, ...]) -> None:
    label_counts = _label_counts(records=records)
    if label_counts.get(_SAFE_LABEL) != label_counts.get(_EXFIL_LABEL):
        raise SelectedChoiceSealedHoldoutError(f"Selected-choice label counts must balance: {label_counts}.")
    family_label_counts: Counter[tuple[str, str]] = Counter((record.family, record.label) for record in records)
    for family in {record.family for record in records}:
        safe_count = family_label_counts[(family, _SAFE_LABEL)]
        exfil_count = family_label_counts[(family, _EXFIL_LABEL)]
        if safe_count != exfil_count:
            raise SelectedChoiceSealedHoldoutError(f"Family {family} label counts must balance.")


def _validate_written_structured_prompts(
    path: Path,
    expected_count: int,
    config: SelectedChoiceSealedHoldoutConfig,
) -> None:
    sealed_profile = _sealed_profile_for_source_profile(source_profile=config.source_profile)
    examples = load_structured_prompt_examples(path)
    if len(examples) != expected_count:
        raise SelectedChoiceSealedHoldoutError(f"Structured prompt count mismatch for {path}.")
    for index, example in enumerate(examples, start=1):
        if SEALED_HOLDOUT_TAG not in example.tags:
            raise SelectedChoiceSealedHoldoutError(f"Structured prompt {index} missing sealed_holdout tag.")
        if f"profile:{sealed_profile}" not in example.tags:
            raise SelectedChoiceSealedHoldoutError(f"Structured prompt {index} missing sealed profile tag.")
        if f"source_profile:{config.source_profile}" not in example.tags:
            raise SelectedChoiceSealedHoldoutError(f"Structured prompt {index} missing source profile tag.")
        if example.selected_choice_token_span is None:
            raise SelectedChoiceSealedHoldoutError(f"Structured prompt {index} missing selected-choice token span.")
        if example.selected_choice_readout_token_indices is None:
            raise SelectedChoiceSealedHoldoutError(
                f"Structured prompt {index} missing selected-choice readout indices."
            )
        if example.fallback_reason is not None:
            raise SelectedChoiceSealedHoldoutError(f"Structured prompt {index} has fallback_reason.")


def _validate_written_runtime_turns(
    path: Path,
    expected_count: int,
    config: SelectedChoiceSealedHoldoutConfig,
) -> None:
    sealed_profile = _sealed_profile_for_source_profile(source_profile=config.source_profile)
    requests = load_runtime_requests_jsonl(path)
    if len(requests) != expected_count:
        raise SelectedChoiceSealedHoldoutError(f"Runtime turn count mismatch for {path}.")
    for index, request in enumerate(requests, start=1):
        eval_metadata = _required_json_object(request.metadata, "eval", f"runtime request {index}.metadata")
        tags = eval_metadata.get("tags")
        if not isinstance(tags, list):
            raise SelectedChoiceSealedHoldoutError(f"Runtime request {index} tags must be a list.")
        if SEALED_HOLDOUT_TAG not in tags:
            raise SelectedChoiceSealedHoldoutError(f"Runtime request {index} missing sealed_holdout tag.")
        if f"profile:{sealed_profile}" not in tags:
            raise SelectedChoiceSealedHoldoutError(f"Runtime request {index} missing sealed profile tag.")
        if f"source_profile:{config.source_profile}" not in tags:
            raise SelectedChoiceSealedHoldoutError(f"Runtime request {index} missing source profile tag.")
        expected_window = eval_metadata.get("expected_cift_window_family")
        if expected_window != _SELECTED_CHOICE:
            raise SelectedChoiceSealedHoldoutError(f"Runtime request {index} is not selected-choice.")
        cift_metadata = _required_json_object(request.metadata, "cift", f"runtime request {index}.metadata")
        for field_name in (
            "selected_choice_char_span",
            "selected_choice_token_span",
            "selected_choice_readout_token_indices",
        ):
            if cift_metadata.get(field_name) is None:
                raise SelectedChoiceSealedHoldoutError(f"Runtime request {index} missing {field_name}.")


def _source_profile_builder(source_profile: SelectedChoiceSealedHoldoutSourceProfile) -> _SourceProfileBuilder:
    builder = _SOURCE_PROFILE_BUILDERS.get(source_profile)
    if builder is None:
        supported = ", ".join(supported_selected_choice_sealed_holdout_source_profiles())
        raise SelectedChoiceSealedHoldoutError(f"source_profile must be one of: {supported}.")
    return builder


def _sealed_profile_for_source_profile(source_profile: SelectedChoiceSealedHoldoutSourceProfile) -> str:
    sealed_profile = _SEALED_PROFILE_BY_SOURCE_PROFILE.get(source_profile)
    if sealed_profile is None:
        supported = ", ".join(supported_selected_choice_sealed_holdout_source_profiles())
        raise SelectedChoiceSealedHoldoutError(f"source_profile must be one of: {supported}.")
    return sealed_profile


def _required_json_object(record: Mapping[str, JsonValue], field_name: str, context: str) -> dict[str, JsonValue]:
    value = record.get(field_name)
    if not isinstance(value, dict):
        raise SelectedChoiceSealedHoldoutError(f"{context}.{field_name} must be an object.")
    return cast(dict[str, JsonValue], value)


def _label_counts(records: tuple[StructuredTracePromptRecord, ...]) -> dict[str, int]:
    return dict(Counter(record.label for record in records))


def _family_counts(records: tuple[StructuredTracePromptRecord, ...]) -> dict[str, int]:
    return dict(Counter(record.family for record in records))


def _write_runtime_turns(path: Path, turns: tuple[Mapping[str, JsonValue], ...]) -> None:
    if len(turns) == 0:
        raise SelectedChoiceSealedHoldoutError("Cannot write an empty runtime turn dataset.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for turn in turns:
            output_file.write(json.dumps(turn, sort_keys=True))
            output_file.write("\n")


def _write_json(path: Path, record: Mapping[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
