from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

PromptLabel: TypeAlias = Literal["benign", "secret_present_safe", "exfiltration_intent"]

_VALID_LABELS: frozenset[str] = frozenset(("benign", "secret_present_safe", "exfiltration_intent"))


class PromptDataError(ValueError):
    """Raised when a prompt dataset entry is malformed."""


@dataclass(frozen=True)
class PromptExample:
    id: str
    label: PromptLabel
    family: str
    text: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class PromptTokenSpan:
    start: int
    end: int


@dataclass(frozen=True)
class StructuredPromptExample:
    id: str
    label: PromptLabel
    family: str
    text: str
    tags: tuple[str, ...]
    secret_token_span: PromptTokenSpan | None
    query_token_span: PromptTokenSpan
    payload_token_span: PromptTokenSpan | None
    readout_token_indices: tuple[int, ...]
    query_tail_readout_token_indices: tuple[int, ...] | None
    selected_choice_token_span: PromptTokenSpan | None
    selected_choice_readout_token_indices: tuple[int, ...] | None
    fallback_reason: str | None


def _as_mapping(value: object, line_number: int) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise PromptDataError(f"Line {line_number}: expected a JSON object.")
    return cast(Mapping[str, object], value)


def _required_string(record: Mapping[str, object], field_name: str, line_number: int) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise PromptDataError(f"Line {line_number}: field '{field_name}' must be a string.")
    if value == "":
        raise PromptDataError(f"Line {line_number}: field '{field_name}' must not be empty.")
    return value


def _optional_string(record: Mapping[str, object], field_name: str, line_number: int) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise PromptDataError(f"Line {line_number}: field '{field_name}' must be a string or null.")
    if value == "":
        raise PromptDataError(f"Line {line_number}: field '{field_name}' must not be empty.")
    return value


def _required_label(record: Mapping[str, object], line_number: int) -> PromptLabel:
    value = _required_string(record, "label", line_number)
    if value not in _VALID_LABELS:
        valid = ", ".join(sorted(_VALID_LABELS))
        raise PromptDataError(f"Line {line_number}: label '{value}' is invalid. Expected one of: {valid}.")
    return cast(PromptLabel, value)


def _required_tags(record: Mapping[str, object], line_number: int) -> tuple[str, ...]:
    value = record.get("tags")
    if not isinstance(value, list):
        raise PromptDataError(f"Line {line_number}: field 'tags' must be a list of strings.")

    tags: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise PromptDataError(f"Line {line_number}: tag at index {index} must be a string.")
        if item == "":
            raise PromptDataError(f"Line {line_number}: tag at index {index} must not be empty.")
        tags.append(item)
    return tuple(tags)


def _required_token_span(record: Mapping[str, object], field_name: str, line_number: int) -> PromptTokenSpan:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise PromptDataError(f"Line {line_number}: field '{field_name}' must be a two-integer list.")
    if len(value) != 2:
        raise PromptDataError(f"Line {line_number}: field '{field_name}' must contain exactly two integers.")
    start, end = value
    if not isinstance(start, int) or not isinstance(end, int):
        raise PromptDataError(f"Line {line_number}: field '{field_name}' must contain integers.")
    if start < 0:
        raise PromptDataError(f"Line {line_number}: field '{field_name}' start must be non-negative.")
    if end <= start:
        raise PromptDataError(f"Line {line_number}: field '{field_name}' must be a non-empty half-open span.")
    return PromptTokenSpan(start=start, end=end)


def _optional_token_span(record: Mapping[str, object], field_name: str, line_number: int) -> PromptTokenSpan | None:
    value = record.get(field_name)
    if value is None:
        return None
    return _required_token_span(record=record, field_name=field_name, line_number=line_number)


def _required_token_indices(record: Mapping[str, object], field_name: str, line_number: int) -> tuple[int, ...]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise PromptDataError(f"Line {line_number}: field '{field_name}' must be a list of integers.")
    if len(value) == 0:
        raise PromptDataError(f"Line {line_number}: field '{field_name}' must not be empty.")

    indices: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int):
            raise PromptDataError(f"Line {line_number}: {field_name} index {index} must be an integer.")
        if item < 0:
            raise PromptDataError(f"Line {line_number}: {field_name} index {index} must be non-negative.")
        indices.append(item)
    if tuple(indices) != tuple(sorted(indices)):
        raise PromptDataError(f"Line {line_number}: {field_name} must be sorted.")
    if len(set(indices)) != len(indices):
        raise PromptDataError(f"Line {line_number}: {field_name} must be unique.")
    return tuple(indices)


def _required_readout_token_indices(record: Mapping[str, object], line_number: int) -> tuple[int, ...]:
    return _required_token_indices(record=record, field_name="readout_token_indices", line_number=line_number)


def _optional_token_indices(
    record: Mapping[str, object],
    field_name: str,
    line_number: int,
) -> tuple[int, ...] | None:
    value = record.get(field_name)
    if value is None:
        return None
    return _required_token_indices(record=record, field_name=field_name, line_number=line_number)


def _validate_readout_window(
    secret_token_span: PromptTokenSpan | None,
    query_token_span: PromptTokenSpan,
    payload_token_span: PromptTokenSpan | None,
    readout_token_indices: tuple[int, ...],
    line_number: int,
) -> None:
    first_readout_index = min(readout_token_indices)
    last_readout_index = max(readout_token_indices)
    secret_end = secret_token_span.end if secret_token_span is not None else 0
    if payload_token_span is None:
        visibility_floor = max(secret_end, query_token_span.start)
        if first_readout_index < visibility_floor:
            raise PromptDataError(
                f"Line {line_number}: readout_token_indices must start after secret end and query start."
            )
        return

    visibility_floor = max(secret_end, query_token_span.end, payload_token_span.start)
    if first_readout_index < visibility_floor:
        raise PromptDataError(
            f"Line {line_number}: payload readout_token_indices must start after secret, query, and payload visibility."
        )
    if last_readout_index >= payload_token_span.end:
        raise PromptDataError(f"Line {line_number}: payload readout_token_indices must stay inside payload span.")


def _validate_optional_selected_choice_window(
    selected_choice_token_span: PromptTokenSpan | None,
    selected_choice_readout_token_indices: tuple[int, ...] | None,
    line_number: int,
) -> None:
    if selected_choice_token_span is None and selected_choice_readout_token_indices is None:
        return
    if selected_choice_token_span is None:
        raise PromptDataError(
            f"Line {line_number}: selected_choice_readout_token_indices require selected_choice_token_span."
        )
    if selected_choice_readout_token_indices is None:
        raise PromptDataError(
            f"Line {line_number}: selected_choice_token_span requires selected_choice_readout_token_indices."
        )
    first_index = min(selected_choice_readout_token_indices)
    last_index = max(selected_choice_readout_token_indices)
    if first_index < selected_choice_token_span.start:
        raise PromptDataError(
            f"Line {line_number}: selected_choice_readout_token_indices must stay inside selected_choice_token_span."
        )
    if last_index >= selected_choice_token_span.end:
        raise PromptDataError(
            f"Line {line_number}: selected_choice_readout_token_indices must stay inside selected_choice_token_span."
        )


def _validate_optional_query_tail_window(
    query_token_span: PromptTokenSpan,
    query_tail_readout_token_indices: tuple[int, ...] | None,
    line_number: int,
) -> None:
    if query_tail_readout_token_indices is None:
        return
    first_index = min(query_tail_readout_token_indices)
    last_index = max(query_tail_readout_token_indices)
    if first_index < query_token_span.start:
        raise PromptDataError(
            f"Line {line_number}: query_tail_readout_token_indices must stay inside query_token_span."
        )
    if last_index >= query_token_span.end:
        raise PromptDataError(
            f"Line {line_number}: query_tail_readout_token_indices must stay inside query_token_span."
        )


def parse_prompt_example(record: Mapping[str, object], line_number: int) -> PromptExample:
    return PromptExample(
        id=_required_string(record, "id", line_number),
        label=_required_label(record, line_number),
        family=_required_string(record, "family", line_number),
        text=_required_string(record, "text", line_number),
        tags=_required_tags(record, line_number),
    )


def parse_structured_prompt_example(record: Mapping[str, object], line_number: int) -> StructuredPromptExample:
    prompt_id = _required_string(record, "id", line_number)
    label = _required_label(record, line_number)
    rendered_prompt = _required_string(record, "rendered_prompt", line_number)
    text = _required_string(record, "text", line_number)
    if text != rendered_prompt:
        raise PromptDataError(f"Line {line_number}: fields 'text' and 'rendered_prompt' must match.")

    example_id = record.get("example_id")
    if example_id is not None and example_id != prompt_id:
        raise PromptDataError(f"Line {line_number}: fields 'id' and 'example_id' must match.")

    secret_token_span = _optional_token_span(record=record, field_name="secret_token_span", line_number=line_number)
    if label != "benign" and secret_token_span is None:
        raise PromptDataError(f"Line {line_number}: non-benign structured prompts require 'secret_token_span'.")
    query_token_span = _required_token_span(record=record, field_name="query_token_span", line_number=line_number)
    payload_token_span = _optional_token_span(record=record, field_name="payload_token_span", line_number=line_number)
    readout_token_indices = _required_readout_token_indices(record=record, line_number=line_number)
    query_tail_readout_token_indices = _optional_token_indices(
        record=record,
        field_name="query_tail_readout_token_indices",
        line_number=line_number,
    )
    selected_choice_token_span = _optional_token_span(
        record=record,
        field_name="selected_choice_token_span",
        line_number=line_number,
    )
    selected_choice_readout_token_indices = _optional_token_indices(
        record=record,
        field_name="selected_choice_readout_token_indices",
        line_number=line_number,
    )
    _validate_readout_window(
        secret_token_span=secret_token_span,
        query_token_span=query_token_span,
        payload_token_span=payload_token_span,
        readout_token_indices=readout_token_indices,
        line_number=line_number,
    )
    _validate_optional_query_tail_window(
        query_token_span=query_token_span,
        query_tail_readout_token_indices=query_tail_readout_token_indices,
        line_number=line_number,
    )
    _validate_optional_selected_choice_window(
        selected_choice_token_span=selected_choice_token_span,
        selected_choice_readout_token_indices=selected_choice_readout_token_indices,
        line_number=line_number,
    )

    return StructuredPromptExample(
        id=prompt_id,
        label=label,
        family=_required_string(record, "family", line_number),
        text=text,
        tags=_required_tags(record, line_number),
        secret_token_span=secret_token_span,
        query_token_span=query_token_span,
        payload_token_span=payload_token_span,
        readout_token_indices=readout_token_indices,
        query_tail_readout_token_indices=query_tail_readout_token_indices,
        selected_choice_token_span=selected_choice_token_span,
        selected_choice_readout_token_indices=selected_choice_readout_token_indices,
        fallback_reason=_optional_string(record, "fallback_reason", line_number),
    )


def load_prompt_examples(path: Path) -> tuple[PromptExample, ...]:
    examples: list[PromptExample] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PromptDataError(f"Line {line_number}: invalid JSON: {exc.msg}.") from exc

            example = parse_prompt_example(_as_mapping(decoded, line_number), line_number)
            if example.id in seen_ids:
                raise PromptDataError(f"Line {line_number}: duplicate prompt id '{example.id}'.")
            seen_ids.add(example.id)
            examples.append(example)

    if len(examples) == 0:
        raise PromptDataError(f"No prompt examples found in {path}.")

    return tuple(examples)


def load_structured_prompt_examples(path: Path) -> tuple[StructuredPromptExample, ...]:
    examples: list[StructuredPromptExample] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PromptDataError(f"Line {line_number}: invalid JSON: {exc.msg}.") from exc

            example = parse_structured_prompt_example(_as_mapping(decoded, line_number), line_number)
            if example.id in seen_ids:
                raise PromptDataError(f"Line {line_number}: duplicate prompt id '{example.id}'.")
            seen_ids.add(example.id)
            examples.append(example)

    if len(examples) == 0:
        raise PromptDataError(f"No structured prompt examples found in {path}.")

    return tuple(examples)
