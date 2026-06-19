from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, TypeAlias, cast


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


def parse_prompt_example(record: Mapping[str, object], line_number: int) -> PromptExample:
    return PromptExample(
        id=_required_string(record, "id", line_number),
        label=_required_label(record, line_number),
        family=_required_string(record, "family", line_number),
        text=_required_string(record, "text", line_number),
        tags=_required_tags(record, line_number),
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
