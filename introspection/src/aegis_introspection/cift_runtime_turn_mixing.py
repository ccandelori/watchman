from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from aegis.core.contracts import JsonValue

_SELECTED_CHOICE_WINDOW = "selected_choice"
_FALLBACK_WINDOW = "payload_query_fallback"
_SELECTED_CHOICE_KEYS: tuple[str, ...] = (
    "selected_choice_char_span",
    "selected_choice_token_span",
    "selected_choice_readout_token_indices",
)


class CiftRuntimeTurnMixingError(ValueError):
    """Raised when mixed-route CIFT runtime turns cannot be built."""


@dataclass(frozen=True)
class CiftRuntimeTurnMixConfig:
    fallback_modulus: int
    fallback_remainder: int

    def __post_init__(self) -> None:
        if self.fallback_modulus < 2:
            raise CiftRuntimeTurnMixingError("fallback_modulus must be at least 2.")
        if self.fallback_remainder < 0:
            raise CiftRuntimeTurnMixingError("fallback_remainder must be non-negative.")
        if self.fallback_remainder >= self.fallback_modulus:
            raise CiftRuntimeTurnMixingError("fallback_remainder must be smaller than fallback_modulus.")


@dataclass(frozen=True)
class CiftRuntimeTurnMixResult:
    turns: tuple[dict[str, JsonValue], ...]
    window_family_counts: dict[str, int]


def load_runtime_turn_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise CiftRuntimeTurnMixingError(f"{path}:{line_number}: expected a JSON object.")
            rows.append(cast(Mapping[str, object], decoded))
    if len(rows) == 0:
        raise CiftRuntimeTurnMixingError(f"No runtime turns found in {path}.")
    return tuple(rows)


def write_runtime_turn_jsonl(path: Path, turns: tuple[Mapping[str, JsonValue], ...]) -> None:
    if len(turns) == 0:
        raise CiftRuntimeTurnMixingError("Cannot write an empty mixed-route runtime turn file.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for turn in turns:
            output_file.write(json.dumps(turn, sort_keys=True))
            output_file.write("\n")


def build_mixed_cift_window_runtime_turns(
    turns: tuple[Mapping[str, object], ...],
    config: CiftRuntimeTurnMixConfig,
) -> CiftRuntimeTurnMixResult:
    if len(turns) == 0:
        raise CiftRuntimeTurnMixingError("turns must not be empty.")

    mixed_turns: list[dict[str, JsonValue]] = []
    route_counts: Counter[str] = Counter()
    label_counters: Counter[str] = Counter()
    for index, turn in enumerate(turns, start=1):
        mutable_turn = _json_safe_copy(turn=turn, context=f"turn {index}")
        metadata = _required_json_object(mutable_turn, "metadata", f"turn {index}")
        eval_metadata = _required_json_object(metadata, "eval", f"turn {index}.metadata")
        label = _required_string(eval_metadata, "label", f"turn {index}.metadata.eval")
        family = _optional_string(eval_metadata, "family")
        label_key = _label_counter_key(label=label, family=family)
        label_index = label_counters[label_key]
        label_counters[label_key] += 1

        window_family = _window_family_for_index(label_index=label_index, config=config)
        cift_metadata = _required_json_object(metadata, "cift", f"turn {index}.metadata")
        if window_family == _SELECTED_CHOICE_WINDOW:
            _validate_selected_choice_metadata(cift_metadata=cift_metadata, context=f"turn {index}.metadata.cift")
        else:
            _validate_fallback_metadata(cift_metadata=cift_metadata, context=f"turn {index}.metadata.cift")
            _strip_selected_choice_metadata(cift_metadata=cift_metadata)
        eval_metadata["expected_cift_window_family"] = window_family
        route_counts[window_family] += 1
        mixed_turns.append(mutable_turn)

    return CiftRuntimeTurnMixResult(
        turns=tuple(mixed_turns),
        window_family_counts=dict(route_counts),
    )


def _json_safe_copy(turn: Mapping[str, object], context: str) -> dict[str, JsonValue]:
    encoded = json.dumps(turn)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise CiftRuntimeTurnMixingError(f"{context}: runtime turn must be a JSON object.")
    return cast(dict[str, JsonValue], decoded)


def _window_family_for_index(label_index: int, config: CiftRuntimeTurnMixConfig) -> str:
    if label_index % config.fallback_modulus == config.fallback_remainder:
        return _FALLBACK_WINDOW
    return _SELECTED_CHOICE_WINDOW


def _strip_selected_choice_metadata(cift_metadata: dict[str, JsonValue]) -> None:
    for key in _SELECTED_CHOICE_KEYS:
        cift_metadata.pop(key, None)


def _validate_selected_choice_metadata(cift_metadata: Mapping[str, JsonValue], context: str) -> None:
    for key in _SELECTED_CHOICE_KEYS:
        if key not in cift_metadata:
            raise CiftRuntimeTurnMixingError(f"{context}.{key} is required for selected-choice rows.")
    indices = cift_metadata.get("selected_choice_readout_token_indices")
    _validate_int_list(value=indices, field_name=f"{context}.selected_choice_readout_token_indices")


def _validate_fallback_metadata(cift_metadata: Mapping[str, JsonValue], context: str) -> None:
    _validate_int_list(value=cift_metadata.get("readout_token_indices"), field_name=f"{context}.readout_token_indices")


def _validate_int_list(value: JsonValue, field_name: str) -> None:
    if not isinstance(value, list):
        raise CiftRuntimeTurnMixingError(f"{field_name} must be a list of integers.")
    if len(value) == 0:
        raise CiftRuntimeTurnMixingError(f"{field_name} must not be empty.")
    for index, item in enumerate(value):
        if not isinstance(item, int):
            raise CiftRuntimeTurnMixingError(f"{field_name}[{index}] must be an integer.")


def _required_json_object(record: Mapping[str, JsonValue], field_name: str, context: str) -> dict[str, JsonValue]:
    value = record.get(field_name)
    if not isinstance(value, dict):
        raise CiftRuntimeTurnMixingError(f"{context}.{field_name} must be a JSON object.")
    for key in value:
        if not isinstance(key, str):
            raise CiftRuntimeTurnMixingError(f"{context}.{field_name} contains a non-string key.")
    return cast(dict[str, JsonValue], value)


def _required_string(record: Mapping[str, JsonValue], field_name: str, context: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise CiftRuntimeTurnMixingError(f"{context}.{field_name} must be a string.")
    if value == "":
        raise CiftRuntimeTurnMixingError(f"{context}.{field_name} must not be empty.")
    return value


def _optional_string(record: Mapping[str, JsonValue], field_name: str) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CiftRuntimeTurnMixingError(f"{field_name} must be a string when present.")
    if value == "":
        raise CiftRuntimeTurnMixingError(f"{field_name} must not be empty when present.")
    return value


def _label_counter_key(label: str, family: str | None) -> str:
    if family is None:
        return label
    return f"{family}:{label}"
