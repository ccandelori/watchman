from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from aegis.core.contracts import JsonValue

NormalizedTurnJson: TypeAlias = dict[str, JsonValue]

_DISALLOWED_SECRET_FIELDS: frozenset[str] = frozenset(
    (
        "secret",
        "secret_value",
        "raw_secret",
        "credential",
        "credential_secret",
        "honeytoken",
        "honeytoken_value",
    )
)


class RuntimeBridgeError(ValueError):
    """Raised when a structured prompt cannot be adapted to the runtime spine shape."""


@dataclass(frozen=True)
class RuntimeBridgeConfig:
    trace_id: str
    session_id: str
    turn_index: int
    capability_mode: str
    model_provider: str
    model_id: str
    revision: str | None
    selected_device: str | None
    sensitive_source: str


def structured_prompt_to_normalized_turn(
    record: Mapping[str, object],
    config: RuntimeBridgeConfig,
) -> NormalizedTurnJson:
    _reject_raw_secret_metadata(record)
    text = _required_matching_text(record=record)
    secret_char_span = _required_int_pair(record=record, field_name="secret_char_span")
    secret_token_span = _required_int_pair(record=record, field_name="secret_token_span")
    query_char_span = _required_int_pair(record=record, field_name="query_char_span")
    query_token_span = _required_int_pair(record=record, field_name="query_token_span")
    payload_char_span = _optional_int_pair(record=record, field_name="payload_char_span")
    payload_token_span = _optional_int_pair(record=record, field_name="payload_token_span")
    readout_token_indices = _required_int_list(record=record, field_name="readout_token_indices")
    query_tail_readout_token_indices = _optional_int_list(
        record=record,
        field_name="query_tail_readout_token_indices",
    )
    selected_choice_char_span = _optional_int_pair(record=record, field_name="selected_choice_char_span")
    selected_choice_token_span = _optional_int_pair(record=record, field_name="selected_choice_token_span")
    selected_choice_readout_token_indices = _optional_int_list(
        record=record,
        field_name="selected_choice_readout_token_indices",
    )
    fallback_reason = _optional_string(record=record, field_name="fallback_reason")
    _validate_selected_choice_geometry(
        selected_choice_char_span=selected_choice_char_span,
        selected_choice_token_span=selected_choice_token_span,
        selected_choice_readout_token_indices=selected_choice_readout_token_indices,
    )

    return {
        "trace_id": config.trace_id,
        "session_id": config.session_id,
        "turn_index": config.turn_index,
        "capability_mode": config.capability_mode,
        "model": {
            "provider": config.model_provider,
            "model_id": config.model_id,
            "revision": config.revision,
            "selected_device": config.selected_device,
        },
        "messages": [{"role": "user", "content": text}],
        "tool_calls": [],
        "sensitive_spans": [
            {
                "kind": "honeytoken",
                "source": config.sensitive_source,
                "char_start": secret_char_span[0],
                "char_end": secret_char_span[1],
                "token_start": secret_token_span[0],
                "token_end": secret_token_span[1],
                "identifier": _required_string(record=record, field_name="honeytoken_id"),
                "metadata": {
                    "credential_type": _required_string(record=record, field_name="credential_type"),
                    "honeytoken_sha256": _required_string(record=record, field_name="honeytoken_sha256"),
                },
            }
        ],
        "metadata": {
            "example_id": _required_string(record=record, field_name="example_id"),
            "eval": {
                "label": _required_string(record=record, field_name="label"),
                "family": _required_string(record=record, field_name="family"),
                "tags": _required_string_list(record=record, field_name="tags"),
            },
            "cift": {
                "secret_token_span": list(secret_token_span),
                "query_char_span": list(query_char_span),
                "query_token_span": list(query_token_span),
                "payload_char_span": _optional_pair_json(payload_char_span),
                "payload_token_span": _optional_pair_json(payload_token_span),
                "readout_token_indices": readout_token_indices,
                "query_tail_readout_token_indices": query_tail_readout_token_indices,
                "selected_choice_char_span": _optional_pair_json(selected_choice_char_span),
                "selected_choice_token_span": _optional_pair_json(selected_choice_token_span),
                "selected_choice_readout_token_indices": selected_choice_readout_token_indices,
                "fallback_reason": fallback_reason,
            },
            "policy_window": _policy_window_metadata(record=record),
            "bridge": {
                "source": "aegis_introspection",
                "shape": "aegis.core.contracts.NormalizedTurn",
                "message_layout": "single_rendered_prompt_message",
                "audit_note": "synthetic_honeytoken_prompt_may_include_sensitive_text",
            },
        },
    }


def _required_matching_text(record: Mapping[str, object]) -> str:
    text = _required_string(record=record, field_name="text")
    rendered_prompt = _required_string(record=record, field_name="rendered_prompt")
    if text != rendered_prompt:
        raise RuntimeBridgeError("Fields 'text' and 'rendered_prompt' must match.")
    return text


def _policy_window_metadata(record: Mapping[str, object]) -> dict[str, JsonValue]:
    policy_window_token_indices = _optional_int_list(record=record, field_name="policy_window_token_indices")
    policy_window_char_spans = _optional_int_pair_list(record=record, field_name="policy_window_char_spans")
    selected_field = _optional_string(record=record, field_name="policy_window_selected_field")
    selected_mode = _optional_string(record=record, field_name="policy_window_selected_mode")
    selected_action = _optional_string(record=record, field_name="policy_window_selected_action")
    kind = _optional_string(record=record, field_name="policy_window_kind")
    return {
        "token_indices": policy_window_token_indices,
        "char_spans": policy_window_char_spans,
        "selected_field": selected_field,
        "selected_mode": selected_mode,
        "selected_action": selected_action,
        "kind": kind,
    }


def _reject_raw_secret_metadata(record: Mapping[str, object]) -> None:
    present = sorted(field_name for field_name in _DISALLOWED_SECRET_FIELDS if field_name in record)
    if len(present) > 0:
        fields = ", ".join(present)
        raise RuntimeBridgeError(f"Structured prompt record contains raw-secret metadata fields: {fields}.")


def _validate_selected_choice_geometry(
    selected_choice_char_span: tuple[int, int] | None,
    selected_choice_token_span: tuple[int, int] | None,
    selected_choice_readout_token_indices: list[JsonValue] | None,
) -> None:
    present = (
        selected_choice_char_span is not None,
        selected_choice_token_span is not None,
        selected_choice_readout_token_indices is not None,
    )
    if any(present) and not all(present):
        raise RuntimeBridgeError(
            "Selected-choice geometry requires selected_choice_char_span, selected_choice_token_span, "
            "and selected_choice_readout_token_indices."
        )
    if selected_choice_token_span is None or selected_choice_readout_token_indices is None:
        return
    token_indices = cast(list[int], selected_choice_readout_token_indices)
    if min(token_indices) < selected_choice_token_span[0]:
        raise RuntimeBridgeError("selected_choice_readout_token_indices must stay inside selected_choice_token_span.")
    if max(token_indices) >= selected_choice_token_span[1]:
        raise RuntimeBridgeError("selected_choice_readout_token_indices must stay inside selected_choice_token_span.")


def _required_string(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise RuntimeBridgeError(f"Field '{field_name}' must be a string.")
    if value == "":
        raise RuntimeBridgeError(f"Field '{field_name}' must not be empty.")
    return value


def _optional_string(record: Mapping[str, object], field_name: str) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeBridgeError(f"Field '{field_name}' must be a string when present.")
    if value == "":
        raise RuntimeBridgeError(f"Field '{field_name}' must not be empty when present.")
    return value


def _required_string_list(record: Mapping[str, object], field_name: str) -> list[JsonValue]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise RuntimeBridgeError(f"Field '{field_name}' must be a list of strings.")
    items: list[JsonValue] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise RuntimeBridgeError(f"Field '{field_name}' item {index} must be a string.")
        if item == "":
            raise RuntimeBridgeError(f"Field '{field_name}' item {index} must not be empty.")
        items.append(item)
    return items


def _required_int_pair(record: Mapping[str, object], field_name: str) -> tuple[int, int]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise RuntimeBridgeError(f"Field '{field_name}' must be a two-integer list.")
    return _int_pair(value=value, field_name=field_name)


def _optional_int_pair(record: Mapping[str, object], field_name: str) -> tuple[int, int] | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, list):
        raise RuntimeBridgeError(f"Field '{field_name}' must be null or a two-integer list.")
    return _int_pair(value=value, field_name=field_name)


def _int_pair(value: list[object], field_name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise RuntimeBridgeError(f"Field '{field_name}' must contain exactly two integers.")
    start, end = value
    if not isinstance(start, int) or not isinstance(end, int):
        raise RuntimeBridgeError(f"Field '{field_name}' must contain integers.")
    if start < 0:
        raise RuntimeBridgeError(f"Field '{field_name}' start must be non-negative.")
    if end <= start:
        raise RuntimeBridgeError(f"Field '{field_name}' must be a non-empty half-open span.")
    return (start, end)


def _required_int_list(record: Mapping[str, object], field_name: str) -> list[JsonValue]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise RuntimeBridgeError(f"Field '{field_name}' must be a list of integers.")
    if len(value) == 0:
        raise RuntimeBridgeError(f"Field '{field_name}' must not be empty.")
    return _int_list(value=value, field_name=field_name)


def _optional_int_list(record: Mapping[str, object], field_name: str) -> list[JsonValue] | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, list):
        raise RuntimeBridgeError(f"Field '{field_name}' must be a list of integers when present.")
    if len(value) == 0:
        raise RuntimeBridgeError(f"Field '{field_name}' must not be empty when present.")
    return _int_list(value=value, field_name=field_name)


def _int_list(value: list[object], field_name: str) -> list[JsonValue]:
    items: list[JsonValue] = []
    for index, item in enumerate(value):
        if not isinstance(item, int):
            raise RuntimeBridgeError(f"Field '{field_name}' item {index} must be an integer.")
        if item < 0:
            raise RuntimeBridgeError(f"Field '{field_name}' item {index} must be non-negative.")
        items.append(item)
    if items != sorted(cast(list[int], items)):
        raise RuntimeBridgeError(f"Field '{field_name}' must be sorted.")
    if len(set(cast(list[int], items))) != len(items):
        raise RuntimeBridgeError(f"Field '{field_name}' must contain unique integers.")
    return items


def _optional_int_pair_list(record: Mapping[str, object], field_name: str) -> list[JsonValue] | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, list):
        raise RuntimeBridgeError(f"Field '{field_name}' must be a list of two-integer lists when present.")
    pairs: list[JsonValue] = []
    for index, item in enumerate(value):
        if not isinstance(item, list):
            raise RuntimeBridgeError(f"Field '{field_name}' item {index} must be a two-integer list.")
        pairs.append(list(_int_pair(value=item, field_name=f"{field_name}[{index}]")))
    return pairs


def _optional_pair_json(value: tuple[int, int] | None) -> list[JsonValue] | None:
    if value is None:
        return None
    return [value[0], value[1]]
