from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from aegis.core.contracts import CapabilityMode, JsonValue, Message, ModelInfo, SensitiveSpan, ToolCall
from aegis.core.orchestrator import RuntimeRequest


class RuntimeRequestJsonlError(ValueError):
    """Raised when runtime request JSONL cannot be decoded."""


def load_runtime_requests_jsonl(path: Path) -> tuple[RuntimeRequest, ...]:
    requests: list[RuntimeRequest] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeRequestJsonlError(f"Line {line_number}: invalid JSON in {path}: {exc.msg}.") from exc
            requests.append(_runtime_request_from_mapping(_as_mapping(decoded, line_number), line_number))
    if len(requests) == 0:
        raise RuntimeRequestJsonlError(f"No runtime requests found in {path}.")
    return tuple(requests)


def _runtime_request_from_mapping(record: Mapping[str, object], line_number: int) -> RuntimeRequest:
    return RuntimeRequest(
        trace_id=_required_string(record=record, field_name="trace_id", line_number=line_number),
        session_id=_required_string(record=record, field_name="session_id", line_number=line_number),
        turn_index=_required_int(record=record, field_name="turn_index", line_number=line_number),
        capability_mode=_capability_mode(
            value=_required_string(record=record, field_name="capability_mode", line_number=line_number),
            line_number=line_number,
        ),
        model=_model_info(record=_required_mapping(record=record, field_name="model", line_number=line_number)),
        messages=_messages(record=_required_list(record=record, field_name="messages", line_number=line_number)),
        tool_calls=_tool_calls(record=_required_list(record=record, field_name="tool_calls", line_number=line_number)),
        sensitive_spans=_sensitive_spans(
            record=_required_list(record=record, field_name="sensitive_spans", line_number=line_number),
        ),
        metadata=_json_object(
            value=_required_mapping(record=record, field_name="metadata", line_number=line_number),
            field_path=f"line {line_number}.metadata",
        ),
    )


def _model_info(record: Mapping[str, object]) -> ModelInfo:
    return ModelInfo(
        provider=_required_string(record=record, field_name="provider", line_number=0),
        model_id=_required_string(record=record, field_name="model_id", line_number=0),
        revision=_optional_string(record=record, field_name="revision", line_number=0),
        selected_device=_optional_string(record=record, field_name="selected_device", line_number=0),
    )


def _messages(record: list[object]) -> tuple[Message, ...]:
    messages: list[Message] = []
    for index, item in enumerate(record):
        message = _as_mapping(item, index)
        messages.append(
            Message(
                role=_required_string(record=message, field_name="role", line_number=index),
                content=_required_string(record=message, field_name="content", line_number=index),
            )
        )
    return tuple(messages)


def _tool_calls(record: list[object]) -> tuple[ToolCall, ...]:
    tool_calls: list[ToolCall] = []
    for index, item in enumerate(record):
        tool_call = _as_mapping(item, index)
        tool_calls.append(
            ToolCall(
                name=_required_string(record=tool_call, field_name="name", line_number=index),
                arguments=_json_object(
                    value=_required_mapping(record=tool_call, field_name="arguments", line_number=index),
                    field_path=f"tool_calls[{index}].arguments",
                ),
            )
        )
    return tuple(tool_calls)


def _sensitive_spans(record: list[object]) -> tuple[SensitiveSpan, ...]:
    spans: list[SensitiveSpan] = []
    for index, item in enumerate(record):
        span = _as_mapping(item, index)
        spans.append(
            SensitiveSpan(
                kind=_required_string(record=span, field_name="kind", line_number=index),
                source=_required_string(record=span, field_name="source", line_number=index),
                char_start=_optional_int(record=span, field_name="char_start", line_number=index),
                char_end=_optional_int(record=span, field_name="char_end", line_number=index),
                token_start=_optional_int(record=span, field_name="token_start", line_number=index),
                token_end=_optional_int(record=span, field_name="token_end", line_number=index),
                identifier=_optional_string(record=span, field_name="identifier", line_number=index),
                metadata=_json_object(
                    value=_required_mapping(record=span, field_name="metadata", line_number=index),
                    field_path=f"sensitive_spans[{index}].metadata",
                ),
            )
        )
    return tuple(spans)


def _capability_mode(value: str, line_number: int) -> CapabilityMode:
    try:
        return CapabilityMode(value)
    except ValueError as exc:
        raise RuntimeRequestJsonlError(f"Line {line_number}: unsupported capability_mode '{value}'.") from exc


def _required_mapping(record: Mapping[str, object], field_name: str, line_number: int) -> Mapping[str, object]:
    return _as_mapping(record.get(field_name), line_number)


def _required_list(record: Mapping[str, object], field_name: str, line_number: int) -> list[object]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise RuntimeRequestJsonlError(f"Line {line_number}: field '{field_name}' must be a list.")
    return value


def _required_string(record: Mapping[str, object], field_name: str, line_number: int) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or value == "":
        raise RuntimeRequestJsonlError(f"Line {line_number}: field '{field_name}' must be a non-empty string.")
    return value


def _optional_string(record: Mapping[str, object], field_name: str, line_number: int) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise RuntimeRequestJsonlError(f"Line {line_number}: field '{field_name}' must be null or a non-empty string.")
    return value


def _required_int(record: Mapping[str, object], field_name: str, line_number: int) -> int:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeRequestJsonlError(f"Line {line_number}: field '{field_name}' must be an integer.")
    return value


def _optional_int(record: Mapping[str, object], field_name: str, line_number: int) -> int | None:
    value = record.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeRequestJsonlError(f"Line {line_number}: field '{field_name}' must be null or an integer.")
    return value


def _as_mapping(value: object, line_number: int) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise RuntimeRequestJsonlError(f"Line {line_number}: expected a JSON object.")
    return cast(Mapping[str, object], value)


def _json_object(value: Mapping[str, object], field_path: str) -> dict[str, JsonValue]:
    parsed: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise RuntimeRequestJsonlError(f"{field_path}: object keys must be strings.")
        parsed[key] = _json_value(value=item, field_path=f"{field_path}.{key}")
    return parsed


def _json_value(value: object, field_path: str) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(value=item, field_path=f"{field_path}[]") for item in value]
    if isinstance(value, dict):
        return _json_object(value=cast(Mapping[str, object], value), field_path=field_path)
    raise RuntimeRequestJsonlError(f"{field_path}: unsupported JSON value type '{type(value).__name__}'.")
