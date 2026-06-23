from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, cast

from aegis_introspection.prompts import PromptLabel

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

_VALID_LABELS: frozenset[str] = frozenset(("benign", "secret_present_safe", "exfiltration_intent"))
_SEMANTIC_INDIRECTION_SPACER = "Compare both neutral entries before applying the final code choice."


class TraceRecordAdapterError(ValueError):
    """Raised when runtime trace records cannot be converted to CIFT prompt records."""


class TokenOffsetEncoder(Protocol):
    def encode_offsets(self, text: str) -> tuple[TokenOffset, ...]:
        """Return tokenizer offsets for the exact prompt text passed to activation extraction."""


@dataclass(frozen=True)
class TracePromptConversionConfig:
    readout_token_count: int

    def __post_init__(self) -> None:
        if self.readout_token_count < 1:
            raise TraceRecordAdapterError("readout_token_count must be positive.")


@dataclass(frozen=True)
class CharSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise TraceRecordAdapterError("char span start must be non-negative.")
        if self.end <= self.start:
            raise TraceRecordAdapterError("char span must be a non-empty half-open span.")

    def to_json(self) -> list[JsonValue]:
        return [self.start, self.end]


@dataclass(frozen=True)
class TokenOffset:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise TraceRecordAdapterError("token offset start must be non-negative.")
        if self.end < self.start:
            raise TraceRecordAdapterError("token offset end must be greater than or equal to start.")


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise TraceRecordAdapterError("token span start must be non-negative.")
        if self.end <= self.start:
            raise TraceRecordAdapterError("token span must be a non-empty half-open span.")

    def to_json(self) -> list[JsonValue]:
        return [self.start, self.end]


@dataclass(frozen=True)
class SensitiveSpanRecord:
    kind: str
    source: str
    char_span: CharSpan
    identifier: str
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class MessageRecord:
    role: str
    content: str


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class MessageSegment:
    index: int
    role: str
    content: str
    content_span: CharSpan


@dataclass(frozen=True)
class ToolArgumentSegment:
    tool_call_index: int
    tool_call_name: str
    argument_path: str
    value: str
    value_span: CharSpan


@dataclass(frozen=True)
class RenderedTracePrompt:
    text: str
    message_segments: tuple[MessageSegment, ...]
    tool_argument_segments: tuple[ToolArgumentSegment, ...]


@dataclass(frozen=True)
class ToolPayloadReadout:
    payload_char_span: CharSpan
    payload_token_span: TokenSpan
    readout_char_span: CharSpan
    tool_call_name: str
    argument_path: str


@dataclass(frozen=True)
class SelectedChoiceReadout:
    selected_choice_char_span: CharSpan
    selected_choice_token_span: TokenSpan
    selected_choice_readout_token_indices: tuple[int, ...]


@dataclass(frozen=True)
class StructuredTracePromptRecord:
    id: str
    example_id: str
    label: PromptLabel
    family: str
    text: str
    rendered_prompt: str
    tags: tuple[str, ...]
    secret_char_span: CharSpan | None
    query_char_span: CharSpan
    payload_char_span: CharSpan | None
    secret_token_span: TokenSpan | None
    query_token_span: TokenSpan
    payload_token_span: TokenSpan | None
    readout_token_indices: tuple[int, ...]
    query_tail_readout_token_indices: tuple[int, ...]
    selected_choice_char_span: CharSpan | None
    selected_choice_token_span: TokenSpan | None
    selected_choice_readout_token_indices: tuple[int, ...] | None
    honeytoken_id: str | None
    credential_type: str
    honeytoken_sha256: str | None
    source_trace_id: str
    source_session_id: str
    source_turn_index: int
    tool_call_name: str | None
    tool_argument_path: str | None

    def to_dict(self) -> JsonObject:
        secret_char_span: list[JsonValue] | None = None
        payload_char_span: list[JsonValue] | None = None
        selected_choice_char_span: list[JsonValue] | None = None
        secret_token_span: list[JsonValue] | None = None
        payload_token_span: list[JsonValue] | None = None
        selected_choice_token_span: list[JsonValue] | None = None
        if self.secret_char_span is not None:
            secret_char_span = self.secret_char_span.to_json()
        if self.payload_char_span is not None:
            payload_char_span = self.payload_char_span.to_json()
        if self.selected_choice_char_span is not None:
            selected_choice_char_span = self.selected_choice_char_span.to_json()
        if self.secret_token_span is not None:
            secret_token_span = self.secret_token_span.to_json()
        if self.payload_token_span is not None:
            payload_token_span = self.payload_token_span.to_json()
        if self.selected_choice_token_span is not None:
            selected_choice_token_span = self.selected_choice_token_span.to_json()
        return {
            "id": self.id,
            "example_id": self.example_id,
            "label": self.label,
            "family": self.family,
            "text": self.text,
            "rendered_prompt": self.rendered_prompt,
            "tags": list(self.tags),
            "secret_char_span": secret_char_span,
            "query_char_span": self.query_char_span.to_json(),
            "payload_char_span": payload_char_span,
            "selected_choice_char_span": selected_choice_char_span,
            "secret_token_span": secret_token_span,
            "query_token_span": self.query_token_span.to_json(),
            "payload_token_span": payload_token_span,
            "readout_token_indices": list(self.readout_token_indices),
            "query_tail_readout_token_indices": list(self.query_tail_readout_token_indices),
            "selected_choice_token_span": selected_choice_token_span,
            "selected_choice_readout_token_indices": list(self.selected_choice_readout_token_indices)
            if self.selected_choice_readout_token_indices is not None
            else None,
            "honeytoken_id": self.honeytoken_id,
            "credential_type": self.credential_type,
            "honeytoken_sha256": self.honeytoken_sha256,
            "source_trace_id": self.source_trace_id,
            "source_session_id": self.source_session_id,
            "source_turn_index": self.source_turn_index,
            "tool_call_name": self.tool_call_name,
            "tool_argument_path": self.tool_argument_path,
        }


@dataclass(frozen=True)
class SkippedTraceRecord:
    record_id: str
    reason: str


@dataclass(frozen=True)
class TracePromptConversionResult:
    records: tuple[StructuredTracePromptRecord, ...]
    skipped_records: tuple[SkippedTraceRecord, ...]


@dataclass(frozen=True)
class _FlatToolArgument:
    path: str
    rendered_value: str


def load_trace_records_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceRecordAdapterError(f"Line {line_number}: invalid JSON: {exc.msg}.") from exc
            records.append(_as_mapping(value=parsed, context=f"{path}:{line_number}"))
    if len(records) == 0:
        raise TraceRecordAdapterError(f"No trace records found in {path}.")
    return tuple(records)


def write_structured_prompt_jsonl(path: Path, records: tuple[StructuredTracePromptRecord, ...]) -> None:
    if len(records) == 0:
        raise TraceRecordAdapterError("Cannot write an empty structured prompt dataset.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record.to_dict(), sort_keys=True))
            output_file.write("\n")


def structured_prompt_records_from_trace_records(
    records: tuple[Mapping[str, object], ...],
    encoder: TokenOffsetEncoder,
    config: TracePromptConversionConfig,
) -> TracePromptConversionResult:
    converted: list[StructuredTracePromptRecord] = []
    skipped: list[SkippedTraceRecord] = []
    for index, record in enumerate(records, start=1):
        trace_id = _trace_id_for_record(record=record, context=f"record {index}")
        label = _required_label(record=record, context=f"record {index}")
        secret_span = _find_source_span(record=record, source="dp_honey", context=f"record {index}")
        if secret_span is None and label != "benign":
            skipped.append(SkippedTraceRecord(record_id=trace_id, reason="no_dp_honey_secret_span"))
            continue
        converted.append(
            _convert_trace_record(
                record=record,
                secret_span=secret_span,
                encoder=encoder,
                config=config,
                context=f"record {index}",
            )
        )
    return TracePromptConversionResult(records=tuple(converted), skipped_records=tuple(skipped))


def _convert_trace_record(
    record: Mapping[str, object],
    secret_span: SensitiveSpanRecord | None,
    encoder: TokenOffsetEncoder,
    config: TracePromptConversionConfig,
    context: str,
) -> StructuredTracePromptRecord:
    label = _required_label(record=record, context=context)
    family = _required_string(record=record, field_name="family", context=context)
    turn = _required_mapping(record=record, field_name="normalized_turn", context=context)
    collection = _required_collection_metadata(turn=turn, context=context)
    messages = _messages_from_turn(turn=turn, context=context)
    tool_calls = _tool_calls_from_turn(turn=turn, context=context)
    rendered = _render_trace_prompt(messages=messages, tool_calls=tool_calls)
    offsets = encoder.encode_offsets(rendered.text)

    query_char_span = _query_char_span(message_segments=rendered.message_segments, context=context)
    secret_char_span = None
    secret_token_span = None
    if secret_span is not None:
        secret_char_span = _absolute_message_span(
            span=secret_span,
            message_segments=rendered.message_segments,
            context=context,
        )
        secret_token_span = _token_span_for_char_span(offsets=offsets, char_span=secret_char_span, context=context)
    query_token_span = _token_span_for_char_span(offsets=offsets, char_span=query_char_span, context=context)
    query_tail_readout_token_indices = _readout_indices_for_char_span(
        offsets=offsets,
        char_span=query_char_span,
        lower_bound=max(_optional_token_span_end(secret_token_span), query_token_span.start),
        readout_token_count=config.readout_token_count,
        context=context,
    )
    tool_span = _find_source_span(record=record, source="tool_call_argument", context=context)
    payload_readout: ToolPayloadReadout | None = None
    if tool_span is not None:
        payload_readout = _tool_payload_readout(
            span=tool_span,
            tool_argument_segments=rendered.tool_argument_segments,
            offsets=offsets,
            context=context,
        )
        readout_tag = "readout:payload_secret"
    else:
        payload_readout = _safe_tool_payload_readout(
            tool_argument_segments=rendered.tool_argument_segments,
            offsets=offsets,
            context=context,
        )
        readout_tag = "readout:safe_payload" if payload_readout is not None else "readout:query_tail"

    if payload_readout is None:
        readout_token_indices = _readout_indices_for_char_span(
            offsets=offsets,
            char_span=query_char_span,
            lower_bound=max(_optional_token_span_end(secret_token_span), query_token_span.start),
            readout_token_count=config.readout_token_count,
            context=context,
        )
        tags = _tags_for_record(
            record=record,
            collection=collection,
            readout_tag="readout:query_tail",
            context=context,
        )
        payload_char_span = None
        payload_token_span = None
        tool_call_name = None
        tool_argument_path = None
    else:
        readout_token_indices = _readout_indices_for_char_span(
            offsets=offsets,
            char_span=payload_readout.readout_char_span,
            lower_bound=max(
                _optional_token_span_end(secret_token_span),
                query_token_span.end,
                payload_readout.payload_token_span.start,
            ),
            readout_token_count=config.readout_token_count,
            context=context,
        )
        tags = _tags_for_record(
            record=record,
            collection=collection,
            readout_tag=readout_tag,
            context=context,
        )
        payload_char_span = payload_readout.payload_char_span
        payload_token_span = payload_readout.payload_token_span
        tool_call_name = payload_readout.tool_call_name
        tool_argument_path = payload_readout.argument_path

    selected_choice_readout = _selected_choice_readout(
        message_segments=rendered.message_segments,
        offsets=offsets,
        readout_token_count=config.readout_token_count,
        context=context,
    )

    return StructuredTracePromptRecord(
        id=_required_string(record=turn, field_name="trace_id", context=context),
        example_id=_required_string(record=turn, field_name="trace_id", context=context),
        label=label,
        family=family,
        text=rendered.text,
        rendered_prompt=rendered.text,
        tags=tags,
        secret_char_span=secret_char_span,
        query_char_span=query_char_span,
        payload_char_span=payload_char_span,
        secret_token_span=secret_token_span,
        query_token_span=query_token_span,
        payload_token_span=payload_token_span,
        readout_token_indices=readout_token_indices,
        query_tail_readout_token_indices=query_tail_readout_token_indices,
        selected_choice_char_span=selected_choice_readout.selected_choice_char_span
        if selected_choice_readout is not None
        else None,
        selected_choice_token_span=selected_choice_readout.selected_choice_token_span
        if selected_choice_readout is not None
        else None,
        selected_choice_readout_token_indices=selected_choice_readout.selected_choice_readout_token_indices
        if selected_choice_readout is not None
        else None,
        honeytoken_id=secret_span.identifier if secret_span is not None else None,
        credential_type=_required_string(record=collection, field_name="credential_type", context=context),
        honeytoken_sha256=_required_string(record=secret_span.metadata, field_name="sha256", context=context)
        if secret_span is not None
        else None,
        source_trace_id=_required_string(record=turn, field_name="trace_id", context=context),
        source_session_id=_required_string(record=turn, field_name="session_id", context=context),
        source_turn_index=_required_int(record=turn, field_name="turn_index", context=context),
        tool_call_name=tool_call_name,
        tool_argument_path=tool_argument_path,
    )


def _optional_token_span_end(span: TokenSpan | None) -> int:
    if span is None:
        return 0
    return span.end


def _render_trace_prompt(
    messages: tuple[MessageRecord, ...],
    tool_calls: tuple[ToolCallRecord, ...],
) -> RenderedTracePrompt:
    parts: list[str] = []
    message_segments: list[MessageSegment] = []
    tool_argument_segments: list[ToolArgumentSegment] = []
    cursor = 0

    def append(text: str) -> None:
        nonlocal cursor
        parts.append(text)
        cursor += len(text)

    for message_index, message in enumerate(messages):
        append(f"[message:{message.role}:{message_index}]\n")
        content_start = cursor
        append(message.content)
        content_end = cursor
        message_segments.append(
            MessageSegment(
                index=message_index,
                role=message.role,
                content=message.content,
                content_span=CharSpan(start=content_start, end=content_end),
            )
        )
        append("\n\n")

    for tool_call_index, tool_call in enumerate(tool_calls):
        append(f"[tool_call:{tool_call.name}:{tool_call_index}]\n")
        for flat_argument in _flatten_tool_arguments(value=tool_call.arguments, path="arguments"):
            append(f"{flat_argument.path}: ")
            value_start = cursor
            append(flat_argument.rendered_value)
            value_end = cursor
            tool_argument_segments.append(
                ToolArgumentSegment(
                    tool_call_index=tool_call_index,
                    tool_call_name=tool_call.name,
                    argument_path=flat_argument.path,
                    value=flat_argument.rendered_value,
                    value_span=CharSpan(start=value_start, end=value_end),
                )
            )
            append("\n")
        append("\n")

    return RenderedTracePrompt(
        text="".join(parts),
        message_segments=tuple(message_segments),
        tool_argument_segments=tuple(tool_argument_segments),
    )


def _flatten_tool_arguments(value: object, path: str) -> tuple[_FlatToolArgument, ...]:
    if isinstance(value, str):
        return (_FlatToolArgument(path=path, rendered_value=value),)
    if isinstance(value, list):
        flattened: list[_FlatToolArgument] = []
        for index, item in enumerate(value):
            flattened.extend(_flatten_tool_arguments(value=item, path=f"{path}[{index}]"))
        if len(flattened) == 0:
            return (_FlatToolArgument(path=path, rendered_value="[]"),)
        return tuple(flattened)
    if isinstance(value, dict):
        flattened = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise TraceRecordAdapterError(f"{path} contains a non-string key.")
            flattened.extend(_flatten_tool_arguments(value=value[key], path=f"{path}.{key}"))
        if len(flattened) == 0:
            return (_FlatToolArgument(path=path, rendered_value="{}"),)
        return tuple(flattened)
    return (_FlatToolArgument(path=path, rendered_value=json.dumps(value, sort_keys=True)),)


def _find_source_span(record: Mapping[str, object], source: str, context: str) -> SensitiveSpanRecord | None:
    turn = _required_mapping(record=record, field_name="normalized_turn", context=context)
    span_rows = _required_list(record=turn, field_name="sensitive_spans", context=context)
    for index, span_row in enumerate(span_rows):
        span = _sensitive_span_from_row(value=span_row, context=f"{context}.sensitive_spans[{index}]")
        if span.source == source:
            return span
    return None


def _sensitive_span_from_row(value: object, context: str) -> SensitiveSpanRecord:
    row = _as_mapping(value=value, context=context)
    char_start = _required_int(record=row, field_name="char_start", context=context)
    char_end = _required_int(record=row, field_name="char_end", context=context)
    return SensitiveSpanRecord(
        kind=_required_string(record=row, field_name="kind", context=context),
        source=_required_string(record=row, field_name="source", context=context),
        char_span=CharSpan(start=char_start, end=char_end),
        identifier=_required_string(record=row, field_name="identifier", context=context),
        metadata=_required_mapping(record=row, field_name="metadata", context=context),
    )


def _absolute_message_span(
    span: SensitiveSpanRecord,
    message_segments: tuple[MessageSegment, ...],
    context: str,
) -> CharSpan:
    eligible_segments = tuple(
        segment
        for segment in message_segments
        if span.char_span.end <= len(segment.content) and segment.role == "system"
    )
    fallback_segments = tuple(segment for segment in message_segments if span.char_span.end <= len(segment.content))
    candidates = eligible_segments if len(eligible_segments) > 0 else fallback_segments
    if len(candidates) == 0:
        raise TraceRecordAdapterError(f"{context}: source span does not fit any rendered message segment.")
    segment = candidates[0]
    return CharSpan(
        start=segment.content_span.start + span.char_span.start,
        end=segment.content_span.start + span.char_span.end,
    )


def _query_char_span(message_segments: tuple[MessageSegment, ...], context: str) -> CharSpan:
    for segment in message_segments:
        if segment.role == "user":
            return segment.content_span
    raise TraceRecordAdapterError(f"{context}: normalized_turn.messages must include a user message.")


def _tool_payload_readout(
    span: SensitiveSpanRecord,
    tool_argument_segments: tuple[ToolArgumentSegment, ...],
    offsets: tuple[TokenOffset, ...],
    context: str,
) -> ToolPayloadReadout:
    tool_call_name = _required_string(record=span.metadata, field_name="tool_call_name", context=context)
    argument_path = _required_string(record=span.metadata, field_name="argument_path", context=context)
    matches = tuple(
        segment
        for segment in tool_argument_segments
        if segment.tool_call_name == tool_call_name and segment.argument_path == argument_path
    )
    if len(matches) == 0:
        raise TraceRecordAdapterError(f"{context}: no rendered tool argument segment matches {argument_path}.")
    segment = matches[0]
    if span.char_span.end > len(segment.value):
        raise TraceRecordAdapterError(f"{context}: tool sensitive span exceeds rendered argument value length.")
    payload_char_span = segment.value_span
    payload_secret_char_span = CharSpan(
        start=segment.value_span.start + span.char_span.start,
        end=segment.value_span.start + span.char_span.end,
    )
    return ToolPayloadReadout(
        payload_char_span=payload_char_span,
        payload_token_span=_token_span_for_char_span(offsets=offsets, char_span=payload_char_span, context=context),
        readout_char_span=payload_secret_char_span,
        tool_call_name=tool_call_name,
        argument_path=argument_path,
    )


def _safe_tool_payload_readout(
    tool_argument_segments: tuple[ToolArgumentSegment, ...],
    offsets: tuple[TokenOffset, ...],
    context: str,
) -> ToolPayloadReadout | None:
    if len(tool_argument_segments) == 0:
        return None
    credential_segments = tuple(
        segment for segment in tool_argument_segments if segment.argument_path.endswith(".credential")
    )
    segment = credential_segments[0] if len(credential_segments) > 0 else tool_argument_segments[0]
    return ToolPayloadReadout(
        payload_char_span=segment.value_span,
        payload_token_span=_token_span_for_char_span(offsets=offsets, char_span=segment.value_span, context=context),
        readout_char_span=segment.value_span,
        tool_call_name=segment.tool_call_name,
        argument_path=segment.argument_path,
    )


def _selected_choice_readout(
    message_segments: tuple[MessageSegment, ...],
    offsets: tuple[TokenOffset, ...],
    readout_token_count: int,
    context: str,
) -> SelectedChoiceReadout | None:
    user_segments = tuple(segment for segment in message_segments if segment.role == "user")
    if len(user_segments) == 0:
        return None
    selected_choice_char_span = _semantic_indirection_selected_choice_span(
        user_content=user_segments[0].content,
        user_content_span=user_segments[0].content_span,
        context=context,
    )
    if selected_choice_char_span is None:
        return None
    selected_choice_token_span = _token_span_for_char_span(
        offsets=offsets,
        char_span=selected_choice_char_span,
        context=context,
    )
    return SelectedChoiceReadout(
        selected_choice_char_span=selected_choice_char_span,
        selected_choice_token_span=selected_choice_token_span,
        selected_choice_readout_token_indices=_readout_indices_for_char_span(
            offsets=offsets,
            char_span=selected_choice_char_span,
            lower_bound=selected_choice_token_span.start,
            readout_token_count=readout_token_count,
            context=context,
        ),
    )


def _semantic_indirection_selected_choice_span(
    user_content: str,
    user_content_span: CharSpan,
    context: str,
) -> CharSpan | None:
    spacer_start = user_content.find(_SEMANTIC_INDIRECTION_SPACER)
    if spacer_start < 0:
        return None
    selected_clause_start = spacer_start + len(_SEMANTIC_INDIRECTION_SPACER)
    selected_clause_end = user_content.find(";", selected_clause_start)
    if selected_clause_end < 0:
        raise TraceRecordAdapterError(f"{context}: semantic indirection prompt is missing selected-choice terminator.")
    while selected_clause_start < selected_clause_end and user_content[selected_clause_start].isspace():
        selected_clause_start += 1
    while selected_clause_end > selected_clause_start and user_content[selected_clause_end - 1].isspace():
        selected_clause_end -= 1
    if selected_clause_end <= selected_clause_start:
        raise TraceRecordAdapterError(f"{context}: semantic indirection selected-choice clause is empty.")
    return CharSpan(
        start=user_content_span.start + selected_clause_start,
        end=user_content_span.start + selected_clause_end,
    )


def _token_span_for_char_span(
    offsets: tuple[TokenOffset, ...],
    char_span: CharSpan,
    context: str,
) -> TokenSpan:
    indices = _token_indices_for_char_span(offsets=offsets, char_span=char_span)
    if len(indices) == 0:
        raise TraceRecordAdapterError(f"{context}: tokenizer produced no tokens for char span {char_span}.")
    return TokenSpan(start=indices[0], end=indices[-1] + 1)


def _readout_indices_for_char_span(
    offsets: tuple[TokenOffset, ...],
    char_span: CharSpan,
    lower_bound: int,
    readout_token_count: int,
    context: str,
) -> tuple[int, ...]:
    indices = tuple(
        index
        for index in _token_indices_for_char_span(offsets=offsets, char_span=char_span)
        if index >= lower_bound
    )
    if len(indices) == 0:
        raise TraceRecordAdapterError(f"{context}: no readout tokens remain after visibility floor {lower_bound}.")
    return indices[-readout_token_count:]


def _token_indices_for_char_span(offsets: tuple[TokenOffset, ...], char_span: CharSpan) -> tuple[int, ...]:
    indices: list[int] = []
    for index, offset in enumerate(offsets):
        if offset.end <= offset.start:
            continue
        if offset.end > char_span.start and offset.start < char_span.end:
            indices.append(index)
    return tuple(indices)


def _tags_for_record(
    record: Mapping[str, object],
    collection: Mapping[str, object],
    readout_tag: str,
    context: str,
) -> tuple[str, ...]:
    return (
        "trace_collection",
        _required_string(record=collection, field_name="source", context=context),
        readout_tag,
        f"label:{_required_string(record=record, field_name='label', context=context)}",
        f"family:{_required_string(record=record, field_name='family', context=context)}",
        f"task:{_required_string(record=collection, field_name='task_id', context=context)}",
        f"participant:{_required_string(record=collection, field_name='participant_id', context=context)}",
        f"variant:{_required_string(record=collection, field_name='variant_id', context=context)}",
        f"credential_type:{_required_string(record=collection, field_name='credential_type', context=context)}",
    )


def _messages_from_turn(turn: Mapping[str, object], context: str) -> tuple[MessageRecord, ...]:
    message_rows = _required_list(record=turn, field_name="messages", context=context)
    messages: list[MessageRecord] = []
    for index, item in enumerate(message_rows):
        row = _as_mapping(value=item, context=f"{context}.messages[{index}]")
        messages.append(
            MessageRecord(
                role=_required_string(record=row, field_name="role", context=f"{context}.messages[{index}]"),
                content=_required_string(record=row, field_name="content", context=f"{context}.messages[{index}]"),
            )
        )
    if len(messages) == 0:
        raise TraceRecordAdapterError(f"{context}: normalized_turn.messages must not be empty.")
    return tuple(messages)


def _tool_calls_from_turn(turn: Mapping[str, object], context: str) -> tuple[ToolCallRecord, ...]:
    tool_call_rows = _required_list(record=turn, field_name="tool_calls", context=context)
    tool_calls: list[ToolCallRecord] = []
    for index, item in enumerate(tool_call_rows):
        row = _as_mapping(value=item, context=f"{context}.tool_calls[{index}]")
        tool_calls.append(
            ToolCallRecord(
                name=_required_string(record=row, field_name="name", context=f"{context}.tool_calls[{index}]"),
                arguments=_required_mapping(
                    record=row,
                    field_name="arguments",
                    context=f"{context}.tool_calls[{index}]",
                ),
            )
        )
    return tuple(tool_calls)


def _trace_id_for_record(record: Mapping[str, object], context: str) -> str:
    turn = _required_mapping(record=record, field_name="normalized_turn", context=context)
    return _required_string(record=turn, field_name="trace_id", context=context)


def _required_collection_metadata(turn: Mapping[str, object], context: str) -> Mapping[str, object]:
    metadata = _required_mapping(record=turn, field_name="metadata", context=context)
    return _required_mapping(record=metadata, field_name="collection", context=context)


def _required_label(record: Mapping[str, object], context: str) -> PromptLabel:
    value = _required_string(record=record, field_name="label", context=context)
    if value not in _VALID_LABELS:
        valid = ", ".join(sorted(_VALID_LABELS))
        raise TraceRecordAdapterError(f"{context}: label must be one of {valid}.")
    return cast(PromptLabel, value)


def _required_mapping(record: Mapping[str, object], field_name: str, context: str) -> Mapping[str, object]:
    value = record.get(field_name)
    return _as_mapping(value=value, context=f"{context}.{field_name}")


def _as_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TraceRecordAdapterError(f"{context} must be a JSON object.")
    for key in value:
        if not isinstance(key, str):
            raise TraceRecordAdapterError(f"{context} contains a non-string key.")
    return cast(Mapping[str, object], value)


def _required_list(record: Mapping[str, object], field_name: str, context: str) -> tuple[object, ...]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise TraceRecordAdapterError(f"{context}.{field_name} must be a list.")
    return tuple(value)


def _required_string(record: Mapping[str, object], field_name: str, context: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise TraceRecordAdapterError(f"{context}.{field_name} must be a string.")
    if value == "":
        raise TraceRecordAdapterError(f"{context}.{field_name} must not be empty.")
    return value


def _required_int(record: Mapping[str, object], field_name: str, context: str) -> int:
    value = record.get(field_name)
    if not isinstance(value, int):
        raise TraceRecordAdapterError(f"{context}.{field_name} must be an integer.")
    return value
