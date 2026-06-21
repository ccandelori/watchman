from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from aegis.canaries.ledger import Honeytoken, HoneytokenLedger, inject_honeytokens
from aegis.core.contracts import CapabilityMode, JsonValue, Message, ModelInfo, NormalizedTurn, SensitiveSpan, ToolCall
from aegis.detectors.canary import CanaryRecord

TraceLabel: TypeAlias = Literal["benign", "secret_present_safe", "exfiltration_intent"]

_SCHEMA_VERSION = "trace_collection/v1"
_TRACE_LABELS: tuple[TraceLabel, ...] = ("benign", "secret_present_safe", "exfiltration_intent")
_PLACEHOLDER_PATTERN = re.compile(r"\{\{CREDENTIAL:([^:}]+):([^}]+)\}\}")
_SAFE_IDENTIFIER_PATTERN = re.compile(r"[^A-Za-z0-9_-]")


class TraceCollectionError(ValueError):
    """Raised when trace collection inputs violate the harness contract."""


@dataclass(frozen=True)
class TraceCollectionTask:
    task_id: str
    family: str
    credential_slot: str
    credential_type: str
    protected_context_template: str
    benign_context_template: str
    task_brief: str
    benign_instruction: str
    safe_instruction: str
    attack_instruction: str

    def __post_init__(self) -> None:
        _validate_non_empty("task_id", self.task_id)
        _validate_non_empty("family", self.family)
        _validate_non_empty("credential_slot", self.credential_slot)
        _validate_non_empty("credential_type", self.credential_type)
        _validate_non_empty("protected_context_template", self.protected_context_template)
        _validate_non_empty("benign_context_template", self.benign_context_template)
        _validate_non_empty("task_brief", self.task_brief)
        _validate_non_empty("benign_instruction", self.benign_instruction)
        _validate_non_empty("safe_instruction", self.safe_instruction)
        _validate_non_empty("attack_instruction", self.attack_instruction)


@dataclass(frozen=True)
class TraceCollectionInput:
    participant_id: str
    label: TraceLabel
    operator_prompt: str
    model_output_text: str | None
    tool_calls: tuple[ToolCall, ...]

    def __post_init__(self) -> None:
        _validate_non_empty("participant_id", self.participant_id)
        _validate_trace_label(self.label)
        _validate_non_empty("operator_prompt", self.operator_prompt)


@dataclass(frozen=True)
class TraceCollectionAssignment:
    assignment_id: str
    participant_id: str
    task_id: str
    label: TraceLabel
    family: str
    task_brief: str
    operator_instruction: str
    credential_type: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "assignment_id": self.assignment_id,
            "participant_id": self.participant_id,
            "task_id": self.task_id,
            "label": self.label,
            "family": self.family,
            "task_brief": self.task_brief,
            "operator_instruction": self.operator_instruction,
            "credential_type": self.credential_type,
        }


@dataclass(frozen=True)
class TraceCollectionRecord:
    schema_version: str
    label: TraceLabel
    family: str
    task_id: str
    participant_id: str
    normalized_turn: NormalizedTurn
    model_output_text: str | None
    canary_records: tuple[dict[str, JsonValue], ...]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "family": self.family,
            "task_id": self.task_id,
            "participant_id": self.participant_id,
            "normalized_turn": self.normalized_turn.to_dict(),
            "model_output_text": self.model_output_text,
            "canary_records": list(self.canary_records),
        }


@dataclass(frozen=True)
class _ArgumentInjectionResult:
    value: JsonValue
    sensitive_spans: tuple[SensitiveSpan, ...]


@dataclass(frozen=True)
class _ToolCallInjectionResult:
    tool_calls: tuple[ToolCall, ...]
    sensitive_spans: tuple[SensitiveSpan, ...]


def build_trace_collection_assignments(
    participant_ids: tuple[str, ...],
    tasks: tuple[TraceCollectionTask, ...],
) -> tuple[TraceCollectionAssignment, ...]:
    if len(participant_ids) == 0:
        raise TraceCollectionError("participant_ids must not be empty.")
    if len(tasks) == 0:
        raise TraceCollectionError("tasks must not be empty.")

    assignments: list[TraceCollectionAssignment] = []
    for participant_id in participant_ids:
        _validate_non_empty("participant_id", participant_id)
        for task in tasks:
            for label in _TRACE_LABELS:
                assignments.append(
                    TraceCollectionAssignment(
                        assignment_id=_assignment_id(
                            participant_id=participant_id,
                            task_id=task.task_id,
                            label=label,
                        ),
                        participant_id=participant_id,
                        task_id=task.task_id,
                        label=label,
                        family=task.family,
                        task_brief=task.task_brief,
                        operator_instruction=_instruction_for_label(task=task, label=label),
                        credential_type=task.credential_type,
                    )
                )
    return tuple(assignments)


def build_trace_collection_record(
    task: TraceCollectionTask,
    collection_input: TraceCollectionInput,
    model: ModelInfo,
    capability_mode: CapabilityMode,
    ledger: HoneytokenLedger,
) -> TraceCollectionRecord:
    turn_index = 0
    raw_messages = (
        Message(role="system", content=_context_for_label(task=task, label=collection_input.label)),
        Message(role="user", content=collection_input.operator_prompt),
    )
    message_injection = inject_honeytokens(messages=raw_messages, ledger=ledger, turn_index=turn_index)
    tool_injection = _inject_tool_call_honeytokens(
        tool_calls=collection_input.tool_calls,
        ledger=ledger,
        turn_index=turn_index,
    )
    trace_id = _trace_id(
        participant_id=collection_input.participant_id,
        task_id=task.task_id,
        label=collection_input.label,
    )
    turn = NormalizedTurn(
        trace_id=trace_id,
        session_id=ledger.session_id,
        turn_index=turn_index,
        capability_mode=capability_mode,
        model=model,
        messages=message_injection.messages,
        tool_calls=tool_injection.tool_calls,
        sensitive_spans=message_injection.sensitive_spans + tool_injection.sensitive_spans,
        metadata=_turn_metadata(task=task, collection_input=collection_input),
    )

    return TraceCollectionRecord(
        schema_version=_SCHEMA_VERSION,
        label=collection_input.label,
        family=task.family,
        task_id=task.task_id,
        participant_id=collection_input.participant_id,
        normalized_turn=turn,
        model_output_text=collection_input.model_output_text,
        canary_records=tuple(_canary_record_summary(record) for record in ledger.canary_records()),
    )


def write_trace_collection_jsonl(path: Path, records: tuple[TraceCollectionRecord, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record.to_dict(), sort_keys=True))
            output_file.write("\n")


def write_trace_collection_assignments_jsonl(
    path: Path,
    assignments: tuple[TraceCollectionAssignment, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for assignment in assignments:
            output_file.write(json.dumps(assignment.to_dict(), sort_keys=True))
            output_file.write("\n")


def _inject_tool_call_honeytokens(
    tool_calls: tuple[ToolCall, ...],
    ledger: HoneytokenLedger,
    turn_index: int,
) -> _ToolCallInjectionResult:
    transformed_tool_calls: list[ToolCall] = []
    sensitive_spans: list[SensitiveSpan] = []
    for tool_call in tool_calls:
        transformed_arguments: dict[str, JsonValue] = {}
        for argument_name, argument_value in tool_call.arguments.items():
            result = _inject_argument_value(
                value=argument_value,
                ledger=ledger,
                turn_index=turn_index,
                tool_call_name=tool_call.name,
                argument_path=f"arguments.{argument_name}",
            )
            transformed_arguments[argument_name] = result.value
            sensitive_spans.extend(result.sensitive_spans)
        transformed_tool_calls.append(ToolCall(name=tool_call.name, arguments=transformed_arguments))
    return _ToolCallInjectionResult(tool_calls=tuple(transformed_tool_calls), sensitive_spans=tuple(sensitive_spans))


def _inject_argument_value(
    value: JsonValue,
    ledger: HoneytokenLedger,
    turn_index: int,
    tool_call_name: str,
    argument_path: str,
) -> _ArgumentInjectionResult:
    if isinstance(value, str):
        return _inject_string_argument(
            value=value,
            ledger=ledger,
            turn_index=turn_index,
            tool_call_name=tool_call_name,
            argument_path=argument_path,
        )
    if isinstance(value, list):
        transformed_items: list[JsonValue] = []
        list_sensitive_spans: list[SensitiveSpan] = []
        for index, item in enumerate(value):
            result = _inject_argument_value(
                value=item,
                ledger=ledger,
                turn_index=turn_index,
                tool_call_name=tool_call_name,
                argument_path=f"{argument_path}[{index}]",
            )
            transformed_items.append(result.value)
            list_sensitive_spans.extend(result.sensitive_spans)
        return _ArgumentInjectionResult(value=transformed_items, sensitive_spans=tuple(list_sensitive_spans))
    if isinstance(value, dict):
        transformed_values: dict[str, JsonValue] = {}
        dict_sensitive_spans: list[SensitiveSpan] = []
        for key, nested_value in value.items():
            result = _inject_argument_value(
                value=nested_value,
                ledger=ledger,
                turn_index=turn_index,
                tool_call_name=tool_call_name,
                argument_path=f"{argument_path}.{key}",
            )
            transformed_values[key] = result.value
            dict_sensitive_spans.extend(result.sensitive_spans)
        return _ArgumentInjectionResult(value=transformed_values, sensitive_spans=tuple(dict_sensitive_spans))
    return _ArgumentInjectionResult(value=value, sensitive_spans=())


def _inject_string_argument(
    value: str,
    ledger: HoneytokenLedger,
    turn_index: int,
    tool_call_name: str,
    argument_path: str,
) -> _ArgumentInjectionResult:
    parts: list[str] = []
    sensitive_spans: list[SensitiveSpan] = []
    cursor = 0
    for match in _PLACEHOLDER_PATTERN.finditer(value):
        slot_name = match.group(1)
        credential_type = match.group(2)
        honeytoken = ledger.plant(slot_name=slot_name, credential_type=credential_type, turn_index=turn_index)
        parts.append(value[cursor : match.start()])
        char_start = sum(len(part) for part in parts)
        parts.append(honeytoken.value)
        char_end = char_start + len(honeytoken.value)
        sensitive_spans.append(
            _tool_call_sensitive_span(
                honeytoken=honeytoken,
                char_start=char_start,
                char_end=char_end,
                tool_call_name=tool_call_name,
                argument_path=argument_path,
            )
        )
        cursor = match.end()
    parts.append(value[cursor:])
    return _ArgumentInjectionResult(value="".join(parts), sensitive_spans=tuple(sensitive_spans))


def _tool_call_sensitive_span(
    honeytoken: Honeytoken,
    char_start: int,
    char_end: int,
    tool_call_name: str,
    argument_path: str,
) -> SensitiveSpan:
    return SensitiveSpan(
        kind="honeytoken",
        source="tool_call_argument",
        char_start=char_start,
        char_end=char_end,
        token_start=None,
        token_end=None,
        identifier=honeytoken.canary_id,
        metadata={
            "slot_name": honeytoken.slot_name,
            "credential_type": honeytoken.credential_type,
            "sha256": honeytoken.sha256,
            "turn_planted": honeytoken.turn_planted,
            "tool_call_name": tool_call_name,
            "argument_path": argument_path,
        },
    )


def _turn_metadata(task: TraceCollectionTask, collection_input: TraceCollectionInput) -> dict[str, JsonValue]:
    return {
        "collection": {
            "schema_version": _SCHEMA_VERSION,
            "label": collection_input.label,
            "family": task.family,
            "task_id": task.task_id,
            "participant_id": collection_input.participant_id,
            "credential_slot": task.credential_slot,
            "credential_type": task.credential_type,
        },
        "cift": {
            "tokenization_status": "pending",
            "readout_window_status": "pending_tokenization",
            "requires_tokenization": True,
        },
    }


def _context_for_label(task: TraceCollectionTask, label: TraceLabel) -> str:
    _validate_trace_label(label)
    if label == "benign":
        return task.benign_context_template
    return task.protected_context_template


def _instruction_for_label(task: TraceCollectionTask, label: TraceLabel) -> str:
    _validate_trace_label(label)
    if label == "benign":
        return task.benign_instruction
    if label == "secret_present_safe":
        return task.safe_instruction
    return task.attack_instruction


def _assignment_id(participant_id: str, task_id: str, label: TraceLabel) -> str:
    return f"assignment-{_safe_identifier(participant_id)}-{_safe_identifier(task_id)}-{label}"


def _trace_id(participant_id: str, task_id: str, label: TraceLabel) -> str:
    return f"trace-{_safe_identifier(participant_id)}-{_safe_identifier(task_id)}-{label}"


def _safe_identifier(value: str) -> str:
    return _SAFE_IDENTIFIER_PATTERN.sub("_", value)


def _canary_record_summary(record: CanaryRecord) -> dict[str, JsonValue]:
    return {
        "canary_id": record.canary_id,
        "credential_type": record.credential_type,
        "sha256": record.sha256,
        "source": record.source,
        "metadata": record.metadata,
    }


def _validate_trace_label(label: TraceLabel) -> None:
    if label not in _TRACE_LABELS:
        raise TraceCollectionError(f"unsupported trace label: {label}")


def _validate_non_empty(field_name: str, value: str) -> None:
    if value == "":
        raise TraceCollectionError(f"{field_name} must not be empty.")
