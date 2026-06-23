from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from aegis.core.contracts import (
    CapabilityMode,
    JsonValue,
    Message,
    ModelInfo,
    SensitiveSpan,
    ToolCall,
)
from aegis.core.orchestrator import AegisRuntime, AegisRuntimeResponse, RuntimeRequest
from aegis.detectors.cift_candidate import CiftCandidateScore


class OfflineReplayError(ValueError):
    """Raised when offline replay fixtures do not match runtime contracts."""


def load_runtime_requests_jsonl(path: Path) -> tuple[RuntimeRequest, ...]:
    return tuple(_runtime_request_from_json(record, line_number) for line_number, record in _read_jsonl(path))


def load_cift_candidate_scores_jsonl(path: Path) -> dict[str, CiftCandidateScore]:
    scores_by_example_id: dict[str, CiftCandidateScore] = {}
    for line_number, record in _read_jsonl(path):
        score = _cift_candidate_score_from_json(record, line_number)
        if score.example_id in scores_by_example_id:
            raise OfflineReplayError(f"Line {line_number}: duplicate CIFT score example_id '{score.example_id}'.")
        scores_by_example_id[score.example_id] = score
    if len(scores_by_example_id) == 0:
        raise OfflineReplayError(f"No CIFT candidate scores found in {path}.")
    return scores_by_example_id


def replay_requests(
    runtime: AegisRuntime,
    requests: tuple[RuntimeRequest, ...],
) -> tuple[AegisRuntimeResponse, ...]:
    return tuple(runtime.evaluate_turn(request) for request in requests)


def _read_jsonl(path: Path) -> tuple[tuple[int, Mapping[str, object]], ...]:
    records: list[tuple[int, Mapping[str, object]]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            decoded = json.loads(line)
            records.append((line_number, _as_mapping(decoded, f"Line {line_number}")))
    if len(records) == 0:
        raise OfflineReplayError(f"No JSONL records found in {path}.")
    return tuple(records)


def _runtime_request_from_json(record: Mapping[str, object], line_number: int) -> RuntimeRequest:
    model = _as_mapping(record.get("model"), f"Line {line_number} field 'model'")
    return RuntimeRequest(
        trace_id=_required_string(record, "trace_id", line_number),
        session_id=_required_string(record, "session_id", line_number),
        turn_index=_required_int(record, "turn_index", line_number),
        capability_mode=_capability_mode(_required_string(record, "capability_mode", line_number), line_number),
        model=ModelInfo(
            provider=_required_string(model, "provider", line_number),
            model_id=_required_string(model, "model_id", line_number),
            revision=_optional_string(model, "revision", line_number),
            selected_device=_optional_string(model, "selected_device", line_number),
        ),
        messages=tuple(
            _message_from_json(item, line_number) for item in _required_list(record, "messages", line_number)
        ),
        tool_calls=tuple(
            _tool_call_from_json(item, line_number) for item in _required_list(record, "tool_calls", line_number)
        ),
        sensitive_spans=tuple(
            _sensitive_span_from_json(item, line_number)
            for item in _required_list(record, "sensitive_spans", line_number)
        ),
        metadata=_json_object(_as_mapping(record.get("metadata"), f"Line {line_number} field 'metadata'")),
    )


def _cift_candidate_score_from_json(record: Mapping[str, object], line_number: int) -> CiftCandidateScore:
    example_id = _required_string(record, "example_id", line_number)
    detector_result = _as_mapping(record.get("detector_result"), f"Line {line_number} field 'detector_result'")
    evidence = _json_object(_as_mapping(detector_result.get("evidence"), f"Line {line_number} detector evidence"))
    return CiftCandidateScore(
        example_id=example_id,
        score=_required_float(detector_result, "score", line_number),
        confidence=_required_float(detector_result, "confidence", line_number),
        evidence=evidence,
    )


def _message_from_json(value: object, line_number: int) -> Message:
    record = _as_mapping(value, f"Line {line_number} message")
    return Message(
        role=_required_string(record, "role", line_number),
        content=_required_string(record, "content", line_number),
    )


def _tool_call_from_json(value: object, line_number: int) -> ToolCall:
    record = _as_mapping(value, f"Line {line_number} tool call")
    arguments = _json_object(_as_mapping(record.get("arguments"), f"Line {line_number} tool call arguments"))
    return ToolCall(name=_required_string(record, "name", line_number), arguments=arguments)


def _sensitive_span_from_json(value: object, line_number: int) -> SensitiveSpan:
    record = _as_mapping(value, f"Line {line_number} sensitive span")
    metadata = _json_object(_as_mapping(record.get("metadata"), f"Line {line_number} sensitive span metadata"))
    return SensitiveSpan(
        kind=_required_string(record, "kind", line_number),
        source=_required_string(record, "source", line_number),
        char_start=_optional_int(record, "char_start", line_number),
        char_end=_optional_int(record, "char_end", line_number),
        token_start=_optional_int(record, "token_start", line_number),
        token_end=_optional_int(record, "token_end", line_number),
        identifier=_optional_string(record, "identifier", line_number),
        metadata=metadata,
    )


def _capability_mode(value: str, line_number: int) -> CapabilityMode:
    try:
        return CapabilityMode(value)
    except ValueError as exc:
        raise OfflineReplayError(f"Line {line_number}: unsupported capability_mode '{value}'.") from exc


def _as_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise OfflineReplayError(f"{description}: expected a JSON object.")
    return cast(Mapping[str, object], value)


def _required_string(record: Mapping[str, object], field_name: str, line_number: int) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or value == "":
        raise OfflineReplayError(f"Line {line_number}: field '{field_name}' must be a non-empty string.")
    return value


def _optional_string(record: Mapping[str, object], field_name: str, line_number: int) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise OfflineReplayError(f"Line {line_number}: field '{field_name}' must be null or a non-empty string.")
    return value


def _required_int(record: Mapping[str, object], field_name: str, line_number: int) -> int:
    value = record.get(field_name)
    if not isinstance(value, int):
        raise OfflineReplayError(f"Line {line_number}: field '{field_name}' must be an integer.")
    return value


def _optional_int(record: Mapping[str, object], field_name: str, line_number: int) -> int | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, int):
        raise OfflineReplayError(f"Line {line_number}: field '{field_name}' must be null or an integer.")
    return value


def _required_float(record: Mapping[str, object], field_name: str, line_number: int) -> float:
    value = record.get(field_name)
    if not isinstance(value, int) and not isinstance(value, float):
        raise OfflineReplayError(f"Line {line_number}: field '{field_name}' must be numeric.")
    return float(value)


def _required_list(record: Mapping[str, object], field_name: str, line_number: int) -> list[object]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise OfflineReplayError(f"Line {line_number}: field '{field_name}' must be a list.")
    return value


def _json_object(record: Mapping[str, object]) -> dict[str, JsonValue]:
    return {key: _json_value(value) for key, value in record.items()}


def _json_value(value: object) -> JsonValue:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return None
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return _json_object(cast(Mapping[str, object], value))
    raise OfflineReplayError(f"Unsupported JSON value type '{type(value).__name__}'.")
