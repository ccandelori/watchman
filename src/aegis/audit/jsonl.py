from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from aegis.audit.memory import InMemoryAuditSink
from aegis.core.contracts import AuditEvent, JsonValue


class JsonlAuditSinkError(ValueError):
    """Raised when a durable audit JSONL record cannot be read or written."""


class JsonlAuditSink(InMemoryAuditSink):
    def __init__(self, path: Path) -> None:
        if str(path) == "":
            raise JsonlAuditSinkError("audit JSONL path must not be empty.")
        super().__init__()
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: AuditEvent) -> None:
        record = event.to_dict()
        super().write(event)
        with self._path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, sort_keys=True))
            output_file.write("\n")

    def recent_records(self, limit: int, session_id: str | None = None) -> tuple[dict[str, JsonValue], ...]:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        records = _read_records(self._path)
        if session_id is not None:
            records = tuple(record for record in records if record.get("session_id") == session_id)
        return tuple(reversed(records[-limit:]))

    def find_record(self, trace_id: str | None, session_id: str | None) -> dict[str, JsonValue] | None:
        return find_jsonl_audit_record(path=self._path, trace_id=trace_id, session_id=session_id)

    def durable_path(self) -> str | None:
        return str(self._path)

    def clear(self) -> None:
        super().clear()
        self._path.write_text("", encoding="utf-8")

    def clear_session(self, session_id: str) -> None:
        super().clear_session(session_id)
        records = tuple(record for record in _read_records(self._path) if record.get("session_id") != session_id)
        with self._path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, sort_keys=True))
                output_file.write("\n")


def _read_records(path: Path) -> tuple[dict[str, JsonValue], ...]:
    if not path.exists():
        return ()
    records: list[dict[str, JsonValue]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if raw_line.strip() == "":
            continue
        try:
            decoded = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise JsonlAuditSinkError(f"{path}:{line_number}: invalid JSONL audit record: {exc.msg}.") from exc
        if not isinstance(decoded, dict):
            raise JsonlAuditSinkError(f"{path}:{line_number}: audit record must be a JSON object.")
        records.append(cast(dict[str, JsonValue], decoded))
    return tuple(records)


def read_jsonl_audit_records(path: Path) -> tuple[dict[str, JsonValue], ...]:
    return _read_records(path)


def find_jsonl_audit_record(path: Path, trace_id: str | None, session_id: str | None) -> dict[str, JsonValue] | None:
    if trace_id is None and session_id is None:
        raise ValueError("trace_id or session_id must be provided.")
    records = _read_records(path)
    for record in reversed(records):
        if trace_id is not None and record.get("trace_id") != trace_id:
            continue
        if session_id is not None and record.get("session_id") != session_id:
            continue
        return record
    return None
