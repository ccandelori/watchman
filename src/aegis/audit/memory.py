from __future__ import annotations

from aegis.core.contracts import AuditEvent, JsonValue


class InMemoryAuditSink:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def write(self, event: AuditEvent) -> None:
        self._events.append(event)

    def recent(self, limit: int, session_id: str | None = None) -> tuple[AuditEvent, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        events = self._events
        if session_id is not None:
            events = [event for event in events if event.session_id == session_id]
        return tuple(reversed(events[-limit:]))

    def recent_records(self, limit: int, session_id: str | None = None) -> tuple[dict[str, JsonValue], ...]:
        return tuple(event.to_dict() for event in self.recent(limit=limit, session_id=session_id))

    def find_record(self, trace_id: str | None, session_id: str | None) -> dict[str, JsonValue] | None:
        if trace_id is None and session_id is None:
            raise ValueError("trace_id or session_id must be provided.")
        for event in reversed(self._events):
            if trace_id is not None and event.trace_id != trace_id:
                continue
            if session_id is not None and event.session_id != session_id:
                continue
            return event.to_dict()
        return None

    def durable_path(self) -> str | None:
        return None

    def clear(self) -> None:
        self._events.clear()

    def clear_session(self, session_id: str) -> None:
        self._events = [event for event in self._events if event.session_id != session_id]
