from __future__ import annotations

from aegis.core.contracts import AuditEvent


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

    def clear(self) -> None:
        self._events.clear()

    def clear_session(self, session_id: str) -> None:
        self._events = [event for event in self._events if event.session_id != session_id]
