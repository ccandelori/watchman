from __future__ import annotations

from aegis.core.contracts import AuditEvent


class InMemoryAuditSink:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def write(self, event: AuditEvent) -> None:
        self._events.append(event)

    def recent(self, limit: int) -> tuple[AuditEvent, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        return tuple(reversed(self._events[-limit:]))
