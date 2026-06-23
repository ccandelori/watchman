from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

from aegis.core.contracts import JsonValue
from aegis.proxy.mock_app import create_default_proxy
from aegis.redteam.targets import HttpAegisTarget, InProcessAegisTarget


class _AegisTargetHandler(BaseHTTPRequestHandler):
    events: ClassVar[list[dict[str, JsonValue]]] = []

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json({"status": "ok"})
            return
        if self.path == "/audit/recent":
            self._write_json({"events": self.events})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        event = {"trace_id": "trace-http-target-1"}
        self.events.append(event)
        self._write_json(
            {
                "choices": [{"message": {"role": "assistant", "content": "adapter response"}}],
                "aegis": {"trace_id": "trace-http-target-1", "policy_decision": {}, "detector_results": []},
            }
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _write_json(self, payload: dict[str, JsonValue]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def _http_target_server() -> Iterator[str]:
    _AegisTargetHandler.events = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AegisTargetHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_in_process_target_exercises_aegis_proxy_contract() -> None:
    target = InProcessAegisTarget(create_default_proxy())

    result = target.send_chat_completion(
        {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"trace_id": "trace-redteam-1", "session_id": "session-redteam-1"},
        }
    )
    audit = target.recent_audit()

    assert target.health() == {"status": "ok"}
    assert result.status_code == 200
    assert result.assistant_text == "Aegis mock response."
    assert result.aegis["trace_id"] == "trace-redteam-1"
    assert audit["events"][0]["trace_id"] == "trace-redteam-1"


def test_http_target_exercises_http_contract() -> None:
    with _http_target_server() as base_url:
        target = HttpAegisTarget(base_url=base_url, timeout_seconds=2.0)

        result = target.send_chat_completion(
            {
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
                "metadata": {"trace_id": "trace-http-target-1", "session_id": "session-http-target-1"},
            }
        )
        health = target.health()
        audit = target.recent_audit()

    assert health == {"status": "ok"}
    assert result.assistant_text == "adapter response"
    assert result.aegis["trace_id"] == "trace-http-target-1"
    assert audit["events"][0]["trace_id"] == "trace-http-target-1"
