from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import cast

from aegis.core.contracts import JsonValue

_REQUEST_SCHEMA_VERSION = "aegis.loopback_openai_provider_request/v1"
_MAX_BODY_BYTES = 1_000_000


class LoopbackOpenAIProviderError(ValueError):
    """Raised when the loopback OpenAI-compatible provider is configured incorrectly."""


@dataclass(frozen=True)
class LoopbackOpenAIProviderConfig:
    host: str
    port: int
    response_content: str
    request_log_path: Path | None
    expected_bearer_token: str | None
    forbidden_substrings: tuple[str, ...]


def parse_args(argv: Sequence[str]) -> LoopbackOpenAIProviderConfig:
    parser = argparse.ArgumentParser(
        description="Run a local OpenAI-compatible loopback provider for Aegis gateway smoke evidence."
    )
    parser.add_argument("--host", required=True, help="Host interface to bind, for example 127.0.0.1.")
    parser.add_argument("--port", required=True, type=int, help="TCP port to bind.")
    parser.add_argument("--response-content", required=True, help="Assistant content returned for chat completions.")
    parser.add_argument(
        "--request-log",
        required=False,
        help="Optional JSONL path for redacted provider request receipts.",
    )
    parser.add_argument(
        "--expected-bearer-token",
        required=False,
        help="Optional bearer token the gateway must send in the Authorization header.",
    )
    parser.add_argument(
        "--forbidden-substring",
        action="append",
        required=False,
        help="Sensitive marker to detect without logging the marker or request body. May be supplied repeatedly.",
    )
    args = parser.parse_args(argv)
    forbidden_substrings = tuple(str(item) for item in (args.forbidden_substring or ()))
    config = LoopbackOpenAIProviderConfig(
        host=str(args.host),
        port=int(args.port),
        response_content=str(args.response_content),
        request_log_path=None if args.request_log is None else Path(str(args.request_log)),
        expected_bearer_token=None if args.expected_bearer_token is None else str(args.expected_bearer_token),
        forbidden_substrings=forbidden_substrings,
    )
    validate_config(config)
    return config


def validate_config(config: LoopbackOpenAIProviderConfig) -> None:
    if not _is_loopback_host(config.host):
        raise LoopbackOpenAIProviderError("loopback provider host must be localhost or a loopback IP address.")
    if config.port <= 0 or config.port > 65535:
        raise LoopbackOpenAIProviderError("loopback provider port must be between 1 and 65535.")
    if config.response_content == "":
        raise LoopbackOpenAIProviderError("loopback provider response content must not be empty.")
    for forbidden_substring in config.forbidden_substrings:
        if forbidden_substring == "":
            raise LoopbackOpenAIProviderError("loopback provider forbidden substrings must not be empty.")


def openai_chat_completion_response(payload: dict[str, JsonValue], response_content: str) -> dict[str, JsonValue]:
    model = payload.get("model")
    if not isinstance(model, str) or model == "":
        model = "loopback-model"
    return {
        "id": "chatcmpl-aegis-loopback",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_content},
                "finish_reason": "stop",
            }
        ],
    }


def loopback_request_record(
    method: str,
    path: str,
    body: bytes,
    payload: dict[str, JsonValue] | None,
    authorization_status: str,
    forbidden_substrings: tuple[str, ...],
) -> dict[str, JsonValue]:
    body_text = body.decode("utf-8", errors="replace")
    forbidden_presence = tuple(marker in body_text for marker in forbidden_substrings)
    model: JsonValue = None
    message_count: JsonValue = None
    if payload is not None:
        model = payload.get("model")
        messages = payload.get("messages")
        if isinstance(messages, list):
            message_count = len(messages)
    return {
        "schema_version": _REQUEST_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "method": method,
        "path": path,
        "authorization_status": authorization_status,
        "request_body_sha256": hashlib.sha256(body).hexdigest(),
        "request_body_bytes": len(body),
        "model": model,
        "message_count": message_count,
        "forbidden_substring_count": len(forbidden_substrings),
        "forbidden_substring_present": any(forbidden_presence),
    }


def make_server(config: LoopbackOpenAIProviderConfig) -> HTTPServer:
    validate_config(config)
    if config.request_log_path is not None:
        config.request_log_path.parent.mkdir(parents=True, exist_ok=True)
    return HTTPServer((config.host, config.port), _handler_class(config))


def run_server(config: LoopbackOpenAIProviderConfig) -> None:
    server = make_server(config)
    server.serve_forever()


def main() -> None:
    try:
        run_server(parse_args(tuple(sys.argv[1:])))
    except KeyboardInterrupt:
        return
    except LoopbackOpenAIProviderError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(2) from exc


def _handler_class(config: LoopbackOpenAIProviderConfig) -> type[BaseHTTPRequestHandler]:
    class LoopbackOpenAIRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                _send_json(self, 200, {"status": "ok"})
                return
            _send_json(self, 404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            _handle_post(self, config)

        def log_message(self, format: str, *args: object) -> None:
            return

    return LoopbackOpenAIRequestHandler


def _handle_post(handler: BaseHTTPRequestHandler, config: LoopbackOpenAIProviderConfig) -> None:
    try:
        body = _read_body(handler)
    except LoopbackOpenAIProviderError as exc:
        _send_json(handler, 400, {"error": {"message": str(exc)}})
        return
    decoded = _decode_json_object(body)
    authorization_status = _authorization_status(
        authorization_header=handler.headers.get("Authorization"),
        expected_bearer_token=config.expected_bearer_token,
    )
    _write_request_record(
        path=config.request_log_path,
        record=loopback_request_record(
            method="POST",
            path=handler.path,
            body=body,
            payload=decoded if isinstance(decoded, dict) else None,
            authorization_status=authorization_status,
            forbidden_substrings=config.forbidden_substrings,
        ),
    )
    if handler.path != "/v1/chat/completions":
        _send_json(handler, 404, {"error": {"message": "not found"}})
        return
    if config.expected_bearer_token is not None and authorization_status != "matched_expected":
        _send_json(handler, 401, {"error": {"message": "authorization bearer token mismatch"}})
        return
    if not isinstance(decoded, dict):
        _send_json(handler, 400, {"error": {"message": "request body must be a JSON object"}})
        return
    _send_json(handler, 200, openai_chat_completion_response(decoded, config.response_content))


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    content_length = handler.headers.get("Content-Length")
    if content_length is None:
        return b""
    try:
        size = int(content_length)
    except ValueError as exc:
        raise LoopbackOpenAIProviderError("Content-Length must be an integer.") from exc
    if size < 0 or size > _MAX_BODY_BYTES:
        raise LoopbackOpenAIProviderError("request body is outside the supported size limit.")
    return handler.rfile.read(size)


def _decode_json_object(body: bytes) -> dict[str, JsonValue] | None:
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    return cast(dict[str, JsonValue], decoded)


def _authorization_status(authorization_header: str | None, expected_bearer_token: str | None) -> str:
    if authorization_header is None or authorization_header == "":
        return "missing"
    if expected_bearer_token is None:
        return "present"
    if authorization_header == f"Bearer {expected_bearer_token}":
        return "matched_expected"
    return "mismatched"


def _is_loopback_host(host: str) -> bool:
    if host in ("localhost", "::1"):
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _write_request_record(path: Path | None, record: dict[str, JsonValue]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, sort_keys=True))
        output_file.write("\n")


def _send_json(handler: BaseHTTPRequestHandler, status_code: int, payload: dict[str, JsonValue]) -> None:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
