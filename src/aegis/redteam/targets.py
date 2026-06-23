"""Target adapters used by redteam scenarios."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from aegis.core.contracts import JsonValue
from aegis.proxy.mock_app import MockProxyApp


class RedteamTargetError(RuntimeError):
    """Raised when a redteam target returns an unusable response."""


@dataclass(frozen=True)
class RedteamResult:
    status_code: int
    assistant_text: str
    aegis: dict[str, JsonValue]
    raw_response: dict[str, JsonValue]


@dataclass(frozen=True)
class HttpJsonResponse:
    status_code: int
    payload: dict[str, JsonValue]


class AegisTarget(Protocol):
    def health(self) -> dict[str, JsonValue]:
        """Return the Aegis health payload."""

    def send_chat_completion(self, body: dict[str, JsonValue]) -> RedteamResult:
        """Send an OpenAI-compatible chat completion request."""

    def recent_audit(self) -> dict[str, JsonValue]:
        """Return recent audit events."""


class InProcessAegisTarget:
    def __init__(self, proxy: MockProxyApp) -> None:
        self._proxy = proxy

    def health(self) -> dict[str, JsonValue]:
        status_code, payload = self._proxy.handle(method="GET", path="/health", body={})
        return _successful_payload(status_code=status_code, payload=payload, operation="health")

    def send_chat_completion(self, body: dict[str, JsonValue]) -> RedteamResult:
        status_code, payload = self._proxy.handle(method="POST", path="/v1/chat/completions", body=dict(body))
        return _redteam_result_from_response(status_code=status_code, payload=payload)

    def recent_audit(self) -> dict[str, JsonValue]:
        status_code, payload = self._proxy.handle(method="GET", path="/audit/recent", body={})
        return _successful_payload(status_code=status_code, payload=payload, operation="recent audit")


class HttpAegisTarget:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        if base_url == "":
            raise RedteamTargetError("base_url must be a non-empty string.")
        if timeout_seconds <= 0:
            raise RedteamTargetError("timeout_seconds must be greater than zero.")
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout_seconds = timeout_seconds

    def health(self) -> dict[str, JsonValue]:
        response = self._request_json(method="GET", path="/health", body=None)
        return _successful_payload(status_code=response.status_code, payload=response.payload, operation="health")

    def send_chat_completion(self, body: dict[str, JsonValue]) -> RedteamResult:
        response = self._request_json(method="POST", path="/v1/chat/completions", body=body)
        return _redteam_result_from_response(status_code=response.status_code, payload=response.payload)

    def recent_audit(self) -> dict[str, JsonValue]:
        response = self._request_json(method="GET", path="/audit/recent", body=None)
        return _successful_payload(status_code=response.status_code, payload=response.payload, operation="recent audit")

    def _request_json(self, method: str, path: str, body: dict[str, JsonValue] | None) -> HttpJsonResponse:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"

        request = Request(urljoin(self._base_url, path.lstrip("/")), data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = _decode_json_response(response.read().decode("utf-8"))
                return HttpJsonResponse(status_code=response.status, payload=payload)
        except HTTPError as exc:
            payload = _decode_json_response(exc.read().decode("utf-8"))
            return HttpJsonResponse(status_code=exc.code, payload=payload)
        except URLError as exc:
            raise RedteamTargetError(f"HTTP target request failed for {method} {path}: {exc.reason}") from exc


def _successful_payload(status_code: int, payload: dict[str, JsonValue], operation: str) -> dict[str, JsonValue]:
    if status_code != 200:
        raise RedteamTargetError(f"Aegis {operation} returned status {status_code}: {payload}")
    return dict(payload)


def _redteam_result_from_response(status_code: int, payload: dict[str, JsonValue]) -> RedteamResult:
    if status_code != 200:
        raise RedteamTargetError(f"Aegis chat completion returned status {status_code}: {payload}")
    return RedteamResult(
        status_code=status_code,
        assistant_text=_assistant_text(payload),
        aegis=_json_object_field(payload, "aegis"),
        raw_response=dict(payload),
    )


def _assistant_text(payload: Mapping[str, JsonValue]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        raise RedteamTargetError("Aegis chat completion response must include at least one choice.")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RedteamTargetError("Aegis chat completion choice must be an object.")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RedteamTargetError("Aegis chat completion choice must include a message object.")
    content = message.get("content")
    if not isinstance(content, str):
        raise RedteamTargetError("Aegis chat completion message content must be a string.")
    return content


def _json_object_field(payload: Mapping[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise RedteamTargetError(f"Aegis response field '{key}' must be an object.")
    return value


def _decode_json_response(raw_body: str) -> dict[str, JsonValue]:
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise RedteamTargetError(f"Aegis response body is not valid JSON: {raw_body}") from exc
    if not isinstance(payload, dict):
        raise RedteamTargetError(f"Aegis response body must be a JSON object: {raw_body}")
    return cast(dict[str, JsonValue], payload)
