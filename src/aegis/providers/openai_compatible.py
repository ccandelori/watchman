from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from aegis.core.contracts import JsonValue, NormalizedTurn
from aegis.core.orchestrator import ModelResponse

OpenAICompatibleSender = Callable[[str, dict[str, JsonValue], dict[str, str], float], dict[str, JsonValue]]


class OpenAICompatibleProviderError(RuntimeError):
    """Raised when the OpenAI-compatible provider cannot return a usable response."""


@dataclass(frozen=True)
class OpenAICompatibleProviderConfig:
    base_url: str
    api_key: str
    default_model: str | None
    timeout_seconds: float


class OpenAICompatibleProvider:
    def __init__(self, config: OpenAICompatibleProviderConfig, sender: OpenAICompatibleSender) -> None:
        if config.base_url == "":
            raise OpenAICompatibleProviderError("OpenAI-compatible provider base_url must not be empty.")
        if config.api_key == "":
            raise OpenAICompatibleProviderError("OpenAI-compatible provider api_key must not be empty.")
        if config.timeout_seconds <= 0:
            raise OpenAICompatibleProviderError("OpenAI-compatible provider timeout_seconds must be positive.")
        self._config = config
        self._sender = sender

    def generate(self, turn: NormalizedTurn) -> ModelResponse:
        model_id = self._config.default_model or turn.model.model_id
        payload: dict[str, JsonValue] = {
            "model": model_id,
            "messages": [message.to_dict() for message in turn.messages],
        }
        response = self._sender(
            _chat_completions_url(self._config.base_url),
            payload,
            {"Authorization": f"Bearer {self._config.api_key}", "Content-Type": "application/json"},
            self._config.timeout_seconds,
        )
        output_text = _extract_output_text(response)
        return ModelResponse(
            output_text=output_text,
            metadata={
                "provider": "openai_compatible",
                "base_url": _redacted_base_url(self._config.base_url),
                "model_id": model_id,
            },
        )


def urllib_openai_sender(
    url: str,
    payload: dict[str, JsonValue],
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, JsonValue]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            status_code = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        raise OpenAICompatibleProviderError(
            f"OpenAI-compatible provider returned HTTP {exc.code}: {_safe_body_excerpt(body)}"
        ) from exc
    except urllib.error.URLError as exc:
        raise OpenAICompatibleProviderError(f"OpenAI-compatible provider request failed: {exc.reason}") from exc
    if status_code < 200 or status_code >= 300:
        raise OpenAICompatibleProviderError(
            f"OpenAI-compatible provider returned HTTP {status_code}: {_safe_body_excerpt(body)}"
        )
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OpenAICompatibleProviderError("OpenAI-compatible provider returned invalid JSON.") from exc
    if not isinstance(decoded, dict):
        raise OpenAICompatibleProviderError("OpenAI-compatible provider response must be a JSON object.")
    return cast(dict[str, JsonValue], decoded)


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _extract_output_text(response: dict[str, JsonValue]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        raise OpenAICompatibleProviderError("OpenAI-compatible response must include non-empty choices.")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise OpenAICompatibleProviderError("OpenAI-compatible choice must be an object.")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise OpenAICompatibleProviderError("OpenAI-compatible choice must include message object.")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _extract_text_parts(content)
    raise OpenAICompatibleProviderError("OpenAI-compatible message content must be string or content part list.")


def _extract_text_parts(content: list[JsonValue]) -> str:
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    if len(parts) == 0:
        raise OpenAICompatibleProviderError("OpenAI-compatible content part list did not include text.")
    return "".join(parts)


def _safe_body_excerpt(body: str) -> str:
    return body[:500]


def _redacted_base_url(base_url: str) -> str:
    return base_url.rstrip("/")
