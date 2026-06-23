from dataclasses import dataclass

import pytest

from aegis.core.contracts import CapabilityMode, JsonValue, Message, ModelInfo, NormalizedTurn
from aegis.providers.openai_compatible import (
    OpenAICompatibleProvider,
    OpenAICompatibleProviderConfig,
    OpenAICompatibleProviderError,
)


@dataclass(frozen=True)
class CapturedProviderCall:
    url: str
    payload: dict[str, JsonValue]
    headers: dict[str, str]
    timeout_seconds: float


def test_openai_compatible_provider_sends_chat_completion_payload() -> None:
    calls: list[CapturedProviderCall] = []

    def sender(
        url: str,
        payload: dict[str, JsonValue],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, JsonValue]:
        calls.append(
            CapturedProviderCall(
                url=url,
                payload=payload,
                headers=headers,
                timeout_seconds=timeout_seconds,
            )
        )
        return {"choices": [{"message": {"content": "provider response"}}]}

    provider = OpenAICompatibleProvider(
        config=OpenAICompatibleProviderConfig(
            base_url="https://provider.example/v1",
            api_key="test-key",
            default_model=None,
            timeout_seconds=12.5,
        ),
        sender=sender,
    )

    response = provider.generate(_turn(model_id="runtime-model"))

    assert response.output_text == "provider response"
    assert response.metadata == {
        "provider": "openai_compatible",
        "base_url": "https://provider.example/v1",
        "model_id": "runtime-model",
    }
    assert calls == [
        CapturedProviderCall(
            url="https://provider.example/v1/chat/completions",
            payload={
                "model": "runtime-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer test-key", "Content-Type": "application/json"},
            timeout_seconds=12.5,
        )
    ]


def test_openai_compatible_provider_uses_configured_model_override() -> None:
    def sender(
        url: str,
        payload: dict[str, JsonValue],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, JsonValue]:
        assert payload["model"] == "configured-model"
        return {"choices": [{"message": {"content": "ok"}}]}

    provider = OpenAICompatibleProvider(
        config=OpenAICompatibleProviderConfig(
            base_url="https://provider.example",
            api_key="test-key",
            default_model="configured-model",
            timeout_seconds=5.0,
        ),
        sender=sender,
    )

    response = provider.generate(_turn(model_id="runtime-model"))

    assert response.metadata["model_id"] == "configured-model"


def test_openai_compatible_provider_extracts_text_content_parts() -> None:
    def sender(
        url: str,
        payload: dict[str, JsonValue],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, JsonValue]:
        return {"choices": [{"message": {"content": [{"type": "text", "text": "hello"}, {"text": " world"}]}}]}

    provider = OpenAICompatibleProvider(
        config=OpenAICompatibleProviderConfig(
            base_url="https://provider.example",
            api_key="test-key",
            default_model=None,
            timeout_seconds=5.0,
        ),
        sender=sender,
    )

    response = provider.generate(_turn(model_id="runtime-model"))

    assert response.output_text == "hello world"


def test_openai_compatible_provider_rejects_invalid_config() -> None:
    with pytest.raises(OpenAICompatibleProviderError, match="api_key"):
        OpenAICompatibleProvider(
            config=OpenAICompatibleProviderConfig(
                base_url="https://provider.example",
                api_key="",
                default_model=None,
                timeout_seconds=5.0,
            ),
            sender=_unused_sender,
        )


def test_openai_compatible_provider_rejects_invalid_response_shape() -> None:
    def sender(
        url: str,
        payload: dict[str, JsonValue],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, JsonValue]:
        return {"choices": []}

    provider = OpenAICompatibleProvider(
        config=OpenAICompatibleProviderConfig(
            base_url="https://provider.example",
            api_key="test-key",
            default_model=None,
            timeout_seconds=5.0,
        ),
        sender=sender,
    )

    with pytest.raises(OpenAICompatibleProviderError, match="choices"):
        provider.generate(_turn(model_id="runtime-model"))


def _turn(model_id: str) -> NormalizedTurn:
    return NormalizedTurn(
        trace_id="trace-provider-test",
        session_id="session-provider-test",
        turn_index=1,
        capability_mode=CapabilityMode.BLACK_BOX,
        model=ModelInfo(provider="openai_compatible", model_id=model_id, revision=None, selected_device=None),
        messages=(Message(role="user", content="hello"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={},
    )


def _unused_sender(
    url: str,
    payload: dict[str, JsonValue],
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, JsonValue]:
    raise AssertionError("sender should not be called.")
