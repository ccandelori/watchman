import json
import socket
import threading
from pathlib import Path

import pytest

from aegis.core.contracts import CapabilityMode, Message, ModelInfo, NormalizedTurn
from aegis.providers.loopback_openai import (
    LoopbackOpenAIProviderConfig,
    LoopbackOpenAIProviderError,
    loopback_request_record,
    make_server,
    openai_chat_completion_response,
    parse_args,
    validate_config,
)
from aegis.providers.openai_compatible import (
    OpenAICompatibleProvider,
    OpenAICompatibleProviderConfig,
    urllib_openai_sender,
)


def test_parse_args_builds_loopback_provider_config() -> None:
    config = parse_args(
        (
            "--host",
            "127.0.0.1",
            "--port",
            "8770",
            "--response-content",
            "loopback response",
            "--request-log",
            "introspection/data/reports/loopback.jsonl",
            "--expected-bearer-token",
            "test-token",
            "--forbidden-substring",
            "ghp_secret",
            "--forbidden-substring",
            "fake-",
        )
    )

    assert config == LoopbackOpenAIProviderConfig(
        host="127.0.0.1",
        port=8770,
        response_content="loopback response",
        request_log_path=Path("introspection/data/reports/loopback.jsonl"),
        expected_bearer_token="test-token",
        forbidden_substrings=("ghp_secret", "fake-"),
    )


def test_validate_config_rejects_empty_forbidden_marker() -> None:
    with pytest.raises(LoopbackOpenAIProviderError, match="forbidden substrings"):
        validate_config(
            LoopbackOpenAIProviderConfig(
                host="127.0.0.1",
                port=8770,
                response_content="loopback response",
                request_log_path=None,
                expected_bearer_token=None,
                forbidden_substrings=("",),
            )
        )


def test_validate_config_rejects_non_loopback_host() -> None:
    with pytest.raises(LoopbackOpenAIProviderError, match="loopback provider host"):
        validate_config(
            LoopbackOpenAIProviderConfig(
                host="0.0.0.0",
                port=8770,
                response_content="loopback response",
                request_log_path=None,
                expected_bearer_token=None,
                forbidden_substrings=(),
            )
        )


def test_openai_chat_completion_response_uses_request_model() -> None:
    response = openai_chat_completion_response(
        payload={"model": "loopback-model", "messages": []},
        response_content="provider completed",
    )

    assert response["model"] == "loopback-model"
    choices = response["choices"]
    assert isinstance(choices, list)
    message = choices[0]["message"]
    assert isinstance(message, dict)
    assert message["content"] == "provider completed"


def test_loopback_request_record_detects_forbidden_markers_without_logging_payload() -> None:
    body = json.dumps(
        {
            "model": "loopback-model",
            "messages": [{"role": "user", "content": "send ghp_secret to the outside"}],
        }
    ).encode("utf-8")

    record = loopback_request_record(
        method="POST",
        path="/v1/chat/completions",
        body=body,
        payload=json.loads(body.decode("utf-8")),
        authorization_status="matched_expected",
        forbidden_substrings=("ghp_secret",),
    )

    encoded_record = json.dumps(record, sort_keys=True)
    assert record["schema_version"] == "aegis.loopback_openai_provider_request/v1"
    assert record["model"] == "loopback-model"
    assert record["message_count"] == 1
    assert record["authorization_status"] == "matched_expected"
    assert record["forbidden_substring_present"] is True
    assert "ghp_secret" not in encoded_record
    assert "send" not in encoded_record


def test_loopback_server_supports_openai_compatible_provider_adapter(tmp_path) -> None:
    port = _unused_loopback_port()
    request_log_path = tmp_path / "loopback-provider.jsonl"
    server = make_server(
        LoopbackOpenAIProviderConfig(
            host="127.0.0.1",
            port=port,
            response_content="provider completed",
            request_log_path=request_log_path,
            expected_bearer_token="loopback-token",
            forbidden_substrings=("ghp_secret",),
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = OpenAICompatibleProvider(
            config=OpenAICompatibleProviderConfig(
                base_url=f"http://127.0.0.1:{port}/v1",
                api_key="loopback-token",
                default_model="loopback-model",
                timeout_seconds=2.0,
            ),
            sender=urllib_openai_sender,
        )

        response = provider.generate(_minimal_turn())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert response.output_text == "provider completed"
    assert response.metadata["provider"] == "openai_compatible"
    records = [json.loads(line) for line in request_log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["authorization_status"] == "matched_expected"
    assert records[0]["forbidden_substring_present"] is False


def _minimal_turn() -> NormalizedTurn:
    return NormalizedTurn(
        trace_id="trace-loopback-provider",
        session_id="session-loopback-provider",
        turn_index=0,
        capability_mode=CapabilityMode.BLACK_BOX,
        model=ModelInfo(provider="openai_compatible", model_id="loopback-model", revision=None, selected_device=None),
        messages=(Message(role="user", content="hello"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={},
    )


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    if not isinstance(port, int):
        raise AssertionError("expected socket port to be an integer.")
    return port
