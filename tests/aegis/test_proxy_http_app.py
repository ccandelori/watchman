from fastapi.testclient import TestClient

from aegis.proxy.http_app import create_http_app
from aegis.proxy.mock_app import create_default_proxy
from aegis.proxy.server import ProxyServerConfig, parse_args, run_server


def test_http_health_route_returns_ok() -> None:
    client = TestClient(create_http_app(create_default_proxy()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_http_chat_completion_returns_aegis_block_and_audit_event() -> None:
    client = TestClient(create_http_app(create_default_proxy()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"trace_id": "trace-http-1", "session_id": "session-http-1"},
        },
    )
    audit_response = client.get("/audit/recent")

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "Aegis mock response."
    assert payload["aegis"]["trace_id"] == "trace-http-1"
    assert "policy_decision" in payload["aegis"]
    assert "detector_results" in payload["aegis"]
    assert audit_response.status_code == 200
    assert audit_response.json()["events"][0]["trace_id"] == "trace-http-1"


def test_http_chat_completion_rejects_non_object_body() -> None:
    client = TestClient(create_http_app(create_default_proxy()))

    response = client.post("/v1/chat/completions", json=["not", "an", "object"])

    assert response.status_code == 400
    assert "JSON object" in response.json()["error"]


def test_proxy_server_parses_explicit_bind_arguments() -> None:
    config = parse_args(("--host", "127.0.0.1", "--port", "8000"))

    assert config == ProxyServerConfig(host="127.0.0.1", port=8000)


def test_proxy_server_invokes_uvicorn_factory(monkeypatch) -> None:
    calls = []

    def fake_run(app_reference: str, *, factory: bool, host: str, port: int) -> None:
        calls.append(
            {
                "app_reference": app_reference,
                "factory": factory,
                "host": host,
                "port": port,
            }
        )

    monkeypatch.setattr("aegis.proxy.server.uvicorn.run", fake_run)

    run_server(ProxyServerConfig(host="127.0.0.1", port=8000))

    assert calls == [
        {
            "app_reference": "aegis.proxy.http_app:create_default_http_app",
            "factory": True,
            "host": "127.0.0.1",
            "port": 8000,
        }
    ]
