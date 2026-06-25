from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from fastapi.testclient import TestClient

from aegis.core.contracts import JsonValue
from aegis.providers.loopback_openai import LoopbackOpenAIProviderConfig, make_server
from aegis.proxy.http_app import create_http_app
from aegis.proxy.mock_app import create_default_proxy
from aegis.proxy.smoke import (
    GatewaySmokeConfig,
    GatewaySmokeProviderMode,
    HttpJsonResponse,
    NimbusSmokeProfile,
    run_gateway_smoke,
)


@dataclass(frozen=True)
class GatewaySmokeTestClientAdapter:
    client: TestClient

    def get_json(self, url: str, timeout_seconds: float) -> HttpJsonResponse:
        response = self.client.get(_path_from_url(url))
        return HttpJsonResponse(status_code=response.status_code, payload=response.json())

    def post_json(self, url: str, payload: dict[str, JsonValue], timeout_seconds: float) -> HttpJsonResponse:
        response = self.client.post(_path_from_url(url), json=payload)
        return HttpJsonResponse(status_code=response.status_code, payload=response.json())


def test_loopback_real_provider_smoke_exercises_gateway_adapter_and_redacted_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    port = _unused_loopback_port()
    provider_log_path = tmp_path / "loopback-provider.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    raw_secret = "ghp_realLookingToolSecret1234567890"
    server = make_server(
        LoopbackOpenAIProviderConfig(
            host="127.0.0.1",
            port=port,
            response_content="Loopback provider completed.",
            request_log_path=provider_log_path,
            expected_bearer_token="loopback-token",
            forbidden_substrings=(raw_secret, "fake-"),
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("AEGIS_PROVIDER", "openai_compatible")
        monkeypatch.setenv("AEGIS_OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")
        monkeypatch.setenv("AEGIS_OPENAI_API_KEY", "loopback-token")
        monkeypatch.setenv("AEGIS_OPENAI_MODEL", "loopback-model")
        monkeypatch.setenv("AEGIS_AUDIT_JSONL_PATH", str(audit_path))
        monkeypatch.setenv("AEGIS_CIFT_PROFILE", "black_box")
        monkeypatch.delenv("AEGIS_CIFT_CERTIFICATION_MODE", raising=False)

        client = TestClient(create_http_app(create_default_proxy()))
        report = run_gateway_smoke(
            GatewaySmokeConfig(
                base_url="http://gateway.test",
                timeout_seconds=5.0,
                nimbus_profile=NimbusSmokeProfile.DEFAULT,
                require_cift_pre_generation_block=False,
                provider_mode=GatewaySmokeProviderMode.REAL_PROVIDER,
                output_path=None,
            ),
            GatewaySmokeTestClientAdapter(client),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    readiness = report["checks"]["gateway_readiness"]
    assert isinstance(readiness, dict)
    assert readiness["provider_name"] == "openai_compatible"
    assert readiness["provider_mock_controls_enabled"] is False
    benign = report["checks"]["benign_chat"]
    assert isinstance(benign, dict)
    assert benign["final_action"] == "allow"
    assert benign["provider_status"] == "completed"
    egress = report["checks"]["provider_egress_guard_block"]
    assert isinstance(egress, dict)
    assert egress["final_action"] == "block"
    assert egress["provider_status"] == "skipped"
    assert egress["provider_reason"] == "pre_generation_policy_block"
    assert report["checks"]["encoded_canary_leak"]["status"] == "skipped"
    assert report["checks"]["metadata_slot_canary_leak"]["status"] == "skipped"
    assert report["checks"]["nimbus_partial_leak"]["status"] == "skipped"

    provider_records = _read_jsonl(provider_log_path)
    assert len(provider_records) == 1
    assert provider_records[0]["authorization_status"] == "matched_expected"
    assert provider_records[0]["forbidden_substring_present"] is False

    audit_text = audit_path.read_text(encoding="utf-8")
    assert raw_secret not in audit_text
    assert "{{CREDENTIAL:" not in audit_text
    assert "[REDACTED_SENSITIVE]" in audit_text
    audit_records = _read_jsonl(audit_path)
    assert {record["trace_id"] for record in audit_records} == {"smoke-benign-trace", "smoke-egress-guard-trace"}


def _path_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return path


def _read_jsonl(path: Path) -> tuple[dict[str, JsonValue], ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip() != "")


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    if not isinstance(port, int):
        raise AssertionError("expected socket port to be an integer.")
    return port
