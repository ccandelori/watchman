from __future__ import annotations

import json
from pathlib import Path

from aegis.proxy.provider_preflight import ProviderPreflightConfig, parse_args, provider_preflight_report


def test_provider_preflight_parses_require_real_provider_and_output() -> None:
    config = parse_args(("--require-real-provider", "--output", "introspection/data/reports/provider.json"))

    assert config == ProviderPreflightConfig(
        require_real_provider=True,
        output_path=Path("introspection/data/reports/provider.json"),
    )


def test_provider_preflight_reports_real_provider_ready_without_network_or_secret_leakage() -> None:
    report = provider_preflight_report(
        config=ProviderPreflightConfig(require_real_provider=True, output_path=None),
        env={
            "AEGIS_PROVIDER": "openai_compatible",
            "AEGIS_OPENAI_BASE_URL": "https://provider.example",
            "AEGIS_OPENAI_API_KEY": "secret-test-key",
            "AEGIS_OPENAI_MODEL": "configured-model",
            "AEGIS_OPENAI_TIMEOUT_SECONDS": "7.5",
        },
    )
    encoded = json.dumps(report, sort_keys=True)

    assert report["schema_version"] == "aegis.provider_preflight/v1"
    assert report["ready"] is True
    assert report["status"] == "ready"
    assert report["provider_kind"] == "openai_compatible"
    assert report["provider_name"] == "openai_compatible"
    assert report["mock_controls_enabled"] is False
    assert report["network_access"] == "not_attempted"
    assert report["env"]["AEGIS_OPENAI_API_KEY"] == "set"
    assert "secret-test-key" not in encoded


def test_provider_preflight_requires_real_provider_when_requested() -> None:
    report = provider_preflight_report(
        config=ProviderPreflightConfig(require_real_provider=True, output_path=None),
        env={},
    )

    assert report["ready"] is False
    assert report["status"] == "not_ready"
    assert report["provider_kind"] == "mock"
    assert report["mock_controls_enabled"] is True
    assert _check_status(report, "provider_config") == "pass"
    assert _check_status(report, "real_provider_required") == "fail"
    assert _check_status(report, "network_access") == "pass"


def test_provider_preflight_reports_invalid_real_provider_config_without_secret_leakage() -> None:
    report = provider_preflight_report(
        config=ProviderPreflightConfig(require_real_provider=True, output_path=None),
        env={
            "AEGIS_PROVIDER": "openai_compatible",
            "AEGIS_OPENAI_BASE_URL": "https://provider.example?token=secret",
            "AEGIS_OPENAI_API_KEY": "secret-test-key",
        },
    )
    encoded = json.dumps(report, sort_keys=True)

    assert report["ready"] is False
    assert report["status"] == "not_ready"
    assert report["provider_kind"] == "invalid"
    assert _check_status(report, "provider_config") == "fail"
    assert "secret-test-key" not in encoded
    assert "token=secret" not in encoded


def _check_status(report: dict[str, object], name: str) -> object:
    checks = report["checks"]
    if not isinstance(checks, list):
        raise AssertionError("checks must be a list")
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            return check.get("status")
    raise AssertionError(f"missing check {name}")
