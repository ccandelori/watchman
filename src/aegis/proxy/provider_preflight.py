from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from aegis.core.contracts import JsonValue
from aegis.proxy.config import ProviderKind, ProxyConfigError, provider_config_from_env

_SCHEMA_VERSION = "aegis.provider_preflight/v1"
_ENV_KEYS = (
    "AEGIS_PROVIDER",
    "AEGIS_OPENAI_BASE_URL",
    "AEGIS_OPENAI_API_KEY",
    "AEGIS_OPENAI_MODEL",
    "AEGIS_OPENAI_TIMEOUT_SECONDS",
)


@dataclass(frozen=True)
class ProviderPreflightConfig:
    require_real_provider: bool
    output_path: Path | None


def parse_args(argv: Sequence[str]) -> ProviderPreflightConfig:
    parser = argparse.ArgumentParser(description="Validate Aegis provider configuration without network access.")
    parser.add_argument(
        "--require-real-provider",
        action="store_true",
        help="Fail unless AEGIS_PROVIDER resolves to openai_compatible.",
    )
    parser.add_argument("--output", required=False, help="Optional path to write the provider preflight JSON report.")
    args = parser.parse_args(argv)
    return ProviderPreflightConfig(
        require_real_provider=bool(args.require_real_provider),
        output_path=None if args.output is None else Path(str(args.output)),
    )


def provider_preflight_report(config: ProviderPreflightConfig, env: Mapping[str, str]) -> dict[str, JsonValue]:
    checks: list[JsonValue] = []
    provider_kind = "invalid"
    provider_name = None
    mock_controls_enabled = None
    provider_ready = False
    provider_error = None
    try:
        provider_config = provider_config_from_env(env=env)
        provider_kind = provider_config.kind.value
        provider_name = provider_config.provider_name
        mock_controls_enabled = provider_config.mock_controls_enabled
        provider_ready = True
        checks.append(_check(name="provider_config", passed=True, detail="Provider config parsed without network I/O."))
    except ProxyConfigError as exc:
        provider_error = str(exc)
        checks.append(_check(name="provider_config", passed=False, detail=provider_error))

    real_provider_ready = provider_ready and provider_kind == ProviderKind.OPENAI_COMPATIBLE.value
    if config.require_real_provider:
        checks.append(
            _check(
                name="real_provider_required",
                passed=real_provider_ready,
                detail="AEGIS_PROVIDER must resolve to openai_compatible for real-provider smoke.",
            )
        )
    checks.append(_check(name="network_access", passed=True, detail="No provider network request was attempted."))
    ready = all(isinstance(check, dict) and check.get("status") == "pass" for check in checks)
    report: dict[str, JsonValue] = {
        "schema_version": _SCHEMA_VERSION,
        "ready": ready,
        "status": "ready" if ready else "not_ready",
        "require_real_provider": config.require_real_provider,
        "provider_kind": provider_kind,
        "provider_name": provider_name,
        "mock_controls_enabled": mock_controls_enabled,
        "network_access": "not_attempted",
        "env": _redacted_env_status(env=env),
        "checks": checks,
    }
    if provider_error is not None:
        report["error"] = provider_error
    return report


def main() -> None:
    config = parse_args(tuple(sys.argv[1:]))
    report = provider_preflight_report(config=config, env=os.environ)
    report_json = json.dumps(report, sort_keys=True)
    if config.output_path is not None:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(f"{report_json}\n", encoding="utf-8")
    sys.stdout.write(f"{report_json}\n")
    if report["ready"] is not True:
        raise SystemExit(1)


def _check(name: str, passed: bool, detail: str) -> dict[str, JsonValue]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "detail": detail,
    }


def _redacted_env_status(env: Mapping[str, str]) -> dict[str, JsonValue]:
    statuses: dict[str, JsonValue] = {}
    for key in _ENV_KEYS:
        value = env.get(key)
        statuses[key] = "set" if value not in (None, "") else "unset"
    return statuses
