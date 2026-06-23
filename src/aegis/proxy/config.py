from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from aegis.core.orchestrator import ModelProvider
from aegis.providers.mock import MockModelProvider
from aegis.providers.openai_compatible import (
    OpenAICompatibleProvider,
    OpenAICompatibleProviderConfig,
    urllib_openai_sender,
)


class ProxyConfigError(RuntimeError):
    """Raised when proxy runtime configuration is invalid."""


class ProviderKind(StrEnum):
    MOCK = "mock"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass(frozen=True)
class ProxyProviderConfig:
    kind: ProviderKind
    provider_name: str
    model_provider: ModelProvider
    mock_controls_enabled: bool


def provider_config_from_env(env: Mapping[str, str] | None = None) -> ProxyProviderConfig:
    values: Mapping[str, str] = os.environ if env is None else env
    provider_value = values.get("AEGIS_PROVIDER", ProviderKind.MOCK.value)
    try:
        provider_kind = ProviderKind(provider_value)
    except ValueError as exc:
        supported = ", ".join(item.value for item in ProviderKind)
        raise ProxyConfigError(
            f"Unsupported AEGIS_PROVIDER '{provider_value}'. Supported values: {supported}."
        ) from exc

    if provider_kind == ProviderKind.MOCK:
        return ProxyProviderConfig(
            kind=provider_kind,
            provider_name="mock",
            model_provider=MockModelProvider(
                default_content=values.get("AEGIS_MOCK_DEFAULT_CONTENT", "Aegis mock response.")
            ),
            mock_controls_enabled=True,
        )

    if provider_kind == ProviderKind.OPENAI_COMPATIBLE:
        base_url = values.get("AEGIS_OPENAI_BASE_URL", "")
        api_key = values.get("AEGIS_OPENAI_API_KEY", "")
        if base_url == "":
            raise ProxyConfigError("AEGIS_OPENAI_BASE_URL must be set when AEGIS_PROVIDER=openai_compatible.")
        if api_key == "":
            raise ProxyConfigError("AEGIS_OPENAI_API_KEY must be set when AEGIS_PROVIDER=openai_compatible.")
        timeout_seconds = _float_env(values, "AEGIS_OPENAI_TIMEOUT_SECONDS", 30.0)
        return ProxyProviderConfig(
            kind=provider_kind,
            provider_name="openai_compatible",
            model_provider=OpenAICompatibleProvider(
                config=OpenAICompatibleProviderConfig(
                    base_url=base_url,
                    api_key=api_key,
                    default_model=_optional_non_empty(values.get("AEGIS_OPENAI_MODEL")),
                    timeout_seconds=timeout_seconds,
                ),
                sender=urllib_openai_sender,
            ),
            mock_controls_enabled=False,
        )

    raise ProxyConfigError(f"Unhandled provider kind '{provider_kind.value}'.")


def _float_env(values: Mapping[str, str], key: str, default: float) -> float:
    raw_value = values.get(key)
    if raw_value is None:
        return default
    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise ProxyConfigError(f"{key} must be a number.") from exc
    if parsed <= 0:
        raise ProxyConfigError(f"{key} must be positive.")
    return parsed


def _optional_non_empty(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value
