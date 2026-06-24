from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

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


@dataclass(frozen=True)
class ProxyNimbusConfig:
    exact_match_leakage_bits: float
    encoded_match_leakage_bits: float
    partial_match_leakage_bits: float
    partial_match_threshold: float
    confidence: float
    budget_bits: float
    warn_threshold: float
    sanitize_threshold: float
    block_threshold: float
    max_turns: int
    critic_version: str


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


def nimbus_config_from_env(env: Mapping[str, str] | None = None) -> ProxyNimbusConfig:
    values: Mapping[str, str] = os.environ if env is None else env
    config = ProxyNimbusConfig(
        exact_match_leakage_bits=_non_negative_float_env(values, "AEGIS_NIMBUS_EXACT_MATCH_LEAKAGE_BITS", 1.0),
        encoded_match_leakage_bits=_non_negative_float_env(values, "AEGIS_NIMBUS_ENCODED_MATCH_LEAKAGE_BITS", 1.0),
        partial_match_leakage_bits=_non_negative_float_env(values, "AEGIS_NIMBUS_PARTIAL_MATCH_LEAKAGE_BITS", 0.8),
        partial_match_threshold=_probability_env(values, "AEGIS_NIMBUS_PARTIAL_MATCH_THRESHOLD", 0.4),
        confidence=_probability_env(values, "AEGIS_NIMBUS_CONFIDENCE", 0.8),
        budget_bits=_positive_float_env(values, "AEGIS_NIMBUS_BUDGET_BITS", 1.0),
        warn_threshold=_probability_env(values, "AEGIS_NIMBUS_WARN_THRESHOLD", 0.3),
        sanitize_threshold=_probability_env(values, "AEGIS_NIMBUS_SANITIZE_THRESHOLD", 0.6),
        block_threshold=_probability_env(values, "AEGIS_NIMBUS_BLOCK_THRESHOLD", 0.9),
        max_turns=_positive_int_env(values, "AEGIS_NIMBUS_MAX_TURNS", 20),
        critic_version=_string_env(values, "AEGIS_NIMBUS_CRITIC_VERSION", "canary-v0"),
    )
    if not config.warn_threshold <= config.sanitize_threshold <= config.block_threshold:
        raise ProxyConfigError(
            "AEGIS_NIMBUS thresholds must satisfy WARN_THRESHOLD <= SANITIZE_THRESHOLD <= BLOCK_THRESHOLD."
        )
    return config


def _float_env(values: Mapping[str, str], key: str, default: float) -> float:
    return _positive_float_env(values, key, default)


def _positive_float_env(values: Mapping[str, str], key: str, default: float) -> float:
    raw_value = values.get(key)
    if raw_value is None:
        return default
    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise ProxyConfigError(f"{key} must be a number.") from exc
    if parsed <= 0:
        raise ProxyConfigError(f"{key} must be positive.")
    if not isfinite(parsed):
        raise ProxyConfigError(f"{key} must be finite.")
    return parsed


def _non_negative_float_env(values: Mapping[str, str], key: str, default: float) -> float:
    raw_value = values.get(key)
    if raw_value is None:
        return default
    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise ProxyConfigError(f"{key} must be a number.") from exc
    if parsed < 0:
        raise ProxyConfigError(f"{key} must be non-negative.")
    if not isfinite(parsed):
        raise ProxyConfigError(f"{key} must be finite.")
    return parsed


def _probability_env(values: Mapping[str, str], key: str, default: float) -> float:
    parsed = _non_negative_float_env(values, key, default)
    if parsed > 1:
        raise ProxyConfigError(f"{key} must be between 0 and 1.")
    return parsed


def _positive_int_env(values: Mapping[str, str], key: str, default: int) -> int:
    raw_value = values.get(key)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ProxyConfigError(f"{key} must be an integer.") from exc
    if parsed <= 0:
        raise ProxyConfigError(f"{key} must be positive.")
    return parsed


def _string_env(values: Mapping[str, str], key: str, default: str) -> str:
    raw_value = values.get(key)
    if raw_value is None:
        return default
    if raw_value == "":
        raise ProxyConfigError(f"{key} must be non-empty.")
    return raw_value


def _optional_non_empty(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value
