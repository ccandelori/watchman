from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path

from aegis.core.contracts import Action, JsonValue


class ProxyConfigError(RuntimeError):
    """Raised when proxy runtime configuration is invalid."""


class NimbusSmokeProfile(StrEnum):
    DEFAULT = "default"
    STRICT_PARTIAL_BLOCK = "strict-partial-block"


class NimbusCriticKind(StrEnum):
    CANARY = "canary"
    LEARNED_INFONCE_BETA = "learned_infonce_beta"


@dataclass(frozen=True)
class ProxyNimbusConfig:
    critic_kind: NimbusCriticKind
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
    infonce_model_path: Path | None


@dataclass(frozen=True)
class NimbusPartialLeakSmokeExpectation:
    profile: NimbusSmokeProfile
    nimbus_exact_action: Action | None
    nimbus_min_action: Action | None
    final_min_action: Action | None
    final_below_action: Action | None


def nimbus_config_from_env(env: Mapping[str, str] | None = None) -> ProxyNimbusConfig:
    values: Mapping[str, str] = os.environ if env is None else env
    critic_kind = _critic_kind_env(values)
    infonce_model_path = _optional_path_env(values, "AEGIS_NIMBUS_INFONCE_MODEL_PATH")
    if critic_kind == NimbusCriticKind.LEARNED_INFONCE_BETA and infonce_model_path is None:
        raise ProxyConfigError(
            "AEGIS_NIMBUS_INFONCE_MODEL_PATH is required when AEGIS_NIMBUS_CRITIC_KIND=learned_infonce_beta."
        )
    if critic_kind == NimbusCriticKind.CANARY and infonce_model_path is not None:
        raise ProxyConfigError(
            "AEGIS_NIMBUS_INFONCE_MODEL_PATH requires AEGIS_NIMBUS_CRITIC_KIND=learned_infonce_beta."
        )
    config = ProxyNimbusConfig(
        critic_kind=critic_kind,
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
        infonce_model_path=infonce_model_path,
    )
    if not config.warn_threshold <= config.sanitize_threshold <= config.block_threshold:
        raise ProxyConfigError(
            "AEGIS_NIMBUS thresholds must satisfy WARN_THRESHOLD <= SANITIZE_THRESHOLD <= BLOCK_THRESHOLD."
        )
    return config


def nimbus_capabilities(config: ProxyNimbusConfig) -> dict[str, JsonValue]:
    status = "deterministic_beta"
    promotion_status = "deterministic_canary_beta"
    if config.critic_kind == NimbusCriticKind.LEARNED_INFONCE_BETA:
        status = "learned_runtime_beta"
        promotion_status = "learned_runtime_beta_not_promotable"
    capabilities: dict[str, JsonValue] = {
        "status": status,
        "critic_kind": config.critic_kind.value,
        "critic_version": config.critic_version,
        "paper_faithful_learned_critic": False,
        "promotion_status": promotion_status,
        "tool_argument_pre_dispatch_accounting": True,
        "budget_bits": config.budget_bits,
        "max_turns": config.max_turns,
        "thresholds": {
            "warn": config.warn_threshold,
            "sanitize": config.sanitize_threshold,
            "block": config.block_threshold,
        },
        "critic": {
            "exact_match_leakage_bits": config.exact_match_leakage_bits,
            "encoded_match_leakage_bits": config.encoded_match_leakage_bits,
            "partial_match_leakage_bits": config.partial_match_leakage_bits,
            "partial_match_threshold": config.partial_match_threshold,
            "confidence": config.confidence,
        },
    }
    if config.infonce_model_path is not None:
        capabilities["infonce_model_path"] = str(config.infonce_model_path)
    return capabilities


def partial_leak_smoke_expectation(profile: NimbusSmokeProfile) -> NimbusPartialLeakSmokeExpectation:
    if profile == NimbusSmokeProfile.DEFAULT:
        return NimbusPartialLeakSmokeExpectation(
            profile=profile,
            nimbus_exact_action=None,
            nimbus_min_action=Action.WARN,
            final_min_action=None,
            final_below_action=Action.BLOCK,
        )
    if profile == NimbusSmokeProfile.STRICT_PARTIAL_BLOCK:
        return NimbusPartialLeakSmokeExpectation(
            profile=profile,
            nimbus_exact_action=None,
            nimbus_min_action=Action.BLOCK,
            final_min_action=Action.BLOCK,
            final_below_action=None,
        )
    raise ProxyConfigError(f"Unsupported NIMBUS smoke profile '{profile}'.")


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


def _critic_kind_env(values: Mapping[str, str]) -> NimbusCriticKind:
    raw_value = values.get("AEGIS_NIMBUS_CRITIC_KIND")
    if raw_value is None:
        return NimbusCriticKind.CANARY
    try:
        return NimbusCriticKind(raw_value)
    except ValueError as exc:
        supported = ", ".join(item.value for item in NimbusCriticKind)
        raise ProxyConfigError(f"AEGIS_NIMBUS_CRITIC_KIND must be one of: {supported}.") from exc


def _optional_path_env(values: Mapping[str, str], key: str) -> Path | None:
    raw_value = values.get(key)
    if raw_value is None:
        return None
    if raw_value == "":
        raise ProxyConfigError(f"{key} must be non-empty.")
    return Path(raw_value)
