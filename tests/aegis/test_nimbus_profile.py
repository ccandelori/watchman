from __future__ import annotations

import pytest

from aegis.core.contracts import Action
from aegis.proxy.config import ProxyConfigError
from aegis.proxy.nimbus_profile import (
    NimbusSmokeProfile,
    nimbus_capabilities,
    nimbus_config_from_env,
    partial_leak_smoke_expectation,
)


def test_nimbus_profile_renders_capabilities_from_runtime_config() -> None:
    config = nimbus_config_from_env(
        env={
            "AEGIS_NIMBUS_PARTIAL_MATCH_LEAKAGE_BITS": "1.0",
            "AEGIS_NIMBUS_WARN_THRESHOLD": "0.2",
            "AEGIS_NIMBUS_SANITIZE_THRESHOLD": "0.3",
            "AEGIS_NIMBUS_BLOCK_THRESHOLD": "0.36",
            "AEGIS_NIMBUS_CRITIC_VERSION": "canary-strict-test",
        }
    )

    capabilities = nimbus_capabilities(config)

    assert capabilities["status"] == "deterministic_beta"
    assert capabilities["critic_kind"] == "canary"
    assert capabilities["critic_version"] == "canary-strict-test"
    assert capabilities["promotion_status"] == "deterministic_canary_beta"
    assert capabilities["thresholds"] == {"warn": 0.2, "sanitize": 0.3, "block": 0.36}
    assert capabilities["critic"]["partial_match_leakage_bits"] == 1.0


def test_nimbus_profile_renders_learned_infonce_beta_capabilities() -> None:
    config = nimbus_config_from_env(
        env={
            "AEGIS_NIMBUS_CRITIC_KIND": "learned_infonce_beta",
            "AEGIS_NIMBUS_INFONCE_MODEL_PATH": "introspection/data/reports/aegis_nimbus_infonce_model_v0.json",
            "AEGIS_NIMBUS_CRITIC_VERSION": "nimbus-infonce-lexical-v0",
        }
    )

    capabilities = nimbus_capabilities(config)

    assert capabilities["status"] == "learned_runtime_beta"
    assert capabilities["critic_kind"] == "learned_infonce_beta"
    assert capabilities["critic_version"] == "nimbus-infonce-lexical-v0"
    assert capabilities["paper_faithful_learned_critic"] is False
    assert capabilities["promotion_status"] == "learned_runtime_beta_not_promotable"
    assert capabilities["infonce_model_path"] == "introspection/data/reports/aegis_nimbus_infonce_model_v0.json"


def test_nimbus_profile_owns_partial_leak_smoke_expectations() -> None:
    default_expectation = partial_leak_smoke_expectation(NimbusSmokeProfile.DEFAULT)
    strict_expectation = partial_leak_smoke_expectation(NimbusSmokeProfile.STRICT_PARTIAL_BLOCK)

    assert default_expectation.nimbus_exact_action is None
    assert default_expectation.nimbus_min_action == Action.WARN
    assert default_expectation.final_min_action is None
    assert default_expectation.final_below_action == Action.BLOCK
    assert strict_expectation.nimbus_exact_action is None
    assert strict_expectation.nimbus_min_action == Action.BLOCK
    assert strict_expectation.final_min_action == Action.BLOCK
    assert strict_expectation.final_below_action is None


def test_nimbus_profile_rejects_invalid_threshold_order() -> None:
    with pytest.raises(ProxyConfigError, match="WARN_THRESHOLD"):
        nimbus_config_from_env(
            env={
                "AEGIS_NIMBUS_WARN_THRESHOLD": "0.5",
                "AEGIS_NIMBUS_SANITIZE_THRESHOLD": "0.4",
            }
        )
