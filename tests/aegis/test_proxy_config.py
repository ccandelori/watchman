import pytest

from aegis.core.contracts import CapabilityMode, Message, ModelInfo, NormalizedTurn
from aegis.providers.mock import MockModelProvider
from aegis.providers.openai_compatible import OpenAICompatibleProvider
from aegis.proxy.config import ProviderKind, ProxyConfigError, nimbus_config_from_env, provider_config_from_env


def test_provider_config_defaults_to_mock_provider() -> None:
    config = provider_config_from_env(env={})

    assert config.kind == ProviderKind.MOCK
    assert config.provider_name == "mock"
    assert config.mock_controls_enabled is True
    assert isinstance(config.model_provider, MockModelProvider)


def test_provider_config_supports_mock_default_content() -> None:
    config = provider_config_from_env(env={"AEGIS_MOCK_DEFAULT_CONTENT": "configured mock"})

    assert config.model_provider.generate(_minimal_turn()).output_text == "configured mock"


def test_provider_config_rejects_unknown_provider() -> None:
    with pytest.raises(ProxyConfigError, match="Unsupported AEGIS_PROVIDER"):
        provider_config_from_env(env={"AEGIS_PROVIDER": "surprise"})


def test_provider_config_requires_openai_base_url() -> None:
    with pytest.raises(ProxyConfigError, match="AEGIS_OPENAI_BASE_URL"):
        provider_config_from_env(
            env={
                "AEGIS_PROVIDER": "openai_compatible",
                "AEGIS_OPENAI_API_KEY": "test-key",
            }
        )


def test_provider_config_requires_openai_api_key() -> None:
    with pytest.raises(ProxyConfigError, match="AEGIS_OPENAI_API_KEY"):
        provider_config_from_env(
            env={
                "AEGIS_PROVIDER": "openai_compatible",
                "AEGIS_OPENAI_BASE_URL": "https://provider.example",
            }
        )


def test_provider_config_builds_openai_compatible_provider() -> None:
    config = provider_config_from_env(
        env={
            "AEGIS_PROVIDER": "openai_compatible",
            "AEGIS_OPENAI_BASE_URL": "https://provider.example",
            "AEGIS_OPENAI_API_KEY": "test-key",
            "AEGIS_OPENAI_MODEL": "configured-model",
            "AEGIS_OPENAI_TIMEOUT_SECONDS": "7.5",
        }
    )

    assert config.kind == ProviderKind.OPENAI_COMPATIBLE
    assert config.provider_name == "openai_compatible"
    assert config.mock_controls_enabled is False
    assert isinstance(config.model_provider, OpenAICompatibleProvider)


def test_provider_config_rejects_invalid_openai_timeout() -> None:
    with pytest.raises(ProxyConfigError, match="AEGIS_OPENAI_TIMEOUT_SECONDS"):
        provider_config_from_env(
            env={
                "AEGIS_PROVIDER": "openai_compatible",
                "AEGIS_OPENAI_BASE_URL": "https://provider.example",
                "AEGIS_OPENAI_API_KEY": "test-key",
                "AEGIS_OPENAI_TIMEOUT_SECONDS": "0",
            }
        )


def test_nimbus_config_defaults_to_current_runtime_thresholds() -> None:
    config = nimbus_config_from_env(env={})

    assert config.exact_match_leakage_bits == 1.0
    assert config.encoded_match_leakage_bits == 1.0
    assert config.partial_match_leakage_bits == 0.8
    assert config.partial_match_threshold == 0.4
    assert config.confidence == 0.8
    assert config.budget_bits == 1.0
    assert config.warn_threshold == 0.3
    assert config.sanitize_threshold == 0.6
    assert config.block_threshold == 0.9
    assert config.max_turns == 20
    assert config.critic_version == "canary-v0"


def test_nimbus_config_accepts_strict_partial_block_profile() -> None:
    config = nimbus_config_from_env(
        env={
            "AEGIS_NIMBUS_PARTIAL_MATCH_LEAKAGE_BITS": "1.0",
            "AEGIS_NIMBUS_WARN_THRESHOLD": "0.2",
            "AEGIS_NIMBUS_SANITIZE_THRESHOLD": "0.3",
            "AEGIS_NIMBUS_BLOCK_THRESHOLD": "0.36",
            "AEGIS_NIMBUS_MAX_TURNS": "7",
            "AEGIS_NIMBUS_CRITIC_VERSION": "canary-strict-test",
        }
    )

    assert config.partial_match_leakage_bits == 1.0
    assert config.warn_threshold == 0.2
    assert config.sanitize_threshold == 0.3
    assert config.block_threshold == 0.36
    assert config.max_turns == 7
    assert config.critic_version == "canary-strict-test"


def test_nimbus_config_rejects_invalid_threshold_order() -> None:
    with pytest.raises(ProxyConfigError, match="WARN_THRESHOLD"):
        nimbus_config_from_env(
            env={
                "AEGIS_NIMBUS_WARN_THRESHOLD": "0.5",
                "AEGIS_NIMBUS_SANITIZE_THRESHOLD": "0.4",
            }
        )


def test_nimbus_config_rejects_invalid_values() -> None:
    with pytest.raises(ProxyConfigError, match="AEGIS_NIMBUS_PARTIAL_MATCH_THRESHOLD"):
        nimbus_config_from_env(env={"AEGIS_NIMBUS_PARTIAL_MATCH_THRESHOLD": "1.1"})
    with pytest.raises(ProxyConfigError, match="AEGIS_NIMBUS_MAX_TURNS"):
        nimbus_config_from_env(env={"AEGIS_NIMBUS_MAX_TURNS": "0"})


def _minimal_turn() -> NormalizedTurn:
    return NormalizedTurn(
        trace_id="trace-config-test",
        session_id="session-config-test",
        turn_index=1,
        capability_mode=CapabilityMode.BLACK_BOX,
        model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
        messages=(Message(role="user", content="hello"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={},
    )
