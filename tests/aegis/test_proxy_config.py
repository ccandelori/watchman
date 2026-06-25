import pytest

from aegis.audit.jsonl import JsonlAuditSink
from aegis.audit.memory import InMemoryAuditSink
from aegis.core.contracts import CapabilityMode, Message, ModelInfo, NormalizedTurn
from aegis.providers.mock import MockModelProvider
from aegis.providers.openai_compatible import OpenAICompatibleProvider
from aegis.proxy.config import (
    CiftCertificationMode,
    CiftProfile,
    ProviderKind,
    ProxyConfigError,
    audit_sink_from_env,
    cift_config_from_env,
    nimbus_config_from_env,
    provider_config_from_env,
)


def test_provider_config_defaults_to_mock_provider() -> None:
    config = provider_config_from_env(env={})

    assert config.kind == ProviderKind.MOCK
    assert config.provider_name == "mock"
    assert config.mock_controls_enabled is True
    assert isinstance(config.model_provider, MockModelProvider)


def test_provider_config_supports_mock_default_content() -> None:
    config = provider_config_from_env(env={"AEGIS_MOCK_DEFAULT_CONTENT": "configured mock"})

    assert config.model_provider.generate(_minimal_turn()).output_text == "configured mock"


def test_audit_sink_config_defaults_to_memory_and_supports_jsonl(tmp_path) -> None:
    assert isinstance(audit_sink_from_env(env={}), InMemoryAuditSink)

    audit_path = tmp_path / "audit.jsonl"
    sink = audit_sink_from_env(env={"AEGIS_AUDIT_JSONL_PATH": str(audit_path)})

    assert isinstance(sink, JsonlAuditSink)
    assert sink.durable_path() == str(audit_path)


def test_audit_sink_config_rejects_empty_jsonl_path() -> None:
    with pytest.raises(ProxyConfigError, match="AEGIS_AUDIT_JSONL_PATH"):
        audit_sink_from_env(env={"AEGIS_AUDIT_JSONL_PATH": ""})


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


def test_provider_config_accepts_loopback_http_openai_base_url() -> None:
    config = provider_config_from_env(
        env={
            "AEGIS_PROVIDER": "openai_compatible",
            "AEGIS_OPENAI_BASE_URL": "http://127.0.0.1:8080/v1",
            "AEGIS_OPENAI_API_KEY": "test-key",
        }
    )

    assert config.kind == ProviderKind.OPENAI_COMPATIBLE
    assert config.mock_controls_enabled is False


@pytest.mark.parametrize(
    "base_url",
    (
        "http://provider.example",
        "ftp://provider.example",
        "https://user:pass@provider.example",
        "https://provider.example?api_key=leaky",
        "https:///missing-host",
        " https://provider.example",
    ),
)
def test_provider_config_rejects_unsafe_openai_base_urls(base_url: str) -> None:
    with pytest.raises(ProxyConfigError, match="AEGIS_OPENAI_BASE_URL"):
        provider_config_from_env(
            env={
                "AEGIS_PROVIDER": "openai_compatible",
                "AEGIS_OPENAI_BASE_URL": base_url,
                "AEGIS_OPENAI_API_KEY": "test-key",
            }
        )


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


def test_cift_config_defaults_to_black_box_profile() -> None:
    config = cift_config_from_env(env={})

    assert config.profile == CiftProfile.BLACK_BOX
    assert config.selected_choice_model_path is None
    assert config.fallback_model_path is None
    assert config.certification_manifest_sha256 is None
    assert config.certification_report_sha256 is None
    assert config.release_gate_report_path is None
    assert config.release_gate_report_sha256 is None
    assert config.certification_artifact_root is None
    assert config.selected_choice_readout_token_count is None
    assert config.extractor_id is None
    assert config.extractor_base_url is None
    assert config.extractor_api_key is None
    assert config.extractor_timeout_seconds is None


def test_cift_config_rejects_unknown_profile() -> None:
    with pytest.raises(ProxyConfigError, match="Unsupported AEGIS_CIFT_PROFILE"):
        cift_config_from_env(env={"AEGIS_CIFT_PROFILE": "surprise"})


def test_cift_config_requires_self_hosted_window_selector_fields() -> None:
    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH"):
        cift_config_from_env(env={"AEGIS_CIFT_PROFILE": "self_hosted_window_selector"})

    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_EXTRACTOR_ID"):
        cift_config_from_env(
            env={
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
            }
        )

    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_FALLBACK_MODEL_PATH"):
        cift_config_from_env(
            env={
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_FALLBACK_MODEL_PATH": "",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
            }
        )

    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH"):
        cift_config_from_env(
            env={
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
            }
        )

    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_CERTIFICATION_REPORT_PATH"):
        cift_config_from_env(
            env={
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH": "cift-certification.json",
            }
        )

    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT"):
        cift_config_from_env(
            env={
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH": "cift-certification.json",
                "AEGIS_CIFT_CERTIFICATION_REPORT_PATH": "cift-certification-run.json",
            }
        )

    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256"):
        cift_config_from_env(
            env={
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH": "cift-certification.json",
                "AEGIS_CIFT_CERTIFICATION_REPORT_PATH": "cift-certification-run.json",
                "AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT": "cift-artifacts",
            }
        )

    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_CERTIFICATION_REPORT_SHA256"):
        cift_config_from_env(
            env={
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH": "cift-certification.json",
                "AEGIS_CIFT_CERTIFICATION_REPORT_PATH": "cift-certification-run.json",
                "AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT": "cift-artifacts",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256": "a" * 64,
            }
        )

    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_RELEASE_GATE_REPORT_PATH"):
        cift_config_from_env(
            env={
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH": "cift-certification.json",
                "AEGIS_CIFT_CERTIFICATION_REPORT_PATH": "cift-certification-run.json",
                "AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT": "cift-artifacts",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256": "a" * 64,
                "AEGIS_CIFT_CERTIFICATION_REPORT_SHA256": "b" * 64,
            }
        )

    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256"):
        cift_config_from_env(
            env={
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH": "cift-certification.json",
                "AEGIS_CIFT_CERTIFICATION_REPORT_PATH": "cift-certification-run.json",
                "AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT": "cift-artifacts",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256": "a" * 64,
                "AEGIS_CIFT_CERTIFICATION_REPORT_SHA256": "b" * 64,
                "AEGIS_CIFT_RELEASE_GATE_REPORT_PATH": "cift-release-gate.json",
            }
        )

    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_REQUIRED_DEVICE"):
        cift_config_from_env(
            env={
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH": "cift-certification.json",
                "AEGIS_CIFT_CERTIFICATION_REPORT_PATH": "cift-certification-run.json",
                "AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT": "cift-artifacts",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256": "a" * 64,
                "AEGIS_CIFT_CERTIFICATION_REPORT_SHA256": "b" * 64,
                "AEGIS_CIFT_RELEASE_GATE_REPORT_PATH": "cift-release-gate.json",
                "AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256": "c" * 64,
            }
        )

    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT"):
        cift_config_from_env(
            env={
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH": "cift-certification.json",
                "AEGIS_CIFT_CERTIFICATION_REPORT_PATH": "cift-certification-run.json",
                "AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT": "cift-artifacts",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256": "a" * 64,
                "AEGIS_CIFT_CERTIFICATION_REPORT_SHA256": "b" * 64,
                "AEGIS_CIFT_RELEASE_GATE_REPORT_PATH": "cift-release-gate.json",
                "AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256": "c" * 64,
                "AEGIS_CIFT_REQUIRED_DEVICE": "mps",
            }
        )


def test_cift_config_accepts_self_hosted_window_selector_profile() -> None:
    config = cift_config_from_env(
        env={
            "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
            "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
            "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH": "cift-certification.json",
            "AEGIS_CIFT_CERTIFICATION_REPORT_PATH": "cift-certification-run.json",
            "AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT": "cift-artifacts",
            "AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256": "a" * 64,
            "AEGIS_CIFT_CERTIFICATION_REPORT_SHA256": "b" * 64,
            "AEGIS_CIFT_RELEASE_GATE_REPORT_PATH": "cift-release-gate.json",
            "AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256": "c" * 64,
            "AEGIS_CIFT_REQUIRED_DEVICE": "mps",
            "AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT": "4",
            "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
            "AEGIS_CIFT_EXTRACTOR_BASE_URL": "http://127.0.0.1:9000",
            "AEGIS_CIFT_EXTRACTOR_API_KEY": "sidecar-token",
            "AEGIS_CIFT_EXTRACTOR_TIMEOUT_SECONDS": "2.5",
            "AEGIS_CIFT_FEATURE_SOURCE": "self_hosted_activation_extractor",
            "AEGIS_CIFT_DETECTOR_NAME": "cift_prod",
        }
    )

    assert config.profile == CiftProfile.SELF_HOSTED_WINDOW_SELECTOR
    assert config.certification_mode == CiftCertificationMode.STRICT
    assert config.selected_choice_model_path is not None
    assert str(config.selected_choice_model_path) == "selected.json"
    assert config.fallback_model_path is None
    assert config.certification_manifest_path is not None
    assert str(config.certification_manifest_path) == "cift-certification.json"
    assert config.certification_report_path is not None
    assert str(config.certification_report_path) == "cift-certification-run.json"
    assert config.certification_artifact_root is not None
    assert str(config.certification_artifact_root) == "cift-artifacts"
    assert config.certification_manifest_sha256 == "a" * 64
    assert config.certification_report_sha256 == "b" * 64
    assert config.release_gate_report_path is not None
    assert str(config.release_gate_report_path) == "cift-release-gate.json"
    assert config.release_gate_report_sha256 == "c" * 64
    assert config.required_device == "mps"
    assert config.selected_choice_readout_token_count == 4
    assert config.extractor_id == "trusted-activation-sidecar"
    assert config.extractor_base_url == "http://127.0.0.1:9000"
    assert config.extractor_api_key == "sidecar-token"
    assert config.extractor_timeout_seconds == 2.5
    assert config.feature_source == "self_hosted_activation_extractor"
    assert config.detector_name == "cift_prod"


def test_cift_config_rejects_strict_fallback_model_path() -> None:
    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_FALLBACK_MODEL_PATH"):
        cift_config_from_env(
            env={
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_FALLBACK_MODEL_PATH": "fallback.json",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH": "cift-certification.json",
                "AEGIS_CIFT_CERTIFICATION_REPORT_PATH": "cift-certification-run.json",
                "AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT": "cift-artifacts",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256": "a" * 64,
                "AEGIS_CIFT_CERTIFICATION_REPORT_SHA256": "b" * 64,
                "AEGIS_CIFT_RELEASE_GATE_REPORT_PATH": "cift-release-gate.json",
                "AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256": "c" * 64,
                "AEGIS_CIFT_REQUIRED_DEVICE": "mps",
                "AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT": "4",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
            }
        )


def test_cift_config_rejects_untrusted_strict_feature_source() -> None:
    try:
        cift_config_from_env(
            env={
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH": "cift-certification.json",
                "AEGIS_CIFT_CERTIFICATION_REPORT_PATH": "cift-certification-run.json",
                "AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT": "cift-artifacts",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256": "a" * 64,
                "AEGIS_CIFT_CERTIFICATION_REPORT_SHA256": "b" * 64,
                "AEGIS_CIFT_RELEASE_GATE_REPORT_PATH": "cift-release-gate.json",
                "AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256": "c" * 64,
                "AEGIS_CIFT_REQUIRED_DEVICE": "mps",
                "AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT": "4",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
                "AEGIS_CIFT_FEATURE_SOURCE": "offline_replay",
            }
        )
    except ProxyConfigError as exc:
        assert "AEGIS_CIFT_FEATURE_SOURCE" in str(exc)
    else:
        raise AssertionError("strict CIFT must reject untrusted feature sources.")


def test_cift_config_accepts_self_hosted_window_selector_without_fallback() -> None:
    config = cift_config_from_env(
        env={
            "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
            "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
            "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH": "cift-certification.json",
            "AEGIS_CIFT_CERTIFICATION_REPORT_PATH": "cift-certification-run.json",
            "AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT": "cift-artifacts",
            "AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256": "a" * 64,
            "AEGIS_CIFT_CERTIFICATION_REPORT_SHA256": "b" * 64,
            "AEGIS_CIFT_RELEASE_GATE_REPORT_PATH": "cift-release-gate.json",
            "AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256": "c" * 64,
            "AEGIS_CIFT_REQUIRED_DEVICE": "mps",
            "AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT": "4",
            "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
        }
    )

    assert config.profile == CiftProfile.SELF_HOSTED_WINDOW_SELECTOR
    assert config.certification_mode == CiftCertificationMode.STRICT
    assert config.selected_choice_model_path is not None
    assert str(config.selected_choice_model_path) == "selected.json"
    assert config.fallback_model_path is None
    assert config.certification_manifest_path is not None
    assert str(config.certification_manifest_path) == "cift-certification.json"
    assert config.certification_report_path is not None
    assert str(config.certification_report_path) == "cift-certification-run.json"
    assert config.certification_artifact_root is not None
    assert str(config.certification_artifact_root) == "cift-artifacts"
    assert config.certification_manifest_sha256 == "a" * 64
    assert config.certification_report_sha256 == "b" * 64
    assert config.release_gate_report_path is not None
    assert str(config.release_gate_report_path) == "cift-release-gate.json"
    assert config.release_gate_report_sha256 == "c" * 64
    assert config.required_device == "mps"
    assert config.selected_choice_readout_token_count == 4
    assert config.extractor_id == "trusted-activation-sidecar"
    assert config.extractor_base_url is None
    assert config.extractor_api_key is None
    assert config.extractor_timeout_seconds is None


def test_cift_config_rejects_invalid_self_hosted_extractor_sidecar_values() -> None:
    base_env = {
        "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
        "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
        "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH": "cift-certification.json",
        "AEGIS_CIFT_CERTIFICATION_REPORT_PATH": "cift-certification-run.json",
        "AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT": "cift-artifacts",
        "AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256": "a" * 64,
        "AEGIS_CIFT_CERTIFICATION_REPORT_SHA256": "b" * 64,
        "AEGIS_CIFT_RELEASE_GATE_REPORT_PATH": "cift-release-gate.json",
        "AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256": "c" * 64,
        "AEGIS_CIFT_REQUIRED_DEVICE": "mps",
        "AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT": "4",
        "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
    }

    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_EXTRACTOR_BASE_URL"):
        cift_config_from_env(env={**base_env, "AEGIS_CIFT_EXTRACTOR_BASE_URL": ""})
    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_EXTRACTOR_BASE_URL"):
        cift_config_from_env(env={**base_env, "AEGIS_CIFT_EXTRACTOR_BASE_URL": "http://sidecar.example:9000"})
    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_EXTRACTOR_BASE_URL"):
        cift_config_from_env(env={**base_env, "AEGIS_CIFT_EXTRACTOR_BASE_URL": "https://token@sidecar.example"})
    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_EXTRACTOR_BASE_URL"):
        cift_config_from_env(env={**base_env, "AEGIS_CIFT_EXTRACTOR_BASE_URL": "file:///tmp/sidecar.sock"})
    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_EXTRACTOR_API_KEY"):
        cift_config_from_env(env={**base_env, "AEGIS_CIFT_EXTRACTOR_API_KEY": ""})
    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_EXTRACTOR_TIMEOUT_SECONDS"):
        cift_config_from_env(env={**base_env, "AEGIS_CIFT_EXTRACTOR_TIMEOUT_SECONDS": "0"})
    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT"):
        cift_config_from_env(env={**base_env, "AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT": "0"})
    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT"):
        cift_config_from_env(env={**base_env, "AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT": "two"})
    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256"):
        cift_config_from_env(env={**base_env, "AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256": "abc"})
    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_CERTIFICATION_REPORT_SHA256"):
        cift_config_from_env(env={**base_env, "AEGIS_CIFT_CERTIFICATION_REPORT_SHA256": "B" * 64})
    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256"):
        cift_config_from_env(env={**base_env, "AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256": "D" * 64})


def test_cift_config_accepts_gateway_smoke_bootstrap_without_certification_artifacts() -> None:
    config = cift_config_from_env(
        env={
            "AEGIS_PROVIDER": "mock",
            "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
            "AEGIS_CIFT_CERTIFICATION_MODE": "gateway_smoke_bootstrap",
            "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
            "AEGIS_CIFT_REQUIRED_DEVICE": "mps",
            "AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT": "4",
            "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
            "AEGIS_CIFT_EXTRACTOR_BASE_URL": "http://127.0.0.1:9000",
        }
    )

    assert config.profile == CiftProfile.SELF_HOSTED_WINDOW_SELECTOR
    assert config.certification_mode == CiftCertificationMode.GATEWAY_SMOKE_BOOTSTRAP
    assert config.selected_choice_model_path is not None
    assert str(config.selected_choice_model_path) == "selected.json"
    assert config.certification_manifest_path is None
    assert config.certification_report_path is None
    assert config.certification_artifact_root is None
    assert config.certification_manifest_sha256 is None
    assert config.certification_report_sha256 is None
    assert config.release_gate_report_path is None
    assert config.release_gate_report_sha256 is None
    assert config.required_device == "mps"
    assert config.selected_choice_readout_token_count == 4
    assert config.extractor_id == "trusted-activation-sidecar"
    assert config.extractor_base_url == "http://127.0.0.1:9000"


def test_cift_config_rejects_gateway_smoke_bootstrap_mixed_with_certification_artifacts() -> None:
    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH"):
        cift_config_from_env(
            env={
                "AEGIS_PROVIDER": "mock",
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_CERTIFICATION_MODE": "gateway_smoke_bootstrap",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_REQUIRED_DEVICE": "mps",
                "AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT": "4",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
                "AEGIS_CIFT_EXTRACTOR_BASE_URL": "http://127.0.0.1:9000",
                "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH": "certification.json",
            }
        )


def test_cift_config_rejects_gateway_smoke_bootstrap_without_mock_provider() -> None:
    with pytest.raises(ProxyConfigError, match="AEGIS_PROVIDER=mock"):
        cift_config_from_env(
            env={
                "AEGIS_PROVIDER": "openai_compatible",
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_CERTIFICATION_MODE": "gateway_smoke_bootstrap",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_REQUIRED_DEVICE": "mps",
                "AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT": "4",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
                "AEGIS_CIFT_EXTRACTOR_BASE_URL": "http://127.0.0.1:9000",
            }
        )


def test_cift_config_rejects_gateway_smoke_bootstrap_without_sidecar_base_url() -> None:
    with pytest.raises(ProxyConfigError, match="AEGIS_CIFT_EXTRACTOR_BASE_URL"):
        cift_config_from_env(
            env={
                "AEGIS_PROVIDER": "mock",
                "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
                "AEGIS_CIFT_CERTIFICATION_MODE": "gateway_smoke_bootstrap",
                "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH": "selected.json",
                "AEGIS_CIFT_REQUIRED_DEVICE": "mps",
                "AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT": "4",
                "AEGIS_CIFT_EXTRACTOR_ID": "trusted-activation-sidecar",
            }
        )


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
