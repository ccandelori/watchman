from __future__ import annotations

import pytest

from aegis.core.contracts import Action, CapabilityMode, JsonValue
from aegis.core.orchestrator import ModelProvider
from aegis.detectors.cift_runtime import CiftFeatureVectorAnnotator, CiftRuntimeDetector, CiftRuntimeLinearModel
from aegis.providers.mock import MockModelProvider
from aegis.proxy.config import (
    CiftCertificationMode,
    ProviderKind,
    ProxyConfigError,
    ProxyNimbusConfig,
    ProxyProviderConfig,
)
from aegis.proxy.mock_app import create_default_proxy, create_proxy
from aegis.proxy.nimbus_profile import NimbusCriticKind
from aegis.proxy.runtime_factory import ProxyCiftCapability, ProxyCiftRuntimeBinding


def test_default_proxy_reports_black_box_cift_capability() -> None:
    proxy = create_default_proxy()

    status, payload = proxy.handle(method="GET", path="/aegis/capabilities", body={})

    assert status == 200
    assert payload["cift"] == {
        "capability_mode": "black_box",
        "detectors": ["activation_unavailable"],
        "support_reason": (
            "black-box provider mode has no certified hidden-state extractor binding; "
            "DP-HONEY, NIMBUS, and provider egress remain available."
        ),
        "support_scope": "model-specific CIFT enforcement unavailable",
        "support_tier": "unsupported",
        "turn_annotator_count": 0,
    }


def test_proxy_reports_strict_cift_runtime_binding_in_capabilities() -> None:
    proxy = create_proxy(
        provider_config=_mock_provider_config(MockModelProvider(default_content="ok")),
        nimbus_config=_nimbus_config(),
        cift_capability=ProxyCiftCapability(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            turn_annotators=(),
            pre_generation_detectors=(),
            detector_names=("cift_runtime",),
            runtime_binding=ProxyCiftRuntimeBinding(
                certification_mode=CiftCertificationMode.STRICT,
                certification_id="synthetic-certified-cift",
                runtime_model_sha256="a" * 64,
                release_gate_report_sha256="e" * 64,
                model_bundle_id="selected-choice-bundle",
                source_model_id="Qwen/Qwen3-4B",
                source_revision="0123456789abcdef0123456789abcdef01234567",
                source_selected_device="mps",
                source_hidden_size=2560,
                source_layer_count=36,
                tokenizer_fingerprint_sha256="b" * 64,
                special_tokens_map_sha256="c" * 64,
                chat_template_sha256="d" * 64,
                feature_key="selected_choice_window_layer_21",
                feature_count=1024,
                selected_choice_readout_token_count=4,
            ),
        ),
    )

    status, payload = proxy.handle(method="GET", path="/aegis/capabilities", body={})

    assert status == 200
    assert payload["cift"] == {
        "capability_mode": "self_hosted_introspection",
        "detectors": ["cift_runtime"],
        "support_reason": (
            "strict certification binding is loaded; readiness still depends on trusted extractor attestation."
        ),
        "support_scope": "model-specific CIFT enforcement for Qwen/Qwen3-4B on mps",
        "support_tier": "runtime-enforceable",
        "turn_annotator_count": 0,
        "runtime_binding": {
            "certification_mode": "strict",
            "certification_id": "synthetic-certified-cift",
            "runtime_model_sha256": "a" * 64,
            "release_gate_report_sha256": "e" * 64,
            "model_bundle_id": "selected-choice-bundle",
            "source_model_id": "Qwen/Qwen3-4B",
            "source_revision": "0123456789abcdef0123456789abcdef01234567",
            "source_selected_device": "mps",
            "source_hidden_size": 2560,
            "source_layer_count": 36,
            "tokenizer_fingerprint_sha256": "b" * 64,
            "special_tokens_map_sha256": "c" * 64,
            "chat_template_sha256": "d" * 64,
            "feature_key": "selected_choice_window_layer_21",
            "feature_count": 1024,
            "selected_choice_readout_token_count": 4,
        },
    }


def test_default_proxy_fails_closed_when_self_hosted_cift_extractor_is_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AEGIS_CIFT_PROFILE", "self_hosted_window_selector")
    monkeypatch.setenv("AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH", "selected.json")
    monkeypatch.setenv("AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH", "certification.json")
    monkeypatch.setenv("AEGIS_CIFT_CERTIFICATION_REPORT_PATH", "certification-run.json")
    monkeypatch.setenv("AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT", ".")
    monkeypatch.setenv("AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256", "0" * 64)
    monkeypatch.setenv("AEGIS_CIFT_CERTIFICATION_REPORT_SHA256", "1" * 64)
    monkeypatch.setenv("AEGIS_CIFT_RELEASE_GATE_REPORT_PATH", "release-gate.json")
    monkeypatch.setenv("AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256", "2" * 64)
    monkeypatch.setenv("AEGIS_CIFT_REQUIRED_DEVICE", "mps")
    monkeypatch.setenv("AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT", "4")
    monkeypatch.setenv("AEGIS_CIFT_EXTRACTOR_ID", "trusted-activation-sidecar")

    with pytest.raises(ProxyConfigError, match="trusted-activation-sidecar"):
        create_default_proxy()


def test_proxy_can_run_server_configured_self_hosted_cift_capability() -> None:
    proxy = create_proxy(
        provider_config=_mock_provider_config(MockModelProvider(default_content="ok")),
        nimbus_config=_nimbus_config(),
        cift_capability=ProxyCiftCapability(
            capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
            turn_annotators=(
                CiftFeatureVectorAnnotator(
                    feature_key="readout_window_layer_15",
                    extractor=ConstantFeatureExtractor(feature_vector=(3.0, 2.0)),
                    source="test_self_hosted_extractor",
                    selected_choice_window=False,
                ),
            ),
            pre_generation_detectors=(
                CiftRuntimeDetector(
                    detector_name="cift_runtime",
                    model=_runtime_model(),
                    activation_failure_action=Action.BLOCK,
                ),
            ),
            detector_names=("cift_runtime",),
            runtime_binding=None,
        ),
    )

    status, payload = proxy.handle(
        method="POST",
        path="/v1/chat/completions",
        body={
            "model": "mock-qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"trace_id": "trace-cift-proxy", "session_id": "session-cift-proxy", "turn_index": 0},
        },
    )

    assert status == 200
    aegis = payload["aegis"]
    assert isinstance(aegis, dict)
    detector_results = _detector_results(aegis)
    cift_result = _detector_result(detector_results=detector_results, detector_name="cift_runtime")
    assert cift_result["capability_status"] == "active"
    assert cift_result["recommended_action"] == Action.BLOCK.value
    assert cift_result["evidence"]["activation_source"] == "metadata.cift.feature_vectors"
    assert cift_result["evidence"]["feature_source"] == "test_self_hosted_extractor"
    assert payload["choices"][0]["message"]["content"] == "[aegis output withheld]"
    assert aegis["policy_decision"]["final_action"] == Action.BLOCK.value
    cift_stage = _runtime_stage(aegis=aegis, stage_name="cift")
    assert cift_stage == {"stage": "cift", "status": "active", "detectors": ["cift_runtime"]}
    provider_stage = _runtime_stage(aegis=aegis, stage_name="provider")
    assert provider_stage["status"] == "skipped"
    assert provider_stage["reason"] == "pre_generation_policy_block"


class ConstantFeatureExtractor:
    def __init__(self, feature_vector: tuple[float, ...]) -> None:
        self._feature_vector = feature_vector

    def extract_feature_vector(self, turn: object, feature_key: str) -> tuple[float, ...] | None:
        return self._feature_vector


def _mock_provider_config(model_provider: ModelProvider) -> ProxyProviderConfig:
    return ProxyProviderConfig(
        kind=ProviderKind.MOCK,
        provider_name="mock",
        provider_target_url=None,
        model_provider=model_provider,
        mock_controls_enabled=True,
    )


def _nimbus_config() -> ProxyNimbusConfig:
    return ProxyNimbusConfig(
        critic_kind=NimbusCriticKind.CANARY,
        exact_match_leakage_bits=1.0,
        encoded_match_leakage_bits=1.0,
        partial_match_leakage_bits=0.8,
        partial_match_threshold=0.4,
        confidence=0.8,
        budget_bits=1.0,
        warn_threshold=0.3,
        sanitize_threshold=0.6,
        block_threshold=0.9,
        max_turns=20,
        critic_version="canary-v0",
        infonce_model_path=None,
    )


def _runtime_model() -> CiftRuntimeLinearModel:
    return CiftRuntimeLinearModel(
        schema_version="aegis.cift_runtime_linear/v1",
        model_bundle_id="test_bundle",
        source_model_id="test-model",
        source_revision="main",
        source_selected_device="mps",
        source_hidden_size=2,
        source_layer_count=1,
        tokenizer_fingerprint_sha256="b" * 64,
        special_tokens_map_sha256="c" * 64,
        chat_template_sha256="d" * 64,
        training_dataset_id="test-dataset",
        source_artifact_sha256="a" * 64,
        evaluation_report_ids=("test-report",),
        task_name="safe_secret_vs_exfiltration",
        feature_key="readout_window_layer_15",
        feature_count=2,
        label_names=("secret_present_safe", "exfiltration_intent"),
        positive_label="exfiltration_intent",
        positive_class_index=1,
        class_indices=(0, 1),
        decision_threshold=0.5,
        score_semantics="test_probability",
        confidence=0.7,
        candidate_status="runtime_candidate",
        scaler_mean=(1.0, 2.0),
        scaler_scale=(2.0, 4.0),
        logistic_coefficients=(1.0, -0.5),
        logistic_intercept=0.25,
        negative_action=Action.ALLOW,
        positive_action=Action.BLOCK,
    )


def _detector_results(aegis: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    detector_results = aegis["detector_results"]
    if not isinstance(detector_results, list):
        raise AssertionError("detector_results must be a list.")
    return [item for item in detector_results if isinstance(item, dict)]


def _detector_result(
    detector_results: list[dict[str, JsonValue]],
    detector_name: str,
) -> dict[str, JsonValue]:
    matches = [result for result in detector_results if result.get("detector_name") == detector_name]
    if len(matches) != 1:
        raise AssertionError(f"expected one detector result named {detector_name}.")
    return matches[0]


def _runtime_stage(aegis: dict[str, JsonValue], stage_name: str) -> dict[str, JsonValue]:
    runtime_trace = aegis["runtime_trace"]
    if not isinstance(runtime_trace, dict):
        raise AssertionError("runtime_trace must be an object.")
    stages = runtime_trace["stages"]
    if not isinstance(stages, list):
        raise AssertionError("runtime_trace.stages must be a list.")
    matches = [stage for stage in stages if isinstance(stage, dict) and stage.get("stage") == stage_name]
    if len(matches) != 1:
        raise AssertionError(f"expected one runtime stage named {stage_name}.")
    return matches[0]
