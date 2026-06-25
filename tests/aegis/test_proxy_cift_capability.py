from __future__ import annotations

from aegis.core.contracts import Action, CapabilityMode, JsonValue
from aegis.core.orchestrator import ModelProvider
from aegis.detectors.cift_runtime import CiftFeatureVectorAnnotator, CiftRuntimeDetector, CiftRuntimeLinearModel
from aegis.providers.mock import MockModelProvider
from aegis.proxy.config import ProviderKind, ProxyNimbusConfig, ProxyProviderConfig
from aegis.proxy.mock_app import ProxyCiftCapability, create_default_proxy, create_proxy


def test_default_proxy_reports_black_box_cift_capability() -> None:
    proxy = create_default_proxy()

    status, payload = proxy.handle(method="GET", path="/aegis/capabilities", body={})

    assert status == 200
    assert payload["cift"] == {
        "capability_mode": "black_box",
        "detectors": ["activation_unavailable"],
        "turn_annotator_count": 0,
    }


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
                ),
            ),
            pre_generation_detectors=(CiftRuntimeDetector(detector_name="cift_runtime", model=_runtime_model()),),
            detector_names=("cift_runtime",),
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
    assert cift_result["recommended_action"] == Action.WARN.value
    assert cift_result["evidence"]["activation_source"] == "metadata.cift.feature_vectors"
    assert cift_result["evidence"]["feature_source"] == "test_self_hosted_extractor"
    cift_stage = _runtime_stage(aegis=aegis, stage_name="cift")
    assert cift_stage == {"stage": "cift", "status": "active", "detectors": ["cift_runtime"]}


class ConstantFeatureExtractor:
    def __init__(self, feature_vector: tuple[float, ...]) -> None:
        self._feature_vector = feature_vector

    def extract_feature_vector(self, turn: object, feature_key: str) -> tuple[float, ...] | None:
        return self._feature_vector


def _mock_provider_config(model_provider: ModelProvider) -> ProxyProviderConfig:
    return ProxyProviderConfig(
        kind=ProviderKind.MOCK,
        provider_name="mock",
        model_provider=model_provider,
        mock_controls_enabled=True,
    )


def _nimbus_config() -> ProxyNimbusConfig:
    return ProxyNimbusConfig(
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
    )


def _runtime_model() -> CiftRuntimeLinearModel:
    return CiftRuntimeLinearModel(
        schema_version="aegis.cift_runtime_linear/v1",
        model_bundle_id="test_bundle",
        source_model_id="test-model",
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
        positive_action=Action.WARN,
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
