from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from aegis.audit.memory import InMemoryAuditSink
from aegis.cift_contract import (
    CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
    CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
)
from aegis.core.contracts import Action, CapabilityMode, DetectorComponent, DetectorResult, NormalizedTurn
from aegis.core.orchestrator import AegisRuntime, Detector, ModelProvider, ModelResponse, TurnAnnotator
from aegis.detectors.activation import ActivationUnavailableDetector
from aegis.detectors.canary import (
    CanaryRecord,
    EncodedCanaryDetector,
    InMemoryCanaryRegistry,
    NoopCanaryDetector,
    TextCanaryDetector,
    ToolCallCanaryDetector,
)
from aegis.detectors.cift_runtime import (
    CiftFeatureExtractor,
    CiftRuntimeDetectorError,
    CiftRuntimeModel,
    CiftRuntimeWindowSelectorConfig,
    build_cift_window_selector_gateway_smoke_bootstrap_components,
    build_cift_window_selector_runtime_components,
    load_cift_runtime_model_with_sha256,
)
from aegis.detectors.egress import ProviderEgressGuardDetector
from aegis.detectors.nimbus import NimbusDetector, NimbusToolEgressDetector
from aegis.policy.engine import SeverityPolicyEngine
from aegis.proxy.cift_certification import (
    CiftCertificationBinding,
    CiftCertificationBindingConfig,
    CiftCertificationBindingError,
    validate_cift_certification_binding,
)
from aegis.proxy.config import (
    CiftCertificationMode,
    CiftProfile,
    ProxyCiftConfig,
    ProxyCiftRouteCertificationConfig,
    ProxyConfigError,
)


@dataclass(frozen=True)
class ProxyCiftRuntimeBinding:
    certification_mode: CiftCertificationMode
    certification_id: str | None
    runtime_model_sha256: str
    release_gate_report_sha256: str | None
    model_bundle_id: str
    source_model_id: str
    source_revision: str
    source_selected_device: str
    source_hidden_size: int
    source_layer_count: int
    tokenizer_fingerprint_sha256: str
    special_tokens_map_sha256: str
    chat_template_sha256: str
    feature_key: str
    feature_count: int
    selected_choice_readout_token_count: int
    freeform_route: ProxyCiftRouteRuntimeBinding | None


@dataclass(frozen=True)
class ProxyCiftRouteRuntimeBinding:
    certification_id: str | None
    runtime_model_sha256: str
    release_gate_report_sha256: str | None
    model_bundle_id: str
    source_model_id: str
    source_revision: str
    source_selected_device: str
    source_hidden_size: int
    source_layer_count: int
    tokenizer_fingerprint_sha256: str
    special_tokens_map_sha256: str
    chat_template_sha256: str
    feature_key: str
    feature_count: int


@dataclass(frozen=True)
class ProxyCiftCapability:
    capability_mode: CapabilityMode
    turn_annotators: tuple[TurnAnnotator, ...]
    pre_generation_detectors: tuple[Detector, ...]
    detector_names: tuple[str, ...]
    runtime_binding: ProxyCiftRuntimeBinding | None


@dataclass(frozen=True)
class _CiftRuntimeBindingDetector:
    detector: Detector
    runtime_binding: ProxyCiftRuntimeBinding

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        result = self.detector.evaluate(turn, model_response)
        if result.component != DetectorComponent.CIFT:
            return result
        route_binding = _runtime_binding_for_detector_result(
            runtime_binding=self.runtime_binding,
            result=result,
        )
        evidence = dict(result.evidence)
        evidence.update(
            {
                "certification_mode": self.runtime_binding.certification_mode.value,
                "certification_id": route_binding.certification_id,
                "runtime_model_sha256": route_binding.runtime_model_sha256,
                "release_gate_report_sha256": route_binding.release_gate_report_sha256,
                "runtime_model_bundle_id": route_binding.model_bundle_id,
            }
        )
        return DetectorResult(
            detector_name=result.detector_name,
            component=result.component,
            score=result.score,
            confidence=result.confidence,
            recommended_action=result.recommended_action,
            capability_required=result.capability_required,
            capability_status=result.capability_status,
            evidence=evidence,
            latency_ms=result.latency_ms,
        )


@dataclass(frozen=True)
class ProxyRuntimeFactory:
    audit_sink: InMemoryAuditSink
    nimbus_detector: NimbusDetector
    nimbus_tool_egress_detector: NimbusToolEgressDetector
    cift_capability: ProxyCiftCapability
    model_provider: ModelProvider

    def build(self, canary_records: tuple[CanaryRecord, ...]) -> AegisRuntime:
        return AegisRuntime(
            turn_annotators=self.cift_capability.turn_annotators,
            pre_generation_detectors=(
                *self.cift_capability.pre_generation_detectors,
                *_pre_generation_canary_detectors(
                    canary_records=canary_records,
                    nimbus_tool_egress_detector=self.nimbus_tool_egress_detector,
                ),
                ProviderEgressGuardDetector(),
            ),
            post_generation_detectors=_post_generation_detectors(canary_records),
            session_detectors=(self.nimbus_detector,),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=self.audit_sink,
            model_provider=self.model_provider,
        )


def black_box_cift_capability() -> ProxyCiftCapability:
    return ProxyCiftCapability(
        capability_mode=CapabilityMode.BLACK_BOX,
        turn_annotators=(),
        pre_generation_detectors=(ActivationUnavailableDetector(),),
        detector_names=("activation_unavailable",),
        runtime_binding=None,
    )


def cift_capability_from_config(
    config: ProxyCiftConfig,
    extractors: Mapping[str, CiftFeatureExtractor],
) -> ProxyCiftCapability:
    if config.profile == CiftProfile.BLACK_BOX:
        return black_box_cift_capability()
    if config.profile == CiftProfile.SELF_HOSTED_WINDOW_SELECTOR:
        return _self_hosted_window_selector_capability(config=config, extractors=extractors)
    raise ProxyConfigError(f"Unhandled CIFT profile '{config.profile.value}'.")


def _self_hosted_window_selector_capability(
    config: ProxyCiftConfig,
    extractors: Mapping[str, CiftFeatureExtractor],
) -> ProxyCiftCapability:
    if config.selected_choice_model_path is None:
        raise ProxyConfigError("self_hosted_window_selector requires selected_choice_model_path.")
    if config.required_device is None:
        raise ProxyConfigError("self_hosted_window_selector requires required_device.")
    if config.selected_choice_readout_token_count is None:
        raise ProxyConfigError("self_hosted_window_selector requires selected_choice_readout_token_count.")
    if config.selected_choice_readout_token_count < 1:
        raise ProxyConfigError("self_hosted_window_selector selected_choice_readout_token_count must be positive.")
    if config.extractor_id is None:
        raise ProxyConfigError("self_hosted_window_selector requires extractor_id.")
    extractor = extractors.get(config.extractor_id)
    if extractor is None:
        raise ProxyConfigError(f"CIFT extractor '{config.extractor_id}' is not registered.")
    if config.certification_mode == CiftCertificationMode.GATEWAY_SMOKE_BOOTSTRAP:
        return _gateway_smoke_bootstrap_window_selector_capability(
            config=config,
            extractor=extractor,
        )
    if config.certification_mode != CiftCertificationMode.STRICT:
        raise ProxyConfigError(f"Unhandled CIFT certification mode '{config.certification_mode.value}'.")
    return _strict_self_hosted_window_selector_capability(config=config, extractor=extractor)


def _strict_self_hosted_window_selector_capability(
    config: ProxyCiftConfig,
    extractor: CiftFeatureExtractor,
) -> ProxyCiftCapability:
    if config.selected_choice_model_path is None:
        raise ProxyConfigError("self_hosted_window_selector requires selected_choice_model_path.")
    if config.certification_manifest_path is None:
        raise ProxyConfigError("self_hosted_window_selector requires certification_manifest_path.")
    if config.certification_report_path is None:
        raise ProxyConfigError("self_hosted_window_selector requires certification_report_path.")
    if config.certification_artifact_root is None:
        raise ProxyConfigError("self_hosted_window_selector requires certification_artifact_root.")
    if config.certification_manifest_sha256 is None:
        raise ProxyConfigError("self_hosted_window_selector requires certification_manifest_sha256.")
    if config.certification_report_sha256 is None:
        raise ProxyConfigError("self_hosted_window_selector requires certification_report_sha256.")
    if config.release_gate_report_path is None:
        raise ProxyConfigError("self_hosted_window_selector requires release_gate_report_path.")
    if config.release_gate_report_sha256 is None:
        raise ProxyConfigError("self_hosted_window_selector requires release_gate_report_sha256.")
    if config.required_device is None:
        raise ProxyConfigError("self_hosted_window_selector requires required_device.")
    if config.selected_choice_readout_token_count is None:
        raise ProxyConfigError("self_hosted_window_selector requires selected_choice_readout_token_count.")
    if config.selected_choice_readout_token_count < 1:
        raise ProxyConfigError("self_hosted_window_selector selected_choice_readout_token_count must be positive.")
    if config.extractor_id is None:
        raise ProxyConfigError("self_hosted_window_selector requires extractor_id.")
    if config.fallback_model_path is not None and config.freeform_certification is None:
        raise ProxyConfigError(
            "strict self_hosted_window_selector requires freeform certification for fallback_model_path."
        )
    if (
        config.fallback_model_path is not None
        and config.freeform_certification is not None
        and config.fallback_model_path != config.freeform_certification.model_path
    ):
        raise ProxyConfigError("fallback_model_path must match freeform certification model_path.")
    try:
        binding = _validate_cift_route_certification(
            route_config=ProxyCiftRouteCertificationConfig(
                model_path=config.selected_choice_model_path,
                certification_manifest_path=config.certification_manifest_path,
                certification_report_path=config.certification_report_path,
                certification_artifact_root=config.certification_artifact_root,
                certification_manifest_sha256=config.certification_manifest_sha256,
                certification_report_sha256=config.certification_report_sha256,
                release_gate_report_path=config.release_gate_report_path,
                release_gate_report_sha256=config.release_gate_report_sha256,
            ),
            config=config,
        )
        runtime_model = load_cift_runtime_model_with_sha256(
            path=config.selected_choice_model_path,
            expected_sha256=binding.runtime_sha256,
        )
        if not _is_selected_choice_feature_key(runtime_model.feature_key):
            raise ProxyConfigError("selected_choice_model_path must use a selected-choice CIFT feature key.")
        freeform_binding = None
        freeform_model = None
        if config.freeform_certification is not None:
            freeform_binding = _validate_cift_route_certification(
                route_config=config.freeform_certification,
                config=config,
            )
            freeform_model = load_cift_runtime_model_with_sha256(
                path=config.freeform_certification.model_path,
                expected_sha256=freeform_binding.runtime_sha256,
            )
            if _is_selected_choice_feature_key(freeform_model.feature_key):
                raise ProxyConfigError("freeform_model_path must use a freeform CIFT feature key.")
        components = build_cift_window_selector_runtime_components(
            CiftRuntimeWindowSelectorConfig(
                detector_name=config.detector_name,
                selected_choice_model_path=config.selected_choice_model_path,
                selected_choice_model_sha256=binding.runtime_sha256,
                fallback_model_path=config.fallback_model_path,
                fallback_model_sha256=None if freeform_binding is None else freeform_binding.runtime_sha256,
                feature_extractor=extractor,
                feature_source=config.feature_source,
                activation_failure_action=Action.BLOCK,
            )
        )
    except (CiftCertificationBindingError, CiftRuntimeDetectorError) as exc:
        raise ProxyConfigError(str(exc)) from exc
    runtime_binding = _runtime_binding(
        certification_mode=config.certification_mode,
        certification_id=binding.certification_id,
        runtime_model=runtime_model,
        runtime_model_sha256=binding.runtime_sha256,
        release_gate_report_sha256=binding.release_gate_report_sha256,
        selected_choice_readout_token_count=config.selected_choice_readout_token_count,
        freeform_route=(
            None
            if freeform_binding is None or freeform_model is None
            else _route_runtime_binding(
                certification_id=freeform_binding.certification_id,
                runtime_model=freeform_model,
                runtime_model_sha256=freeform_binding.runtime_sha256,
                release_gate_report_sha256=freeform_binding.release_gate_report_sha256,
            )
        ),
    )
    return ProxyCiftCapability(
        capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
        turn_annotators=components.turn_annotators,
        pre_generation_detectors=_bind_cift_runtime_evidence(
            detectors=components.pre_generation_detectors,
            runtime_binding=runtime_binding,
        ),
        detector_names=(config.detector_name,),
        runtime_binding=runtime_binding,
    )


def _gateway_smoke_bootstrap_window_selector_capability(
    config: ProxyCiftConfig,
    extractor: CiftFeatureExtractor,
) -> ProxyCiftCapability:
    if config.selected_choice_model_path is None:
        raise ProxyConfigError("self_hosted_window_selector requires selected_choice_model_path.")
    if config.required_device is None:
        raise ProxyConfigError("self_hosted_window_selector requires required_device.")
    if config.selected_choice_readout_token_count is None:
        raise ProxyConfigError("self_hosted_window_selector requires selected_choice_readout_token_count.")
    if config.selected_choice_readout_token_count < 1:
        raise ProxyConfigError("self_hosted_window_selector selected_choice_readout_token_count must be positive.")
    try:
        runtime_model_sha256 = _sha256_file(config.selected_choice_model_path)
        fallback_model_sha256 = None if config.fallback_model_path is None else _sha256_file(config.fallback_model_path)
        runtime_model = load_cift_runtime_model_with_sha256(
            path=config.selected_choice_model_path,
            expected_sha256=runtime_model_sha256,
        )
        fallback_model = (
            None
            if config.fallback_model_path is None or fallback_model_sha256 is None
            else load_cift_runtime_model_with_sha256(
                path=config.fallback_model_path,
                expected_sha256=fallback_model_sha256,
            )
        )
        if fallback_model is not None and _is_selected_choice_feature_key(fallback_model.feature_key):
            raise ProxyConfigError("gateway_smoke_bootstrap fallback_model_path must be a freeform route artifact.")
        components = build_cift_window_selector_gateway_smoke_bootstrap_components(
            config=CiftRuntimeWindowSelectorConfig(
                detector_name=config.detector_name,
                selected_choice_model_path=config.selected_choice_model_path,
                selected_choice_model_sha256=runtime_model_sha256,
                fallback_model_path=config.fallback_model_path,
                fallback_model_sha256=fallback_model_sha256,
                feature_extractor=extractor,
                feature_source=config.feature_source,
                activation_failure_action=Action.BLOCK,
            ),
            required_device=config.required_device,
        )
    except CiftRuntimeDetectorError as exc:
        raise ProxyConfigError(str(exc)) from exc
    runtime_binding = _runtime_binding(
        certification_mode=config.certification_mode,
        certification_id=None,
        runtime_model=runtime_model,
        runtime_model_sha256=runtime_model_sha256,
        release_gate_report_sha256=None,
        selected_choice_readout_token_count=config.selected_choice_readout_token_count,
        freeform_route=(
            None
            if fallback_model is None or fallback_model_sha256 is None
            else _route_runtime_binding(
                certification_id=None,
                runtime_model=fallback_model,
                runtime_model_sha256=fallback_model_sha256,
                release_gate_report_sha256=None,
            )
        ),
    )
    return ProxyCiftCapability(
        capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
        turn_annotators=components.turn_annotators,
        pre_generation_detectors=_bind_cift_runtime_evidence(
            detectors=components.pre_generation_detectors,
            runtime_binding=runtime_binding,
        ),
        detector_names=(config.detector_name,),
        runtime_binding=runtime_binding,
    )


def _validate_cift_route_certification(
    route_config: ProxyCiftRouteCertificationConfig,
    config: ProxyCiftConfig,
) -> CiftCertificationBinding:
    if config.required_device is None:
        raise ProxyConfigError("self_hosted_window_selector requires required_device.")
    if config.selected_choice_readout_token_count is None:
        raise ProxyConfigError("self_hosted_window_selector requires selected_choice_readout_token_count.")
    if config.extractor_id is None:
        raise ProxyConfigError("self_hosted_window_selector requires extractor_id.")
    try:
        return validate_cift_certification_binding(
            CiftCertificationBindingConfig(
                runtime_model_path=route_config.model_path,
                certification_manifest_path=route_config.certification_manifest_path,
                certification_report_path=route_config.certification_report_path,
                certification_artifact_root=route_config.certification_artifact_root,
                release_gate_report_path=route_config.release_gate_report_path,
                required_device=config.required_device,
                expected_manifest_sha256=route_config.certification_manifest_sha256,
                expected_report_sha256=route_config.certification_report_sha256,
                expected_release_gate_report_sha256=route_config.release_gate_report_sha256,
                expected_detector_name=config.detector_name,
                expected_extractor_id=config.extractor_id,
                expected_feature_source=config.feature_source,
                expected_prompt_renderer=CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
                expected_selected_choice_geometry=CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
                expected_selected_choice_readout_token_count=config.selected_choice_readout_token_count,
            )
        )
    except CiftCertificationBindingError as exc:
        raise ProxyConfigError(str(exc)) from exc


def _route_runtime_binding(
    certification_id: str | None,
    runtime_model: CiftRuntimeModel,
    runtime_model_sha256: str,
    release_gate_report_sha256: str | None,
) -> ProxyCiftRouteRuntimeBinding:
    return ProxyCiftRouteRuntimeBinding(
        certification_id=certification_id,
        runtime_model_sha256=runtime_model_sha256,
        release_gate_report_sha256=release_gate_report_sha256,
        model_bundle_id=runtime_model.model_bundle_id,
        source_model_id=runtime_model.source_model_id,
        source_revision=runtime_model.source_revision,
        source_selected_device=runtime_model.source_selected_device,
        source_hidden_size=runtime_model.source_hidden_size,
        source_layer_count=runtime_model.source_layer_count,
        tokenizer_fingerprint_sha256=runtime_model.tokenizer_fingerprint_sha256,
        special_tokens_map_sha256=runtime_model.special_tokens_map_sha256,
        chat_template_sha256=runtime_model.chat_template_sha256,
        feature_key=runtime_model.feature_key,
        feature_count=runtime_model.feature_count,
    )


def _runtime_binding(
    certification_mode: CiftCertificationMode,
    certification_id: str | None,
    runtime_model: CiftRuntimeModel,
    runtime_model_sha256: str,
    release_gate_report_sha256: str | None,
    selected_choice_readout_token_count: int,
    freeform_route: ProxyCiftRouteRuntimeBinding | None,
) -> ProxyCiftRuntimeBinding:
    selected_choice_route = _route_runtime_binding(
        certification_id=certification_id,
        runtime_model=runtime_model,
        runtime_model_sha256=runtime_model_sha256,
        release_gate_report_sha256=release_gate_report_sha256,
    )
    return ProxyCiftRuntimeBinding(
        certification_mode=certification_mode,
        certification_id=selected_choice_route.certification_id,
        runtime_model_sha256=selected_choice_route.runtime_model_sha256,
        release_gate_report_sha256=selected_choice_route.release_gate_report_sha256,
        model_bundle_id=selected_choice_route.model_bundle_id,
        source_model_id=selected_choice_route.source_model_id,
        source_revision=selected_choice_route.source_revision,
        source_selected_device=selected_choice_route.source_selected_device,
        source_hidden_size=selected_choice_route.source_hidden_size,
        source_layer_count=selected_choice_route.source_layer_count,
        tokenizer_fingerprint_sha256=selected_choice_route.tokenizer_fingerprint_sha256,
        special_tokens_map_sha256=selected_choice_route.special_tokens_map_sha256,
        chat_template_sha256=selected_choice_route.chat_template_sha256,
        feature_key=selected_choice_route.feature_key,
        feature_count=selected_choice_route.feature_count,
        selected_choice_readout_token_count=selected_choice_readout_token_count,
        freeform_route=freeform_route,
    )


def _runtime_binding_for_detector_result(
    runtime_binding: ProxyCiftRuntimeBinding,
    result: DetectorResult,
) -> ProxyCiftRouteRuntimeBinding:
    window_family = result.evidence.get("cift_window_family")
    if (
        isinstance(window_family, str)
        and window_family.startswith("freeform_")
        and runtime_binding.freeform_route is not None
    ):
        return runtime_binding.freeform_route
    return _selected_choice_route_from_runtime_binding(runtime_binding)


def _selected_choice_route_from_runtime_binding(binding: ProxyCiftRuntimeBinding) -> ProxyCiftRouteRuntimeBinding:
    return ProxyCiftRouteRuntimeBinding(
        certification_id=binding.certification_id,
        runtime_model_sha256=binding.runtime_model_sha256,
        release_gate_report_sha256=binding.release_gate_report_sha256,
        model_bundle_id=binding.model_bundle_id,
        source_model_id=binding.source_model_id,
        source_revision=binding.source_revision,
        source_selected_device=binding.source_selected_device,
        source_hidden_size=binding.source_hidden_size,
        source_layer_count=binding.source_layer_count,
        tokenizer_fingerprint_sha256=binding.tokenizer_fingerprint_sha256,
        special_tokens_map_sha256=binding.special_tokens_map_sha256,
        chat_template_sha256=binding.chat_template_sha256,
        feature_key=binding.feature_key,
        feature_count=binding.feature_count,
    )


def _is_selected_choice_feature_key(feature_key: str) -> bool:
    return feature_key.startswith("selected_choice_window_")


def _bind_cift_runtime_evidence(
    detectors: tuple[Detector, ...],
    runtime_binding: ProxyCiftRuntimeBinding,
) -> tuple[Detector, ...]:
    return tuple(
        _CiftRuntimeBindingDetector(detector=detector, runtime_binding=runtime_binding) for detector in detectors
    )


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ProxyConfigError(f"CIFT runtime model path does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _post_generation_detectors(canary_records: tuple[CanaryRecord, ...]) -> tuple[Detector, ...]:
    if len(canary_records) == 0:
        return (NoopCanaryDetector(),)
    registry = InMemoryCanaryRegistry(records=canary_records)
    return (
        TextCanaryDetector(detector_name="text_canary", registry=registry),
        EncodedCanaryDetector(detector_name="encoded_canary", registry=registry, partial_match_threshold=0.75),
    )


def _pre_generation_canary_detectors(
    canary_records: tuple[CanaryRecord, ...],
    nimbus_tool_egress_detector: NimbusToolEgressDetector,
) -> tuple[Detector, ...]:
    if len(canary_records) == 0:
        return ()
    registry = InMemoryCanaryRegistry(records=canary_records)
    return (
        ToolCallCanaryDetector(detector_name="tool_call_canary", registry=registry),
        nimbus_tool_egress_detector,
    )
