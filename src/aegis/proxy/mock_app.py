from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from uuid import uuid4

from aegis.audit.memory import InMemoryAuditSink
from aegis.canaries.ledger import HoneytokenLedger, default_honeytoken_generator, inject_honeytokens
from aegis.core.contracts import (
    CapabilityMode,
    DetectorComponent,
    DetectorResult,
    JsonValue,
    Message,
    ModelInfo,
    SensitiveSpan,
)
from aegis.core.orchestrator import AegisRuntime, AegisRuntimeResponse, Detector, ModelProvider, RuntimeRequest
from aegis.detectors.activation import ActivationUnavailableDetector
from aegis.detectors.canary import (
    CanaryRecord,
    EncodedCanaryDetector,
    InMemoryCanaryRegistry,
    NoopCanaryDetector,
    TextCanaryDetector,
    canary_sha256,
)
from aegis.detectors.egress import ProviderEgressGuardDetector
from aegis.detectors.nimbus import (
    CanaryNimbusCritic,
    CanaryNimbusCriticConfig,
    InMemoryNimbusStateStore,
    NimbusConfig,
    NimbusDetector,
)
from aegis.policy.engine import SeverityPolicyEngine
from aegis.providers.mock import SUPPORTED_MOCK_RESPONSE_MODES, MockModelProvider

_RAW_CREDENTIAL_PATTERN = re.compile(r"(?:AKIA|ghp_|ya29\.|sk_live_|sk-|hny_)[A-Za-z0-9._-]{8,}")


class ProxyRequestError(ValueError):
    """Raised when a mock proxy request cannot be normalized."""


@dataclass(frozen=True)
class ProxyRuntimeRequest:
    runtime_request: RuntimeRequest
    canary_records: tuple[CanaryRecord, ...]


class MockProxyApp:
    def __init__(
        self,
        audit_sink: InMemoryAuditSink,
        nimbus_detector: NimbusDetector,
        nimbus_critic: CanaryNimbusCritic,
        model_provider: ModelProvider,
    ) -> None:
        self._audit_sink = audit_sink
        self._nimbus_detector = nimbus_detector
        self._nimbus_critic = nimbus_critic
        self._model_provider = model_provider

    def handle(self, method: str, path: str, body: dict[str, JsonValue]) -> tuple[int, dict[str, JsonValue]]:
        if method == "GET" and path == "/health":
            return 200, {"status": "ok"}
        if method == "GET" and path == "/audit/recent":
            return 200, {"events": [event.to_dict() for event in self._audit_sink.recent(limit=20)]}
        if method == "POST" and path == "/test/reset":
            try:
                return 200, self._handle_test_reset(body)
            except ProxyRequestError as exc:
                return 400, {"error": str(exc)}
        if method == "POST" and path == "/v1/chat/completions":
            try:
                return 200, self._handle_chat_completions(body)
            except ProxyRequestError as exc:
                return 400, {"error": str(exc)}
        return 404, {"error": f"No route for {method} {path}."}

    def destroy_session(self, session_id: str) -> None:
        self._nimbus_detector.destroy_session(session_id)
        self._nimbus_critic.destroy_session(session_id)

    def _handle_test_reset(self, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
        session_id = _optional_metadata_string(body, "session_id")
        if session_id is not None:
            self.destroy_session(session_id)
        self._audit_sink.clear()
        return {
            "status": "reset",
            "audit_events_cleared": True,
            "session_id": session_id,
        }

    def _handle_chat_completions(self, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
        proxy_request = _runtime_request_from_chat_body(body)
        request = proxy_request.runtime_request
        self._nimbus_critic.register_canary_records(
            session_id=request.session_id,
            records=proxy_request.canary_records,
        )
        runtime = self._runtime_for_canary_records(proxy_request.canary_records)
        response = runtime.evaluate_turn(request)
        return {
            "id": f"chatcmpl-{request.trace_id}",
            "object": "chat.completion",
            "model": request.model.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.output_text},
                    "finish_reason": "stop",
                }
            ],
            "aegis": {
                "trace_id": request.trace_id,
                "runtime_trace": _runtime_trace(request=request, response=response),
                "policy_decision": response.policy_decision.to_dict(),
                "detector_results": [result.to_dict() for result in response.detector_results],
            },
        }

    def _runtime_for_canary_records(self, canary_records: tuple[CanaryRecord, ...]) -> AegisRuntime:
        return AegisRuntime(
            turn_annotators=(),
            pre_generation_detectors=(ActivationUnavailableDetector(), ProviderEgressGuardDetector()),
            post_generation_detectors=_post_generation_detectors(canary_records),
            session_detectors=(self._nimbus_detector,),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=self._audit_sink,
            model_provider=self._model_provider,
        )


def _runtime_request_from_chat_body(body: dict[str, JsonValue]) -> ProxyRuntimeRequest:
    model = body.get("model")
    if not isinstance(model, str) or model == "":
        raise ProxyRequestError("field 'model' must be a non-empty string.")

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) == 0:
        raise ProxyRequestError("field 'messages' must be a non-empty list.")

    messages = tuple(_message_from_raw(item) for item in raw_messages)
    metadata = _metadata_from_raw(body.get("metadata"))
    _validate_mock_response_mode(metadata)
    trace_id = _metadata_string(metadata, "trace_id", f"trace-{uuid4().hex}")
    session_id = _metadata_string(metadata, "session_id", f"session-{uuid4().hex}")
    turn_index = _metadata_int(metadata, "turn_index", 1)

    injection = inject_honeytokens(
        messages=messages,
        ledger=HoneytokenLedger(
            session_id=session_id,
            generator=default_honeytoken_generator,
            source="dp_honey_dev_proxy",
        ),
        turn_index=turn_index,
    )
    raw_credential_spans = _raw_credential_spans(
        messages=injection.messages,
        canary_records=injection.canary_records,
    )
    metadata = _metadata_with_dp_honey_summary(metadata, injection.canary_records)

    return ProxyRuntimeRequest(
        runtime_request=RuntimeRequest(
            trace_id=trace_id,
            session_id=session_id,
            turn_index=turn_index,
            capability_mode=CapabilityMode.BLACK_BOX,
            model=ModelInfo(provider="mock", model_id=model, revision=None, selected_device=None),
            messages=injection.messages,
            tool_calls=(),
            sensitive_spans=injection.sensitive_spans + raw_credential_spans,
            metadata=metadata,
        ),
        canary_records=injection.canary_records,
    )


def _message_from_raw(value: object) -> Message:
    if not isinstance(value, dict):
        raise ProxyRequestError("each message must be an object.")
    role = value.get("role")
    content = value.get("content")
    if not isinstance(role, str) or role == "":
        raise ProxyRequestError("each message must include a non-empty string role.")
    if not isinstance(content, str):
        raise ProxyRequestError("each message must include string content.")
    return Message(role=role, content=content)


def _metadata_from_raw(value: object) -> dict[str, JsonValue]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProxyRequestError("field 'metadata' must be an object when provided.")
    metadata: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ProxyRequestError("metadata keys must be strings.")
        metadata[key] = item
    _validate_no_raw_credentials_in_metadata(metadata)
    return metadata


def _validate_no_raw_credentials_in_metadata(metadata: dict[str, JsonValue]) -> None:
    paths = _credential_shaped_metadata_paths(metadata, path="metadata")
    if len(paths) > 0:
        joined_paths = ", ".join(paths)
        raise ProxyRequestError(f"metadata contains credential-shaped value(s) at {joined_paths}.")


def _credential_shaped_metadata_paths(value: JsonValue, path: str) -> tuple[str, ...]:
    if isinstance(value, str):
        if _RAW_CREDENTIAL_PATTERN.search(value):
            return (path,)
        return ()
    if isinstance(value, list):
        paths: list[str] = []
        for index, item in enumerate(value):
            paths.extend(_credential_shaped_metadata_paths(item, path=f"{path}[{index}]"))
        return tuple(paths)
    if isinstance(value, dict):
        paths = []
        for key, item in value.items():
            paths.extend(_credential_shaped_metadata_paths(item, path=f"{path}.{key}"))
        return tuple(paths)
    return ()


def _validate_mock_response_mode(metadata: Mapping[str, JsonValue]) -> None:
    mode = metadata.get("mock_response_mode")
    if mode is None:
        return
    if not isinstance(mode, str) or mode == "":
        raise ProxyRequestError("metadata field 'mock_response_mode' must be a non-empty string when provided.")
    if mode not in SUPPORTED_MOCK_RESPONSE_MODES:
        supported_modes = ", ".join(sorted(SUPPORTED_MOCK_RESPONSE_MODES))
        raise ProxyRequestError(f"unsupported mock_response_mode '{mode}'. Supported modes: {supported_modes}.")


def _metadata_with_dp_honey_summary(
    metadata: dict[str, JsonValue],
    canary_records: tuple[CanaryRecord, ...],
) -> dict[str, JsonValue]:
    if len(canary_records) == 0:
        return metadata
    updated = dict(metadata)
    updated["dp_honey_canary_count"] = len(canary_records)
    updated["dp_honey_canary_ids"] = [record.canary_id for record in canary_records]
    return updated


def _raw_credential_spans(
    messages: tuple[Message, ...],
    canary_records: tuple[CanaryRecord, ...],
) -> tuple[SensitiveSpan, ...]:
    canary_hashes = frozenset(record.sha256 for record in canary_records)
    spans: list[SensitiveSpan] = []
    for message_index, message in enumerate(messages):
        for match in _RAW_CREDENTIAL_PATTERN.finditer(message.content):
            candidate = match.group(0)
            candidate_sha256 = canary_sha256(candidate)
            if candidate_sha256 in canary_hashes:
                continue
            spans.append(
                _raw_credential_span(
                    char_start=match.start(),
                    char_end=match.end(),
                    sha256=candidate_sha256,
                    message_index=message_index,
                )
            )
    return tuple(spans)


def _raw_credential_span(char_start: int, char_end: int, sha256: str, message_index: int) -> SensitiveSpan:
    return SensitiveSpan(
        kind="credential",
        source="proxy_raw_credential_scanner",
        char_start=char_start,
        char_end=char_end,
        token_start=None,
        token_end=None,
        identifier=None,
        metadata={"sha256": sha256, "message_index": message_index},
    )


def _metadata_string(metadata: Mapping[str, JsonValue], key: str, default: str) -> str:
    value = metadata.get(key)
    if value is None:
        return default
    if not isinstance(value, str) or value == "":
        raise ProxyRequestError(f"metadata field '{key}' must be a non-empty string.")
    return value


def _optional_metadata_string(metadata: Mapping[str, JsonValue], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise ProxyRequestError(f"field '{key}' must be a non-empty string when provided.")
    return value


def _metadata_int(metadata: Mapping[str, JsonValue], key: str, default: int) -> int:
    value = metadata.get(key)
    if value is None:
        return default
    if not isinstance(value, int):
        raise ProxyRequestError(f"metadata field '{key}' must be an integer.")
    return value


def _post_generation_detectors(canary_records: tuple[CanaryRecord, ...]) -> tuple[Detector, ...]:
    if len(canary_records) == 0:
        return (NoopCanaryDetector(),)
    registry = InMemoryCanaryRegistry(records=canary_records)
    return (
        TextCanaryDetector(detector_name="text_canary", registry=registry),
        EncodedCanaryDetector(detector_name="encoded_canary", registry=registry, partial_match_threshold=0.75),
    )


def _runtime_trace(request: RuntimeRequest, response: AegisRuntimeResponse) -> dict[str, JsonValue]:
    detector_results = response.detector_results
    policy_decision = response.policy_decision
    model_response_metadata = response.model_response_metadata
    stages: list[JsonValue] = [
        {"stage": "normalize", "status": "ok"},
        _dp_honey_stage(request),
        _detector_stage(
            stage="cift",
            component=DetectorComponent.CIFT,
            detector_results=detector_results,
        ),
        _detector_stage(
            stage="provider_egress_guard",
            component=DetectorComponent.TOOL_SCANNER,
            detector_results=detector_results,
        ),
        _provider_stage(request=request, model_response_metadata=model_response_metadata),
        _canary_stage(detector_results),
        _detector_stage(
            stage="nimbus",
            component=DetectorComponent.NIMBUS,
            detector_results=detector_results,
        ),
        {
            "stage": "policy",
            "status": "decided",
            "final_action": policy_decision.final_action.value,
        },
        {"stage": "audit", "status": "written"},
    ]
    return {"schema_version": "aegis.runtime_trace/v1", "stages": stages}


def _dp_honey_stage(request: RuntimeRequest) -> dict[str, JsonValue]:
    canary_count = request.metadata.get("dp_honey_canary_count")
    if not isinstance(canary_count, int):
        canary_count = 0
    return {
        "stage": "dp_honey",
        "status": "active" if canary_count > 0 else "not_configured",
        "canary_count": canary_count,
    }


def _provider_stage(request: RuntimeRequest, model_response_metadata: dict[str, JsonValue]) -> dict[str, JsonValue]:
    provider = model_response_metadata.get("provider")
    if not isinstance(provider, str) or provider == "":
        provider = request.model.provider
    status = "skipped" if provider == "skipped" else "completed"
    stage: dict[str, JsonValue] = {
        "stage": "provider",
        "status": status,
        "provider": provider,
        "model_id": request.model.model_id,
    }
    reason = model_response_metadata.get("reason")
    if isinstance(reason, str) and reason != "":
        stage["reason"] = reason
    return stage


def _canary_stage(detector_results: tuple[DetectorResult, ...]) -> dict[str, JsonValue]:
    detector_names = _detector_names_for_component(DetectorComponent.TEXT_CANARY, detector_results)
    if len(detector_names) > 0:
        return {"stage": "canary", "status": "active", "detectors": list(detector_names)}
    noop_names = tuple(result.detector_name for result in detector_results if result.detector_name == "noop_canary")
    if len(noop_names) > 0:
        return {"stage": "canary", "status": "degraded", "detectors": list(noop_names)}
    return {"stage": "canary", "status": "not_configured", "detectors": []}


def _detector_stage(
    stage: str,
    component: DetectorComponent,
    detector_results: tuple[DetectorResult, ...],
) -> dict[str, JsonValue]:
    detector_names = _detector_names_for_component(component, detector_results)
    if len(detector_names) == 0:
        return {"stage": stage, "status": "not_configured", "detectors": []}
    statuses = {result.capability_status.value for result in detector_results if result.component == component}
    if "active" in statuses:
        status = "active"
    elif "degraded" in statuses:
        status = "degraded"
    else:
        status = "unavailable"
    return {"stage": stage, "status": status, "detectors": list(detector_names)}


def _detector_names_for_component(
    component: DetectorComponent,
    detector_results: tuple[DetectorResult, ...],
) -> tuple[str, ...]:
    return tuple(result.detector_name for result in detector_results if result.component == component)


def create_default_proxy() -> MockProxyApp:
    audit_sink = InMemoryAuditSink()
    nimbus_critic = CanaryNimbusCritic(
        CanaryNimbusCriticConfig(
            exact_match_leakage_bits=1.0,
            encoded_match_leakage_bits=1.0,
            partial_match_leakage_bits=0.8,
            partial_match_threshold=0.4,
            confidence=0.8,
        )
    )
    nimbus_detector = NimbusDetector(
        NimbusConfig(
            budget_bits=1.0,
            warn_threshold=0.3,
            sanitize_threshold=0.6,
            block_threshold=0.9,
            max_turns=20,
            critic_version="canary-v0",
        ),
        nimbus_critic,
        InMemoryNimbusStateStore(max_turns=20),
    )
    return MockProxyApp(
        audit_sink=audit_sink,
        nimbus_detector=nimbus_detector,
        nimbus_critic=nimbus_critic,
        model_provider=MockModelProvider(default_content="Aegis mock response."),
    )
