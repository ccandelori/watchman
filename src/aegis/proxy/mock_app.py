from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from uuid import uuid4

from aegis.audit.memory import InMemoryAuditSink
from aegis.canaries.ledger import HoneytokenLedger, default_honeytoken_generator, inject_honeytokens
from aegis.core.contracts import CapabilityMode, JsonValue, Message, ModelInfo
from aegis.core.orchestrator import AegisRuntime, Detector, ModelProvider, RuntimeRequest
from aegis.detectors.activation import ActivationUnavailableDetector
from aegis.detectors.canary import (
    CanaryRecord,
    EncodedCanaryDetector,
    InMemoryCanaryRegistry,
    NoopCanaryDetector,
    TextCanaryDetector,
)
from aegis.detectors.nimbus import BaselineNimbusCritic, InMemoryNimbusStateStore, NimbusConfig, NimbusDetector
from aegis.policy.engine import SeverityPolicyEngine
from aegis.providers.mock import SUPPORTED_MOCK_RESPONSE_MODES, MockModelProvider


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
        model_provider: ModelProvider,
    ) -> None:
        self._audit_sink = audit_sink
        self._nimbus_detector = nimbus_detector
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
                "policy_decision": response.policy_decision.to_dict(),
                "detector_results": [result.to_dict() for result in response.detector_results],
            },
        }

    def _runtime_for_canary_records(self, canary_records: tuple[CanaryRecord, ...]) -> AegisRuntime:
        return AegisRuntime(
            turn_annotators=(),
            pre_generation_detectors=(ActivationUnavailableDetector(),),
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
            sensitive_spans=injection.sensitive_spans,
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
    return metadata


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


def create_default_proxy() -> MockProxyApp:
    audit_sink = InMemoryAuditSink()
    nimbus_detector = NimbusDetector(
        NimbusConfig(
            budget_bits=10.0,
            warn_threshold=0.5,
            sanitize_threshold=0.7,
            block_threshold=0.9,
            max_turns=20,
            critic_version="baseline-v0",
        ),
        BaselineNimbusCritic(fixed_estimated_leakage_bits=0.0, fixed_confidence=0.5),
        InMemoryNimbusStateStore(max_turns=20),
    )
    return MockProxyApp(
        audit_sink=audit_sink,
        nimbus_detector=nimbus_detector,
        model_provider=MockModelProvider(default_content="Aegis mock response."),
    )
