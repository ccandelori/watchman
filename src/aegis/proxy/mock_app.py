from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

from aegis.audit.memory import InMemoryAuditSink
from aegis.core.contracts import CapabilityMode, JsonValue, Message, ModelInfo
from aegis.core.orchestrator import AegisRuntime, RuntimeRequest
from aegis.detectors.activation import ActivationUnavailableDetector
from aegis.detectors.canary import NoopCanaryDetector
from aegis.policy.engine import SeverityPolicyEngine
from aegis.providers.mock import MockModelProvider


class ProxyRequestError(ValueError):
    """Raised when a mock proxy request cannot be normalized."""


class MockProxyApp:
    def __init__(self, runtime: AegisRuntime, audit_sink: InMemoryAuditSink) -> None:
        self._runtime = runtime
        self._audit_sink = audit_sink

    def handle(self, method: str, path: str, body: dict[str, JsonValue]) -> tuple[int, dict[str, JsonValue]]:
        if method == "GET" and path == "/health":
            return 200, {"status": "ok"}
        if method == "GET" and path == "/audit/recent":
            return 200, {"events": [event.to_dict() for event in self._audit_sink.recent(limit=20)]}
        if method == "POST" and path == "/v1/chat/completions":
            try:
                return 200, self._handle_chat_completions(body)
            except ProxyRequestError as exc:
                return 400, {"error": str(exc)}
        return 404, {"error": f"No route for {method} {path}."}

    def _handle_chat_completions(self, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
        request = _runtime_request_from_chat_body(body)
        response = self._runtime.evaluate_turn(request)
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


def _runtime_request_from_chat_body(body: dict[str, JsonValue]) -> RuntimeRequest:
    model = body.get("model")
    if not isinstance(model, str) or model == "":
        raise ProxyRequestError("field 'model' must be a non-empty string.")

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) == 0:
        raise ProxyRequestError("field 'messages' must be a non-empty list.")

    messages = tuple(_message_from_raw(item) for item in raw_messages)
    metadata = _metadata_from_raw(body.get("metadata"))
    trace_id = _metadata_string(metadata, "trace_id", f"trace-{uuid4().hex}")
    session_id = _metadata_string(metadata, "session_id", f"session-{uuid4().hex}")
    turn_index = _metadata_int(metadata, "turn_index", 1)

    return RuntimeRequest(
        trace_id=trace_id,
        session_id=session_id,
        turn_index=turn_index,
        capability_mode=CapabilityMode.BLACK_BOX,
        model=ModelInfo(provider="mock", model_id=model, revision=None, selected_device=None),
        messages=messages,
        tool_calls=(),
        sensitive_spans=(),
        metadata=metadata,
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


def _metadata_string(metadata: Mapping[str, JsonValue], key: str, default: str) -> str:
    value = metadata.get(key)
    if value is None:
        return default
    if not isinstance(value, str) or value == "":
        raise ProxyRequestError(f"metadata field '{key}' must be a non-empty string.")
    return value


def _metadata_int(metadata: Mapping[str, JsonValue], key: str, default: int) -> int:
    value = metadata.get(key)
    if value is None:
        return default
    if not isinstance(value, int):
        raise ProxyRequestError(f"metadata field '{key}' must be an integer.")
    return value


def create_default_proxy() -> MockProxyApp:
    audit_sink = InMemoryAuditSink()
    runtime = AegisRuntime(
        pre_generation_detectors=(ActivationUnavailableDetector(),),
        post_generation_detectors=(NoopCanaryDetector(),),
        session_detectors=(),
        policy_engine=SeverityPolicyEngine(),
        audit_sink=audit_sink,
        model_provider=MockModelProvider(default_content="Aegis mock response."),
    )
    return MockProxyApp(runtime=runtime, audit_sink=audit_sink)
