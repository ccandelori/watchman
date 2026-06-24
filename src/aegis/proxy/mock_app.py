from __future__ import annotations

import math
from collections.abc import Mapping
from uuid import uuid4

from aegis.audit.memory import InMemoryAuditSink
from aegis.core.contracts import AuditEvent, CapabilityMode, JsonValue, Message, ModelInfo, SensitiveSpan
from aegis.core.orchestrator import AegisRuntime, AegisRuntimeResponse, RuntimeRequest
from aegis.detectors.activation import ActivationUnavailableDetector
from aegis.detectors.canary import NoopCanaryDetector
from aegis.detectors.nimbus import BaselineNimbusCritic, InMemoryNimbusStateStore, NimbusConfig, NimbusDetector
from aegis.policy.engine import SeverityPolicyEngine
from aegis.providers.mock import MockModelProvider


class ProxyRequestError(ValueError):
    """Raised when a mock proxy request cannot be normalized."""


class MockProxyApp:
    def __init__(self, runtime: AegisRuntime, audit_sink: InMemoryAuditSink) -> None:
        self._runtime = runtime
        self._audit_sink = audit_sink

    def handle(self, method: str, path: str, body: object) -> tuple[int, dict[str, JsonValue]]:
        if method == "GET" and path == "/health":
            return 200, {"status": "ok"}
        if method == "GET" and path == "/audit/recent":
            return 200, {"events": [_safe_audit_event(event) for event in self._audit_sink.recent(limit=20)]}
        if method == "POST" and path == "/v1/chat/completions":
            try:
                return 200, self._handle_chat_completions(body)
            except ProxyRequestError as exc:
                return 400, {"error": str(exc)}
            except Exception:
                return 500, _internal_proxy_error_payload(body=body)
        return 404, {"error": f"No route for {method} {path}."}

    def _handle_chat_completions(self, body: object) -> dict[str, JsonValue]:
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
            "aegis": _chat_completion_aegis_metadata(request=request, response=response),
        }


def _chat_completion_aegis_metadata(
    request: RuntimeRequest,
    response: AegisRuntimeResponse,
) -> dict[str, JsonValue]:
    detector_results: list[JsonValue] = [result.to_dict() for result in response.detector_results]
    return {
        "schema_version": "aegis.proxy.chat_completion/v1",
        "trace_id": request.trace_id,
        "session_id": request.session_id,
        "turn_index": request.turn_index,
        "capability_mode": request.capability_mode.value,
        "detector_count": len(detector_results),
        "detector_results": detector_results,
        "policy_decision": response.policy_decision.to_dict(),
    }


def _runtime_request_from_chat_body(body: object) -> RuntimeRequest:
    request_body = _json_object_from_raw(value=body, context="request body")
    model = request_body.get("model")
    if not isinstance(model, str) or model == "":
        raise ProxyRequestError("field 'model' must be a non-empty string.")

    raw_messages = request_body.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) == 0:
        raise ProxyRequestError("field 'messages' must be a non-empty list.")

    messages = tuple(_message_from_raw(item) for item in raw_messages)
    metadata = _metadata_from_raw(request_body.get("metadata"))
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
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProxyRequestError(f"metadata field '{key}' must be an integer.")
    if value < 0:
        raise ProxyRequestError(f"metadata field '{key}' must be non-negative.")
    return value


def _json_object_from_raw(value: object, context: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ProxyRequestError(f"{context} must be a JSON object.")
    decoded: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ProxyRequestError(f"{context} keys must be strings.")
        decoded[key] = _json_value_from_raw(value=item, context=f"{context}.{key}")
    return decoded


def _json_value_from_raw(value: object, context: str) -> JsonValue:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProxyRequestError(f"{context} must be finite.")
        return value
    if isinstance(value, list):
        return [_json_value_from_raw(value=item, context=f"{context}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        return _json_object_from_raw(value=value, context=context)
    raise ProxyRequestError(f"{context} must be JSON-compatible.")


def _internal_proxy_error_payload(body: object) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {"error": "internal proxy error"}
    trace_id = _trace_id_from_body(body=body)
    if trace_id is not None:
        payload["aegis"] = {"trace_id": trace_id}
    return payload


def _trace_id_from_body(body: object) -> str | None:
    if not isinstance(body, dict):
        return None
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return None
    trace_id = metadata.get("trace_id")
    if not isinstance(trace_id, str) or trace_id == "":
        return None
    return trace_id


def _safe_audit_event(event: AuditEvent) -> dict[str, JsonValue]:
    turn = event.normalized_turn
    return {
        "trace_id": event.trace_id,
        "session_id": event.session_id,
        "turn_index": event.turn_index,
        "turn_summary": {
            "capability_mode": turn.capability_mode.value,
            "model": turn.model.to_dict(),
            "message_count": len(turn.messages),
            "message_roles": [message.role for message in turn.messages],
            "tool_call_count": len(turn.tool_calls),
            "sensitive_span_count": len(turn.sensitive_spans),
            "metadata_key_count": len(turn.metadata),
        },
        "sensitive_spans": [_safe_sensitive_span(span) for span in turn.sensitive_spans],
        "detector_results": [result.to_dict() for result in event.detector_results],
        "policy_decision": event.policy_decision.to_dict(),
        "latency_ms": event.latency_ms,
        "created_at": event.created_at,
    }


def _safe_sensitive_span(span: SensitiveSpan) -> dict[str, JsonValue]:
    return {
        "kind": span.kind,
        "source": span.source,
        "char_start": span.char_start,
        "char_end": span.char_end,
        "token_start": span.token_start,
        "token_end": span.token_end,
        "identifier": span.identifier,
        "metadata": _safe_sensitive_span_metadata(span.metadata),
    }


def _safe_sensitive_span_metadata(metadata: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    safe_keys = ("sha256", "honeytoken_sha256", "credential_type", "source", "slot_name", "turn_planted")
    return {key: metadata[key] for key in safe_keys if key in metadata}


def create_default_proxy() -> MockProxyApp:
    audit_sink = InMemoryAuditSink()
    runtime = AegisRuntime(
        turn_annotators=(),
        pre_generation_detectors=(ActivationUnavailableDetector(),),
        post_generation_detectors=(NoopCanaryDetector(),),
        session_detectors=(
            NimbusDetector(
                config=NimbusConfig(
                    budget_bits=1.0,
                    warn_threshold=0.3,
                    sanitize_threshold=0.5,
                    block_threshold=0.7,
                    max_turns=20,
                    critic_version="baseline-fixed-v0",
                ),
                critic=BaselineNimbusCritic(fixed_estimated_leakage_bits=0.0, fixed_confidence=0.7),
                state_store=InMemoryNimbusStateStore(max_turns=20),
            ),
        ),
        policy_engine=SeverityPolicyEngine(),
        audit_sink=audit_sink,
        model_provider=MockModelProvider(default_content="Aegis mock response."),
    )
    return MockProxyApp(runtime=runtime, audit_sink=audit_sink)
