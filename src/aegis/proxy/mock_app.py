from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from uuid import uuid4

from aegis.audit.memory import InMemoryAuditSink
from aegis.canaries.ledger import HoneytokenLedger, default_honeytoken_generator, inject_honeytokens
from aegis.core.contracts import (
    Action,
    CapabilityMode,
    DetectorComponent,
    DetectorResult,
    JsonValue,
    Message,
    ModelInfo,
    SensitiveSpan,
)
from aegis.core.orchestrator import AegisRuntimeResponse, RuntimeRequest
from aegis.detectors.canary import (
    CanaryRecord,
    canary_sha256,
)
from aegis.detectors.nimbus import (
    CanaryNimbusCritic,
    CanaryNimbusCriticConfig,
    InMemoryNimbusStateStore,
    NimbusConfig,
    NimbusDetector,
)
from aegis.providers.mock import SUPPORTED_MOCK_RESPONSE_MODES
from aegis.providers.openai_compatible import OpenAICompatibleProviderError
from aegis.proxy.config import ProxyNimbusConfig, ProxyProviderConfig, nimbus_config_from_env, provider_config_from_env
from aegis.proxy.nimbus_profile import nimbus_capabilities
from aegis.proxy.runtime_factory import ProxyCiftCapability, ProxyRuntimeFactory, black_box_cift_capability

_RAW_CREDENTIAL_PATTERN = re.compile(r"(?:AKIA|ghp_|ya29\.|sk_live_|sk-|hny_)[A-Za-z0-9._-]{8,}")
_PROXY_ERROR_SCHEMA_VERSION = "aegis.proxy_error/v1"
_CAPABILITIES_SCHEMA_VERSION = "aegis.proxy_capabilities/v1"
_AEGIS_RESPONSE_SCHEMA_VERSION = "aegis.chat_response/v1"
_TEST_SEED_CANARY_SCHEMA_VERSION = "aegis.test_seed_canary/v1"
_RESERVED_METADATA_PREFIXES = ("aegis_", "cift_", "dp_honey_", "nimbus_")
_TEST_SEEDED_CANARY_SOURCE = "test_seed_canary"
_SUPPORTED_TEST_SEED_CREDENTIAL_TYPES = (
    "aws_access_key",
    "github_pat",
    "oauth_token",
    "openai_key",
    "stripe_key",
)
_TEST_SEED_CANARY_FIELDS = frozenset(("session_id", "slot_name", "credential_type", "turn_index"))
_WITHHELD_OUTPUT_TEXT = "[aegis output withheld]"


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
        runtime_factory: ProxyRuntimeFactory,
        provider_name: str,
        mock_controls_enabled: bool,
        nimbus_config: ProxyNimbusConfig,
    ) -> None:
        self._audit_sink = audit_sink
        self._nimbus_detector = nimbus_detector
        self._nimbus_critic = nimbus_critic
        self._runtime_factory = runtime_factory
        self._provider_name = provider_name
        self._mock_controls_enabled = mock_controls_enabled
        self._nimbus_config = nimbus_config
        self._seeded_canaries_by_session_id: dict[str, tuple[CanaryRecord, ...]] = {}

    def handle(self, method: str, path: str, body: dict[str, JsonValue]) -> tuple[int, dict[str, JsonValue]]:
        if method == "GET" and path == "/health":
            return 200, {"status": "ok"}
        if method == "GET" and path == "/aegis/capabilities":
            return 200, self._capabilities()
        if method == "GET" and path == "/audit/recent":
            try:
                return 200, self._handle_audit_recent(body)
            except ProxyRequestError as exc:
                return 400, proxy_error_payload(code="invalid_request", message=str(exc), details={})
        if method == "POST" and path == "/test/reset":
            try:
                return 200, self._handle_test_reset(body)
            except ProxyRequestError as exc:
                return 400, proxy_error_payload(code="invalid_request", message=str(exc), details={})
        if method == "POST" and path == "/test/seed-canary" and self._mock_controls_enabled:
            try:
                return 200, self._handle_test_seed_canary(body)
            except ProxyRequestError as exc:
                return 400, proxy_error_payload(code="invalid_request", message=str(exc), details={})
        if method == "POST" and path == "/v1/chat/completions":
            try:
                return 200, self._handle_chat_completions(body)
            except ProxyRequestError as exc:
                return 400, proxy_error_payload(code="invalid_request", message=str(exc), details={})
            except OpenAICompatibleProviderError as exc:
                return 502, proxy_error_payload(code="provider_error", message=str(exc), details={})
        return 404, proxy_error_payload(
            code="route_not_found",
            message=f"No route for {method} {path}.",
            details={"method": method, "path": path},
        )

    def destroy_session(self, session_id: str) -> None:
        self._nimbus_detector.destroy_session(session_id)
        self._nimbus_critic.destroy_session(session_id)
        self._seeded_canaries_by_session_id.pop(session_id, None)

    def destroy_all_sessions(self) -> None:
        self._nimbus_detector.clear()
        self._nimbus_critic.clear()
        self._seeded_canaries_by_session_id.clear()

    def _capabilities(self) -> dict[str, JsonValue]:
        routes: list[JsonValue] = [
            {"method": "GET", "path": "/health"},
            {"method": "GET", "path": "/aegis/capabilities"},
            {"method": "POST", "path": "/v1/chat/completions"},
            {"method": "GET", "path": "/audit/recent"},
            {"method": "POST", "path": "/test/reset"},
        ]
        mock_response_modes: list[JsonValue] = []
        test_controls: dict[str, JsonValue] = {}
        if self._mock_controls_enabled:
            routes.append({"method": "POST", "path": "/test/seed-canary"})
            mock_response_modes = list(sorted(SUPPORTED_MOCK_RESPONSE_MODES))
            test_controls = {
                "seed_canary": {
                    "enabled": True,
                    "route": "/test/seed-canary",
                    "schema_version": _TEST_SEED_CANARY_SCHEMA_VERSION,
                    "request_fields": list(sorted(_TEST_SEED_CANARY_FIELDS)),
                    "supported_credential_types": list(_SUPPORTED_TEST_SEED_CREDENTIAL_TYPES),
                }
            }
        detectors: list[JsonValue] = [
            "activation_unavailable",
            "provider_egress_guard",
            "text_canary",
            "encoded_canary",
            "nimbus",
        ]
        return {
            "schema_version": _CAPABILITIES_SCHEMA_VERSION,
            "contract": {
                "chat_response_schema_version": _AEGIS_RESPONSE_SCHEMA_VERSION,
                "runtime_trace_schema_version": "aegis.runtime_trace/v1",
                "error_schema_version": _PROXY_ERROR_SCHEMA_VERSION,
            },
            "provider": {
                "name": self._provider_name,
                "mock_controls_enabled": self._mock_controls_enabled,
            },
            "cift": _cift_capabilities(self._runtime_factory.cift_capability),
            "nimbus": nimbus_capabilities(self._nimbus_config),
            "routes": routes,
            "mock_response_modes": mock_response_modes,
            "detectors": detectors,
            "audit": {
                "recent_default_limit": 20,
                "recent_supports_session_id": True,
            },
            "test_controls": test_controls,
        }

    def _handle_audit_recent(self, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
        limit = _metadata_positive_int(body, "limit", 20)
        session_id = _optional_metadata_string(body, "session_id")
        return {
            "schema_version": "aegis.audit_recent/v1",
            "limit": limit,
            "session_id": session_id,
            "events": [event.to_dict() for event in self._audit_sink.recent(limit=limit, session_id=session_id)],
        }

    def _handle_test_reset(self, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
        session_id = _optional_metadata_string(body, "session_id")
        if session_id is not None:
            self.destroy_session(session_id)
            self._audit_sink.clear_session(session_id)
            return {
                "schema_version": "aegis.test_reset/v1",
                "status": "reset",
                "scope": "session",
                "audit_events_cleared": True,
                "session_id": session_id,
            }
        self.destroy_all_sessions()
        self._audit_sink.clear()
        return {
            "schema_version": "aegis.test_reset/v1",
            "status": "reset",
            "scope": "all",
            "audit_events_cleared": True,
            "session_id": None,
        }

    def _handle_test_seed_canary(self, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _validate_test_seed_canary_body(body)
        session_id = _body_string(body, "session_id")
        slot_name = _body_string(body, "slot_name")
        credential_type = _body_string(body, "credential_type")
        turn_index = _body_int(body, "turn_index", 0)
        if turn_index < 0:
            raise ProxyRequestError("field 'turn_index' must be non-negative.")
        if credential_type not in _SUPPORTED_TEST_SEED_CREDENTIAL_TYPES:
            supported_types = ", ".join(_SUPPORTED_TEST_SEED_CREDENTIAL_TYPES)
            raise ProxyRequestError(
                f"unsupported credential_type '{credential_type}'. Supported types: {supported_types}."
            )

        existing_record = _existing_seeded_canary(
            records=self._seeded_canaries_by_session_id.get(session_id, ()),
            slot_name=slot_name,
        )
        if existing_record is not None:
            if existing_record.credential_type != credential_type:
                raise ProxyRequestError(
                    "a canary is already seeded for this session_id and slot_name with a different credential_type."
                )
            return {
                "schema_version": _TEST_SEED_CANARY_SCHEMA_VERSION,
                "status": "seeded",
                "created": False,
                "session_id": session_id,
                "canary": _canary_record_public_summary(existing_record),
                "mock_response_modes": list(sorted(SUPPORTED_MOCK_RESPONSE_MODES)),
            }
        record = _seeded_canary_record(
            session_id=session_id,
            slot_name=slot_name,
            credential_type=credential_type,
            turn_index=turn_index,
        )
        seeded_records = _merge_canary_records(
            self._seeded_canaries_by_session_id.get(session_id, ()),
            (record,),
        )
        self._seeded_canaries_by_session_id[session_id] = seeded_records
        self._nimbus_critic.register_canary_records(session_id=session_id, records=(record,))
        return {
            "schema_version": _TEST_SEED_CANARY_SCHEMA_VERSION,
            "status": "seeded",
            "created": True,
            "session_id": session_id,
            "canary": _canary_record_public_summary(record),
            "mock_response_modes": list(sorted(SUPPORTED_MOCK_RESPONSE_MODES)),
        }

    def _handle_chat_completions(self, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
        proxy_request = _runtime_request_from_chat_body(
            body=body,
            provider_name=self._provider_name,
            mock_controls_enabled=self._mock_controls_enabled,
            seeded_canary_records_by_session_id=self._seeded_canaries_by_session_id,
            capability_mode=self._runtime_factory.cift_capability.capability_mode,
        )
        request = proxy_request.runtime_request
        self._nimbus_critic.register_canary_records(
            session_id=request.session_id,
            records=proxy_request.canary_records,
        )
        runtime = self._runtime_factory.build(proxy_request.canary_records)
        response = runtime.evaluate_turn(request)
        return {
            "id": f"chatcmpl-{request.trace_id}",
            "object": "chat.completion",
            "model": request.model.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": _response_output_text(response)},
                    "finish_reason": "stop",
                }
            ],
            "aegis": {
                "schema_version": _AEGIS_RESPONSE_SCHEMA_VERSION,
                "trace_id": request.trace_id,
                "runtime_trace": _runtime_trace(request=request, response=response),
                "policy_decision": response.policy_decision.to_dict(),
                "detector_results": [result.to_dict() for result in response.detector_results],
            },
        }


def _response_output_text(response: AegisRuntimeResponse) -> str:
    final_action = response.policy_decision.final_action
    if final_action in (Action.ALLOW, Action.WARN):
        return response.output_text
    if final_action == Action.SANITIZE and response.policy_decision.sanitized_output is not None:
        return response.policy_decision.sanitized_output
    return _WITHHELD_OUTPUT_TEXT


def _runtime_request_from_chat_body(
    body: dict[str, JsonValue],
    provider_name: str,
    mock_controls_enabled: bool,
    seeded_canary_records_by_session_id: Mapping[str, tuple[CanaryRecord, ...]],
    capability_mode: CapabilityMode,
) -> ProxyRuntimeRequest:
    model = body.get("model")
    if not isinstance(model, str) or model == "":
        raise ProxyRequestError("field 'model' must be a non-empty string.")

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) == 0:
        raise ProxyRequestError("field 'messages' must be a non-empty list.")

    messages = tuple(_message_from_raw(item) for item in raw_messages)
    metadata = _metadata_from_raw(body.get("metadata"))
    _validate_provider_metadata(metadata=metadata, mock_controls_enabled=mock_controls_enabled)
    trace_id = _metadata_string(metadata, "trace_id", f"trace-{uuid4().hex}")
    session_id = _metadata_string(metadata, "session_id", f"session-{uuid4().hex}")
    turn_index = _metadata_int(metadata, "turn_index", 1)
    if turn_index < 0:
        raise ProxyRequestError("metadata field 'turn_index' must be non-negative.")

    injection = inject_honeytokens(
        messages=messages,
        ledger=HoneytokenLedger(
            session_id=session_id,
            generator=default_honeytoken_generator,
            source="dp_honey_dev_proxy",
        ),
        turn_index=turn_index,
    )
    seeded_canary_records = _records_not_present(
        records=seeded_canary_records_by_session_id.get(session_id, ()),
        existing_records=injection.canary_records,
    )
    seeded_messages, seeded_spans = _append_seeded_canary_messages(
        messages=injection.messages,
        canary_records=seeded_canary_records,
    )
    canary_records = _merge_canary_records(injection.canary_records, seeded_canary_records)
    raw_credential_spans = _raw_credential_spans(
        messages=seeded_messages,
        canary_records=canary_records,
    )
    metadata = _metadata_with_dp_honey_summary(metadata, injection.canary_records)
    metadata = _metadata_with_test_seed_summary(metadata, seeded_canary_records)

    return ProxyRuntimeRequest(
        runtime_request=RuntimeRequest(
            trace_id=trace_id,
            session_id=session_id,
            turn_index=turn_index,
            capability_mode=capability_mode,
            model=ModelInfo(provider=provider_name, model_id=model, revision=None, selected_device=None),
            messages=seeded_messages,
            tool_calls=(),
            sensitive_spans=injection.sensitive_spans + seeded_spans + raw_credential_spans,
            metadata=metadata,
        ),
        canary_records=canary_records,
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
    _validate_no_reserved_metadata_keys(metadata)
    return metadata


def proxy_error_payload(code: str, message: str, details: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "error": {
            "schema_version": _PROXY_ERROR_SCHEMA_VERSION,
            "code": code,
            "message": message,
            "details": details,
        }
    }


def _validate_test_seed_canary_body(body: dict[str, JsonValue]) -> None:
    extra_fields = tuple(sorted(key for key in body if key not in _TEST_SEED_CANARY_FIELDS))
    if len(extra_fields) > 0:
        joined_fields = ", ".join(extra_fields)
        raise ProxyRequestError(f"unsupported field(s) for test canary seeding: {joined_fields}.")
    paths = _credential_shaped_metadata_paths(body, path="body")
    if len(paths) > 0:
        joined_paths = ", ".join(paths)
        raise ProxyRequestError(f"request body contains credential-shaped value(s) at {joined_paths}.")


def _body_string(body: Mapping[str, JsonValue], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or value == "":
        raise ProxyRequestError(f"field '{key}' must be a non-empty string.")
    return value


def _body_int(body: Mapping[str, JsonValue], key: str, default: int) -> int:
    value = body.get(key)
    if value is None:
        return default
    if not isinstance(value, int):
        raise ProxyRequestError(f"field '{key}' must be an integer.")
    return value


def _existing_seeded_canary(records: tuple[CanaryRecord, ...], slot_name: str) -> CanaryRecord | None:
    for record in records:
        if record.metadata.get("slot_name") == slot_name:
            return record
    return None


def _seeded_canary_record(session_id: str, slot_name: str, credential_type: str, turn_index: int) -> CanaryRecord:
    ledger = HoneytokenLedger(
        session_id=session_id,
        generator=default_honeytoken_generator,
        source=_TEST_SEEDED_CANARY_SOURCE,
    )
    ledger.plant(slot_name=slot_name, credential_type=credential_type, turn_index=turn_index)
    records = ledger.canary_records()
    if len(records) != 1:
        raise ProxyRequestError("test canary seeding failed to create exactly one canary.")
    return records[0]


def _canary_record_public_summary(record: CanaryRecord) -> dict[str, JsonValue]:
    return {
        "canary_id": record.canary_id,
        "slot_name": _canary_slot_name(record),
        "credential_type": record.credential_type,
        "sha256": record.sha256,
        "source": record.source,
        "metadata": {"slot_name": _canary_slot_name(record), "turn_planted": record.metadata.get("turn_planted", 0)},
    }


def _records_not_present(
    records: tuple[CanaryRecord, ...],
    existing_records: tuple[CanaryRecord, ...],
) -> tuple[CanaryRecord, ...]:
    existing_ids = frozenset(record.canary_id for record in existing_records)
    return tuple(record for record in records if record.canary_id not in existing_ids)


def _merge_canary_records(
    left_records: tuple[CanaryRecord, ...],
    right_records: tuple[CanaryRecord, ...],
) -> tuple[CanaryRecord, ...]:
    records_by_id = {record.canary_id: record for record in left_records}
    for record in right_records:
        records_by_id[record.canary_id] = record
    return tuple(records_by_id.values())


def _append_seeded_canary_messages(
    messages: tuple[Message, ...],
    canary_records: tuple[CanaryRecord, ...],
) -> tuple[tuple[Message, ...], tuple[SensitiveSpan, ...]]:
    if len(canary_records) == 0:
        return messages, ()
    appended_messages = list(messages)
    spans: list[SensitiveSpan] = []
    for record in canary_records:
        prefix = "Aegis test seed canary: "
        content = f"{prefix}{record.value}"
        message_index = len(appended_messages)
        appended_messages.append(Message(role="system", content=content))
        spans.append(
            SensitiveSpan(
                kind="honeytoken",
                source=record.source,
                char_start=len(prefix),
                char_end=len(content),
                token_start=None,
                token_end=None,
                identifier=record.canary_id,
                metadata={
                    "slot_name": _canary_slot_name(record),
                    "credential_type": record.credential_type,
                    "sha256": record.sha256,
                    "turn_planted": _canary_turn_planted(record),
                    "message_index": message_index,
                    "audit_redact": True,
                },
            )
        )
    return tuple(appended_messages), tuple(spans)


def _canary_slot_name(record: CanaryRecord) -> str:
    slot_name = record.metadata.get("slot_name")
    if not isinstance(slot_name, str) or slot_name == "":
        raise ProxyRequestError(f"canary record '{record.canary_id}' is missing metadata.slot_name.")
    return slot_name


def _canary_turn_planted(record: CanaryRecord) -> int:
    turn_planted = record.metadata.get("turn_planted")
    if isinstance(turn_planted, int) and turn_planted >= 0:
        return turn_planted
    return 0


def _validate_no_raw_credentials_in_metadata(metadata: dict[str, JsonValue]) -> None:
    paths = _credential_shaped_metadata_paths(metadata, path="metadata")
    if len(paths) > 0:
        joined_paths = ", ".join(paths)
        raise ProxyRequestError(f"metadata contains credential-shaped value(s) at {joined_paths}.")


def _validate_no_reserved_metadata_keys(metadata: dict[str, JsonValue]) -> None:
    paths = _reserved_metadata_paths(metadata, path="metadata")
    if len(paths) > 0:
        joined_paths = ", ".join(paths)
        raise ProxyRequestError(f"metadata contains Aegis-reserved key(s) at {joined_paths}.")


def _reserved_metadata_paths(value: JsonValue, path: str) -> tuple[str, ...]:
    if isinstance(value, list):
        paths: list[str] = []
        for index, item in enumerate(value):
            paths.extend(_reserved_metadata_paths(item, path=f"{path}[{index}]"))
        return tuple(paths)
    if not isinstance(value, dict):
        return ()
    paths = []
    for key, item in value.items():
        key_path = f"{path}.{key}"
        if any(key.startswith(prefix) for prefix in _RESERVED_METADATA_PREFIXES):
            paths.append(key_path)
        paths.extend(_reserved_metadata_paths(item, path=key_path))
    return tuple(paths)


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


def _validate_provider_metadata(metadata: Mapping[str, JsonValue], mock_controls_enabled: bool) -> None:
    if not mock_controls_enabled:
        _reject_mock_controls(metadata)
        return
    _validate_mock_response_mode(metadata)


def _reject_mock_controls(metadata: Mapping[str, JsonValue]) -> None:
    blocked_keys = tuple(key for key in ("mock_response", "mock_response_mode") if key in metadata)
    if len(blocked_keys) == 0:
        return
    joined_keys = ", ".join(blocked_keys)
    raise ProxyRequestError(f"metadata field(s) {joined_keys} are only supported by the mock provider.")


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


def _metadata_with_test_seed_summary(
    metadata: dict[str, JsonValue],
    canary_records: tuple[CanaryRecord, ...],
) -> dict[str, JsonValue]:
    if len(canary_records) == 0:
        return metadata
    updated = dict(metadata)
    updated["test_seed_canary_count"] = len(canary_records)
    updated["test_seed_canary_ids"] = [record.canary_id for record in canary_records]
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


def _metadata_positive_int(metadata: Mapping[str, JsonValue], key: str, default: int) -> int:
    value = _metadata_int(metadata, key, default)
    if value <= 0:
        raise ProxyRequestError(f"metadata field '{key}' must be positive.")
    return value


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


def _cift_capabilities(capability: ProxyCiftCapability) -> dict[str, JsonValue]:
    return {
        "capability_mode": capability.capability_mode.value,
        "detectors": list(capability.detector_names),
        "turn_annotator_count": len(capability.turn_annotators),
    }


def create_proxy(
    provider_config: ProxyProviderConfig,
    nimbus_config: ProxyNimbusConfig,
    cift_capability: ProxyCiftCapability,
) -> MockProxyApp:
    audit_sink = InMemoryAuditSink()
    nimbus_critic = CanaryNimbusCritic(
        CanaryNimbusCriticConfig(
            exact_match_leakage_bits=nimbus_config.exact_match_leakage_bits,
            encoded_match_leakage_bits=nimbus_config.encoded_match_leakage_bits,
            partial_match_leakage_bits=nimbus_config.partial_match_leakage_bits,
            partial_match_threshold=nimbus_config.partial_match_threshold,
            confidence=nimbus_config.confidence,
        )
    )
    nimbus_detector = NimbusDetector(
        NimbusConfig(
            budget_bits=nimbus_config.budget_bits,
            warn_threshold=nimbus_config.warn_threshold,
            sanitize_threshold=nimbus_config.sanitize_threshold,
            block_threshold=nimbus_config.block_threshold,
            max_turns=nimbus_config.max_turns,
            critic_version=nimbus_config.critic_version,
        ),
        nimbus_critic,
        InMemoryNimbusStateStore(max_turns=nimbus_config.max_turns),
    )
    runtime_factory = ProxyRuntimeFactory(
        audit_sink=audit_sink,
        nimbus_detector=nimbus_detector,
        cift_capability=cift_capability,
        model_provider=provider_config.model_provider,
    )
    return MockProxyApp(
        audit_sink=audit_sink,
        nimbus_detector=nimbus_detector,
        nimbus_critic=nimbus_critic,
        runtime_factory=runtime_factory,
        provider_name=provider_config.provider_name,
        mock_controls_enabled=provider_config.mock_controls_enabled,
        nimbus_config=nimbus_config,
    )


def create_default_proxy() -> MockProxyApp:
    return create_proxy(
        provider_config=provider_config_from_env(),
        nimbus_config=nimbus_config_from_env(),
        cift_capability=black_box_cift_capability(),
    )
