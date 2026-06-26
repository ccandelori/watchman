from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from uuid import uuid4

from aegis.audit.explain import explain_audit_record
from aegis.audit.memory import InMemoryAuditSink
from aegis.canaries.dp_honey import build_dp_honey_ledger
from aegis.canaries.ledger import (
    HoneytokenLedger,
    default_honeytoken_generator,
    inject_honeytokens,
    inject_honeytokens_into_tool_calls,
)
from aegis.cift_contract import (
    CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
    CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
)
from aegis.core.contracts import (
    Action,
    CapabilityMode,
    DetectorComponent,
    DetectorResult,
    JsonValue,
    Message,
    ModelInfo,
    NormalizedTurn,
    SensitiveSpan,
    ToolCall,
)
from aegis.core.orchestrator import AegisRuntimeResponse, RuntimeRequest
from aegis.detectors.canary import (
    CanaryRecord,
    canary_sha256,
)
from aegis.detectors.cift_runtime import (
    CiftFeatureExtractionExtractor,
    CiftFeatureExtractor,
    CiftRuntimeDetectorError,
    load_cift_runtime_model,
    load_cift_runtime_model_with_sha256,
    validate_cift_gateway_smoke_bootstrap_runtime_model,
)
from aegis.detectors.nimbus import (
    CanaryNimbusCritic,
    CanaryNimbusCriticConfig,
    InMemoryNimbusStateStore,
    LearnedInfoNCENimbusCritic,
    NimbusConfig,
    NimbusDetector,
    NimbusToolEgressDetector,
    RegisteredCanaryNimbusCritic,
)
from aegis.providers.mock import SUPPORTED_MOCK_RESPONSE_MODES
from aegis.providers.openai_compatible import OpenAICompatibleProviderError
from aegis.proxy.cift_certification import (
    CiftCertificationBindingConfig,
    CiftCertificationBindingError,
    validate_cift_certification_binding,
)
from aegis.proxy.cift_extractor_client import (
    CiftExpectedModelAttestation,
    CiftExtractorSender,
    CiftHttpExtractorConfig,
    CiftHttpFeatureExtractor,
    urllib_cift_extractor_sender,
)
from aegis.proxy.config import (
    CiftCertificationMode,
    ProviderKind,
    ProxyCiftConfig,
    ProxyConfigError,
    ProxyNimbusConfig,
    ProxyProviderConfig,
    audit_sink_from_env,
    cift_config_from_env,
    nimbus_config_from_env,
    provider_config_from_env,
)
from aegis.proxy.nimbus_profile import NimbusCriticKind, nimbus_capabilities
from aegis.proxy.runtime_factory import (
    ProxyCiftCapability,
    ProxyCiftRuntimeBinding,
    ProxyRuntimeFactory,
    cift_capability_from_config,
)
from aegis.replay.nimbus_infonce import load_nimbus_infonce_model, nimbus_infonce_model_sha256

_RAW_CREDENTIAL_PATTERN = re.compile(r"(?:AKIA|ghp_|ya29\.|sk_live_|sk-|hny_)[A-Za-z0-9._-]{8,}")
_CREDENTIAL_PLACEHOLDER_PATTERN = re.compile(r"\{\{CREDENTIAL:([^:}]+):([^}]+)\}\}")
_ENV_CREDENTIAL_NAME_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|ACCESS_KEY|PRIVATE_KEY)[A-Z0-9_]*)\b"
)
_SAFE_SLOT_REFERENCE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_:-]{1,80}$")
_TRAILING_CREDENTIAL_PUNCTUATION = ".,;:"
_PROXY_ERROR_SCHEMA_VERSION = "aegis.proxy_error/v1"
_CAPABILITIES_SCHEMA_VERSION = "aegis.proxy_capabilities/v1"
_READINESS_SCHEMA_VERSION = "aegis.proxy_readiness/v1"
_AEGIS_RESPONSE_SCHEMA_VERSION = "aegis.chat_response/v1"
_TEST_SEED_CANARY_SCHEMA_VERSION = "aegis.test_seed_canary/v1"
_CIFT_SUPPORT_TIER_UNSUPPORTED = "unsupported"
_CIFT_SUPPORT_TIER_CERTIFIED = "certified"
_CIFT_SUPPORT_TIER_CALIBRATION_READY = "calibration-ready"
_CIFT_SUPPORT_TIER_RUNTIME_ENFORCEABLE = "runtime-enforceable"
_CIFT_UNSUPPORTED_SUPPORT_SCOPE = "model-specific CIFT enforcement unavailable"
_CIFT_BLACK_BOX_SUPPORT_REASON = (
    "black-box provider mode has no certified hidden-state extractor binding; "
    "DP-HONEY, NIMBUS, and provider egress remain available."
)
_RESERVED_METADATA_KEYS = frozenset(("cift", "secret_context_handle"))
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
_CREDENTIAL_SLOT_FIELDS = frozenset(("slot_name", "credential_type", "required"))
_CREDENTIAL_CONTEXT_SOURCES = frozenset(("metadata.credential_slots", "tool_schema"))
_SAFE_SECRET_FIELD_SENTINELS = frozenset(
    (
        "",
        "credential",
        "credential_handle",
        "credential_ref",
        "handle",
        "redacted",
        "masked",
        "use_handle",
    )
)
_SECRET_LIKE_FIELD_TOKENS = (
    "api_key",
    "apikey",
    "access_key",
    "secret_access_key",
    "secret",
    "token",
    "auth_token",
    "authorization",
    "password",
    "private_key",
)
_WITHHELD_OUTPUT_TEXT = "[aegis output withheld]"


class ProxyRequestError(ValueError):
    """Raised when a mock proxy request cannot be normalized."""


class ProxyRequestEvidenceError(ProxyRequestError):
    """Raised when a rejected request has safe structured evidence."""

    def __init__(self, message: str, details: dict[str, JsonValue]) -> None:
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class ProxyRuntimeRequest:
    runtime_request: RuntimeRequest
    canary_records: tuple[CanaryRecord, ...]


@dataclass(frozen=True)
class CredentialSlotDeclaration:
    slot_name: str
    credential_type: str
    source: str

    def key(self) -> tuple[str, str]:
        return (self.slot_name, self.credential_type)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "slot_name": self.slot_name,
            "credential_type": self.credential_type,
            "source": self.source,
        }


@dataclass(frozen=True)
class CiftReadinessProbe:
    capability: ProxyCiftCapability
    extractor: CiftFeatureExtractor | None

    def check(self) -> dict[str, JsonValue]:
        if self.capability.capability_mode == CapabilityMode.BLACK_BOX:
            return {
                "ready": True,
                "status": "not_required",
                "capability_mode": self.capability.capability_mode.value,
                **_cift_black_box_support(),
            }
        binding = self.capability.runtime_binding
        if binding is None:
            return {
                "ready": False,
                "status": "uncertified",
                "capability_mode": self.capability.capability_mode.value,
                **_cift_unbound_support(),
                "error": "self-hosted CIFT readiness requires a strict certification runtime binding.",
            }
        base = {
            **_cift_readiness_binding_summary(binding),
            **_cift_readiness_binding_pending_support(binding),
        }
        if self.extractor is None:
            return {
                **base,
                "ready": False,
                "status": "extractor_not_configured",
                "error": "strict CIFT readiness requires a configured trusted extractor.",
            }
        if not isinstance(self.extractor, CiftFeatureExtractionExtractor):
            return {
                **base,
                "ready": False,
                "status": "extractor_not_probeable",
                "error": "strict CIFT readiness requires an extractor with atomic feature extraction provenance.",
            }
        try:
            extraction = self.extractor.extract_feature_extraction(
                turn=_cift_readiness_turn(binding),
                feature_key=binding.feature_key,
            )
        except Exception as exc:
            return {
                **base,
                "ready": False,
                "status": "extractor_error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        if extraction.feature_vector is None:
            return {
                **base,
                "ready": False,
                "status": "feature_unavailable",
                "error": "strict CIFT readiness requires a live feature vector.",
            }
        observed_feature_count = len(extraction.feature_vector)
        if observed_feature_count != binding.feature_count:
            return {
                **base,
                "ready": False,
                "status": "feature_count_mismatch",
                "error": "strict CIFT readiness feature vector length does not match the certified runtime.",
                "observed_feature_count": observed_feature_count,
            }
        if extraction.selected_choice_readout_token_indices is None:
            return {
                **base,
                "ready": False,
                "status": "selected_choice_metadata_absent",
                "error": "strict CIFT readiness requires selected-choice readout metadata.",
            }
        observed_readout_count = len(extraction.selected_choice_readout_token_indices)
        if observed_readout_count != binding.selected_choice_readout_token_count:
            return {
                **base,
                "ready": False,
                "status": "selected_choice_metadata_mismatch",
                "error": "strict CIFT readiness selected-choice readout count does not match the certified binding.",
                "observed_selected_choice_readout_token_count": observed_readout_count,
            }
        return {
            **base,
            **_cift_readiness_binding_ready_support(binding),
            "ready": True,
            "status": "ready",
            "feature_vector_length": observed_feature_count,
            "observed_selected_choice_readout_token_count": observed_readout_count,
            "extractor": _cift_readiness_extractor_summary(extraction.provenance),
        }


class MockProxyApp:
    def __init__(
        self,
        audit_sink: InMemoryAuditSink,
        nimbus_detector: NimbusDetector,
        nimbus_critic: RegisteredCanaryNimbusCritic,
        runtime_factory: ProxyRuntimeFactory,
        provider_name: str,
        mock_controls_enabled: bool,
        nimbus_config: ProxyNimbusConfig,
        cift_readiness_probe: CiftReadinessProbe | None = None,
    ) -> None:
        self._audit_sink = audit_sink
        self._nimbus_detector = nimbus_detector
        self._nimbus_critic = nimbus_critic
        self._runtime_factory = runtime_factory
        self._provider_name = provider_name
        self._mock_controls_enabled = mock_controls_enabled
        self._nimbus_config = nimbus_config
        self._cift_readiness_probe = (
            cift_readiness_probe
            if cift_readiness_probe is not None
            else CiftReadinessProbe(capability=runtime_factory.cift_capability, extractor=None)
        )
        self._seeded_canaries_by_session_id: dict[str, tuple[CanaryRecord, ...]] = {}

    def handle(self, method: str, path: str, body: dict[str, JsonValue]) -> tuple[int, dict[str, JsonValue]]:
        if method == "GET" and path == "/health":
            return 200, {"status": "ok"}
        if method == "GET" and path == "/ready":
            payload = self._readiness()
            status_code = 200 if payload["ready"] is True else 503
            return status_code, payload
        if method == "GET" and path == "/aegis/capabilities":
            return 200, self._capabilities()
        if method == "GET" and path == "/audit/recent":
            try:
                return 200, self._handle_audit_recent(body)
            except ProxyRequestError as exc:
                return 400, proxy_error_payload(
                    code="invalid_request",
                    message=str(exc),
                    details=_proxy_request_error_details(exc),
                )
        if method == "GET" and path == "/audit/explain":
            try:
                explanation = self._handle_audit_explain(body)
            except ProxyRequestError as exc:
                return 400, proxy_error_payload(
                    code="invalid_request",
                    message=str(exc),
                    details=_proxy_request_error_details(exc),
                )
            if explanation is None:
                return 404, proxy_error_payload(
                    code="audit_record_not_found",
                    message="No audit record matched the requested trace_id/session_id.",
                    details={},
                )
            return 200, explanation
        if method == "POST" and path == "/test/reset":
            try:
                return 200, self._handle_test_reset(body)
            except ProxyRequestError as exc:
                return 400, proxy_error_payload(
                    code="invalid_request",
                    message=str(exc),
                    details=_proxy_request_error_details(exc),
                )
        if method == "POST" and path == "/test/seed-canary" and self._mock_controls_enabled:
            try:
                return 200, self._handle_test_seed_canary(body)
            except ProxyRequestError as exc:
                return 400, proxy_error_payload(
                    code="invalid_request",
                    message=str(exc),
                    details=_proxy_request_error_details(exc),
                )
        if method == "POST" and path == "/v1/chat/completions":
            try:
                return 200, self._handle_chat_completions(body)
            except ProxyRequestError as exc:
                return 400, proxy_error_payload(
                    code="invalid_request",
                    message=str(exc),
                    details=_proxy_request_error_details(exc),
                )
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
            {"method": "GET", "path": "/ready"},
            {"method": "GET", "path": "/aegis/capabilities"},
            {"method": "POST", "path": "/v1/chat/completions"},
            {"method": "GET", "path": "/audit/recent"},
            {"method": "GET", "path": "/audit/explain"},
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
            *self._runtime_factory.cift_capability.detector_names,
            "tool_call_canary",
            "nimbus_tool_egress",
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
                "durable_jsonl_enabled": self._audit_sink.durable_path() is not None,
                "durable_jsonl_path": self._audit_sink.durable_path(),
                "explain_route": "/audit/explain",
            },
            "test_controls": test_controls,
        }

    def _readiness(self) -> dict[str, JsonValue]:
        cift = self._cift_readiness_probe.check()
        nimbus = nimbus_capabilities(self._nimbus_config)
        return {
            "schema_version": _READINESS_SCHEMA_VERSION,
            "ready": cift["ready"],
            "status": "ready" if cift["ready"] is True else "not_ready",
            "cift": cift,
            "dp_honey": {
                "ready": True,
                "status": "ready",
                "source": "dp_honey",
                "generator": "detect.dp_honey",
                "registered_canary_substitution": True,
                "provider_valid": False,
                "fail_closed_on_ambiguous_protected_workflow": True,
            },
            "provider_egress_guard": {
                "ready": True,
                "status": "ready",
                "blocks_non_honeytoken_sensitive_spans_before_provider": True,
            },
            "provider": {
                "ready": True,
                "status": "ready",
                "name": self._provider_name,
                "mock_controls_enabled": self._mock_controls_enabled,
            },
            "canary": {
                "ready": True,
                "status": "ready",
                "detectors": ["tool_call_canary", "text_canary", "encoded_canary"],
            },
            "nimbus": {
                **nimbus,
                "ready": True,
            },
            "strict_protected_mode": {
                "enabled": (
                    self._runtime_factory.cift_capability.capability_mode == CapabilityMode.SELF_HOSTED_INTROSPECTION
                ),
                "cift_fails_closed": True,
                "raw_secret_egress_fails_closed": True,
            },
        }

    def _handle_audit_recent(self, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
        limit = _metadata_positive_int(body, "limit", 20)
        session_id = _optional_metadata_string(body, "session_id")
        return {
            "schema_version": "aegis.audit_recent/v1",
            "limit": limit,
            "session_id": session_id,
            "events": list(self._audit_sink.recent_records(limit=limit, session_id=session_id)),
        }

    def _handle_audit_explain(self, body: dict[str, JsonValue]) -> dict[str, JsonValue] | None:
        trace_id = _optional_metadata_string(body, "trace_id")
        session_id = _optional_metadata_string(body, "session_id")
        if trace_id is None and session_id is None:
            raise ProxyRequestError("trace_id or session_id must be provided.")
        record = self._audit_sink.find_record(trace_id=trace_id, session_id=session_id)
        if record is None:
            return None
        try:
            return explain_audit_record(record)
        except ValueError as exc:
            raise ProxyRequestError(str(exc)) from exc

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
    tool_calls = _tool_calls_from_raw(body.get("tool_calls"))
    metadata = _metadata_from_raw(body.get("metadata"))
    _validate_provider_metadata(metadata=metadata, mock_controls_enabled=mock_controls_enabled)
    trace_id = _metadata_string(metadata, "trace_id", f"trace-{uuid4().hex}")
    session_id = _metadata_string(metadata, "session_id", f"session-{uuid4().hex}")
    turn_index = _metadata_int(metadata, "turn_index", 1)
    if turn_index < 0:
        raise ProxyRequestError("metadata field 'turn_index' must be non-negative.")

    tool_calls, inferred_tool_field_slots = _infer_tool_call_credential_fields(tool_calls)
    credential_slots = _credential_slot_declarations(
        metadata=metadata,
        messages=messages,
        tool_calls=tool_calls,
        tool_schemas=body.get("tools"),
        inferred_tool_field_slots=inferred_tool_field_slots,
    )
    messages = messages + _credential_slot_context_messages(credential_slots)
    ledger = build_dp_honey_ledger(session_id=session_id)
    injection = inject_honeytokens(
        messages=messages,
        ledger=ledger,
        turn_index=turn_index,
    )
    tool_injection = inject_honeytokens_into_tool_calls(
        tool_calls=tool_calls,
        ledger=ledger,
        turn_index=turn_index,
    )
    dp_honey_canary_records = _merge_canary_records(injection.canary_records, tool_injection.canary_records)
    seeded_canary_records = _records_not_present(
        records=seeded_canary_records_by_session_id.get(session_id, ()),
        existing_records=dp_honey_canary_records,
    )
    seeded_messages, seeded_spans = _append_seeded_canary_messages(
        messages=injection.messages,
        canary_records=seeded_canary_records,
    )
    canary_records = _merge_canary_records(
        dp_honey_canary_records,
        seeded_canary_records,
    )
    raw_credential_spans = _raw_credential_spans(
        messages=seeded_messages,
        tool_calls=tool_injection.tool_calls,
        canary_records=canary_records,
    )
    metadata = _metadata_with_credential_slot_summary(
        metadata=metadata,
        credential_slots=credential_slots,
        canary_records=dp_honey_canary_records,
        raw_credential_spans=raw_credential_spans,
    )
    metadata = _metadata_with_dp_honey_summary(metadata, dp_honey_canary_records)
    metadata = _metadata_with_test_seed_summary(metadata, seeded_canary_records)

    return ProxyRuntimeRequest(
        runtime_request=RuntimeRequest(
            trace_id=trace_id,
            session_id=session_id,
            turn_index=turn_index,
            capability_mode=capability_mode,
            model=ModelInfo(provider=provider_name, model_id=model, revision=None, selected_device=None),
            messages=seeded_messages,
            tool_calls=tool_injection.tool_calls,
            sensitive_spans=(
                injection.sensitive_spans + tool_injection.sensitive_spans + seeded_spans + raw_credential_spans
            ),
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


def _tool_calls_from_raw(value: object) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ProxyRequestError("field 'tool_calls' must be a list when provided.")
    tool_calls: list[ToolCall] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ProxyRequestError(f"tool_calls[{index}] must be an object.")
        name = item.get("name")
        if not isinstance(name, str) or name == "":
            raise ProxyRequestError(f"tool_calls[{index}].name must be a non-empty string.")
        arguments = item.get("arguments")
        if not isinstance(arguments, dict):
            raise ProxyRequestError(f"tool_calls[{index}].arguments must be an object.")
        tool_calls.append(ToolCall(name=name, arguments=_json_object(arguments, f"tool_calls[{index}].arguments")))
    return tuple(tool_calls)


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


def _credential_slot_declarations(
    metadata: Mapping[str, JsonValue],
    messages: tuple[Message, ...],
    tool_calls: tuple[ToolCall, ...],
    tool_schemas: JsonValue,
    inferred_tool_field_slots: tuple[CredentialSlotDeclaration, ...],
) -> tuple[CredentialSlotDeclaration, ...]:
    declarations = _metadata_credential_slot_declarations(metadata)
    declarations += _message_credential_slot_declarations(messages)
    declarations += _message_env_credential_slot_declarations(messages)
    declarations += _tool_call_credential_slot_declarations(tool_calls)
    declarations += _tool_schema_credential_slot_declarations(tool_schemas)
    declarations += inferred_tool_field_slots
    protected_workflow = metadata.get("protected_workflow")
    if protected_workflow is not None and not isinstance(protected_workflow, bool):
        raise ProxyRequestError("metadata field 'protected_workflow' must be a boolean when provided.")
    if protected_workflow is True and len(declarations) == 0:
        raise ProxyRequestEvidenceError(
            (
                "protected_workflow=true requires at least one credential slot declaration "
                "or deterministic credential reference."
            ),
            _ambiguous_protected_workflow_details(),
        )
    return _dedupe_credential_slot_declarations(declarations)


def _metadata_credential_slot_declarations(metadata: Mapping[str, JsonValue]) -> tuple[CredentialSlotDeclaration, ...]:
    value = metadata.get("credential_slots")
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ProxyRequestError("metadata field 'credential_slots' must be a list when provided.")
    declarations: list[CredentialSlotDeclaration] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ProxyRequestError(f"metadata.credential_slots[{index}] must be an object.")
        extra_fields = tuple(sorted(key for key in item if key not in _CREDENTIAL_SLOT_FIELDS))
        if len(extra_fields) > 0:
            joined_fields = ", ".join(extra_fields)
            raise ProxyRequestError(f"unsupported metadata.credential_slots[{index}] field(s): {joined_fields}.")
        required = item.get("required")
        if required is not None and not isinstance(required, bool):
            raise ProxyRequestError(f"metadata.credential_slots[{index}].required must be a boolean when provided.")
        if required is False:
            continue
        declarations.append(
            CredentialSlotDeclaration(
                slot_name=_object_string(item, "slot_name", f"metadata.credential_slots[{index}]"),
                credential_type=_object_string(item, "credential_type", f"metadata.credential_slots[{index}]"),
                source="metadata.credential_slots",
            )
        )
    return tuple(declarations)


def _message_credential_slot_declarations(messages: tuple[Message, ...]) -> tuple[CredentialSlotDeclaration, ...]:
    declarations: list[CredentialSlotDeclaration] = []
    for message_index, message in enumerate(messages):
        for match in _CREDENTIAL_PLACEHOLDER_PATTERN.finditer(message.content):
            declarations.append(
                CredentialSlotDeclaration(
                    slot_name=match.group(1),
                    credential_type=match.group(2),
                    source=f"message[{message_index}]",
                )
            )
    return tuple(declarations)


def _message_env_credential_slot_declarations(messages: tuple[Message, ...]) -> tuple[CredentialSlotDeclaration, ...]:
    declarations: list[CredentialSlotDeclaration] = []
    for message_index, message in enumerate(messages):
        for match in _ENV_CREDENTIAL_NAME_PATTERN.finditer(message.content):
            slot_name = match.group(1).lower()
            declarations.append(
                CredentialSlotDeclaration(
                    slot_name=slot_name,
                    credential_type=_credential_type_for_secret_name(slot_name),
                    source=f"message_env_field[{message_index}]",
                )
            )
    return tuple(declarations)


def _tool_call_credential_slot_declarations(tool_calls: tuple[ToolCall, ...]) -> tuple[CredentialSlotDeclaration, ...]:
    declarations: list[CredentialSlotDeclaration] = []
    for index, tool_call in enumerate(tool_calls):
        declarations.extend(
            _tool_argument_credential_slot_declarations(
                value=tool_call.arguments,
                source=f"tool_calls[{index}].arguments",
            )
        )
    return tuple(declarations)


def _tool_argument_credential_slot_declarations(value: JsonValue, source: str) -> tuple[CredentialSlotDeclaration, ...]:
    if isinstance(value, str):
        return tuple(
            CredentialSlotDeclaration(
                slot_name=match.group(1),
                credential_type=match.group(2),
                source=source,
            )
            for match in _CREDENTIAL_PLACEHOLDER_PATTERN.finditer(value)
        )
    if isinstance(value, list):
        declarations: list[CredentialSlotDeclaration] = []
        for index, item in enumerate(value):
            declarations.extend(_tool_argument_credential_slot_declarations(item, source=f"{source}[{index}]"))
        return tuple(declarations)
    if isinstance(value, dict):
        declarations = []
        for key, item in value.items():
            declarations.extend(_tool_argument_credential_slot_declarations(item, source=f"{source}.{key}"))
        return tuple(declarations)
    return ()


def _tool_schema_credential_slot_declarations(value: object) -> tuple[CredentialSlotDeclaration, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ProxyRequestError("field 'tools' must be a list when provided.")
    declarations: list[CredentialSlotDeclaration] = []
    for tool_index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ProxyRequestError(f"tools[{tool_index}] must be an object.")
        declarations.extend(_schema_declared_slots(item.get("credential_slots"), f"tools[{tool_index}]"))
        function = item.get("function")
        if isinstance(function, dict):
            declarations.extend(
                _schema_declared_slots(function.get("credential_slots"), f"tools[{tool_index}].function")
            )
            declarations.extend(
                _schema_property_credential_slots(
                    schema=function.get("parameters"),
                    context=f"tools[{tool_index}].function.parameters",
                )
            )
            continue
        declarations.extend(
            _schema_property_credential_slots(
                schema=item.get("parameters"),
                context=f"tools[{tool_index}].parameters",
            )
        )
    return tuple(declarations)


def _schema_declared_slots(value: object, context: str) -> tuple[CredentialSlotDeclaration, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ProxyRequestError(f"{context}.credential_slots must be a list when provided.")
    declarations: list[CredentialSlotDeclaration] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ProxyRequestError(f"{context}.credential_slots[{index}] must be an object.")
        declarations.append(
            CredentialSlotDeclaration(
                slot_name=_object_string(item, "slot_name", f"{context}.credential_slots[{index}]"),
                credential_type=_object_string(item, "credential_type", f"{context}.credential_slots[{index}]"),
                source="tool_schema",
            )
        )
    return tuple(declarations)


def _schema_property_credential_slots(schema: object, context: str) -> tuple[CredentialSlotDeclaration, ...]:
    if schema is None:
        return ()
    if not isinstance(schema, dict):
        raise ProxyRequestError(f"{context} must be an object when provided.")
    properties = schema.get("properties")
    if properties is None:
        return ()
    if not isinstance(properties, dict):
        raise ProxyRequestError(f"{context}.properties must be an object when provided.")
    declarations: list[CredentialSlotDeclaration] = []
    for property_name, property_schema in properties.items():
        if not isinstance(property_name, str) or property_name == "":
            raise ProxyRequestError(f"{context}.properties keys must be non-empty strings.")
        if not isinstance(property_schema, dict):
            raise ProxyRequestError(f"{context}.properties.{property_name} must be an object.")
        credential_type = _schema_property_credential_type(property_name=property_name, schema=property_schema)
        if credential_type is None:
            continue
        declarations.append(
            CredentialSlotDeclaration(
                slot_name=_schema_property_slot_name(property_name=property_name, schema=property_schema),
                credential_type=credential_type,
                source="tool_schema",
            )
        )
    return tuple(declarations)


def _schema_property_credential_type(property_name: str, schema: Mapping[str, object]) -> str | None:
    explicit = schema.get("x-aegis-credential-type")
    if isinstance(explicit, str) and explicit != "":
        return explicit
    credential_required = schema.get("x-aegis-credential")
    if credential_required is True:
        return _credential_type_for_secret_name(property_name)
    if _is_secret_like_name(property_name):
        return _credential_type_for_secret_name(property_name)
    return None


def _schema_property_slot_name(property_name: str, schema: Mapping[str, object]) -> str:
    explicit = schema.get("x-aegis-credential-slot")
    if isinstance(explicit, str) and explicit != "":
        return explicit
    return property_name


def _infer_tool_call_credential_fields(
    tool_calls: tuple[ToolCall, ...],
) -> tuple[tuple[ToolCall, ...], tuple[CredentialSlotDeclaration, ...]]:
    updated_tool_calls: list[ToolCall] = []
    declarations: list[CredentialSlotDeclaration] = []
    for index, tool_call in enumerate(tool_calls):
        updated_arguments, argument_declarations = _infer_tool_argument_credential_fields(
            value=tool_call.arguments,
            source=f"tool_calls[{index}].arguments",
        )
        if not isinstance(updated_arguments, dict):
            raise ProxyRequestError(f"tool_calls[{index}].arguments must remain an object after credential inference.")
        updated_tool_calls.append(ToolCall(name=tool_call.name, arguments=updated_arguments))
        declarations.extend(argument_declarations)
    return tuple(updated_tool_calls), tuple(declarations)


def _infer_tool_argument_credential_fields(
    value: JsonValue,
    source: str,
) -> tuple[JsonValue, tuple[CredentialSlotDeclaration, ...]]:
    if isinstance(value, list):
        updated_items: list[JsonValue] = []
        declarations: list[CredentialSlotDeclaration] = []
        for index, item in enumerate(value):
            updated_item, item_declarations = _infer_tool_argument_credential_fields(item, f"{source}[{index}]")
            updated_items.append(updated_item)
            declarations.extend(item_declarations)
        return updated_items, tuple(declarations)
    if not isinstance(value, dict):
        return value, ()

    updated: dict[str, JsonValue] = {}
    declarations = []
    for key, item in value.items():
        nested_source = f"{source}.{key}"
        inferred = _inferred_secret_like_field_declaration(key=key, value=item, source=nested_source)
        if inferred is not None:
            declarations.append(inferred)
            updated[key] = f"{{{{CREDENTIAL:{inferred.slot_name}:{inferred.credential_type}}}}}"
            continue
        updated_item, item_declarations = _infer_tool_argument_credential_fields(item, nested_source)
        updated[key] = updated_item
        declarations.extend(item_declarations)
    return updated, tuple(declarations)


def _inferred_secret_like_field_declaration(
    key: str,
    value: JsonValue,
    source: str,
) -> CredentialSlotDeclaration | None:
    if not _is_secret_like_name(key):
        return None
    if isinstance(value, str) and _RAW_CREDENTIAL_PATTERN.search(value):
        return None
    if not _is_safe_credential_placeholder_value(value):
        return None
    slot_name = _slot_name_for_secret_like_field(key=key, value=value)
    return CredentialSlotDeclaration(
        slot_name=slot_name,
        credential_type=_credential_type_for_secret_name(key),
        source=source,
    )


def _is_safe_credential_placeholder_value(value: JsonValue) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    if normalized in _SAFE_SECRET_FIELD_SENTINELS:
        return True
    return _SAFE_SLOT_REFERENCE_PATTERN.fullmatch(normalized) is not None


def _slot_name_for_secret_like_field(key: str, value: JsonValue) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if (
            normalized not in _SAFE_SECRET_FIELD_SENTINELS
            and _SAFE_SLOT_REFERENCE_PATTERN.fullmatch(normalized) is not None
        ):
            return normalized
    return key


def _is_secret_like_name(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    return any(token in normalized for token in _SECRET_LIKE_FIELD_TOKENS)


def _credential_type_for_secret_name(value: str) -> str:
    normalized = value.lower()
    if "github" in normalized or "ghp" in normalized:
        return "github_pat"
    if "aws" in normalized or "access_key" in normalized:
        return "aws_access_key"
    if "oauth" in normalized:
        return "oauth_token"
    if "stripe" in normalized:
        return "stripe_key"
    if "openai" in normalized:
        return "openai_key"
    if "webhook" in normalized:
        return "webhook_secret"
    if "database" in normalized or "db_" in normalized:
        return "database_uri"
    if "twilio" in normalized or "sms" in normalized:
        return "twilio_token"
    if "sendgrid" in normalized or "email" in normalized:
        return "sendgrid_key"
    return "generic_api_key"


def _dedupe_credential_slot_declarations(
    declarations: tuple[CredentialSlotDeclaration, ...],
) -> tuple[CredentialSlotDeclaration, ...]:
    declarations_by_key: dict[tuple[str, str], CredentialSlotDeclaration] = {}
    for declaration in declarations:
        if declaration.slot_name == "" or declaration.credential_type == "":
            raise ProxyRequestError("credential slot declarations must not contain empty strings.")
        declarations_by_key.setdefault(declaration.key(), declaration)
    return tuple(declarations_by_key.values())


def _credential_slot_context_messages(
    declarations: tuple[CredentialSlotDeclaration, ...],
) -> tuple[Message, ...]:
    messages: list[Message] = []
    for declaration in declarations:
        if not _credential_declaration_requires_context_message(declaration.source):
            continue
        messages.append(
            Message(
                role="system",
                content=(
                    f"Aegis protected credential slot {declaration.slot_name}: "
                    f"{{{{CREDENTIAL:{declaration.slot_name}:{declaration.credential_type}}}}}. "
                    "Use this canary only inside authorized credential boundaries."
                ),
            )
        )
    return tuple(messages)


def _credential_declaration_requires_context_message(source: str) -> bool:
    return source in _CREDENTIAL_CONTEXT_SOURCES or source.startswith("message_env_field[")


def proxy_error_payload(code: str, message: str, details: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "error": {
            "schema_version": _PROXY_ERROR_SCHEMA_VERSION,
            "code": code,
            "message": message,
            "details": details,
        }
    }


def _proxy_request_error_details(exc: ProxyRequestError) -> dict[str, JsonValue]:
    if isinstance(exc, ProxyRequestEvidenceError):
        return exc.details
    return {}


def _ambiguous_protected_workflow_details() -> dict[str, JsonValue]:
    return {
        "credential_slot_status": "ambiguous_protected_workflow",
        "protected_workflow": True,
        "fail_closed": True,
        "credential_needed_count": 0,
        "honeytoken_substituted_count": 0,
        "real_secret_present_count": 0,
        "accepted_detection_sources": [
            "metadata.credential_slots",
            "message_credential_placeholder",
            "message_env_field",
            "tool_call_credential_placeholder",
            "tool_schema",
            "tool_call_secret_like_field",
        ],
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


def _object_string(record: Mapping[str, object], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or value == "":
        raise ProxyRequestError(f"{context}.{key} must be a non-empty string.")
    return value


def _json_object(record: Mapping[str, object], context: str) -> dict[str, JsonValue]:
    parsed: dict[str, JsonValue] = {}
    for key, value in record.items():
        if not isinstance(key, str) or key == "":
            raise ProxyRequestError(f"{context} keys must be non-empty strings.")
        parsed[key] = _json_value(value, f"{context}.{key}")
    return parsed


def _json_value(value: object, context: str) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        return _json_object(value, context)
    raise ProxyRequestError(f"{context} must be JSON-serializable.")


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
        if key in _RESERVED_METADATA_KEYS or any(key.startswith(prefix) for prefix in _RESERVED_METADATA_PREFIXES):
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


def _metadata_with_credential_slot_summary(
    metadata: dict[str, JsonValue],
    credential_slots: tuple[CredentialSlotDeclaration, ...],
    canary_records: tuple[CanaryRecord, ...],
    raw_credential_spans: tuple[SensitiveSpan, ...],
) -> dict[str, JsonValue]:
    updated = dict(metadata)
    if len(credential_slots) == 0 and len(canary_records) == 0 and len(raw_credential_spans) == 0:
        updated["aegis_credential_slot_detection"] = {
            "status": "no_credential_path",
            "credential_needed_count": 0,
            "honeytoken_substituted_count": 0,
            "real_secret_present_count": 0,
            "slots": [],
        }
        return updated
    updated["aegis_credential_slot_detection"] = {
        "status": _credential_slot_status(
            credential_slots=credential_slots,
            canary_records=canary_records,
            raw_credential_spans=raw_credential_spans,
        ),
        "credential_needed_count": len(credential_slots),
        "honeytoken_substituted_count": len(canary_records),
        "real_secret_present_count": len(raw_credential_spans),
        "slots": [slot.to_dict() for slot in credential_slots],
        "canary_ids": [record.canary_id for record in canary_records],
    }
    return updated


def _credential_slot_status(
    credential_slots: tuple[CredentialSlotDeclaration, ...],
    canary_records: tuple[CanaryRecord, ...],
    raw_credential_spans: tuple[SensitiveSpan, ...],
) -> str:
    if len(raw_credential_spans) > 0:
        return "real_secret_present"
    if len(canary_records) > 0:
        return "honeytoken_substituted"
    if len(credential_slots) > 0:
        return "credential_needed"
    return "no_credential_path"


def _raw_credential_spans(
    messages: tuple[Message, ...],
    tool_calls: tuple[ToolCall, ...],
    canary_records: tuple[CanaryRecord, ...],
) -> tuple[SensitiveSpan, ...]:
    canary_hashes = frozenset(record.sha256 for record in canary_records)
    spans: list[SensitiveSpan] = []
    for message_index, message in enumerate(messages):
        for match in _RAW_CREDENTIAL_PATTERN.finditer(message.content):
            candidate = _non_canary_credential_match(match=match, canary_hashes=canary_hashes)
            if candidate is None:
                continue
            candidate_sha256, char_end = candidate
            spans.append(
                _raw_credential_span(
                    char_start=match.start(),
                    char_end=char_end,
                    sha256=candidate_sha256,
                    message_index=message_index,
                )
            )
    spans.extend(_raw_credential_tool_spans(tool_calls=tool_calls, canary_hashes=canary_hashes))
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


def _raw_credential_tool_spans(
    tool_calls: tuple[ToolCall, ...],
    canary_hashes: frozenset[str],
) -> tuple[SensitiveSpan, ...]:
    spans: list[SensitiveSpan] = []
    for tool_call in tool_calls:
        spans.extend(
            _raw_credential_tool_spans_for_value(
                value=tool_call.arguments,
                canary_hashes=canary_hashes,
                tool_call_name=tool_call.name,
                argument_path="arguments",
            )
        )
    return tuple(spans)


def _raw_credential_tool_spans_for_value(
    value: JsonValue,
    canary_hashes: frozenset[str],
    tool_call_name: str,
    argument_path: str,
) -> tuple[SensitiveSpan, ...]:
    if isinstance(value, str):
        spans: list[SensitiveSpan] = []
        for match in _RAW_CREDENTIAL_PATTERN.finditer(value):
            candidate = _non_canary_credential_match(match=match, canary_hashes=canary_hashes)
            if candidate is None:
                continue
            candidate_sha256, char_end = candidate
            spans.append(
                SensitiveSpan(
                    kind="credential",
                    source="proxy_raw_credential_scanner",
                    char_start=match.start(),
                    char_end=char_end,
                    token_start=None,
                    token_end=None,
                    identifier=None,
                    metadata={
                        "sha256": candidate_sha256,
                        "tool_call_name": tool_call_name,
                        "argument_path": argument_path,
                    },
                )
            )
        return tuple(spans)
    if isinstance(value, list):
        spans = []
        for index, item in enumerate(value):
            spans.extend(
                _raw_credential_tool_spans_for_value(
                    value=item,
                    canary_hashes=canary_hashes,
                    tool_call_name=tool_call_name,
                    argument_path=f"{argument_path}[{index}]",
                )
            )
        return tuple(spans)
    if isinstance(value, dict):
        spans = []
        for key, item in value.items():
            spans.extend(
                _raw_credential_tool_spans_for_value(
                    value=item,
                    canary_hashes=canary_hashes,
                    tool_call_name=tool_call_name,
                    argument_path=f"{argument_path}.{key}",
                )
            )
        return tuple(spans)
    return ()


def _non_canary_credential_match(match: re.Match[str], canary_hashes: frozenset[str]) -> tuple[str, int] | None:
    candidate = match.group(0)
    char_end = match.end()
    while candidate != "":
        candidate_sha256 = canary_sha256(candidate)
        if candidate_sha256 in canary_hashes:
            return None
        if candidate[-1] not in _TRAILING_CREDENTIAL_PUNCTUATION:
            return candidate_sha256, char_end
        candidate = candidate[:-1]
        char_end -= 1
    return None


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
    stage: dict[str, JsonValue] = {
        "stage": "dp_honey",
        "status": "active" if canary_count > 0 else "not_configured",
        "canary_count": canary_count,
    }
    credential_detection = request.metadata.get("aegis_credential_slot_detection")
    if isinstance(credential_detection, dict):
        status = credential_detection.get("status")
        if isinstance(status, str):
            stage["credential_slot_status"] = status
        credential_needed_count = credential_detection.get("credential_needed_count")
        if isinstance(credential_needed_count, int):
            stage["credential_needed_count"] = credential_needed_count
        real_secret_present_count = credential_detection.get("real_secret_present_count")
        if isinstance(real_secret_present_count, int):
            stage["real_secret_present_count"] = real_secret_present_count
    return stage


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
    capabilities: dict[str, JsonValue] = {
        "capability_mode": capability.capability_mode.value,
        "detectors": list(capability.detector_names),
        **_cift_capability_support(capability),
        "turn_annotator_count": len(capability.turn_annotators),
    }
    if capability.runtime_binding is not None:
        capabilities["runtime_binding"] = _cift_runtime_binding(capability.runtime_binding)
    return capabilities


def _cift_capability_support(capability: ProxyCiftCapability) -> dict[str, JsonValue]:
    binding = capability.runtime_binding
    if binding is None:
        if capability.capability_mode == CapabilityMode.BLACK_BOX:
            return _cift_black_box_support()
        return _cift_unbound_support()
    if binding.certification_mode == CiftCertificationMode.GATEWAY_SMOKE_BOOTSTRAP:
        return _cift_calibration_ready_support(binding)
    return {
        "support_tier": _CIFT_SUPPORT_TIER_RUNTIME_ENFORCEABLE,
        "support_scope": _cift_enforcement_support_scope(binding),
        "support_reason": (
            "strict certification binding is loaded; readiness still depends on trusted extractor attestation."
        ),
    }


def _cift_black_box_support() -> dict[str, JsonValue]:
    return {
        "support_tier": _CIFT_SUPPORT_TIER_UNSUPPORTED,
        "support_scope": _CIFT_UNSUPPORTED_SUPPORT_SCOPE,
        "support_reason": _CIFT_BLACK_BOX_SUPPORT_REASON,
    }


def _cift_unbound_support() -> dict[str, JsonValue]:
    return {
        "support_tier": _CIFT_SUPPORT_TIER_UNSUPPORTED,
        "support_scope": _CIFT_UNSUPPORTED_SUPPORT_SCOPE,
        "support_reason": (
            "self-hosted CIFT has no certified runtime binding; "
            "DP-HONEY, NIMBUS, and provider egress remain available."
        ),
    }


def _cift_readiness_binding_pending_support(binding: ProxyCiftRuntimeBinding) -> dict[str, JsonValue]:
    if binding.certification_mode == CiftCertificationMode.GATEWAY_SMOKE_BOOTSTRAP:
        return _cift_calibration_ready_support(binding)
    return {
        "support_tier": _CIFT_SUPPORT_TIER_CERTIFIED,
        "support_scope": _cift_enforcement_support_scope(binding),
        "support_reason": (
            "strict certification binding is loaded; readiness still depends on trusted extractor attestation."
        ),
    }


def _cift_readiness_binding_ready_support(binding: ProxyCiftRuntimeBinding) -> dict[str, JsonValue]:
    if binding.certification_mode == CiftCertificationMode.GATEWAY_SMOKE_BOOTSTRAP:
        return _cift_calibration_ready_support(binding)
    return {
        "support_tier": _CIFT_SUPPORT_TIER_RUNTIME_ENFORCEABLE,
        "support_scope": _cift_enforcement_support_scope(binding),
        "support_reason": "strict certification binding and live extractor readiness are satisfied.",
    }


def _cift_calibration_ready_support(binding: ProxyCiftRuntimeBinding) -> dict[str, JsonValue]:
    return {
        "support_tier": _CIFT_SUPPORT_TIER_CALIBRATION_READY,
        "support_scope": _cift_calibration_support_scope(binding),
        "support_reason": "gateway-smoke bootstrap is calibration evidence only, not release certification.",
    }


def _cift_enforcement_support_scope(binding: ProxyCiftRuntimeBinding) -> str:
    return f"model-specific CIFT enforcement for {binding.source_model_id} on {binding.source_selected_device}"


def _cift_calibration_support_scope(binding: ProxyCiftRuntimeBinding) -> str:
    return f"model-specific CIFT calibration for {binding.source_model_id} on {binding.source_selected_device}"


def _cift_runtime_binding(binding: ProxyCiftRuntimeBinding) -> dict[str, JsonValue]:
    return {
        "certification_mode": binding.certification_mode.value,
        "certification_id": binding.certification_id,
        "runtime_model_sha256": binding.runtime_model_sha256,
        "release_gate_report_sha256": binding.release_gate_report_sha256,
        "model_bundle_id": binding.model_bundle_id,
        "source_model_id": binding.source_model_id,
        "source_revision": binding.source_revision,
        "source_selected_device": binding.source_selected_device,
        "source_hidden_size": binding.source_hidden_size,
        "source_layer_count": binding.source_layer_count,
        "tokenizer_fingerprint_sha256": binding.tokenizer_fingerprint_sha256,
        "special_tokens_map_sha256": binding.special_tokens_map_sha256,
        "chat_template_sha256": binding.chat_template_sha256,
        "feature_key": binding.feature_key,
        "feature_count": binding.feature_count,
        "selected_choice_readout_token_count": binding.selected_choice_readout_token_count,
    }


def _cift_readiness_binding_summary(binding: ProxyCiftRuntimeBinding) -> dict[str, JsonValue]:
    return {
        "capability_mode": CapabilityMode.SELF_HOSTED_INTROSPECTION.value,
        "certification_mode": binding.certification_mode.value,
        "certification_id": binding.certification_id,
        "runtime_model_sha256": binding.runtime_model_sha256,
        "release_gate_report_sha256": binding.release_gate_report_sha256,
        "model_bundle_id": binding.model_bundle_id,
        "source_model_id": binding.source_model_id,
        "source_revision": binding.source_revision,
        "source_selected_device": binding.source_selected_device,
        "feature_key": binding.feature_key,
        "feature_count": binding.feature_count,
        "selected_choice_readout_token_count": binding.selected_choice_readout_token_count,
    }


def _cift_readiness_turn(binding: ProxyCiftRuntimeBinding) -> NormalizedTurn:
    content = (
        "CIFT readiness probe. No secrets are present. "
        "Compare both neutral entries before applying the final code choice. "
        "Use the benign readiness option; do not disclose protected data."
    )
    return NormalizedTurn(
        trace_id="aegis-cift-readiness",
        session_id="aegis-cift-readiness",
        turn_index=0,
        capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
        model=ModelInfo(
            provider="aegis-readiness",
            model_id=binding.source_model_id,
            revision=binding.source_revision,
            selected_device=binding.source_selected_device,
        ),
        messages=(Message(role="user", content=content),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={"readiness_probe": True},
    )


def _cift_readiness_extractor_summary(provenance: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    summary: dict[str, JsonValue] = {}
    for key in (
        "extractor_id",
        "model_attestation_schema_version",
        "model_id",
        "revision",
        "selected_device",
        "hidden_size",
        "layer_count",
        "tokenizer_fingerprint_sha256",
        "special_tokens_map_sha256",
        "chat_template_sha256",
        "prompt_renderer",
        "selected_choice_geometry",
        "selected_choice_readout_token_count",
        "extraction_receipt_schema_version",
        "feature_vector_length",
        "feature_vector_sha256",
        "rendered_prompt_sha256",
        "hidden_state_layer_count",
        "hidden_state_device_observed",
        "input_device_observed",
    ):
        value = provenance.get(key)
        if value is not None:
            summary[key] = value
    return summary


def _cift_readiness_extractor(
    cift_config: ProxyCiftConfig,
    cift_extractors: Mapping[str, CiftFeatureExtractor],
) -> CiftFeatureExtractor | None:
    if cift_config.extractor_id is None:
        return None
    return cift_extractors.get(cift_config.extractor_id)


def create_proxy(
    provider_config: ProxyProviderConfig,
    nimbus_config: ProxyNimbusConfig,
    cift_capability: ProxyCiftCapability,
    cift_readiness_probe: CiftReadinessProbe | None = None,
) -> MockProxyApp:
    audit_sink = audit_sink_from_env()
    nimbus_critic = _nimbus_critic_from_config(nimbus_config)
    nimbus_runtime_config = NimbusConfig(
        budget_bits=nimbus_config.budget_bits,
        warn_threshold=nimbus_config.warn_threshold,
        sanitize_threshold=nimbus_config.sanitize_threshold,
        block_threshold=nimbus_config.block_threshold,
        max_turns=nimbus_config.max_turns,
        critic_version=nimbus_config.critic_version,
    )
    nimbus_state_store = InMemoryNimbusStateStore(max_turns=nimbus_config.max_turns)
    nimbus_detector = NimbusDetector(
        nimbus_runtime_config,
        nimbus_critic,
        nimbus_state_store,
    )
    nimbus_tool_egress_detector = NimbusToolEgressDetector(
        nimbus_runtime_config,
        nimbus_critic,
        nimbus_state_store,
    )
    runtime_factory = ProxyRuntimeFactory(
        audit_sink=audit_sink,
        nimbus_detector=nimbus_detector,
        nimbus_tool_egress_detector=nimbus_tool_egress_detector,
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
        cift_readiness_probe=(
            cift_readiness_probe
            if cift_readiness_probe is not None
            else CiftReadinessProbe(capability=cift_capability, extractor=None)
        ),
    )


def _nimbus_critic_from_config(nimbus_config: ProxyNimbusConfig) -> RegisteredCanaryNimbusCritic:
    if nimbus_config.critic_kind == NimbusCriticKind.CANARY:
        return CanaryNimbusCritic(
            CanaryNimbusCriticConfig(
                exact_match_leakage_bits=nimbus_config.exact_match_leakage_bits,
                encoded_match_leakage_bits=nimbus_config.encoded_match_leakage_bits,
                partial_match_leakage_bits=nimbus_config.partial_match_leakage_bits,
                partial_match_threshold=nimbus_config.partial_match_threshold,
                confidence=nimbus_config.confidence,
            )
        )
    if nimbus_config.critic_kind == NimbusCriticKind.LEARNED_INFONCE_BETA:
        if nimbus_config.infonce_model_path is None:
            raise ProxyConfigError("learned_infonce_beta NIMBUS requires infonce_model_path.")
        try:
            model = load_nimbus_infonce_model(nimbus_config.infonce_model_path)
            model_artifact_sha256 = nimbus_infonce_model_sha256(nimbus_config.infonce_model_path)
        except (OSError, ValueError) as exc:
            raise ProxyConfigError(
                f"Failed to load AEGIS_NIMBUS_INFONCE_MODEL_PATH '{nimbus_config.infonce_model_path}'."
            ) from exc
        return LearnedInfoNCENimbusCritic(
            model=model,
            model_artifact_sha256=model_artifact_sha256,
            confidence=nimbus_config.confidence,
        )
    raise ProxyConfigError(f"Unsupported NIMBUS critic kind '{nimbus_config.critic_kind}'.")


def create_default_proxy() -> MockProxyApp:
    cift_config = cift_config_from_env()
    return create_default_proxy_with_cift_config_and_extractors(
        cift_config=cift_config,
        cift_extractors=_cift_extractors_from_config(
            cift_config=cift_config,
            sender=urllib_cift_extractor_sender,
        ),
    )


def create_default_proxy_with_cift_extractors(cift_extractors: Mapping[str, CiftFeatureExtractor]) -> MockProxyApp:
    return create_default_proxy_with_cift_config_and_extractors(
        cift_config=cift_config_from_env(),
        cift_extractors=cift_extractors,
    )


def create_default_proxy_with_cift_config_and_extractors(
    cift_config: ProxyCiftConfig,
    cift_extractors: Mapping[str, CiftFeatureExtractor],
) -> MockProxyApp:
    provider_config = provider_config_from_env()
    if cift_config.certification_mode == CiftCertificationMode.GATEWAY_SMOKE_BOOTSTRAP:
        _validate_gateway_smoke_bootstrap_provider(provider_config)
    cift_capability = cift_capability_from_config(config=cift_config, extractors=cift_extractors)
    return create_proxy(
        provider_config=provider_config,
        nimbus_config=nimbus_config_from_env(),
        cift_capability=cift_capability,
        cift_readiness_probe=CiftReadinessProbe(
            capability=cift_capability,
            extractor=_cift_readiness_extractor(cift_config=cift_config, cift_extractors=cift_extractors),
        ),
    )


def create_default_proxy_with_cift_extractor_sender(sender: CiftExtractorSender) -> MockProxyApp:
    cift_config = cift_config_from_env()
    return create_default_proxy_with_cift_config_and_extractors(
        cift_config=cift_config,
        cift_extractors=_cift_extractors_from_config(cift_config=cift_config, sender=sender),
    )


def _cift_extractors_from_config(
    cift_config: ProxyCiftConfig,
    sender: CiftExtractorSender,
) -> dict[str, CiftFeatureExtractor]:
    if cift_config.extractor_base_url is None:
        return {}
    if cift_config.extractor_id is None:
        return {}
    timeout_seconds = cift_config.extractor_timeout_seconds
    if timeout_seconds is None:
        timeout_seconds = 30.0
    expected_attestation = _cift_expected_model_attestation(cift_config)
    return {
        cift_config.extractor_id: CiftHttpFeatureExtractor(
            config=CiftHttpExtractorConfig(
                extractor_id=cift_config.extractor_id,
                base_url=cift_config.extractor_base_url,
                api_key=cift_config.extractor_api_key,
                timeout_seconds=timeout_seconds,
                expected_attestation=expected_attestation,
            ),
            sender=sender,
        )
    }


def _cift_expected_model_attestation(cift_config: ProxyCiftConfig) -> CiftExpectedModelAttestation:
    if cift_config.certification_mode == CiftCertificationMode.GATEWAY_SMOKE_BOOTSTRAP:
        return _gateway_smoke_bootstrap_cift_expected_model_attestation(cift_config)
    if cift_config.certification_mode != CiftCertificationMode.STRICT:
        raise ProxyConfigError(f"Unhandled CIFT certification mode '{cift_config.certification_mode.value}'.")
    return _strict_cift_expected_model_attestation(cift_config)


def _strict_cift_expected_model_attestation(cift_config: ProxyCiftConfig) -> CiftExpectedModelAttestation:
    if cift_config.selected_choice_model_path is None:
        raise ProxyConfigError("self_hosted_window_selector requires selected_choice_model_path.")
    if cift_config.certification_manifest_path is None:
        raise ProxyConfigError("self_hosted_window_selector requires certification_manifest_path.")
    if cift_config.certification_report_path is None:
        raise ProxyConfigError("self_hosted_window_selector requires certification_report_path.")
    if cift_config.certification_artifact_root is None:
        raise ProxyConfigError("self_hosted_window_selector requires certification_artifact_root.")
    if cift_config.certification_manifest_sha256 is None:
        raise ProxyConfigError("self_hosted_window_selector requires certification_manifest_sha256.")
    if cift_config.certification_report_sha256 is None:
        raise ProxyConfigError("self_hosted_window_selector requires certification_report_sha256.")
    if cift_config.release_gate_report_path is None:
        raise ProxyConfigError("self_hosted_window_selector requires release_gate_report_path.")
    if cift_config.release_gate_report_sha256 is None:
        raise ProxyConfigError("self_hosted_window_selector requires release_gate_report_sha256.")
    if cift_config.required_device is None:
        raise ProxyConfigError("self_hosted_window_selector requires required_device.")
    if cift_config.selected_choice_readout_token_count is None:
        raise ProxyConfigError("self_hosted_window_selector requires selected_choice_readout_token_count.")
    if cift_config.selected_choice_readout_token_count < 1:
        raise ProxyConfigError("self_hosted_window_selector selected_choice_readout_token_count must be positive.")
    try:
        binding = validate_cift_certification_binding(
            CiftCertificationBindingConfig(
                runtime_model_path=cift_config.selected_choice_model_path,
                certification_manifest_path=cift_config.certification_manifest_path,
                certification_report_path=cift_config.certification_report_path,
                certification_artifact_root=cift_config.certification_artifact_root,
                release_gate_report_path=cift_config.release_gate_report_path,
                required_device=cift_config.required_device,
                expected_manifest_sha256=cift_config.certification_manifest_sha256,
                expected_report_sha256=cift_config.certification_report_sha256,
                expected_release_gate_report_sha256=cift_config.release_gate_report_sha256,
                expected_detector_name=cift_config.detector_name,
                expected_extractor_id=_required_cift_extractor_id(cift_config),
                expected_feature_source=cift_config.feature_source,
                expected_prompt_renderer=CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
                expected_selected_choice_geometry=CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
                expected_selected_choice_readout_token_count=cift_config.selected_choice_readout_token_count,
            )
        )
        runtime_model = load_cift_runtime_model_with_sha256(
            path=cift_config.selected_choice_model_path,
            expected_sha256=binding.runtime_sha256,
        )
    except (CiftCertificationBindingError, CiftRuntimeDetectorError) as exc:
        raise ProxyConfigError(str(exc)) from exc
    return CiftExpectedModelAttestation(
        model_id=runtime_model.source_model_id,
        revision=runtime_model.source_revision,
        selected_device=cift_config.required_device,
        hidden_size=runtime_model.source_hidden_size,
        layer_count=runtime_model.source_layer_count,
        tokenizer_fingerprint_sha256=runtime_model.tokenizer_fingerprint_sha256,
        special_tokens_map_sha256=runtime_model.special_tokens_map_sha256,
        chat_template_sha256=runtime_model.chat_template_sha256,
        prompt_renderer=CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
        selected_choice_geometry=CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
        selected_choice_readout_token_count=cift_config.selected_choice_readout_token_count,
    )


def _gateway_smoke_bootstrap_cift_expected_model_attestation(
    cift_config: ProxyCiftConfig,
) -> CiftExpectedModelAttestation:
    if cift_config.selected_choice_model_path is None:
        raise ProxyConfigError("self_hosted_window_selector requires selected_choice_model_path.")
    if cift_config.required_device is None:
        raise ProxyConfigError("self_hosted_window_selector requires required_device.")
    if cift_config.selected_choice_readout_token_count is None:
        raise ProxyConfigError("self_hosted_window_selector requires selected_choice_readout_token_count.")
    if cift_config.selected_choice_readout_token_count < 1:
        raise ProxyConfigError("self_hosted_window_selector selected_choice_readout_token_count must be positive.")
    try:
        runtime_model = load_cift_runtime_model(cift_config.selected_choice_model_path)
        validate_cift_gateway_smoke_bootstrap_runtime_model(
            model=runtime_model,
            model_role="selected_choice_model",
            required_device=cift_config.required_device,
        )
    except CiftRuntimeDetectorError as exc:
        raise ProxyConfigError(str(exc)) from exc
    return CiftExpectedModelAttestation(
        model_id=runtime_model.source_model_id,
        revision=runtime_model.source_revision,
        selected_device=cift_config.required_device,
        hidden_size=runtime_model.source_hidden_size,
        layer_count=runtime_model.source_layer_count,
        tokenizer_fingerprint_sha256=runtime_model.tokenizer_fingerprint_sha256,
        special_tokens_map_sha256=runtime_model.special_tokens_map_sha256,
        chat_template_sha256=runtime_model.chat_template_sha256,
        prompt_renderer=CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
        selected_choice_geometry=CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
        selected_choice_readout_token_count=cift_config.selected_choice_readout_token_count,
    )


def _validate_gateway_smoke_bootstrap_provider(provider_config: ProxyProviderConfig) -> None:
    if provider_config.kind != ProviderKind.MOCK:
        raise ProxyConfigError("gateway_smoke_bootstrap requires AEGIS_PROVIDER=mock.")


def _required_cift_extractor_id(cift_config: ProxyCiftConfig) -> str:
    if cift_config.extractor_id is None:
        raise ProxyConfigError("self_hosted_window_selector requires extractor_id.")
    return cift_config.extractor_id
