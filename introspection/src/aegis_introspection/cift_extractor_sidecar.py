from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from aegis.cift_contract import (
    CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
    CIFT_FEATURE_EXTRACT_REQUEST_SCHEMA_VERSION,
    CIFT_FEATURE_EXTRACT_RESPONSE_SCHEMA_VERSION,
    CIFT_MODEL_ATTESTATION_SCHEMA_VERSION,
    CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
    CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
)
from aegis.core.contracts import (
    CapabilityMode,
    JsonValue,
    Message,
    ModelInfo,
    NormalizedTurn,
    SensitiveSpan,
    ToolCall,
)
from aegis.detectors.cift_runtime import CiftFeatureExtraction, CiftFeatureExtractionExtractor
from aegis_introspection.trace_record_adapter import (
    CharSpan,
    MessageRecord,
    MessageSegment,
    RenderedTracePrompt,
    SelectedChoiceReadout,
    TokenOffset,
    TokenOffsetEncoder,
    TokenSpan,
    ToolArgumentSegment,
    ToolCallRecord,
    TraceRecordAdapterError,
    render_trace_prompt,
    semantic_indirection_selected_choice_readout,
)

ERROR_SCHEMA_VERSION = "aegis.cift_feature_extract_error/v1"
_SELECTED_CHOICE_POOLING_METHODS = frozenset(("selected_choice_window", "combined_readout_window"))
_SUPPORTED_POOLING_METHODS = (
    "final_token",
    "mean_pool",
    "readout_window",
    "query_tail_window",
    "selected_choice_window",
    "combined_readout_window",
)


class CiftSidecarFeatureExtractor(Protocol):
    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        """Return a trusted CIFT feature vector for a normalized turn."""


class CiftSidecarTurnAdapter(Protocol):
    @property
    def prompt_renderer(self) -> str:
        """Return the prompt rendering contract used by this adapter."""

    @property
    def selected_choice_geometry(self) -> str:
        """Return the selected-choice geometry contract used by this adapter."""

    @property
    def selected_choice_readout_token_count(self) -> int:
        """Return the selected-choice live readout token count used by this adapter."""

    def prepare_turn(self, turn: NormalizedTurn, feature_key: str) -> NormalizedTurn:
        """Return the trusted turn shape used for live hidden-state extraction."""


class CiftSidecarRequestError(ValueError):
    """Raised when a sidecar request does not satisfy the CIFT sidecar contract."""


class CiftSidecarTurnPreparationError(ValueError):
    """Raised when the sidecar cannot prepare a turn for hidden-state extraction."""


class CiftSidecarFeatureKeyError(ValueError):
    """Raised when a requested CIFT feature key is unsupported."""


@dataclass(frozen=True)
class CiftSidecarModelAttestation:
    model_id: str
    revision: str
    selected_device: str
    hidden_size: int
    layer_count: int
    tokenizer_fingerprint_sha256: str
    special_tokens_map_sha256: str
    chat_template_sha256: str
    prompt_renderer: str
    selected_choice_geometry: str
    selected_choice_readout_token_count: int


@dataclass(frozen=True)
class FreeformReadout:
    readout_token_indices: tuple[int, ...]
    query_tail_readout_token_indices: tuple[int, ...] | None
    readout_window_source: str


@dataclass(frozen=True)
class CiftRenderedPromptTurnAdapter:
    offset_encoder: TokenOffsetEncoder
    selected_choice_readout_token_count: int

    def __post_init__(self) -> None:
        if self.selected_choice_readout_token_count < 1:
            raise CiftSidecarTurnPreparationError("selected_choice_readout_token_count must be positive.")

    @property
    def prompt_renderer(self) -> str:
        return CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1

    @property
    def selected_choice_geometry(self) -> str:
        return CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1

    def prepare_turn(self, turn: NormalizedTurn, feature_key: str) -> NormalizedTurn:
        try:
            rendered_prompt = _rendered_prompt_for_turn(turn)
        except TraceRecordAdapterError as exc:
            raise CiftSidecarTurnPreparationError(str(exc)) from exc
        offsets = self.offset_encoder.encode_offsets(rendered_prompt.text)
        selected_choice_readout = self._selected_choice_readout(
            turn=turn,
            feature_key=feature_key,
            rendered_prompt=rendered_prompt,
            offsets=offsets,
        )
        freeform_readout = _freeform_readout(
            feature_key=feature_key,
            rendered_prompt=rendered_prompt,
            offsets=offsets,
            readout_token_count=self.selected_choice_readout_token_count,
            context=f"sidecar turn {turn.trace_id}",
        )
        metadata = _metadata_with_selected_choice_readout(
            metadata=turn.metadata,
            selected_choice_readout=selected_choice_readout,
            freeform_readout=freeform_readout,
            selected_choice_readout_token_count=self.selected_choice_readout_token_count,
        )
        return NormalizedTurn(
            trace_id=turn.trace_id,
            session_id=turn.session_id,
            turn_index=turn.turn_index,
            capability_mode=turn.capability_mode,
            model=turn.model,
            messages=(Message(role="user", content=rendered_prompt.text),),
            tool_calls=(),
            sensitive_spans=turn.sensitive_spans,
            metadata=metadata,
        )

    def _selected_choice_readout(
        self,
        turn: NormalizedTurn,
        feature_key: str,
        rendered_prompt: RenderedTracePrompt,
        offsets: tuple[TokenOffset, ...],
    ) -> SelectedChoiceReadout | None:
        if not _requires_selected_choice_indices(feature_key):
            return None
        try:
            return semantic_indirection_selected_choice_readout(
                rendered_prompt=rendered_prompt,
                offsets=offsets,
                readout_token_count=self.selected_choice_readout_token_count,
                context=f"sidecar turn {turn.trace_id}",
            )
        except TraceRecordAdapterError as exc:
            raise CiftSidecarTurnPreparationError(str(exc)) from exc


@dataclass(frozen=True)
class CiftFeatureExtractRequest:
    feature_key: str
    turn: NormalizedTurn


@dataclass(frozen=True)
class CiftSidecarHttpResponse:
    status_code: int
    payload: dict[str, JsonValue]


def _rendered_prompt_for_turn(turn: NormalizedTurn) -> RenderedTracePrompt:
    if _is_single_rendered_prompt_turn(turn):
        if len(turn.messages) != 1:
            raise TraceRecordAdapterError("single_rendered_prompt_message turns must contain exactly one message.")
        prompt = turn.messages[0].content
        return RenderedTracePrompt(
            text=prompt,
            message_segments=(
                MessageSegment(
                    index=0,
                    role="user",
                    content=prompt,
                    content_span=CharSpan(start=0, end=len(prompt)),
                ),
            ),
            tool_argument_segments=(),
        )
    return render_trace_prompt(
        messages=tuple(MessageRecord(role=message.role, content=message.content) for message in turn.messages),
        tool_calls=tuple(
            ToolCallRecord(name=tool_call.name, arguments=tool_call.arguments) for tool_call in turn.tool_calls
        ),
    )


def _is_single_rendered_prompt_turn(turn: NormalizedTurn) -> bool:
    bridge_metadata = turn.metadata.get("bridge")
    if not isinstance(bridge_metadata, dict):
        return False
    return bridge_metadata.get("message_layout") == "single_rendered_prompt_message"


def _metadata_with_selected_choice_readout(
    metadata: Mapping[str, JsonValue],
    selected_choice_readout: SelectedChoiceReadout | None,
    freeform_readout: FreeformReadout | None,
    selected_choice_readout_token_count: int,
) -> dict[str, JsonValue]:
    copied_metadata = {key: value for key, value in metadata.items() if key != "cift"}
    cift_metadata: dict[str, JsonValue] = {}
    if freeform_readout is not None:
        cift_metadata["readout_token_indices"] = list(freeform_readout.readout_token_indices)
        if freeform_readout.query_tail_readout_token_indices is not None:
            cift_metadata["query_tail_readout_token_indices"] = list(freeform_readout.query_tail_readout_token_indices)
        cift_metadata["readout_window_source"] = freeform_readout.readout_window_source
        cift_metadata["readout_source"] = {
            "source": "sidecar_freeform",
            "readout_window": freeform_readout.readout_window_source,
            "readout_token_count": len(freeform_readout.readout_token_indices),
        }
    if selected_choice_readout is None:
        if cift_metadata:
            copied_metadata["cift"] = cift_metadata
        return copied_metadata
    cift_metadata["selected_choice_char_span"] = selected_choice_readout.selected_choice_char_span.to_json()
    cift_metadata["selected_choice_token_span"] = selected_choice_readout.selected_choice_token_span.to_json()
    cift_metadata["selected_choice_readout_token_indices"] = list(
        selected_choice_readout.selected_choice_readout_token_indices
    )
    cift_metadata["selected_choice_readout_source"] = {
        "source": "sidecar_semantic_indirection",
        "readout_token_count": selected_choice_readout_token_count,
    }
    copied_metadata["cift"] = cift_metadata
    return copied_metadata


def _freeform_readout(
    feature_key: str,
    rendered_prompt: RenderedTracePrompt,
    offsets: tuple[TokenOffset, ...],
    readout_token_count: int,
    context: str,
) -> FreeformReadout | None:
    if feature_key.startswith("final_token_"):
        if len(offsets) == 0:
            raise CiftSidecarTurnPreparationError(f"{context}: tokenizer produced no prompt tokens.")
        final_token_indices = (len(offsets) - 1,)
        query_tail_indices = _query_tail_indices_or_none(
            rendered_prompt=rendered_prompt,
            offsets=offsets,
            readout_token_count=readout_token_count,
            context=context,
        )
        return FreeformReadout(
            readout_token_indices=final_token_indices,
            query_tail_readout_token_indices=query_tail_indices,
            readout_window_source="final_token",
        )
    query_tail_indices = _query_tail_indices_or_none(
        rendered_prompt=rendered_prompt,
        offsets=offsets,
        readout_token_count=readout_token_count,
        context=context,
    )
    if query_tail_indices is None:
        return None
    payload_readout = _payload_readout(
        tool_argument_segments=rendered_prompt.tool_argument_segments,
        offsets=offsets,
        readout_token_count=readout_token_count,
        context=context,
    )
    if payload_readout is None:
        return FreeformReadout(
            readout_token_indices=query_tail_indices,
            query_tail_readout_token_indices=query_tail_indices,
            readout_window_source="query_tail",
        )
    return FreeformReadout(
        readout_token_indices=payload_readout,
        query_tail_readout_token_indices=query_tail_indices,
        readout_window_source="tool_payload",
    )


def _query_tail_indices_or_none(
    rendered_prompt: RenderedTracePrompt,
    offsets: tuple[TokenOffset, ...],
    readout_token_count: int,
    context: str,
) -> tuple[int, ...] | None:
    query_span = _query_char_span(rendered_prompt.message_segments)
    if query_span is None:
        return None
    query_token_span = _token_span_for_char_span(offsets=offsets, char_span=query_span, context=context)
    return _readout_indices_for_char_span(
        offsets=offsets,
        char_span=query_span,
        lower_bound=query_token_span.start,
        readout_token_count=readout_token_count,
        context=context,
    )


def _query_char_span(message_segments: tuple[MessageSegment, ...]) -> CharSpan | None:
    for segment in message_segments:
        if segment.role == "user":
            return segment.content_span
    return None


def _payload_readout(
    tool_argument_segments: tuple[ToolArgumentSegment, ...],
    offsets: tuple[TokenOffset, ...],
    readout_token_count: int,
    context: str,
) -> tuple[int, ...] | None:
    if len(tool_argument_segments) == 0:
        return None
    credential_segments = tuple(
        segment for segment in tool_argument_segments if segment.argument_path.endswith(".credential")
    )
    segment = credential_segments[0] if len(credential_segments) > 0 else tool_argument_segments[0]
    return _readout_indices_for_char_span(
        offsets=offsets,
        char_span=segment.value_span,
        lower_bound=_token_span_for_char_span(offsets=offsets, char_span=segment.value_span, context=context).start,
        readout_token_count=readout_token_count,
        context=context,
    )


def _token_span_for_char_span(
    offsets: tuple[TokenOffset, ...],
    char_span: CharSpan,
    context: str,
) -> TokenSpan:
    indices = _token_indices_for_char_span(offsets=offsets, char_span=char_span)
    if len(indices) == 0:
        raise CiftSidecarTurnPreparationError(f"{context}: tokenizer produced no tokens for char span {char_span}.")
    return TokenSpan(start=indices[0], end=indices[-1] + 1)


def _readout_indices_for_char_span(
    offsets: tuple[TokenOffset, ...],
    char_span: CharSpan,
    lower_bound: int,
    readout_token_count: int,
    context: str,
) -> tuple[int, ...]:
    indices = tuple(
        index for index in _token_indices_for_char_span(offsets=offsets, char_span=char_span) if index >= lower_bound
    )
    if len(indices) == 0:
        raise CiftSidecarTurnPreparationError(f"{context}: no readout tokens remain after visibility floor.")
    return indices[-readout_token_count:]


def _token_indices_for_char_span(offsets: tuple[TokenOffset, ...], char_span: CharSpan) -> tuple[int, ...]:
    return tuple(
        index
        for index, offset in enumerate(offsets)
        if offset.end > char_span.start and offset.start < char_span.end
    )


class CiftExtractorSidecarService:
    def __init__(
        self,
        extractor: CiftSidecarFeatureExtractor,
        expected_api_key: str | None,
        model_attestation: CiftSidecarModelAttestation,
        turn_adapter: CiftSidecarTurnAdapter,
    ) -> None:
        if expected_api_key == "":
            raise CiftSidecarRequestError("expected_api_key must not be empty when provided.")
        _validate_model_attestation(model_attestation)
        _validate_attestation_matches_turn_adapter(model_attestation=model_attestation, turn_adapter=turn_adapter)
        self._extractor = extractor
        self._expected_api_key = expected_api_key
        self._model_attestation = model_attestation
        self._turn_adapter = turn_adapter

    def extract_features(self, raw_payload: object, authorization_header: str | None) -> CiftSidecarHttpResponse:
        authorization_error = self._authorization_error(authorization_header)
        if authorization_error is not None:
            return authorization_error

        try:
            json_value = _json_value(raw_payload, "request body")
        except CiftSidecarRequestError as exc:
            return cift_sidecar_error_response(status_code=400, code="invalid_json", message=str(exc), details={})
        if not isinstance(json_value, dict):
            return cift_sidecar_error_response(
                status_code=400,
                code="invalid_request",
                message="Request body must be a JSON object.",
                details={},
            )

        try:
            feature_request = _decode_feature_extract_request(json_value)
        except CiftSidecarRequestError as exc:
            return cift_sidecar_error_response(status_code=400, code="invalid_request", message=str(exc), details={})

        try:
            requires_selected_choice = _requires_selected_choice_indices(feature_request.feature_key)
        except CiftSidecarFeatureKeyError as exc:
            return cift_sidecar_error_response(
                status_code=400,
                code="invalid_feature_key",
                message=str(exc),
                details={},
            )

        try:
            prepared_turn = self._turn_adapter.prepare_turn(
                turn=feature_request.turn,
                feature_key=feature_request.feature_key,
            )
        except CiftSidecarTurnPreparationError as exc:
            return cift_sidecar_error_response(
                status_code=500,
                code="turn_preparation_failed",
                message=f"CIFT turn preparation failed: {exc}",
                details={},
            )

        selected_choice_readout_token_indices = _selected_choice_readout_token_indices_or_none(turn=prepared_turn)
        if requires_selected_choice and selected_choice_readout_token_indices is None:
            return _feature_response(
                feature_key=feature_request.feature_key,
                feature_vector=None,
                selected_choice_readout_token_indices=None,
                unavailable_reason="selected_choice_geometry_missing",
                model_attestation=self._model_attestation,
                extraction_receipt=None,
            )

        try:
            extraction = _feature_extraction_from_extractor(
                extractor=self._extractor,
                turn=prepared_turn,
                feature_key=feature_request.feature_key,
            )
        except Exception as exc:
            return cift_sidecar_error_response(
                status_code=500,
                code="feature_extraction_failed",
                message=f"CIFT feature extraction failed with {exc.__class__.__name__}.",
                details={},
            )

        feature_vector = extraction.feature_vector
        try:
            extraction_receipt = _extraction_receipt_payload(
                feature_key=feature_request.feature_key,
                prepared_turn=prepared_turn,
                feature_vector=feature_vector,
                selected_choice_readout_token_indices=selected_choice_readout_token_indices,
                model_attestation=self._model_attestation,
                provenance=extraction.provenance,
            )
        except CiftSidecarRequestError as exc:
            return cift_sidecar_error_response(
                status_code=500,
                code="extraction_receipt_failed",
                message=f"CIFT extraction receipt failed: {exc}",
                details={},
            )
        unavailable_reason = None if feature_vector is not None else "feature_vector_unavailable"
        return _feature_response(
            feature_key=feature_request.feature_key,
            feature_vector=feature_vector,
            selected_choice_readout_token_indices=selected_choice_readout_token_indices,
            unavailable_reason=unavailable_reason,
            model_attestation=self._model_attestation,
            extraction_receipt=extraction_receipt,
        )

    def _authorization_error(self, authorization_header: str | None) -> CiftSidecarHttpResponse | None:
        if self._expected_api_key is None:
            return None
        expected_header = f"Bearer {self._expected_api_key}"
        if authorization_header == expected_header:
            return None
        return cift_sidecar_error_response(
            status_code=401,
            code="unauthorized",
            message="Missing or invalid CIFT sidecar bearer token.",
            details={},
        )


def create_cift_extractor_sidecar_app(
    extractor: CiftSidecarFeatureExtractor,
    expected_api_key: str | None,
    model_attestation: CiftSidecarModelAttestation,
    turn_adapter: CiftSidecarTurnAdapter,
) -> object:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    globals()["JSONResponse"] = JSONResponse
    globals()["Request"] = Request

    service = CiftExtractorSidecarService(
        extractor=extractor,
        expected_api_key=expected_api_key,
        model_attestation=model_attestation,
        turn_adapter=turn_adapter,
    )

    app = FastAPI(
        title="Aegis CIFT Extractor Sidecar",
        description="Trusted hidden-state feature extraction service for self-hosted CIFT.",
    )

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(status_code=200, content={"status": "ok"})

    @app.post("/v1/cift/features")
    async def extract_features(request: Request) -> JSONResponse:
        try:
            raw_body = await request.json()
        except ValueError:
            sidecar_response = cift_sidecar_error_response(
                status_code=400,
                code="invalid_json",
                message="Request body must be valid JSON.",
                details={},
            )
        else:
            sidecar_response = service.extract_features(
                raw_payload=raw_body,
                authorization_header=request.headers.get("authorization"),
            )
        return JSONResponse(status_code=sidecar_response.status_code, content=sidecar_response.payload)

    return app


def _decode_feature_extract_request(payload: Mapping[str, JsonValue]) -> CiftFeatureExtractRequest:
    schema_version = payload.get("schema_version")
    if schema_version != CIFT_FEATURE_EXTRACT_REQUEST_SCHEMA_VERSION:
        raise CiftSidecarRequestError(f"schema_version must be {CIFT_FEATURE_EXTRACT_REQUEST_SCHEMA_VERSION}.")
    feature_key = _required_string(payload, "feature_key", "request")
    if feature_key == "":
        raise CiftSidecarRequestError("request.feature_key must not be empty.")
    turn = _required_object(payload, "turn", "request")
    return CiftFeatureExtractRequest(feature_key=feature_key, turn=_normalized_turn(turn, "request.turn"))


def _normalized_turn(payload: Mapping[str, JsonValue], context: str) -> NormalizedTurn:
    capability_mode = _capability_mode(_required_string(payload, "capability_mode", context), context)
    return NormalizedTurn(
        trace_id=_required_string(payload, "trace_id", context),
        session_id=_required_string(payload, "session_id", context),
        turn_index=_required_non_negative_int(payload, "turn_index", context),
        capability_mode=capability_mode,
        model=_model_info(_required_object(payload, "model", context), f"{context}.model"),
        messages=_messages(_required_list(payload, "messages", context), f"{context}.messages"),
        tool_calls=_tool_calls(_required_list(payload, "tool_calls", context), f"{context}.tool_calls"),
        sensitive_spans=_sensitive_spans(
            _required_list(payload, "sensitive_spans", context),
            f"{context}.sensitive_spans",
        ),
        metadata=_required_object(payload, "metadata", context),
    )


def _capability_mode(raw_value: str, context: str) -> CapabilityMode:
    try:
        return CapabilityMode(raw_value)
    except ValueError as exc:
        raise CiftSidecarRequestError(f"{context}.capability_mode is not supported: {raw_value}.") from exc


def _model_info(payload: Mapping[str, JsonValue], context: str) -> ModelInfo:
    return ModelInfo(
        provider=_required_string(payload, "provider", context),
        model_id=_required_string(payload, "model_id", context),
        revision=_optional_string(payload, "revision", context),
        selected_device=_optional_string(payload, "selected_device", context),
    )


def _messages(items: list[JsonValue], context: str) -> tuple[Message, ...]:
    messages: list[Message] = []
    for index, item in enumerate(items):
        item_context = f"{context}[{index}]"
        payload = _object_value(item, item_context)
        messages.append(
            Message(
                role=_required_string(payload, "role", item_context),
                content=_required_string(payload, "content", item_context),
            )
        )
    return tuple(messages)


def _tool_calls(items: list[JsonValue], context: str) -> tuple[ToolCall, ...]:
    tool_calls: list[ToolCall] = []
    for index, item in enumerate(items):
        item_context = f"{context}[{index}]"
        payload = _object_value(item, item_context)
        tool_calls.append(
            ToolCall(
                name=_required_string(payload, "name", item_context),
                arguments=_required_object(payload, "arguments", item_context),
            )
        )
    return tuple(tool_calls)


def _sensitive_spans(items: list[JsonValue], context: str) -> tuple[SensitiveSpan, ...]:
    spans: list[SensitiveSpan] = []
    for index, item in enumerate(items):
        item_context = f"{context}[{index}]"
        payload = _object_value(item, item_context)
        spans.append(
            SensitiveSpan(
                kind=_required_string(payload, "kind", item_context),
                source=_required_string(payload, "source", item_context),
                char_start=_optional_non_negative_int(payload, "char_start", item_context),
                char_end=_optional_non_negative_int(payload, "char_end", item_context),
                token_start=_optional_non_negative_int(payload, "token_start", item_context),
                token_end=_optional_non_negative_int(payload, "token_end", item_context),
                identifier=_optional_string(payload, "identifier", item_context),
                metadata=_required_object(payload, "metadata", item_context),
            )
        )
    return tuple(spans)


def _requires_selected_choice_indices(feature_key: str) -> bool:
    return any(
        _pooling_method(source_feature_key) in _SELECTED_CHOICE_POOLING_METHODS
        for source_feature_key in _source_feature_keys(feature_key)
    )


def _selected_choice_readout_token_indices_or_none(turn: NormalizedTurn) -> tuple[int, ...] | None:
    cift_metadata = turn.metadata.get("cift")
    if not isinstance(cift_metadata, dict):
        return None
    token_indices = cift_metadata.get("selected_choice_readout_token_indices")
    if not isinstance(token_indices, list):
        return None
    if len(token_indices) == 0:
        return None
    values: list[int] = []
    for token_index in token_indices:
        if isinstance(token_index, bool) or not isinstance(token_index, int):
            return None
        if token_index < 0:
            return None
        values.append(token_index)
    return tuple(values)


def _feature_extraction_from_extractor(
    extractor: CiftSidecarFeatureExtractor,
    turn: NormalizedTurn,
    feature_key: str,
) -> CiftFeatureExtraction:
    if isinstance(extractor, CiftFeatureExtractionExtractor):
        return extractor.extract_feature_extraction(turn=turn, feature_key=feature_key)
    return CiftFeatureExtraction(
        feature_vector=extractor.extract_feature_vector(turn=turn, feature_key=feature_key),
        selected_choice_readout_token_indices=None,
        provenance={},
    )


def _source_feature_keys(feature_key: str) -> tuple[str, ...]:
    prefix = "concat("
    suffix = ")"
    if feature_key.startswith(prefix):
        if not feature_key.endswith(suffix):
            raise CiftSidecarFeatureKeyError(f"Feature expression '{feature_key}' is missing a closing parenthesis.")
        inner_value = feature_key[len(prefix) : -len(suffix)]
        source_feature_keys = tuple(item.strip() for item in inner_value.split(",") if item.strip() != "")
        if len(source_feature_keys) < 2:
            raise CiftSidecarFeatureKeyError(
                f"Feature expression '{feature_key}' must concatenate at least two source features."
            )
        return source_feature_keys
    return (feature_key,)


def _pooling_method(source_feature_key: str) -> str:
    for pooling_method in _SUPPORTED_POOLING_METHODS:
        prefix = f"{pooling_method}_layer_"
        if source_feature_key.startswith(prefix):
            _layer_index(raw_value=source_feature_key[len(prefix) :], feature_key=source_feature_key)
            return pooling_method
    raise CiftSidecarFeatureKeyError(f"Unsupported live CIFT source feature '{source_feature_key}'.")


def _layer_index(raw_value: str, feature_key: str) -> int:
    if raw_value == "":
        raise CiftSidecarFeatureKeyError(f"Feature '{feature_key}' is missing a layer index.")
    try:
        layer_index = int(raw_value)
    except ValueError as exc:
        raise CiftSidecarFeatureKeyError(f"Feature '{feature_key}' has non-integer layer index '{raw_value}'.") from exc
    if layer_index < 0:
        raise CiftSidecarFeatureKeyError(f"Feature '{feature_key}' must use a non-negative layer index.")
    return layer_index


def _feature_response(
    feature_key: str,
    feature_vector: tuple[float, ...] | None,
    selected_choice_readout_token_indices: tuple[int, ...] | None,
    unavailable_reason: str | None,
    model_attestation: CiftSidecarModelAttestation,
    extraction_receipt: dict[str, JsonValue] | None,
) -> CiftSidecarHttpResponse:
    content: dict[str, JsonValue] = {
        "schema_version": CIFT_FEATURE_EXTRACT_RESPONSE_SCHEMA_VERSION,
        "feature_key": feature_key,
        "feature_vector": _optional_float_list(feature_vector),
        "selected_choice_readout_token_indices": _optional_int_list(selected_choice_readout_token_indices),
        "model_attestation": _model_attestation_payload(model_attestation),
        "extraction_receipt": extraction_receipt,
    }
    if unavailable_reason is not None:
        content["unavailable_reason"] = unavailable_reason
    return CiftSidecarHttpResponse(status_code=200, payload=content)


def _extraction_receipt_payload(
    feature_key: str,
    prepared_turn: NormalizedTurn,
    feature_vector: tuple[float, ...] | None,
    selected_choice_readout_token_indices: tuple[int, ...] | None,
    model_attestation: CiftSidecarModelAttestation,
    provenance: Mapping[str, JsonValue],
) -> dict[str, JsonValue] | None:
    if feature_vector is None:
        return None
    hidden_state_device_observed = _required_provenance_string(
        provenance=provenance,
        field_name="hidden_state_device_observed",
    )
    hidden_state_layer_count = _required_provenance_positive_int(
        provenance=provenance,
        field_name="hidden_state_layer_count",
    )
    input_device_observed = _required_provenance_string(provenance=provenance, field_name="input_device_observed")
    receipt: dict[str, JsonValue] = {
        "schema_version": CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
        "feature_key": feature_key,
        "feature_vector_length": len(feature_vector),
        "feature_vector_sha256": _json_sha256([float(value) for value in feature_vector]),
        "rendered_prompt_sha256": _rendered_prompt_sha256(prepared_turn),
        "selected_choice_readout_token_indices": _optional_int_list(selected_choice_readout_token_indices),
        "selected_choice_readout_token_count": 0
        if selected_choice_readout_token_indices is None
        else len(selected_choice_readout_token_indices),
        "hidden_state_layer_count": hidden_state_layer_count,
        "hidden_state_device_observed": hidden_state_device_observed,
        "input_device_observed": input_device_observed,
        "model_id": model_attestation.model_id,
        "revision": model_attestation.revision,
        "selected_device": model_attestation.selected_device,
        "hidden_size": model_attestation.hidden_size,
        "layer_count": model_attestation.layer_count,
        "tokenizer_fingerprint_sha256": model_attestation.tokenizer_fingerprint_sha256,
        "special_tokens_map_sha256": model_attestation.special_tokens_map_sha256,
        "chat_template_sha256": model_attestation.chat_template_sha256,
        "prompt_renderer": model_attestation.prompt_renderer,
        "selected_choice_geometry": model_attestation.selected_choice_geometry,
        "selected_choice_readout_configured_token_count": model_attestation.selected_choice_readout_token_count,
    }
    token_indices_digest = _optional_provenance_string(
        provenance=provenance,
        field_name="selected_choice_readout_token_indices_sha256",
    )
    if selected_choice_readout_token_indices is not None:
        expected_digest = _json_sha256([int(token_index) for token_index in selected_choice_readout_token_indices])
        if token_indices_digest is not None and token_indices_digest != expected_digest:
            raise CiftSidecarRequestError(
                "extractor provenance selected_choice_readout_token_indices_sha256 must match prepared turn indices."
            )
        receipt["selected_choice_readout_token_indices_sha256"] = expected_digest
    _copy_cift_readout_receipt_fields(receipt=receipt, prepared_turn=prepared_turn)
    return receipt


def _copy_cift_readout_receipt_fields(
    receipt: dict[str, JsonValue],
    prepared_turn: NormalizedTurn,
) -> None:
    cift_metadata = prepared_turn.metadata.get("cift")
    if not isinstance(cift_metadata, dict):
        return
    for field_name in (
        "readout_token_indices",
        "query_tail_readout_token_indices",
        "readout_window_source",
        "readout_source",
        "selected_choice_char_span",
        "selected_choice_token_span",
        "selected_choice_readout_source",
    ):
        value = cift_metadata.get(field_name)
        if isinstance(value, str | int | float | bool) or value is None:
            receipt[field_name] = value
        elif isinstance(value, list):
            receipt[field_name] = list(value)
            if field_name in ("readout_token_indices", "query_tail_readout_token_indices"):
                receipt[f"{field_name}_sha256"] = _json_sha256(list(value))
        elif isinstance(value, dict):
            receipt[field_name] = dict(value)


def _rendered_prompt_sha256(turn: NormalizedTurn) -> str:
    if len(turn.messages) != 1:
        raise CiftSidecarRequestError("prepared turn must contain exactly one rendered prompt message.")
    return hashlib.sha256(turn.messages[0].content.encode("utf-8")).hexdigest()


def _required_provenance_string(provenance: Mapping[str, JsonValue], field_name: str) -> str:
    value = provenance.get(field_name)
    if not isinstance(value, str) or value == "":
        raise CiftSidecarRequestError(f"extractor provenance {field_name} must be a non-empty string.")
    return value


def _optional_provenance_string(provenance: Mapping[str, JsonValue], field_name: str) -> str | None:
    value = provenance.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise CiftSidecarRequestError(f"extractor provenance {field_name} must be a non-empty string when present.")
    return value


def _required_provenance_positive_int(provenance: Mapping[str, JsonValue], field_name: str) -> int:
    value = provenance.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftSidecarRequestError(f"extractor provenance {field_name} must be an integer.")
    if value < 1:
        raise CiftSidecarRequestError(f"extractor provenance {field_name} must be positive.")
    return value


def _validate_model_attestation(model_attestation: CiftSidecarModelAttestation) -> None:
    if model_attestation.model_id == "":
        raise CiftSidecarRequestError("model_attestation.model_id must not be empty.")
    if model_attestation.revision == "":
        raise CiftSidecarRequestError("model_attestation.revision must not be empty.")
    if model_attestation.selected_device == "":
        raise CiftSidecarRequestError("model_attestation.selected_device must not be empty.")
    if model_attestation.hidden_size < 1:
        raise CiftSidecarRequestError("model_attestation.hidden_size must be positive.")
    if model_attestation.layer_count < 1:
        raise CiftSidecarRequestError("model_attestation.layer_count must be positive.")
    _validate_attestation_sha256(
        value=model_attestation.tokenizer_fingerprint_sha256,
        field_name="tokenizer_fingerprint_sha256",
    )
    _validate_attestation_sha256(
        value=model_attestation.special_tokens_map_sha256,
        field_name="special_tokens_map_sha256",
    )
    _validate_attestation_sha256(
        value=model_attestation.chat_template_sha256,
        field_name="chat_template_sha256",
    )
    if model_attestation.prompt_renderer == "":
        raise CiftSidecarRequestError("model_attestation.prompt_renderer must not be empty.")
    if model_attestation.selected_choice_geometry == "":
        raise CiftSidecarRequestError("model_attestation.selected_choice_geometry must not be empty.")
    if model_attestation.selected_choice_readout_token_count < 1:
        raise CiftSidecarRequestError("model_attestation.selected_choice_readout_token_count must be positive.")


def _validate_attestation_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CiftSidecarRequestError(f"model_attestation.{field_name} must be a lowercase SHA-256 digest.")


def _validate_attestation_matches_turn_adapter(
    model_attestation: CiftSidecarModelAttestation,
    turn_adapter: CiftSidecarTurnAdapter,
) -> None:
    if model_attestation.prompt_renderer != turn_adapter.prompt_renderer:
        raise CiftSidecarRequestError("model_attestation.prompt_renderer must match turn_adapter.prompt_renderer.")
    if model_attestation.selected_choice_geometry != turn_adapter.selected_choice_geometry:
        raise CiftSidecarRequestError(
            "model_attestation.selected_choice_geometry must match turn_adapter.selected_choice_geometry."
        )
    if model_attestation.selected_choice_readout_token_count != turn_adapter.selected_choice_readout_token_count:
        raise CiftSidecarRequestError(
            "model_attestation.selected_choice_readout_token_count must match "
            "turn_adapter.selected_choice_readout_token_count."
        )


def _model_attestation_payload(model_attestation: CiftSidecarModelAttestation) -> dict[str, JsonValue]:
    return {
        "schema_version": CIFT_MODEL_ATTESTATION_SCHEMA_VERSION,
        "model_id": model_attestation.model_id,
        "revision": model_attestation.revision,
        "selected_device": model_attestation.selected_device,
        "hidden_size": model_attestation.hidden_size,
        "layer_count": model_attestation.layer_count,
        "tokenizer_fingerprint_sha256": model_attestation.tokenizer_fingerprint_sha256,
        "special_tokens_map_sha256": model_attestation.special_tokens_map_sha256,
        "chat_template_sha256": model_attestation.chat_template_sha256,
        "prompt_renderer": model_attestation.prompt_renderer,
        "selected_choice_geometry": model_attestation.selected_choice_geometry,
        "selected_choice_readout_token_count": model_attestation.selected_choice_readout_token_count,
    }


def cift_sidecar_error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, JsonValue],
) -> CiftSidecarHttpResponse:
    return CiftSidecarHttpResponse(
        status_code=status_code,
        payload={
            "error": {
                "schema_version": ERROR_SCHEMA_VERSION,
                "code": code,
                "message": message,
                "details": details,
            }
        },
    )


def _optional_float_list(value: tuple[float, ...] | None) -> list[JsonValue] | None:
    if value is None:
        return None
    return [float(item) for item in value]


def _optional_int_list(value: tuple[int, ...] | None) -> list[JsonValue] | None:
    if value is None:
        return None
    return [int(item) for item in value]


def _json_sha256(value: JsonValue) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_string(payload: Mapping[str, JsonValue], field_name: str, context: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise CiftSidecarRequestError(f"{context}.{field_name} must be a string.")
    return value


def _optional_string(payload: Mapping[str, JsonValue], field_name: str, context: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CiftSidecarRequestError(f"{context}.{field_name} must be a string when present.")
    return value


def _required_non_negative_int(payload: Mapping[str, JsonValue], field_name: str, context: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftSidecarRequestError(f"{context}.{field_name} must be an integer.")
    if value < 0:
        raise CiftSidecarRequestError(f"{context}.{field_name} must be non-negative.")
    return value


def _optional_non_negative_int(payload: Mapping[str, JsonValue], field_name: str, context: str) -> int | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftSidecarRequestError(f"{context}.{field_name} must be an integer when present.")
    if value < 0:
        raise CiftSidecarRequestError(f"{context}.{field_name} must be non-negative when present.")
    return value


def _required_object(payload: Mapping[str, JsonValue], field_name: str, context: str) -> dict[str, JsonValue]:
    return _object_value(payload.get(field_name), f"{context}.{field_name}")


def _required_list(payload: Mapping[str, JsonValue], field_name: str, context: str) -> list[JsonValue]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise CiftSidecarRequestError(f"{context}.{field_name} must be a list.")
    return value


def _object_value(value: JsonValue, context: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise CiftSidecarRequestError(f"{context} must be an object.")
    return value


def _json_value(value: object, context: str) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{context}[]") for item in value]
    if isinstance(value, dict):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CiftSidecarRequestError(f"{context} object keys must be strings.")
            normalized[key] = _json_value(item, f"{context}.{key}")
        return normalized
    raise CiftSidecarRequestError(f"{context} contains unsupported JSON value {type(value).__name__}.")
