from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from aegis.cift_contract import (
    CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
    CIFT_FEATURE_EXTRACT_REQUEST_SCHEMA_VERSION,
    CIFT_FEATURE_EXTRACT_RESPONSE_SCHEMA_VERSION,
    CIFT_MODEL_ATTESTATION_SCHEMA_VERSION,
)
from aegis.core.contracts import JsonValue, NormalizedTurn
from aegis.detectors.cift_runtime import CiftFeatureExtraction

CiftExtractorSender = Callable[[str, dict[str, JsonValue], dict[str, str], float], dict[str, JsonValue]]


class CiftHttpExtractorError(RuntimeError):
    """Raised when a trusted CIFT extractor sidecar cannot return usable features."""


@dataclass(frozen=True)
class CiftExpectedModelAttestation:
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
class CiftHttpExtractorConfig:
    extractor_id: str
    base_url: str
    api_key: str | None
    timeout_seconds: float
    expected_attestation: CiftExpectedModelAttestation


@dataclass(frozen=True)
class CiftHttpExtractorResponse:
    feature_key: str
    feature_vector: tuple[float, ...] | None
    selected_choice_readout_token_indices: tuple[int, ...] | None
    unavailable_reason: str | None
    provenance: dict[str, JsonValue]


class CiftHttpFeatureExtractor:
    def __init__(self, config: CiftHttpExtractorConfig, sender: CiftExtractorSender) -> None:
        if config.extractor_id == "":
            raise CiftHttpExtractorError("CIFT extractor extractor_id must not be empty.")
        if config.base_url == "":
            raise CiftHttpExtractorError("CIFT extractor base_url must not be empty.")
        if config.api_key == "":
            raise CiftHttpExtractorError("CIFT extractor api_key must not be empty when provided.")
        if config.timeout_seconds <= 0:
            raise CiftHttpExtractorError("CIFT extractor timeout_seconds must be positive.")
        _validate_expected_attestation(config.expected_attestation)
        self._config = config
        self._sender = sender

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        response = self._response(turn=turn, feature_key=feature_key)
        if response.feature_vector is None and response.unavailable_reason is not None:
            raise CiftHttpExtractorError(f"CIFT extractor feature unavailable: {response.unavailable_reason}.")
        return response.feature_vector

    def extract_selected_choice_readout_token_indices(
        self,
        turn: NormalizedTurn,
        feature_key: str,
    ) -> tuple[int, ...] | None:
        return self._response(turn=turn, feature_key=feature_key).selected_choice_readout_token_indices

    def extract_feature_extraction(self, turn: NormalizedTurn, feature_key: str) -> CiftFeatureExtraction:
        response = self._response(turn=turn, feature_key=feature_key)
        if response.feature_vector is None and response.unavailable_reason is not None:
            raise CiftHttpExtractorError(f"CIFT extractor feature unavailable: {response.unavailable_reason}.")
        return CiftFeatureExtraction(
            feature_vector=response.feature_vector,
            selected_choice_readout_token_indices=response.selected_choice_readout_token_indices,
            provenance=response.provenance,
        )

    def _response(self, turn: NormalizedTurn, feature_key: str) -> CiftHttpExtractorResponse:
        payload = _request_payload(turn=turn, feature_key=feature_key)
        response = self._sender(
            _feature_extract_url(self._config.base_url),
            payload,
            _headers(self._config.api_key),
            self._config.timeout_seconds,
        )
        return _decode_response(
            response=response,
            requested_feature_key=feature_key,
            expected_attestation=self._config.expected_attestation,
            extractor_id=self._config.extractor_id,
        )


def urllib_cift_extractor_sender(
    url: str,
    payload: dict[str, JsonValue],
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, JsonValue]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            status_code = response.status
    except urllib.error.HTTPError as exc:
        raise CiftHttpExtractorError(f"CIFT extractor returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise CiftHttpExtractorError(f"CIFT extractor request failed: {exc.reason}") from exc
    if status_code < 200 or status_code >= 300:
        raise CiftHttpExtractorError(f"CIFT extractor returned HTTP {status_code}.")
    try:
        decoded = json.loads(body, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise CiftHttpExtractorError("CIFT extractor returned invalid JSON.") from exc
    except ValueError as exc:
        raise CiftHttpExtractorError("CIFT extractor returned invalid JSON.") from exc
    if not isinstance(decoded, dict):
        raise CiftHttpExtractorError("CIFT extractor response must be a JSON object.")
    return cast(dict[str, JsonValue], decoded)


def _request_payload(turn: NormalizedTurn, feature_key: str) -> dict[str, JsonValue]:
    if feature_key == "":
        raise CiftHttpExtractorError("feature_key must not be empty.")
    return {
        "schema_version": CIFT_FEATURE_EXTRACT_REQUEST_SCHEMA_VERSION,
        "feature_key": feature_key,
        "turn": turn.to_dict(),
    }


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _feature_extract_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1/cift/features"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/cift/features"
    return f"{normalized}/v1/cift/features"


def _decode_response(
    response: dict[str, JsonValue],
    requested_feature_key: str,
    expected_attestation: CiftExpectedModelAttestation,
    extractor_id: str,
) -> CiftHttpExtractorResponse:
    schema_version = response.get("schema_version")
    if schema_version != CIFT_FEATURE_EXTRACT_RESPONSE_SCHEMA_VERSION:
        raise CiftHttpExtractorError(
            f"CIFT extractor response schema_version must be {CIFT_FEATURE_EXTRACT_RESPONSE_SCHEMA_VERSION}."
        )
    model_attestation = _validate_response_attestation(
        value=response.get("model_attestation"),
        expected_attestation=expected_attestation,
    )
    feature_key = response.get("feature_key")
    if feature_key != requested_feature_key:
        raise CiftHttpExtractorError("CIFT extractor response feature_key does not match the request.")
    feature_vector = _optional_float_tuple(response.get("feature_vector"), "feature_vector")
    selected_choice_readout_token_indices = _optional_int_tuple(
        response.get("selected_choice_readout_token_indices"),
        "selected_choice_readout_token_indices",
    )
    if (
        selected_choice_readout_token_indices is not None
        and len(selected_choice_readout_token_indices) != expected_attestation.selected_choice_readout_token_count
    ):
        raise CiftHttpExtractorError(
            "CIFT extractor response selected_choice_readout_token_indices length must match "
            "model_attestation.selected_choice_readout_token_count."
        )
    unavailable_reason = _optional_non_empty_string(response.get("unavailable_reason"), "unavailable_reason")
    extraction_receipt = _validate_extraction_receipt(
        value=response.get("extraction_receipt"),
        requested_feature_key=requested_feature_key,
        expected_attestation=expected_attestation,
        feature_vector=feature_vector,
        selected_choice_readout_token_indices=selected_choice_readout_token_indices,
    )
    return CiftHttpExtractorResponse(
        feature_key=feature_key,
        feature_vector=feature_vector,
        selected_choice_readout_token_indices=selected_choice_readout_token_indices,
        unavailable_reason=unavailable_reason,
        provenance=_response_provenance(
            extractor_id=extractor_id,
            model_attestation=model_attestation,
            extraction_receipt=extraction_receipt,
        ),
    )


def _validate_expected_attestation(expected_attestation: CiftExpectedModelAttestation) -> None:
    if expected_attestation.model_id == "":
        raise CiftHttpExtractorError("CIFT expected attestation model_id must not be empty.")
    if expected_attestation.revision == "":
        raise CiftHttpExtractorError("CIFT expected attestation revision must not be empty.")
    if expected_attestation.selected_device == "":
        raise CiftHttpExtractorError("CIFT expected attestation selected_device must not be empty.")
    if expected_attestation.hidden_size < 1:
        raise CiftHttpExtractorError("CIFT expected attestation hidden_size must be positive.")
    if expected_attestation.layer_count < 1:
        raise CiftHttpExtractorError("CIFT expected attestation layer_count must be positive.")
    _validate_expected_sha256(
        value=expected_attestation.tokenizer_fingerprint_sha256,
        field_name="tokenizer_fingerprint_sha256",
    )
    _validate_expected_sha256(
        value=expected_attestation.special_tokens_map_sha256,
        field_name="special_tokens_map_sha256",
    )
    _validate_expected_sha256(
        value=expected_attestation.chat_template_sha256,
        field_name="chat_template_sha256",
    )
    if expected_attestation.prompt_renderer == "":
        raise CiftHttpExtractorError("CIFT expected attestation prompt_renderer must not be empty.")
    if expected_attestation.selected_choice_geometry == "":
        raise CiftHttpExtractorError("CIFT expected attestation selected_choice_geometry must not be empty.")
    if expected_attestation.selected_choice_readout_token_count < 1:
        raise CiftHttpExtractorError("CIFT expected attestation selected_choice_readout_token_count must be positive.")


def _validate_expected_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CiftHttpExtractorError(f"CIFT expected attestation {field_name} must be a lowercase SHA-256 digest.")


def _validate_response_attestation(
    value: JsonValue,
    expected_attestation: CiftExpectedModelAttestation,
) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise CiftHttpExtractorError("CIFT extractor response model_attestation must be an object.")
    schema_version = value.get("schema_version")
    if schema_version != CIFT_MODEL_ATTESTATION_SCHEMA_VERSION:
        raise CiftHttpExtractorError(
            f"CIFT extractor response model_attestation.schema_version must be {CIFT_MODEL_ATTESTATION_SCHEMA_VERSION}."
        )
    _expect_attestation_string(
        value=value.get("model_id"),
        field_name="model_id",
        expected_value=expected_attestation.model_id,
    )
    _expect_attestation_string(
        value=value.get("revision"),
        field_name="revision",
        expected_value=expected_attestation.revision,
    )
    _expect_attestation_string(
        value=value.get("selected_device"),
        field_name="selected_device",
        expected_value=expected_attestation.selected_device,
    )
    _expect_attestation_int(
        value=value.get("hidden_size"),
        field_name="hidden_size",
        expected_value=expected_attestation.hidden_size,
    )
    _expect_attestation_int(
        value=value.get("layer_count"),
        field_name="layer_count",
        expected_value=expected_attestation.layer_count,
    )
    _expect_attestation_string(
        value=value.get("tokenizer_fingerprint_sha256"),
        field_name="tokenizer_fingerprint_sha256",
        expected_value=expected_attestation.tokenizer_fingerprint_sha256,
    )
    _expect_attestation_string(
        value=value.get("special_tokens_map_sha256"),
        field_name="special_tokens_map_sha256",
        expected_value=expected_attestation.special_tokens_map_sha256,
    )
    _expect_attestation_string(
        value=value.get("chat_template_sha256"),
        field_name="chat_template_sha256",
        expected_value=expected_attestation.chat_template_sha256,
    )
    _expect_attestation_string(
        value=value.get("prompt_renderer"),
        field_name="prompt_renderer",
        expected_value=expected_attestation.prompt_renderer,
    )
    _expect_attestation_string(
        value=value.get("selected_choice_geometry"),
        field_name="selected_choice_geometry",
        expected_value=expected_attestation.selected_choice_geometry,
    )
    _expect_attestation_int(
        value=value.get("selected_choice_readout_token_count"),
        field_name="selected_choice_readout_token_count",
        expected_value=expected_attestation.selected_choice_readout_token_count,
    )
    return {
        "schema_version": CIFT_MODEL_ATTESTATION_SCHEMA_VERSION,
        "model_id": expected_attestation.model_id,
        "revision": expected_attestation.revision,
        "selected_device": expected_attestation.selected_device,
        "hidden_size": expected_attestation.hidden_size,
        "layer_count": expected_attestation.layer_count,
        "tokenizer_fingerprint_sha256": expected_attestation.tokenizer_fingerprint_sha256,
        "special_tokens_map_sha256": expected_attestation.special_tokens_map_sha256,
        "chat_template_sha256": expected_attestation.chat_template_sha256,
        "prompt_renderer": expected_attestation.prompt_renderer,
        "selected_choice_geometry": expected_attestation.selected_choice_geometry,
        "selected_choice_readout_token_count": expected_attestation.selected_choice_readout_token_count,
    }


def _response_provenance(
    extractor_id: str,
    model_attestation: dict[str, JsonValue],
    extraction_receipt: dict[str, JsonValue] | None,
) -> dict[str, JsonValue]:
    provenance: dict[str, JsonValue] = {
        "extractor_id": extractor_id,
        "model_attestation_schema_version": model_attestation["schema_version"],
        "model_id": model_attestation["model_id"],
        "revision": model_attestation["revision"],
        "selected_device": model_attestation["selected_device"],
        "hidden_size": model_attestation["hidden_size"],
        "layer_count": model_attestation["layer_count"],
        "tokenizer_fingerprint_sha256": model_attestation["tokenizer_fingerprint_sha256"],
        "special_tokens_map_sha256": model_attestation["special_tokens_map_sha256"],
        "chat_template_sha256": model_attestation["chat_template_sha256"],
        "prompt_renderer": model_attestation["prompt_renderer"],
        "selected_choice_geometry": model_attestation["selected_choice_geometry"],
        "selected_choice_readout_token_count": model_attestation["selected_choice_readout_token_count"],
    }
    if extraction_receipt is not None:
        provenance.update(
            {
                "extraction_receipt_schema_version": extraction_receipt["schema_version"],
                "feature_vector_length": extraction_receipt["feature_vector_length"],
                "feature_vector_sha256": extraction_receipt["feature_vector_sha256"],
                "rendered_prompt_sha256": extraction_receipt["rendered_prompt_sha256"],
                "selected_choice_readout_token_indices": extraction_receipt["selected_choice_readout_token_indices"],
                "selected_choice_readout_token_indices_sha256": extraction_receipt.get(
                    "selected_choice_readout_token_indices_sha256"
                ),
                "hidden_state_layer_count": extraction_receipt["hidden_state_layer_count"],
                "hidden_state_device_observed": extraction_receipt["hidden_state_device_observed"],
                "input_device_observed": extraction_receipt["input_device_observed"],
            }
        )
        for field_name in (
            "readout_token_indices",
            "readout_token_indices_sha256",
            "query_tail_readout_token_indices",
            "query_tail_readout_token_indices_sha256",
            "readout_window_source",
            "readout_source",
        ):
            if field_name in extraction_receipt:
                provenance[field_name] = extraction_receipt[field_name]
    return provenance


def _validate_extraction_receipt(
    value: JsonValue,
    requested_feature_key: str,
    expected_attestation: CiftExpectedModelAttestation,
    feature_vector: tuple[float, ...] | None,
    selected_choice_readout_token_indices: tuple[int, ...] | None,
) -> dict[str, JsonValue] | None:
    if feature_vector is None:
        if value is not None:
            raise CiftHttpExtractorError("CIFT extractor response extraction_receipt must be null without a feature.")
        return None
    if not isinstance(value, dict):
        raise CiftHttpExtractorError("CIFT extractor response extraction_receipt must be an object.")
    receipt = value
    _expect_receipt_string(receipt, "schema_version", CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION)
    _expect_receipt_string(receipt, "feature_key", requested_feature_key)
    _expect_receipt_string(receipt, "model_id", expected_attestation.model_id)
    _expect_receipt_string(receipt, "revision", expected_attestation.revision)
    _expect_receipt_string(receipt, "selected_device", expected_attestation.selected_device)
    _expect_receipt_int(receipt, "hidden_size", expected_attestation.hidden_size)
    _expect_receipt_int(receipt, "layer_count", expected_attestation.layer_count)
    _expect_receipt_string(
        receipt,
        "tokenizer_fingerprint_sha256",
        expected_attestation.tokenizer_fingerprint_sha256,
    )
    _expect_receipt_string(
        receipt,
        "special_tokens_map_sha256",
        expected_attestation.special_tokens_map_sha256,
    )
    _expect_receipt_string(receipt, "chat_template_sha256", expected_attestation.chat_template_sha256)
    _expect_receipt_string(receipt, "prompt_renderer", expected_attestation.prompt_renderer)
    _expect_receipt_string(receipt, "selected_choice_geometry", expected_attestation.selected_choice_geometry)
    _expect_receipt_int(
        receipt,
        "selected_choice_readout_configured_token_count",
        expected_attestation.selected_choice_readout_token_count,
    )
    _expect_receipt_int(receipt, "feature_vector_length", len(feature_vector))
    _expect_receipt_string(receipt, "feature_vector_sha256", _json_sha256([float(value) for value in feature_vector]))
    _require_sha256_string(receipt, "rendered_prompt_sha256")
    hidden_state_layer_count = _required_receipt_positive_int(receipt, "hidden_state_layer_count")
    if hidden_state_layer_count < expected_attestation.layer_count:
        raise CiftHttpExtractorError(
            "CIFT extractor response extraction_receipt.hidden_state_layer_count must be at least "
            "model_attestation.layer_count."
        )
    hidden_state_device_observed = _required_receipt_string(receipt, "hidden_state_device_observed")
    if not _device_matches_expected(hidden_state_device_observed, expected_attestation.selected_device):
        raise CiftHttpExtractorError(
            "CIFT extractor response extraction_receipt.hidden_state_device_observed must match "
            "model_attestation.selected_device."
        )
    input_device_observed = _required_receipt_string(receipt, "input_device_observed")
    if not _device_matches_expected(input_device_observed, expected_attestation.selected_device):
        raise CiftHttpExtractorError(
            "CIFT extractor response extraction_receipt.input_device_observed must match "
            "model_attestation.selected_device."
        )
    receipt_token_indices = _optional_int_tuple(
        receipt.get("selected_choice_readout_token_indices"),
        "extraction_receipt.selected_choice_readout_token_indices",
    )
    if receipt_token_indices != selected_choice_readout_token_indices:
        raise CiftHttpExtractorError(
            "CIFT extractor response extraction_receipt.selected_choice_readout_token_indices must match "
            "selected_choice_readout_token_indices."
        )
    selected_choice_count = _required_receipt_non_negative_int(receipt, "selected_choice_readout_token_count")
    if selected_choice_readout_token_indices is None:
        if selected_choice_count != 0:
            raise CiftHttpExtractorError(
                "CIFT extractor response extraction_receipt.selected_choice_readout_token_count must be zero "
                "without selected-choice indices."
            )
    else:
        if selected_choice_count != len(selected_choice_readout_token_indices):
            raise CiftHttpExtractorError(
                "CIFT extractor response extraction_receipt.selected_choice_readout_token_count must match "
                "selected-choice index count."
            )
        _expect_receipt_string(
            receipt,
            "selected_choice_readout_token_indices_sha256",
            _json_sha256([int(token_index) for token_index in selected_choice_readout_token_indices]),
        )
    return receipt


def _required_receipt_string(receipt: Mapping[str, JsonValue], field_name: str) -> str:
    value = receipt.get(field_name)
    if not isinstance(value, str) or value == "":
        raise CiftHttpExtractorError(f"CIFT extractor response extraction_receipt.{field_name} must be a string.")
    return value


def _required_receipt_positive_int(receipt: Mapping[str, JsonValue], field_name: str) -> int:
    value = receipt.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftHttpExtractorError(f"CIFT extractor response extraction_receipt.{field_name} must be an integer.")
    if value < 1:
        raise CiftHttpExtractorError(f"CIFT extractor response extraction_receipt.{field_name} must be positive.")
    return value


def _required_receipt_non_negative_int(receipt: Mapping[str, JsonValue], field_name: str) -> int:
    value = receipt.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftHttpExtractorError(f"CIFT extractor response extraction_receipt.{field_name} must be an integer.")
    if value < 0:
        raise CiftHttpExtractorError(f"CIFT extractor response extraction_receipt.{field_name} must be non-negative.")
    return value


def _expect_receipt_string(receipt: Mapping[str, JsonValue], field_name: str, expected_value: str) -> None:
    value = _required_receipt_string(receipt, field_name)
    if value != expected_value:
        raise CiftHttpExtractorError(
            f"CIFT extractor response extraction_receipt.{field_name} must be {expected_value}, got {value}."
        )


def _expect_receipt_int(receipt: Mapping[str, JsonValue], field_name: str, expected_value: int) -> None:
    value = _required_receipt_positive_int(receipt, field_name)
    if value != expected_value:
        raise CiftHttpExtractorError(
            f"CIFT extractor response extraction_receipt.{field_name} must be {expected_value}, got {value}."
        )


def _require_sha256_string(receipt: Mapping[str, JsonValue], field_name: str) -> str:
    value = _required_receipt_string(receipt, field_name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CiftHttpExtractorError(
            f"CIFT extractor response extraction_receipt.{field_name} must be a lowercase SHA-256 digest."
        )
    return value


def _device_matches_expected(observed_device: str, expected_device: str) -> bool:
    if expected_device == "cpu":
        return observed_device == "cpu"
    return observed_device == expected_device or observed_device.startswith(f"{expected_device}:")


def _expect_attestation_string(value: JsonValue, field_name: str, expected_value: str) -> None:
    if not isinstance(value, str):
        raise CiftHttpExtractorError(f"CIFT extractor response model_attestation.{field_name} must be a string.")
    if value != expected_value:
        raise CiftHttpExtractorError(
            f"CIFT extractor response model_attestation.{field_name} must be {expected_value}, got {value}."
        )


def _expect_attestation_int(value: JsonValue, field_name: str, expected_value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftHttpExtractorError(f"CIFT extractor response model_attestation.{field_name} must be an integer.")
    if value != expected_value:
        raise CiftHttpExtractorError(
            f"CIFT extractor response model_attestation.{field_name} must be {expected_value}, got {value}."
        )


def _optional_float_tuple(value: JsonValue, field_name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise CiftHttpExtractorError(f"CIFT extractor response {field_name} must be a list when present.")
    values: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CiftHttpExtractorError(f"CIFT extractor response {field_name}[{index}] must be a number.")
        parsed_item = float(item)
        if not math.isfinite(parsed_item):
            raise CiftHttpExtractorError(f"CIFT extractor response {field_name}[{index}] must be finite.")
        values.append(parsed_item)
    return tuple(values)


def _optional_int_tuple(value: JsonValue, field_name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise CiftHttpExtractorError(f"CIFT extractor response {field_name} must be a list when present.")
    values: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise CiftHttpExtractorError(f"CIFT extractor response {field_name}[{index}] must be an integer.")
        if item < 0:
            raise CiftHttpExtractorError(f"CIFT extractor response {field_name}[{index}] must be non-negative.")
        values.append(item)
    return tuple(values)


def _optional_non_empty_string(value: JsonValue, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CiftHttpExtractorError(f"CIFT extractor response {field_name} must be a string when present.")
    if value == "":
        raise CiftHttpExtractorError(f"CIFT extractor response {field_name} must not be empty when present.")
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"CIFT extractor returned non-finite JSON number {value}.")


def _json_sha256(value: JsonValue) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
