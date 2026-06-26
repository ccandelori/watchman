from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from aegis.cift_contract import (
    CIFT_SUPPORT_STATE_CALIBRATION_READY,
    CIFT_SUPPORT_STATE_DISCOVERED,
    CIFT_SUPPORT_STATE_HIDDEN_STATE_CAPABLE,
)

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
_VALID_DTYPE_NAMES = frozenset(("auto", "device", "float32", "float16", "bfloat16"))


class CiftModelMetadataError(ValueError):
    """Raised when model metadata cannot support CIFT certification."""


@dataclass(frozen=True)
class CiftModelMetadataConfig:
    model_id: str
    revision: str
    requested_device: str
    dtype_name: str
    selected_readout_candidates: tuple[str, ...]
    local_files_only: bool
    trust_remote_code: bool


@dataclass(frozen=True)
class CiftModelMetadataReport:
    schema_version: str
    support_state: str
    model_id: str
    revision: str
    resolved_revision: str
    model_type: str
    hidden_size: int
    layer_count: int
    requested_device: str
    selected_device: str
    dtype_name: str
    resolved_torch_dtype: str
    hidden_state_support: str
    hidden_state_capable: bool
    selected_readout_candidates: tuple[str, ...]
    failure_reason: str | None
    tokenizer_class: str
    tokenizer_vocab_size: int
    tokenizer_fingerprint_sha256: str
    special_tokens_map_sha256: str
    chat_template_present: bool
    chat_template_sha256: str


def discover_cift_model_metadata(config: CiftModelMetadataConfig) -> CiftModelMetadataReport:
    _validate_config(config)
    from transformers import AutoConfig, AutoTokenizer

    model_config = AutoConfig.from_pretrained(
        config.model_id,
        revision=config.revision,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    return cift_model_metadata_report_from_loaded_objects(
        config=config,
        model_config=model_config,
        tokenizer=tokenizer,
    )


def cift_model_metadata_report_from_loaded_objects(
    config: CiftModelMetadataConfig,
    model_config: object,
    tokenizer: object,
) -> CiftModelMetadataReport:
    _validate_config(config)
    chat_template = _chat_template(tokenizer)
    hidden_state_support = _hidden_state_support(model_config)
    hidden_state_capable = hidden_state_support != "unknown"
    selected_device, resolved_dtype = _resolve_device_and_dtype(
        requested_device=config.requested_device,
        dtype_name=config.dtype_name,
    )
    selected_readout_candidates = _selected_readout_candidates(config.selected_readout_candidates)
    return CiftModelMetadataReport(
        schema_version="aegis_introspection.cift_model_metadata/v1",
        support_state=_support_state(
            hidden_state_capable=hidden_state_capable,
            selected_readout_candidates=selected_readout_candidates,
        ),
        model_id=config.model_id,
        revision=config.revision,
        resolved_revision=_resolved_revision(config=config, model_config=model_config, tokenizer=tokenizer),
        model_type=_model_type(model_config),
        hidden_size=_hidden_size(model_config),
        layer_count=_layer_count(model_config),
        requested_device=config.requested_device,
        selected_device=selected_device,
        dtype_name=config.dtype_name,
        resolved_torch_dtype=str(resolved_dtype),
        hidden_state_support=hidden_state_support,
        hidden_state_capable=hidden_state_capable,
        selected_readout_candidates=selected_readout_candidates,
        failure_reason=None,
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_vocab_size=_tokenizer_vocab_size(tokenizer),
        tokenizer_fingerprint_sha256=_tokenizer_fingerprint_sha256(tokenizer),
        special_tokens_map_sha256=_special_tokens_map_sha256(tokenizer),
        chat_template_present=chat_template != "",
        chat_template_sha256=_sha256_text(chat_template),
    )


def cift_model_metadata_report_to_json(report: CiftModelMetadataReport) -> dict[str, JsonValue]:
    return {
        "schema_version": report.schema_version,
        "support_state": report.support_state,
        "model_id": report.model_id,
        "revision": report.revision,
        "resolved_revision": report.resolved_revision,
        "model_type": report.model_type,
        "hidden_size": report.hidden_size,
        "layer_count": report.layer_count,
        "requested_device": report.requested_device,
        "selected_device": report.selected_device,
        "dtype_name": report.dtype_name,
        "resolved_torch_dtype": report.resolved_torch_dtype,
        "hidden_state_support": report.hidden_state_support,
        "hidden_state_capable": report.hidden_state_capable,
        "selected_readout_candidates": list(report.selected_readout_candidates),
        "failure_reason": report.failure_reason,
        "tokenizer_class": report.tokenizer_class,
        "tokenizer_vocab_size": report.tokenizer_vocab_size,
        "tokenizer_fingerprint_sha256": report.tokenizer_fingerprint_sha256,
        "special_tokens_map_sha256": report.special_tokens_map_sha256,
        "chat_template_present": report.chat_template_present,
        "chat_template_sha256": report.chat_template_sha256,
    }


def _validate_config(config: CiftModelMetadataConfig) -> None:
    if config.model_id == "":
        raise CiftModelMetadataError("model_id must not be empty.")
    if config.revision == "":
        raise CiftModelMetadataError("revision must not be empty.")
    if config.requested_device == "":
        raise CiftModelMetadataError("requested_device must not be empty.")
    if config.dtype_name == "":
        raise CiftModelMetadataError("dtype_name must not be empty.")
    _validate_dtype_name(config.dtype_name)
    _selected_readout_candidates(config.selected_readout_candidates)


def _model_type(model_config: object) -> str:
    model_type = getattr(model_config, "model_type", None)
    if not isinstance(model_type, str) or model_type == "":
        raise CiftModelMetadataError("model config must expose non-empty model_type.")
    return model_type


def _hidden_size(model_config: object) -> int:
    return _positive_int_attribute(
        owner=model_config,
        attribute_names=("hidden_size", "n_embd", "d_model"),
        field_name="hidden_size",
    )


def _layer_count(model_config: object) -> int:
    return _positive_int_attribute(
        owner=model_config,
        attribute_names=("num_hidden_layers", "n_layer", "num_layers"),
        field_name="layer_count",
    )


def _positive_int_attribute(owner: object, attribute_names: tuple[str, ...], field_name: str) -> int:
    for attribute_name in attribute_names:
        value = getattr(owner, attribute_name, None)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value < 1:
            raise CiftModelMetadataError(f"model config {attribute_name} must be positive.")
        return value
    joined_names = ", ".join(attribute_names)
    raise CiftModelMetadataError(f"model config must expose {field_name} through one of: {joined_names}.")


def _tokenizer_vocab_size(tokenizer: object) -> int:
    value = getattr(tokenizer, "vocab_size", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CiftModelMetadataError("tokenizer must expose positive vocab_size.")
    return value


def _tokenizer_fingerprint_sha256(tokenizer: object) -> str:
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    backend_to_str = getattr(backend_tokenizer, "to_str", None)
    if callable(backend_to_str):
        backend_json = backend_to_str()
        if not isinstance(backend_json, str) or backend_json == "":
            raise CiftModelMetadataError("tokenizer backend_tokenizer.to_str() must return non-empty text.")
        return _sha256_text(backend_json)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if not isinstance(vocab, dict) or len(vocab) == 0:
            raise CiftModelMetadataError("tokenizer get_vocab() must return a non-empty dictionary.")
        return _sha256_json(cast(Mapping[str, object], vocab))
    raise CiftModelMetadataError("tokenizer must expose backend_tokenizer.to_str() or get_vocab().")


def _special_tokens_map_sha256(tokenizer: object) -> str:
    special_tokens_map = getattr(tokenizer, "special_tokens_map", None)
    if not isinstance(special_tokens_map, dict):
        raise CiftModelMetadataError("tokenizer must expose special_tokens_map as a dictionary.")
    return _sha256_json(cast(Mapping[str, object], special_tokens_map))


def _chat_template(tokenizer: object) -> str:
    value = getattr(tokenizer, "chat_template", None)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise CiftModelMetadataError("tokenizer chat_template must be a string when present.")
    return value


def _hidden_state_support(model_config: object) -> str:
    if hasattr(model_config, "output_hidden_states"):
        return "configurable_output_hidden_states"
    to_dict = getattr(model_config, "to_dict", None)
    if callable(to_dict):
        decoded = to_dict()
        if isinstance(decoded, dict) and "output_hidden_states" in decoded:
            return "configurable_output_hidden_states"
    return "unknown"


def _support_state(hidden_state_capable: bool, selected_readout_candidates: tuple[str, ...]) -> str:
    if not hidden_state_capable:
        return CIFT_SUPPORT_STATE_DISCOVERED
    if len(selected_readout_candidates) == 0:
        return CIFT_SUPPORT_STATE_HIDDEN_STATE_CAPABLE
    return CIFT_SUPPORT_STATE_CALIBRATION_READY


def _selected_readout_candidates(raw_candidates: tuple[str, ...]) -> tuple[str, ...]:
    candidates: list[str] = []
    for raw_candidate in raw_candidates:
        candidate = raw_candidate.strip()
        if candidate == "":
            continue
        if candidate in candidates:
            raise CiftModelMetadataError(f"duplicate selected readout candidate: {candidate}.")
        candidates.append(candidate)
    return tuple(candidates)


def _validate_dtype_name(dtype_name: str) -> None:
    if dtype_name not in _VALID_DTYPE_NAMES:
        valid_names = ", ".join(sorted(_VALID_DTYPE_NAMES))
        raise CiftModelMetadataError(f"dtype_name must be one of: {valid_names}.")


def _resolve_device_and_dtype(requested_device: str, dtype_name: str) -> tuple[str, str]:
    try:
        from aegis_introspection.model_loader import parse_model_dtype, resolve_model_load_dtype, select_device
    except ModuleNotFoundError as exc:
        if requested_device != "cpu":
            raise CiftModelMetadataError(
                f"{requested_device} was requested, but torch-backed device validation is unavailable: {exc}."
            ) from exc
        return "cpu", _cpu_fallback_dtype(dtype_name)
    device = select_device(requested_device)
    parsed_dtype = parse_model_dtype(dtype_name)
    return device.name, str(resolve_model_load_dtype(dtype_name=parsed_dtype, device=device))


def _cpu_fallback_dtype(dtype_name: str) -> str:
    if dtype_name in ("auto", "device", "float32"):
        return "torch.float32"
    if dtype_name == "float16":
        return "torch.float16"
    if dtype_name == "bfloat16":
        return "torch.bfloat16"
    _validate_dtype_name(dtype_name)
    raise CiftModelMetadataError(f"dtype_name {dtype_name!r} could not be resolved.")


def _resolved_revision(config: CiftModelMetadataConfig, model_config: object, tokenizer: object) -> str:
    config_commit = getattr(model_config, "_commit_hash", None)
    if isinstance(config_commit, str) and config_commit != "":
        return config_commit
    tokenizer_init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(tokenizer_init_kwargs, dict):
        tokenizer_commit = tokenizer_init_kwargs.get("_commit_hash")
        if isinstance(tokenizer_commit, str) and tokenizer_commit != "":
            return tokenizer_commit
    return config.revision


def _sha256_json(record: Mapping[str, object]) -> str:
    return _sha256_text(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
