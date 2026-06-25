from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class CiftModelMetadataError(ValueError):
    """Raised when model metadata cannot support CIFT certification."""


@dataclass(frozen=True)
class CiftModelMetadataConfig:
    model_id: str
    revision: str
    local_files_only: bool
    trust_remote_code: bool


@dataclass(frozen=True)
class CiftModelMetadataReport:
    schema_version: str
    model_id: str
    revision: str
    model_type: str
    hidden_size: int
    layer_count: int
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
    return CiftModelMetadataReport(
        schema_version="aegis_introspection.cift_model_metadata/v1",
        model_id=config.model_id,
        revision=config.revision,
        model_type=_model_type(model_config),
        hidden_size=_hidden_size(model_config),
        layer_count=_layer_count(model_config),
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
        "model_id": report.model_id,
        "revision": report.revision,
        "model_type": report.model_type,
        "hidden_size": report.hidden_size,
        "layer_count": report.layer_count,
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


def _sha256_json(record: Mapping[str, object]) -> str:
    return _sha256_text(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
