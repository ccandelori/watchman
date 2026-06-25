from __future__ import annotations

from typing import cast

from aegis_introspection.trace_record_adapter import TokenOffset, TraceRecordAdapterError


class HuggingFaceOffsetEncoder:
    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer

    def encode_offsets(self, text: str) -> tuple[TokenOffset, ...]:
        if not callable(self._tokenizer):
            raise TraceRecordAdapterError("tokenizer must be callable.")
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        get = getattr(encoded, "get", None)
        if get is None:
            raise TraceRecordAdapterError("Tokenizer output must expose get().")
        offset_mapping = get("offset_mapping")
        if not isinstance(offset_mapping, list):
            raise TraceRecordAdapterError("Tokenizer did not return a list offset_mapping.")

        offsets: list[TokenOffset] = []
        for index, item in enumerate(offset_mapping):
            if not isinstance(item, list | tuple):
                raise TraceRecordAdapterError(f"offset_mapping[{index}] must be a two-item sequence.")
            if len(item) != 2:
                raise TraceRecordAdapterError(f"offset_mapping[{index}] must contain exactly two values.")
            offsets.append(TokenOffset(start=int(item[0]), end=int(item[1])))
        return tuple(offsets)


def load_huggingface_tokenizer(model_id: str, revision: str, local_files_only: bool) -> object:
    from transformers import AutoTokenizer

    return cast(
        object,
        AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        ),
    )
