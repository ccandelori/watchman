from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.trace_record_adapter import (  # noqa: E402
    TokenOffset,
    TracePromptConversionConfig,
    TraceRecordAdapterError,
    load_trace_records_jsonl,
    structured_prompt_records_from_trace_records,
    write_structured_prompt_jsonl,
)


class HuggingFaceOffsetEncoder:
    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer

    def encode_offsets(self, text: str) -> tuple[TokenOffset, ...]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offset_mapping = encoded.get("offset_mapping")
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Aegis trace collection records into structured CIFT prompt records."
    )
    parser.add_argument("--records", required=True, help="Input trace collection JSONL path.")
    parser.add_argument("--output", required=True, help="Output structured prompt JSONL path.")
    parser.add_argument("--model-id", required=False, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", required=False, default="main")
    parser.add_argument("--readout-token-count", required=False, type=int, default=8)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def _load_tokenizer(model_id: str, revision: str, local_files_only: bool) -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
    )


def main(argv: Sequence[str]) -> None:
    namespace = _build_parser().parse_args(argv)
    records_path = Path(str(namespace.records))
    output_path = Path(str(namespace.output))
    model_id = str(namespace.model_id)
    revision = str(namespace.revision)
    readout_token_count = int(namespace.readout_token_count)
    local_files_only = not bool(namespace.allow_download)

    tokenizer = _load_tokenizer(
        model_id=model_id,
        revision=revision,
        local_files_only=local_files_only,
    )
    conversion = structured_prompt_records_from_trace_records(
        records=load_trace_records_jsonl(records_path),
        encoder=HuggingFaceOffsetEncoder(tokenizer),
        config=TracePromptConversionConfig(readout_token_count=readout_token_count),
    )
    write_structured_prompt_jsonl(path=output_path, records=conversion.records)
    print(f"Wrote {len(conversion.records)} structured prompt records to {output_path}")
    print(f"Skipped {len(conversion.skipped_records)} trace records")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
