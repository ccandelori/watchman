from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence, TypeAlias, cast

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from transformers import AutoTokenizer

from aegis_introspection.honeytokens import TokenOffset
from aegis_introspection.policy_windows import V3PolicyWindow, derive_v3_policy_window, derive_v3_selector_window
from aegis_introspection.probe import JsonValue
from aegis_introspection.prompts import load_structured_prompt_examples


PolicyWindowKind: TypeAlias = Literal["decision_path", "selector"]


class OffsetTokenizer(Protocol):
    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        add_special_tokens: bool,
    ) -> Mapping[str, object]:
        ...


@dataclass(frozen=True)
class GenerateV3PolicyWindowPromptsConfig:
    input_path: Path
    output_path: Path
    model_id: str
    revision: str
    local_files_only: bool
    window_kind: PolicyWindowKind


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DP-HONEY-lite V3 prompt rows with policy-window readouts.")
    parser.add_argument(
        "--input",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "prompts_dp_honey_lite_v3.jsonl"),
    )
    parser.add_argument(
        "--output",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "prompts_dp_honey_lite_v3_policy_windows.jsonl"),
    )
    parser.add_argument("--model-id", required=False, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", required=False, default="main")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--window-kind", required=False, choices=("decision_path", "selector"), default="decision_path")
    return parser


def _parse_args(argv: Sequence[str]) -> GenerateV3PolicyWindowPromptsConfig:
    namespace = _build_parser().parse_args(argv)
    return GenerateV3PolicyWindowPromptsConfig(
        input_path=Path(namespace.input),
        output_path=Path(namespace.output),
        model_id=str(namespace.model_id),
        revision=str(namespace.revision),
        local_files_only=not bool(namespace.allow_download),
        window_kind=cast(PolicyWindowKind, namespace.window_kind),
    )


def _load_tokenizer(config: GenerateV3PolicyWindowPromptsConfig) -> OffsetTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        trust_remote_code=True,
        local_files_only=config.local_files_only,
    )
    return cast(OffsetTokenizer, tokenizer)


def _offset_sequence(value: object, field_name: str) -> tuple[TokenOffset, ...]:
    if not isinstance(value, list):
        raise TypeError(f"Tokenizer field '{field_name}' must be a list.")
    offsets: list[TokenOffset] = []
    for index, item in enumerate(value):
        if not isinstance(item, tuple) and not isinstance(item, list):
            raise TypeError(f"Tokenizer field '{field_name}' item {index} must be a pair.")
        if len(item) != 2:
            raise TypeError(f"Tokenizer field '{field_name}' item {index} must contain two values.")
        start, end = item
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError(f"Tokenizer field '{field_name}' item {index} must contain integers.")
        offsets.append(TokenOffset(start=start, end=end))
    return tuple(offsets)


def _token_offsets(tokenizer: OffsetTokenizer, text: str) -> tuple[TokenOffset, ...]:
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    return _offset_sequence(value=encoded.get("offset_mapping"), field_name="offset_mapping")


def _load_jsonl(path: Path) -> tuple[dict[str, JsonValue], ...]:
    records: list[dict[str, JsonValue]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise TypeError(f"Line {line_number}: expected a JSON object.")
            records.append(cast(dict[str, JsonValue], decoded))
    if len(records) == 0:
        raise ValueError(f"No prompt records found in {path}.")
    return tuple(records)


def _record_text(record: Mapping[str, JsonValue], line_number: int) -> str:
    text = record.get("text")
    rendered_prompt = record.get("rendered_prompt")
    if not isinstance(text, str) or text == "":
        raise TypeError(f"Line {line_number}: field 'text' must be a non-empty string.")
    if rendered_prompt != text:
        raise TypeError(f"Line {line_number}: fields 'text' and 'rendered_prompt' must match.")
    return text


def _policy_window_record(
    record: Mapping[str, JsonValue],
    text: str,
    offsets: tuple[TokenOffset, ...],
    window_kind: PolicyWindowKind,
) -> dict[str, JsonValue]:
    window = _derive_window(text=text, offsets=offsets, window_kind=window_kind)
    policy_window_token_indices = list(window.token_indices)
    policy_window_char_spans: list[JsonValue] = [[span.start, span.end] for span in window.char_spans]
    updated = dict(record)
    updated["readout_token_indices"] = policy_window_token_indices
    updated["policy_window_token_indices"] = policy_window_token_indices
    updated["policy_window_char_spans"] = policy_window_char_spans
    updated["policy_window_selected_field"] = window.selected_field
    updated["policy_window_selected_mode"] = window.selected_mode
    updated["policy_window_selected_action"] = window.selected_action
    updated["policy_window_kind"] = window_kind
    return updated


def _derive_window(text: str, offsets: tuple[TokenOffset, ...], window_kind: PolicyWindowKind) -> V3PolicyWindow:
    if window_kind == "decision_path":
        return derive_v3_policy_window(text=text, offsets=offsets)
    if window_kind == "selector":
        return derive_v3_selector_window(text=text, offsets=offsets)
    raise ValueError(f"Unsupported policy window kind '{window_kind}'.")


def run_generation(config: GenerateV3PolicyWindowPromptsConfig) -> None:
    tokenizer = _load_tokenizer(config)
    records = _load_jsonl(config.input_path)
    updated_records: list[dict[str, JsonValue]] = []
    for line_number, record in enumerate(records, start=1):
        text = _record_text(record=record, line_number=line_number)
        updated_records.append(
            _policy_window_record(
                record=record,
                text=text,
                offsets=_token_offsets(tokenizer=tokenizer, text=text),
                window_kind=config.window_kind,
            )
        )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as file:
        for record in updated_records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")

    load_structured_prompt_examples(config.output_path)
    print(f"Wrote {len(updated_records)} V3 policy-window prompt rows to {config.output_path}")


def main(argv: Sequence[str]) -> None:
    run_generation(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
