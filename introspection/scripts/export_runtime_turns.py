from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
SRC_PATH = INTROSPECTION_ROOT / "src"
WORKSPACE_SRC_PATH = WORKSPACE_ROOT / "src"
for source_path in (SRC_PATH, WORKSPACE_SRC_PATH):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from aegis_introspection.runtime_bridge import RuntimeBridgeConfig, structured_prompt_to_normalized_turn  # noqa: E402
from aegis_introspection.sealed_holdout_policy import (  # noqa: E402
    SEALED_HOLDOUT_TAG,
    add_unseal_flag,
    assert_unsealed_jsonl_tags,
    assert_unsealed_paths,
)

from aegis.core.contracts import JsonValue  # noqa: E402


@dataclass(frozen=True)
class ExportRuntimeTurnsConfig:
    input_path: Path
    output_path: Path
    capability_mode: str
    model_provider: str
    model_id: str
    revision: str | None
    selected_device: str | None
    sensitive_source: str
    session_id: str
    allow_sealed_holdout: bool
    sealed_holdout_split_id: str | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export structured prompts as Aegis NormalizedTurn-shaped JSONL.")
    parser.add_argument(
        "--input",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "prompts_dp_honey_lite_v3_selector_windows.jsonl"),
    )
    parser.add_argument(
        "--output",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "runtime_turns_dp_honey_lite_v3_selector_windows.jsonl"),
    )
    parser.add_argument("--capability-mode", required=False, default="offline_eval")
    parser.add_argument("--model-provider", required=False, default="huggingface")
    parser.add_argument("--model-id", required=False, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", required=False, default="main")
    parser.add_argument("--selected-device", required=False, default="cpu")
    parser.add_argument("--sensitive-source", required=False, default="dp_honey_lite")
    parser.add_argument("--session-id", required=False, default="introspection-offline-eval")
    parser.add_argument(
        "--sealed-holdout-split-id",
        required=False,
        help="When set, tag exported runtime turns with sealed_holdout and split:<id>.",
    )
    add_unseal_flag(parser)
    return parser


def _parse_args(argv: Sequence[str]) -> ExportRuntimeTurnsConfig:
    namespace = _build_parser().parse_args(argv)
    return ExportRuntimeTurnsConfig(
        input_path=Path(namespace.input),
        output_path=Path(namespace.output),
        capability_mode=str(namespace.capability_mode),
        model_provider=str(namespace.model_provider),
        model_id=str(namespace.model_id),
        revision=str(namespace.revision),
        selected_device=str(namespace.selected_device),
        sensitive_source=str(namespace.sensitive_source),
        session_id=str(namespace.session_id),
        allow_sealed_holdout=bool(namespace.allow_sealed_holdout),
        sealed_holdout_split_id=_optional_string(namespace.sealed_holdout_split_id),
    )


def _load_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise TypeError(f"Line {line_number}: expected a JSON object.")
            records.append(cast(Mapping[str, object], decoded))
    if len(records) == 0:
        raise ValueError(f"No records found in {path}.")
    return tuple(records)


def _trace_id(record: Mapping[str, object], turn_index: int) -> str:
    example_id = record.get("example_id")
    if isinstance(example_id, str) and example_id != "":
        return f"trace-{example_id}"
    return f"trace-{turn_index}"


def _runtime_bridge_config(
    record: Mapping[str, object],
    export_config: ExportRuntimeTurnsConfig,
    turn_index: int,
) -> RuntimeBridgeConfig:
    return RuntimeBridgeConfig(
        trace_id=_trace_id(record=record, turn_index=turn_index),
        session_id=export_config.session_id,
        turn_index=turn_index,
        capability_mode=export_config.capability_mode,
        model_provider=export_config.model_provider,
        model_id=export_config.model_id,
        revision=export_config.revision,
        selected_device=export_config.selected_device,
        sensitive_source=export_config.sensitive_source,
    )


def run_export(config: ExportRuntimeTurnsConfig) -> None:
    _validate_sealed_tagging(config)
    assert_unsealed_paths(
        paths=(config.input_path, config.output_path),
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="runtime turn export",
    )
    assert_unsealed_jsonl_tags(
        path=config.input_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="runtime turn export",
    )
    records = _load_jsonl(config.input_path)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as file:
        for turn_index, record in enumerate(records, start=1):
            turn = structured_prompt_to_normalized_turn(
                record=record,
                config=_runtime_bridge_config(record=record, export_config=config, turn_index=turn_index),
            )
            _tag_sealed_holdout_turn(turn=turn, config=config)
            json.dump(cast(dict[str, JsonValue], turn), file, ensure_ascii=False)
            file.write("\n")
    print(f"Wrote {len(records)} runtime turn rows to {config.output_path}")


def _validate_sealed_tagging(config: ExportRuntimeTurnsConfig) -> None:
    if config.sealed_holdout_split_id is None:
        return
    if config.sealed_holdout_split_id == "":
        raise ValueError("sealed_holdout_split_id must not be empty when set.")
    if config.sealed_holdout_split_id.startswith("split:"):
        raise ValueError("sealed_holdout_split_id must not include the 'split:' tag prefix.")
    if not config.allow_sealed_holdout:
        raise ValueError("sealed_holdout_split_id requires --allow-sealed-holdout.")


def _tag_sealed_holdout_turn(turn: dict[str, JsonValue], config: ExportRuntimeTurnsConfig) -> None:
    if config.sealed_holdout_split_id is None:
        return
    eval_metadata = _required_json_object(
        record=_required_json_object(record=turn, field_name="metadata", context="runtime turn"),
        field_name="eval",
        context="runtime turn.metadata",
    )
    tags = _required_string_list(record=eval_metadata, field_name="tags", context="runtime turn.metadata.eval")
    tagged = sorted(set(tags) | {SEALED_HOLDOUT_TAG, f"split:{config.sealed_holdout_split_id}"})
    eval_metadata["tags"] = tagged


def _required_json_object(
    record: Mapping[str, JsonValue],
    field_name: str,
    context: str,
) -> dict[str, JsonValue]:
    value = record.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{field_name} must be an object.")
    for key in value:
        if not isinstance(key, str):
            raise ValueError(f"{context}.{field_name} keys must be strings.")
    return cast(dict[str, JsonValue], value)


def _required_string_list(record: Mapping[str, JsonValue], field_name: str, context: str) -> tuple[str, ...]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise ValueError(f"{context}.{field_name} must be a list of strings.")
    strings: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or item == "":
            raise ValueError(f"{context}.{field_name}[{index}] must be a non-empty string.")
        strings.append(item)
    return tuple(strings)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional string argument must be a string when set.")
    return value


def main(argv: Sequence[str]) -> None:
    run_export(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
