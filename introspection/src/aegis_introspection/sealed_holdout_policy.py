from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

from aegis.core.contracts import JsonValue

SEALED_HOLDOUT_TAG = "sealed_holdout"
UNSEAL_FLAG = "--allow-sealed-holdout"


class SealedHoldoutPolicyError(ValueError):
    """Raised when sealed holdout data is used without an explicit unseal override."""


def path_is_sealed_holdout(path: Path) -> bool:
    path_tokens = tuple(token for token in re.split(r"[^a-z0-9]+", path.name.lower()) if token != "")
    return "sealed" in path_tokens


def tags_are_sealed_holdout(tags: Iterable[str]) -> bool:
    return SEALED_HOLDOUT_TAG in set(tags)


def tag_rows_are_sealed_holdout(tag_rows: Iterable[Iterable[str]]) -> bool:
    return any(tags_are_sealed_holdout(tags=row) for row in tag_rows)


def add_unseal_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(UNSEAL_FLAG, action="store_true")


def assert_unsealed_path(path: Path, allow_sealed_holdout: bool, context: str) -> None:
    if allow_sealed_holdout:
        return
    if path_is_sealed_holdout(path):
        raise SealedHoldoutPolicyError(_message(context=context, detail=f"path '{path}' is marked sealed"))


def assert_unsealed_paths(paths: Iterable[Path], allow_sealed_holdout: bool, context: str) -> None:
    for path in paths:
        assert_unsealed_path(path=path, allow_sealed_holdout=allow_sealed_holdout, context=context)


def assert_unsealed_tag_rows(tag_rows: Iterable[Iterable[str]], allow_sealed_holdout: bool, context: str) -> None:
    if allow_sealed_holdout:
        return
    if tag_rows_are_sealed_holdout(tag_rows=tag_rows):
        raise SealedHoldoutPolicyError(_message(context=context, detail=f"row tags include '{SEALED_HOLDOUT_TAG}'"))


def assert_unsealed_jsonl_tags(path: Path, allow_sealed_holdout: bool, context: str) -> None:
    if allow_sealed_holdout:
        return
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SealedHoldoutPolicyError(f"Line {line_number}: invalid JSON in {path}: {exc.msg}.") from exc
            if not isinstance(decoded, dict):
                raise SealedHoldoutPolicyError(f"Line {line_number}: expected a JSON object in {path}.")
            assert_unsealed_tag_rows(
                tag_rows=(_record_tags(record=cast(Mapping[str, JsonValue], decoded), line_number=line_number),),
                allow_sealed_holdout=False,
                context=context,
            )


def _message(context: str, detail: str) -> str:
    return (
        f"Refusing to use sealed holdout data for {context}: {detail}. "
        f"Pass {UNSEAL_FLAG} only after recording the V4.3 unseal decision."
    )


def _record_tags(record: Mapping[str, JsonValue], line_number: int) -> Sequence[str]:
    return _top_level_tags(record=record, line_number=line_number) + _metadata_eval_tags(
        record=record,
        line_number=line_number,
    )


def _top_level_tags(record: Mapping[str, JsonValue], line_number: int) -> tuple[str, ...]:
    tags = record.get("tags")
    if tags is None:
        return ()
    return _string_list(value=tags, line_number=line_number, field_path="tags")


def _metadata_eval_tags(record: Mapping[str, JsonValue], line_number: int) -> tuple[str, ...]:
    metadata = record.get("metadata")
    if metadata is None:
        return ()
    if not isinstance(metadata, dict):
        raise SealedHoldoutPolicyError(f"Line {line_number}: field 'metadata' must be an object when present.")
    eval_record = metadata.get("eval")
    if eval_record is None:
        return ()
    if not isinstance(eval_record, dict):
        raise SealedHoldoutPolicyError(f"Line {line_number}: field 'metadata.eval' must be an object when present.")
    tags = eval_record.get("tags")
    if tags is None:
        return ()
    return _string_list(value=tags, line_number=line_number, field_path="metadata.eval.tags")


def _string_list(value: object, line_number: int, field_path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SealedHoldoutPolicyError(f"Line {line_number}: field '{field_path}' must be a list when present.")
    parsed_tags: list[str] = []
    for tag_index, tag in enumerate(value):
        if not isinstance(tag, str):
            raise SealedHoldoutPolicyError(f"Line {line_number}: tag {tag_index} must be a string.")
        parsed_tags.append(tag)
    return tuple(parsed_tags)
