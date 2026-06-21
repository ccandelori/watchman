from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence, cast

from aegis_introspection.artifacts import ActivationArtifact, load_activation_artifact_allowing_sealed_holdout
from aegis_introspection.probe import JsonValue


SEALED_HOLDOUT_TAG = "sealed_holdout"
UNSEAL_FLAG = "--allow-sealed-holdout"


class SealedHoldoutError(ValueError):
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
        raise SealedHoldoutError(_message(context=context, detail=f"path '{path}' is marked sealed"))


def assert_unsealed_paths(paths: Iterable[Path], allow_sealed_holdout: bool, context: str) -> None:
    for path in paths:
        assert_unsealed_path(path=path, allow_sealed_holdout=allow_sealed_holdout, context=context)


def assert_unsealed_tag_rows(tag_rows: Iterable[Iterable[str]], allow_sealed_holdout: bool, context: str) -> None:
    if allow_sealed_holdout:
        return
    if tag_rows_are_sealed_holdout(tag_rows=tag_rows):
        raise SealedHoldoutError(_message(context=context, detail=f"row tags include '{SEALED_HOLDOUT_TAG}'"))


def assert_unsealed_jsonl_tags(path: Path, allow_sealed_holdout: bool, context: str) -> None:
    if allow_sealed_holdout:
        return
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise SealedHoldoutError(f"Line {line_number}: expected a JSON object in {path}.")
            assert_unsealed_tag_rows(
                tag_rows=(_record_tags(record=cast(Mapping[str, JsonValue], decoded), line_number=line_number),),
                allow_sealed_holdout=False,
                context=context,
            )


def assert_unsealed_activation_artifact_tags(
    artifact: ActivationArtifact,
    allow_sealed_holdout: bool,
    context: str,
) -> None:
    assert_unsealed_tag_rows(
        tag_rows=artifact["tags"],
        allow_sealed_holdout=allow_sealed_holdout,
        context=context,
    )


def assert_unsealed_activation_artifact_path(path: Path, allow_sealed_holdout: bool, context: str) -> None:
    assert_unsealed_path(path=path, allow_sealed_holdout=allow_sealed_holdout, context=context)
    if allow_sealed_holdout:
        return
    assert_unsealed_activation_artifact_tags(
        artifact=load_activation_artifact_allowing_sealed_holdout(path),
        allow_sealed_holdout=False,
        context=context,
    )


def load_activation_artifact_with_unseal_policy(
    path: Path,
    allow_sealed_holdout: bool,
    context: str,
) -> ActivationArtifact:
    assert_unsealed_path(path=path, allow_sealed_holdout=allow_sealed_holdout, context=context)
    artifact = load_activation_artifact_allowing_sealed_holdout(path)
    assert_unsealed_activation_artifact_tags(
        artifact=artifact,
        allow_sealed_holdout=allow_sealed_holdout,
        context=context,
    )
    return artifact


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
        raise SealedHoldoutError(f"Line {line_number}: field 'metadata' must be an object when present.")
    eval_record = metadata.get("eval")
    if eval_record is None:
        return ()
    if not isinstance(eval_record, dict):
        raise SealedHoldoutError(f"Line {line_number}: field 'metadata.eval' must be an object when present.")
    tags = eval_record.get("tags")
    if tags is None:
        return ()
    return _string_list(value=tags, line_number=line_number, field_path="metadata.eval.tags")


def _string_list(value: object, line_number: int, field_path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SealedHoldoutError(f"Line {line_number}: field '{field_path}' must be a list when present.")
    parsed_tags: list[str] = []
    for tag_index, tag in enumerate(value):
        if not isinstance(tag, str):
            raise SealedHoldoutError(f"Line {line_number}: tag {tag_index} must be a string.")
        parsed_tags.append(tag)
    return tuple(parsed_tags)
