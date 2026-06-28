from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class MergeJsonlError(ValueError):
    """Raised when JSONL files cannot be merged safely."""


@dataclass(frozen=True)
class MergeJsonlConfig:
    input_paths: tuple[Path, ...]
    output_path: Path
    overwrite: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge JSONL object files after validating every row.")
    parser.add_argument("--input", required=True, action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_args(argv: Sequence[str]) -> MergeJsonlConfig:
    namespace = _build_parser().parse_args(argv)
    input_paths = tuple(Path(str(value)) for value in namespace.input)
    if len(input_paths) < 1:
        raise MergeJsonlError("at least one --input path is required.")
    return MergeJsonlConfig(
        input_paths=input_paths,
        output_path=Path(str(namespace.output)),
        overwrite=bool(namespace.overwrite),
    )


def merge_jsonl(config: MergeJsonlConfig) -> int:
    if config.output_path.exists() and not config.overwrite:
        raise MergeJsonlError(f"output path already exists: {config.output_path}.")
    rows: list[str] = []
    for input_path in config.input_paths:
        if not input_path.is_file():
            raise MergeJsonlError(f"input path does not exist: {input_path}.")
        for line_number, raw_line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if line == "":
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MergeJsonlError(f"{input_path}:{line_number}: invalid JSON: {exc.msg}.") from exc
            if not isinstance(decoded, dict):
                raise MergeJsonlError(f"{input_path}:{line_number}: expected a JSON object.")
            rows.append(json.dumps(decoded, sort_keys=True))
    if len(rows) == 0:
        raise MergeJsonlError("merged output would be empty.")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def main(argv: Sequence[str]) -> int:
    try:
        count = merge_jsonl(_parse_args(argv))
    except MergeJsonlError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    print(f"Wrote {count} JSONL records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
