from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
SRC_PATH = INTROSPECTION_ROOT / "src"
WORKSPACE_SRC_PATH = WORKSPACE_ROOT / "src"
for source_path in (SRC_PATH, WORKSPACE_SRC_PATH):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from aegis_introspection.cift_runtime_turn_mixing import (  # noqa: E402
    CiftRuntimeTurnMixConfig,
    build_mixed_cift_window_runtime_turns,
    load_runtime_turn_jsonl,
    write_runtime_turn_jsonl,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic mixed-route CIFT runtime-turn JSONL fixture.")
    parser.add_argument("--input", required=True, help="Input runtime turns JSONL path.")
    parser.add_argument("--output", required=True, help="Output mixed-route runtime turns JSONL path.")
    parser.add_argument("--fallback-modulus", required=True, type=int, help="Per-label route modulus.")
    parser.add_argument("--fallback-remainder", required=True, type=int, help="Remainder assigned to fallback route.")
    return parser


def main(argv: tuple[str, ...]) -> None:
    namespace = _build_parser().parse_args(list(argv))
    turns = load_runtime_turn_jsonl(path=Path(str(namespace.input)))
    result = build_mixed_cift_window_runtime_turns(
        turns=turns,
        config=CiftRuntimeTurnMixConfig(
            fallback_modulus=int(namespace.fallback_modulus),
            fallback_remainder=int(namespace.fallback_remainder),
        ),
    )
    write_runtime_turn_jsonl(path=Path(str(namespace.output)), turns=result.turns)
    print(f"Wrote {len(result.turns)} mixed-route runtime turns to {namespace.output}")
    print(f"Window family counts: {result.window_family_counts}")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
