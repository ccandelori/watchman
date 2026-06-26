from __future__ import annotations

import sys
from collections.abc import Sequence

from aegis_introspection.cli.script_loader import load_introspection_script


def main() -> None:
    module = load_introspection_script("certify_cift_local_model.py", "CIFT certification script")
    exit_code = module.main(tuple(sys.argv[1:]))
    raise SystemExit(exit_code)


def run(argv: Sequence[str]) -> int:
    module = load_introspection_script("certify_cift_local_model.py", "CIFT certification script")
    return int(module.main(tuple(argv)))
