from __future__ import annotations

import sys

from aegis_introspection.cli.script_loader import load_introspection_script


def main() -> None:
    module = load_introspection_script("discover_cift_model_metadata.py", "CIFT model metadata script")
    exit_code = module.main(tuple(sys.argv[1:]))
    raise SystemExit(exit_code)
