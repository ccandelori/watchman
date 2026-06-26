from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_introspection_script(script_name: str, error_label: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"aegis_introspection.cli._{script_name}", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {error_label}: {script_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"aegis_introspection.cli._{script_name}"] = module
    spec.loader.exec_module(module)
    return module
