from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


def _load_import_boundary_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_import_boundaries.py"
    spec = importlib.util.spec_from_file_location("check_import_boundaries", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load check_import_boundaries.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_import_boundaries = _load_import_boundary_module()
boundary_violations_for_file = check_import_boundaries.boundary_violations_for_file
import_is_forbidden = check_import_boundaries.import_is_forbidden
module_name_for_path = check_import_boundaries.module_name_for_path


class ImportBoundaryTest(unittest.TestCase):
    def test_runtime_package_importing_introspection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "src"
            source_path = source_root / "aegis" / "detectors" / "bad.py"
            source_path.parent.mkdir(parents=True)
            source_path.write_text("import aegis_introspection.runtime_bridge\n", encoding="utf-8")

            violations = boundary_violations_for_file(source_path=source_path, source_root=source_root)

        self.assertEqual(1, len(violations))
        self.assertEqual("aegis.detectors.bad", violations[0].source_module)
        self.assertEqual("aegis_introspection.runtime_bridge", violations[0].imported_module)

    def test_detector_importing_policy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "src"
            source_path = source_root / "aegis" / "detectors" / "bad.py"
            source_path.parent.mkdir(parents=True)
            source_path.write_text("from aegis.policy.engine import SeverityPolicyEngine\n", encoding="utf-8")

            violations = boundary_violations_for_file(source_path=source_path, source_root=source_root)

        self.assertEqual(1, len(violations))
        self.assertEqual("aegis.policy.engine", violations[0].imported_module)

    def test_detector_importing_core_contracts_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "src"
            source_path = source_root / "aegis" / "detectors" / "good.py"
            source_path.parent.mkdir(parents=True)
            source_path.write_text("from aegis.core.contracts import DetectorResult\n", encoding="utf-8")

            violations = boundary_violations_for_file(source_path=source_path, source_root=source_root)

        self.assertEqual((), violations)

    def test_module_name_for_init_file_uses_package_name(self) -> None:
        source_root = Path("/tmp/repo/src")
        source_path = source_root / "aegis" / "core" / "__init__.py"

        self.assertEqual("aegis.core", module_name_for_path(source_path=source_path, source_root=source_root))

    def test_import_prefix_matching_rejects_nested_forbidden_module(self) -> None:
        self.assertTrue(import_is_forbidden("aegis.proxy.mock_app", ("aegis.proxy",)))
        self.assertFalse(import_is_forbidden("aegis.proxy_mock", ("aegis.proxy",)))


if __name__ == "__main__":
    unittest.main()
