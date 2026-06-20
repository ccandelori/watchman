from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoundaryRule:
    package: str
    forbidden_imports: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ImportedModule:
    name: str
    line_number: int


@dataclass(frozen=True)
class BoundaryViolation:
    source_path: Path
    source_module: str
    imported_module: str
    line_number: int
    reason: str


BOUNDARY_RULES: tuple[BoundaryRule, ...] = (
    BoundaryRule(
        package="aegis.core",
        forbidden_imports=(
            "aegis.audit",
            "aegis.detectors",
            "aegis.policy",
            "aegis.providers",
            "aegis.proxy",
            "aegis.sdk",
            "aegis_introspection",
        ),
        reason="core contracts must not depend on runtime adapters or research code",
    ),
    BoundaryRule(
        package="aegis.audit",
        forbidden_imports=("aegis.detectors", "aegis.policy", "aegis.providers", "aegis.proxy", "aegis.sdk"),
        reason="audit sinks record events without owning detector, policy, provider, proxy, or SDK behavior",
    ),
    BoundaryRule(
        package="aegis.detectors",
        forbidden_imports=("aegis.audit", "aegis.policy", "aegis.providers", "aegis.proxy", "aegis.sdk"),
        reason="detectors emit evidence and recommendations, never final policy or transport behavior",
    ),
    BoundaryRule(
        package="aegis.policy",
        forbidden_imports=("aegis.audit", "aegis.detectors", "aegis.providers", "aegis.proxy", "aegis.sdk"),
        reason="policy combines detector results without depending on detector implementations or transports",
    ),
    BoundaryRule(
        package="aegis.providers",
        forbidden_imports=("aegis.audit", "aegis.detectors", "aegis.policy", "aegis.proxy", "aegis.sdk"),
        reason="model providers generate outputs and should stay independent of policy, audit, and transports",
    ),
    BoundaryRule(
        package="aegis.sdk",
        forbidden_imports=("aegis.audit", "aegis.detectors", "aegis.policy", "aegis.providers", "aegis.proxy"),
        reason="the SDK exposes runtime contracts without taking dependencies on concrete adapters",
    ),
    BoundaryRule(
        package="aegis",
        forbidden_imports=("aegis_introspection",),
        reason="research code must enter runtime through an explicit adapter, not direct package imports",
    ),
)


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "src"
    source_paths = tuple(sorted((source_root / "aegis").rglob("*.py")))
    violations = tuple(
        violation
        for source_path in source_paths
        for violation in boundary_violations_for_file(source_path=source_path, source_root=source_root)
    )

    if len(violations) == 0:
        return 0

    for violation in violations:
        sys.stderr.write(
            f"{violation.source_path}:{violation.line_number}: "
            f"{violation.source_module} must not import {violation.imported_module}; "
            f"{violation.reason}.\n"
        )
    return 1


def boundary_violations_for_file(source_path: Path, source_root: Path) -> tuple[BoundaryViolation, ...]:
    source_module = module_name_for_path(source_path=source_path, source_root=source_root)
    imported_modules = imports_for_file(source_path)
    return tuple(
        BoundaryViolation(
            source_path=source_path,
            source_module=source_module,
            imported_module=imported_module.name,
            line_number=imported_module.line_number,
            reason=rule.reason,
        )
        for rule in BOUNDARY_RULES
        if module_matches_prefix(module_name=source_module, prefix=rule.package)
        for imported_module in imported_modules
        if import_is_forbidden(imported_module=imported_module.name, forbidden_imports=rule.forbidden_imports)
    )


def import_is_forbidden(imported_module: str, forbidden_imports: tuple[str, ...]) -> bool:
    return any(module_matches_prefix(module_name=imported_module, prefix=prefix) for prefix in forbidden_imports)


def module_name_for_path(source_path: Path, source_root: Path) -> str:
    relative_path = source_path.relative_to(source_root)
    path_parts = relative_path.with_suffix("").parts
    if path_parts[-1] == "__init__":
        path_parts = path_parts[:-1]
    return ".".join(path_parts)


def imports_for_file(source_path: Path) -> tuple[ImportedModule, ...]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported_modules: list[ImportedModule] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(ImportedModule(name=alias.name, line_number=node.lineno) for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            imported_modules.append(ImportedModule(name=node.module, line_number=node.lineno))
    return tuple(imported_modules)


def module_matches_prefix(module_name: str, prefix: str) -> bool:
    return module_name == prefix or module_name.startswith(f"{prefix}.")


if __name__ == "__main__":
    raise SystemExit(main())
