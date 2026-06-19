# Experiment Lineage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small, typed lineage system that records datasets, activation artifacts, reports, hashes, and relationships so Aegis introspection experiments remain traceable and regression comparisons stay honest.

**Architecture:** Store lineage state in a canonical JSON manifest at `introspection/data/lineage.json`. Validate it with a typed Python module in `introspection/src/aegis_introspection/lineage.py`, plus a read-only CLI at `introspection/scripts/validate_lineage.py`. Keep existing experiment files in place and register their current paths; future datasets and artifacts should be added as new records rather than replacing baseline records.

**Tech Stack:** Python 3.12, standard-library `json`, `hashlib`, `pathlib`, `dataclasses`, `unittest`, existing `PYTHONPATH`-based script pattern.

---

## File Structure

- Create `introspection/src/aegis_introspection/lineage.py`
  - Owns lineage dataclasses, JSON parsing, schema validation, path existence checks, SHA256 checks, and referential integrity checks.
- Create `introspection/scripts/validate_lineage.py`
  - CLI wrapper that loads `introspection/data/lineage.json`, validates it, and prints a compact summary.
- Create `introspection/tests/test_lineage.py`
  - Unit tests for hash calculation, valid manifest parsing, missing file rejection, hash mismatch rejection, and unknown reference rejection.
- Create `introspection/data/lineage.json`
  - Canonical manifest for the current baseline dataset, activation artifacts, and reports.
- Optionally create `introspection/data/reports/lineage_summary.md` during implementation if we decide the CLI should write a human-readable summary. Do not add that file in the first pass unless the JSON manifest and validator are already green.

No existing dataset, activation artifact, or report should be renamed in this implementation slice. The manifest records current state; future experiment files can use clearer dataset-specific names.

## Design Decisions

- Use JSON instead of YAML so no new dependency is required.
- Use workspace-relative paths such as `introspection/data/prompts.jsonl`.
- Hash every dataset, artifact, and machine-readable report with SHA256.
- Treat Markdown progress notes as narrative references, not required hash-locked records in the first manifest.
- Make `lineage.json` hand-editable but strict: missing fields, unknown references, missing files, and hash mismatches raise explicit errors.
- Keep this first lineage system descriptive. It records what exists and how records relate; it does not regenerate artifacts.

## Task 1: Write Lineage Tests

**Files:**
- Create: `introspection/tests/test_lineage.py`

- [ ] **Step 1: Add tests for hashing, parsing, and validation**

Create `introspection/tests/test_lineage.py` with:

```python
import json
import tempfile
import unittest
from pathlib import Path

from aegis_introspection.lineage import (
    LineageError,
    load_lineage_manifest,
    sha256_file,
    validate_lineage_manifest,
)


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256_file(path)


class LineageTest(unittest.TestCase):
    def test_sha256_file_hashes_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text("aegis\n", encoding="utf-8")

            digest = sha256_file(path)

        self.assertEqual("58c06cc319165dce825f0a57c52384e0ad730b4883852fc1bbf092714d72d16a", digest)

    def test_load_lineage_manifest_accepts_valid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_hash = _write_text(root / "data" / "prompts.jsonl", "{}\n")
            artifact_hash = _write_text(root / "data" / "activations" / "features.pt", "artifact\n")
            report_hash = _write_text(root / "data" / "reports" / "binary.json", "{}\n")
            manifest_path = root / "lineage.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "created_on": "2026-06-18",
                        "datasets": [
                            {
                                "id": "baseline_prompts",
                                "path": "data/prompts.jsonl",
                                "sha256": dataset_hash,
                                "purpose": "Baseline prompt dataset.",
                                "label_counts": {
                                    "benign": 1,
                                    "secret_present_safe": 1,
                                    "exfiltration_intent": 1,
                                },
                                "family_count": 3,
                            }
                        ],
                        "artifacts": [
                            {
                                "id": "baseline_features",
                                "path": "data/activations/features.pt",
                                "sha256": artifact_hash,
                                "dataset_id": "baseline_prompts",
                                "model_id": "Qwen/Qwen3-0.6B",
                                "feature_count": 10,
                            }
                        ],
                        "reports": [
                            {
                                "id": "baseline_binary",
                                "path": "data/reports/binary.json",
                                "sha256": report_hash,
                                "dataset_id": "baseline_prompts",
                                "artifact_id": "baseline_features",
                                "evaluation_strategy": "stratified_kfold",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_lineage_manifest(manifest_path)
            validate_lineage_manifest(manifest, root)

        self.assertEqual("baseline_prompts", manifest.datasets[0].id)
        self.assertEqual("baseline_features", manifest.artifacts[0].id)
        self.assertEqual("baseline_binary", manifest.reports[0].id)

    def test_validate_lineage_manifest_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "lineage.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "created_on": "2026-06-18",
                        "datasets": [
                            {
                                "id": "baseline_prompts",
                                "path": "data/missing.jsonl",
                                "sha256": "0" * 64,
                                "purpose": "Baseline prompt dataset.",
                                "label_counts": {
                                    "benign": 1,
                                    "secret_present_safe": 1,
                                    "exfiltration_intent": 1,
                                },
                                "family_count": 3,
                            }
                        ],
                        "artifacts": [],
                        "reports": [],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_lineage_manifest(manifest_path)
            with self.assertRaises(LineageError):
                validate_lineage_manifest(manifest, root)

    def test_validate_lineage_manifest_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_text(root / "data" / "prompts.jsonl", "{}\n")
            manifest_path = root / "lineage.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "created_on": "2026-06-18",
                        "datasets": [
                            {
                                "id": "baseline_prompts",
                                "path": "data/prompts.jsonl",
                                "sha256": "0" * 64,
                                "purpose": "Baseline prompt dataset.",
                                "label_counts": {
                                    "benign": 1,
                                    "secret_present_safe": 1,
                                    "exfiltration_intent": 1,
                                },
                                "family_count": 3,
                            }
                        ],
                        "artifacts": [],
                        "reports": [],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_lineage_manifest(manifest_path)
            with self.assertRaises(LineageError):
                validate_lineage_manifest(manifest, root)

    def test_validate_lineage_manifest_rejects_unknown_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_hash = _write_text(root / "data" / "activations" / "features.pt", "artifact\n")
            manifest_path = root / "lineage.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "created_on": "2026-06-18",
                        "datasets": [],
                        "artifacts": [
                            {
                                "id": "baseline_features",
                                "path": "data/activations/features.pt",
                                "sha256": artifact_hash,
                                "dataset_id": "missing_dataset",
                                "model_id": "Qwen/Qwen3-0.6B",
                                "feature_count": 10,
                            }
                        ],
                        "reports": [],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_lineage_manifest(manifest_path)
            with self.assertRaises(LineageError):
                validate_lineage_manifest(manifest, root)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python -m unittest introspection.tests.test_lineage
```

Expected: import failure because `aegis_introspection.lineage` does not exist.

## Task 2: Implement Lineage Module

**Files:**
- Create: `introspection/src/aegis_introspection/lineage.py`
- Test: `introspection/tests/test_lineage.py`

- [ ] **Step 1: Add typed lineage models and validators**

Create `introspection/src/aegis_introspection/lineage.py` with:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias, cast


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class LineageError(ValueError):
    """Raised when experiment lineage data is malformed or inconsistent."""


@dataclass(frozen=True)
class DatasetRecord:
    id: str
    path: Path
    sha256: str
    purpose: str
    label_counts: dict[str, int]
    family_count: int


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    path: Path
    sha256: str
    dataset_id: str
    model_id: str
    feature_count: int


@dataclass(frozen=True)
class ReportRecord:
    id: str
    path: Path
    sha256: str
    dataset_id: str
    artifact_id: str
    evaluation_strategy: str


@dataclass(frozen=True)
class LineageManifest:
    schema_version: int
    created_on: str
    datasets: tuple[DatasetRecord, ...]
    artifacts: tuple[ArtifactRecord, ...]
    reports: tuple[ReportRecord, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise LineageError(f"Expected {description} to be a JSON object.")
    return cast(Mapping[str, object], value)


def _required_string(record: Mapping[str, object], field_name: str, description: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise LineageError(f"Expected {description} field '{field_name}' to be a string.")
    if value == "":
        raise LineageError(f"Expected {description} field '{field_name}' to be non-empty.")
    return value


def _required_int(record: Mapping[str, object], field_name: str, description: str) -> int:
    value = record.get(field_name)
    if not isinstance(value, int):
        raise LineageError(f"Expected {description} field '{field_name}' to be an integer.")
    if value < 0:
        raise LineageError(f"Expected {description} field '{field_name}' to be non-negative.")
    return value


def _required_list(record: Mapping[str, object], field_name: str, description: str) -> list[object]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise LineageError(f"Expected {description} field '{field_name}' to be a list.")
    return value


def _required_label_counts(record: Mapping[str, object], description: str) -> dict[str, int]:
    value = record.get("label_counts")
    mapping = _as_mapping(value, f"{description} field 'label_counts'")
    label_counts: dict[str, int] = {}
    for label, count in mapping.items():
        if not isinstance(label, str):
            raise LineageError(f"Expected {description} label_counts keys to be strings.")
        if not isinstance(count, int) or count < 0:
            raise LineageError(f"Expected {description} label_counts value for '{label}' to be a non-negative integer.")
        label_counts[label] = count
    if len(label_counts) == 0:
        raise LineageError(f"Expected {description} label_counts to contain at least one label.")
    return label_counts


def _validate_sha256_text(value: str, description: str) -> None:
    if len(value) != 64:
        raise LineageError(f"Expected {description} sha256 to contain 64 hexadecimal characters.")
    valid_chars = set("0123456789abcdef")
    if any(character not in valid_chars for character in value):
        raise LineageError(f"Expected {description} sha256 to be lowercase hexadecimal.")


def _dataset_record(value: object, index: int) -> DatasetRecord:
    description = f"dataset record {index}"
    record = _as_mapping(value, description)
    sha256 = _required_string(record, "sha256", description)
    _validate_sha256_text(sha256, description)
    return DatasetRecord(
        id=_required_string(record, "id", description),
        path=Path(_required_string(record, "path", description)),
        sha256=sha256,
        purpose=_required_string(record, "purpose", description),
        label_counts=_required_label_counts(record, description),
        family_count=_required_int(record, "family_count", description),
    )


def _artifact_record(value: object, index: int) -> ArtifactRecord:
    description = f"artifact record {index}"
    record = _as_mapping(value, description)
    sha256 = _required_string(record, "sha256", description)
    _validate_sha256_text(sha256, description)
    return ArtifactRecord(
        id=_required_string(record, "id", description),
        path=Path(_required_string(record, "path", description)),
        sha256=sha256,
        dataset_id=_required_string(record, "dataset_id", description),
        model_id=_required_string(record, "model_id", description),
        feature_count=_required_int(record, "feature_count", description),
    )


def _report_record(value: object, index: int) -> ReportRecord:
    description = f"report record {index}"
    record = _as_mapping(value, description)
    sha256 = _required_string(record, "sha256", description)
    _validate_sha256_text(sha256, description)
    return ReportRecord(
        id=_required_string(record, "id", description),
        path=Path(_required_string(record, "path", description)),
        sha256=sha256,
        dataset_id=_required_string(record, "dataset_id", description),
        artifact_id=_required_string(record, "artifact_id", description),
        evaluation_strategy=_required_string(record, "evaluation_strategy", description),
    )


def _unique_ids(ids: tuple[str, ...], description: str) -> None:
    seen: set[str] = set()
    for record_id in ids:
        if record_id in seen:
            raise LineageError(f"Duplicate {description} id '{record_id}'.")
        seen.add(record_id)


def load_lineage_manifest(path: Path) -> LineageManifest:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LineageError(f"Invalid lineage JSON in {path}: {exc.msg}.") from exc

    record = _as_mapping(decoded, "lineage manifest")
    schema_version = _required_int(record, "schema_version", "lineage manifest")
    if schema_version != 1:
        raise LineageError(f"Unsupported lineage schema_version {schema_version}.")

    datasets = tuple(_dataset_record(item, index) for index, item in enumerate(_required_list(record, "datasets", "lineage manifest")))
    artifacts = tuple(_artifact_record(item, index) for index, item in enumerate(_required_list(record, "artifacts", "lineage manifest")))
    reports = tuple(_report_record(item, index) for index, item in enumerate(_required_list(record, "reports", "lineage manifest")))
    _unique_ids(tuple(item.id for item in datasets), "dataset")
    _unique_ids(tuple(item.id for item in artifacts), "artifact")
    _unique_ids(tuple(item.id for item in reports), "report")

    return LineageManifest(
        schema_version=schema_version,
        created_on=_required_string(record, "created_on", "lineage manifest"),
        datasets=datasets,
        artifacts=artifacts,
        reports=reports,
    )


def _validate_file_hash(root_path: Path, relative_path: Path, expected_sha256: str, record_id: str) -> None:
    full_path = root_path / relative_path
    if not full_path.exists():
        raise LineageError(f"Lineage record '{record_id}' points to missing file {relative_path}.")
    actual_sha256 = sha256_file(full_path)
    if actual_sha256 != expected_sha256:
        raise LineageError(
            f"Lineage record '{record_id}' hash mismatch for {relative_path}: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )


def validate_lineage_manifest(manifest: LineageManifest, root_path: Path) -> None:
    dataset_ids = {dataset.id for dataset in manifest.datasets}
    artifact_ids = {artifact.id for artifact in manifest.artifacts}

    for dataset in manifest.datasets:
        _validate_file_hash(root_path, dataset.path, dataset.sha256, dataset.id)

    for artifact in manifest.artifacts:
        if artifact.dataset_id not in dataset_ids:
            raise LineageError(f"Artifact '{artifact.id}' references unknown dataset '{artifact.dataset_id}'.")
        _validate_file_hash(root_path, artifact.path, artifact.sha256, artifact.id)

    for report in manifest.reports:
        if report.dataset_id not in dataset_ids:
            raise LineageError(f"Report '{report.id}' references unknown dataset '{report.dataset_id}'.")
        if report.artifact_id not in artifact_ids:
            raise LineageError(f"Report '{report.id}' references unknown artifact '{report.artifact_id}'.")
        _validate_file_hash(root_path, report.path, report.sha256, report.id)
```

- [ ] **Step 2: Run lineage tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python -m unittest introspection.tests.test_lineage
```

Expected: `Ran 5 tests` and `OK`.

## Task 3: Add Lineage CLI

**Files:**
- Create: `introspection/scripts/validate_lineage.py`
- Modify: `introspection/tests/test_lineage.py`

- [ ] **Step 1: Add a CLI-oriented test for summary values**

Add this test to `LineageTest` in `introspection/tests/test_lineage.py`:

```python
    def test_validate_lineage_manifest_counts_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_hash = _write_text(root / "data" / "prompts.jsonl", "{}\n")
            artifact_hash = _write_text(root / "data" / "activations" / "features.pt", "artifact\n")
            report_hash = _write_text(root / "data" / "reports" / "binary.json", "{}\n")
            manifest_path = root / "lineage.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "created_on": "2026-06-18",
                        "datasets": [
                            {
                                "id": "baseline_prompts",
                                "path": "data/prompts.jsonl",
                                "sha256": dataset_hash,
                                "purpose": "Baseline prompt dataset.",
                                "label_counts": {
                                    "benign": 1,
                                    "secret_present_safe": 1,
                                    "exfiltration_intent": 1,
                                },
                                "family_count": 3,
                            }
                        ],
                        "artifacts": [
                            {
                                "id": "baseline_features",
                                "path": "data/activations/features.pt",
                                "sha256": artifact_hash,
                                "dataset_id": "baseline_prompts",
                                "model_id": "Qwen/Qwen3-0.6B",
                                "feature_count": 10,
                            }
                        ],
                        "reports": [
                            {
                                "id": "baseline_binary",
                                "path": "data/reports/binary.json",
                                "sha256": report_hash,
                                "dataset_id": "baseline_prompts",
                                "artifact_id": "baseline_features",
                                "evaluation_strategy": "stratified_kfold",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_lineage_manifest(manifest_path)
            validate_lineage_manifest(manifest, root)

        self.assertEqual(1, len(manifest.datasets))
        self.assertEqual(1, len(manifest.artifacts))
        self.assertEqual(1, len(manifest.reports))
```

- [ ] **Step 2: Run the focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python -m unittest introspection.tests.test_lineage
```

Expected: still `OK`; this test uses existing module behavior.

- [ ] **Step 3: Create the CLI**

Create `introspection/scripts/validate_lineage.py` with:

```python
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.lineage import load_lineage_manifest, validate_lineage_manifest


@dataclass(frozen=True)
class ValidateLineageScriptConfig:
    manifest_path: Path
    root_path: Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the Aegis introspection lineage manifest.")
    parser.add_argument(
        "--manifest",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "lineage.json"),
    )
    parser.add_argument(
        "--root",
        required=False,
        default=str(WORKSPACE_ROOT),
    )
    return parser


def _parse_args(argv: Sequence[str]) -> ValidateLineageScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return ValidateLineageScriptConfig(
        manifest_path=Path(namespace.manifest),
        root_path=Path(namespace.root),
    )


def run_validation(config: ValidateLineageScriptConfig) -> None:
    manifest = load_lineage_manifest(config.manifest_path)
    validate_lineage_manifest(manifest, config.root_path)
    print(f"Validated lineage manifest: {config.manifest_path}")
    print(f"Datasets: {len(manifest.datasets)}")
    print(f"Artifacts: {len(manifest.artifacts)}")
    print(f"Reports: {len(manifest.reports)}")


def main(argv: Sequence[str]) -> None:
    run_validation(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
```

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python -m unittest introspection.tests.test_lineage
```

Expected: `Ran 6 tests` and `OK`.

## Task 4: Add Current Baseline Manifest

**Files:**
- Create: `introspection/data/lineage.json`

- [ ] **Step 1: Add `lineage.json` for current baseline records**

Create `introspection/data/lineage.json` with:

```json
{
  "schema_version": 1,
  "created_on": "2026-06-18",
  "datasets": [
    {
      "id": "baseline_prompts_v1",
      "path": "introspection/data/prompts.jsonl",
      "sha256": "dfebd7d7254c99fa09f38622da76f0c9f4fac402bab1e8b14a0b3c3632afedbd",
      "purpose": "Baseline hand-authored prompt dataset for early Aegis model-introspection checkpoints.",
      "label_counts": {
        "benign": 30,
        "secret_present_safe": 30,
        "exfiltration_intent": 30
      },
      "family_count": 30
    }
  ],
  "artifacts": [
    {
      "id": "qwen3_0_6b_baseline_sampled_layers_v1",
      "path": "introspection/data/activations/qwen3_0_6b_features.pt",
      "sha256": "30ec6e4e6f9c3aa65165cd43bdba710111a23c036bc73d910dd277fd15455f89",
      "dataset_id": "baseline_prompts_v1",
      "model_id": "Qwen/Qwen3-0.6B",
      "feature_count": 10
    },
    {
      "id": "qwen3_0_6b_baseline_all_layers_v1",
      "path": "introspection/data/activations/qwen3_0_6b_all_layers.pt",
      "sha256": "2843c190e6829e13754472ff4da923f0ae4dfd94e81d1c34df228306a7e19db0",
      "dataset_id": "baseline_prompts_v1",
      "model_id": "Qwen/Qwen3-0.6B",
      "feature_count": 58
    }
  ],
  "reports": [
    {
      "id": "baseline_probe_sampled_layers_v1",
      "path": "introspection/data/reports/probe_baseline.json",
      "sha256": "0e63d50aa43abca46f6c4a724e5de51cf906661fe4df4d65c6f418db4acf60e0",
      "dataset_id": "baseline_prompts_v1",
      "artifact_id": "qwen3_0_6b_baseline_sampled_layers_v1",
      "evaluation_strategy": "stratified_kfold"
    },
    {
      "id": "baseline_text_word_tfidf_v1",
      "path": "introspection/data/reports/text_baseline.json",
      "sha256": "5e06a5e404c95dea1af43ca810b08f944549007884f6df113122d498902b3eeb",
      "dataset_id": "baseline_prompts_v1",
      "artifact_id": "qwen3_0_6b_baseline_sampled_layers_v1",
      "evaluation_strategy": "stratified_kfold"
    },
    {
      "id": "baseline_probe_all_layers_v1",
      "path": "introspection/data/reports/probe_all_layers.json",
      "sha256": "feb4a722700da8fb27aa1d15e6b9ac5c8d69a7bc14159271b485353713778237",
      "dataset_id": "baseline_prompts_v1",
      "artifact_id": "qwen3_0_6b_baseline_all_layers_v1",
      "evaluation_strategy": "stratified_kfold"
    },
    {
      "id": "baseline_binary_random_v1",
      "path": "introspection/data/reports/binary_tasks.json",
      "sha256": "d683674ac5f62bb2a32839b627b9de0d0855e6e7626f53c07bc5e03be6d213d5",
      "dataset_id": "baseline_prompts_v1",
      "artifact_id": "qwen3_0_6b_baseline_all_layers_v1",
      "evaluation_strategy": "stratified_kfold"
    },
    {
      "id": "baseline_binary_grouped_v1",
      "path": "introspection/data/reports/binary_tasks_grouped.json",
      "sha256": "4b55f7f032a1862612f23efde5318265b74dedb6a77ecc8ec783355183763f0c",
      "dataset_id": "baseline_prompts_v1",
      "artifact_id": "qwen3_0_6b_baseline_all_layers_v1",
      "evaluation_strategy": "stratified_group_kfold"
    }
  ]
}
```

- [ ] **Step 2: Validate the manifest**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/validate_lineage.py
```

Expected:

```text
Validated lineage manifest: /Users/sheep/Desktop/Gauntlet/Capstone/introspection/data/lineage.json
Datasets: 1
Artifacts: 2
Reports: 5
```

## Task 5: Full Verification

**Files:**
- Verify: all changed files

- [ ] **Step 1: Run lineage validation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/validate_lineage.py
```

Expected:

```text
Validated lineage manifest: /Users/sheep/Desktop/Gauntlet/Capstone/introspection/data/lineage.json
Datasets: 1
Artifacts: 2
Reports: 5
```

- [ ] **Step 2: Run the full test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python -m unittest discover -s introspection/tests
```

Expected: all tests pass.

- [ ] **Step 3: Check for cache noise**

Run:

```bash
find introspection -name __pycache__ -type d
```

Expected: no output.

If Python creates cache directories despite `PYTHONDONTWRITEBYTECODE=1`, remove only those generated cache directories after confirming their paths:

```bash
rm -rf introspection/tests/__pycache__ introspection/src/aegis_introspection/__pycache__ introspection/scripts/__pycache__
```

This workspace is not currently a git repository, so this plan intentionally uses verification checkpoints rather than commit steps.

## Self-Review

- Spec coverage: The plan creates a manifest, typed validator, CLI, baseline records, hash checks, and referential integrity checks. It preserves existing files and does not replace prior experiment outputs.
- Placeholder scan: The plan contains concrete paths, command lines, expected outputs, and JSON content. No deferred fields are left unresolved.
- Type consistency: `DatasetRecord`, `ArtifactRecord`, `ReportRecord`, and `LineageManifest` are introduced once and used consistently across tests, implementation, and CLI.
