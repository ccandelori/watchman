"""Tests for the typed model artifact status interface."""

from __future__ import annotations

import json
from pathlib import Path

from detect.dp_honey import build_model, model_to_dict
from detect.dp_honey.artifact_status import SnapshotStatus, inspect_artifact, validate_artifact

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "dp_honey" / "golden_model.json"


def _write_artifact(path: Path, mutate=None) -> Path:
    data = model_to_dict(build_model("github-ghp", corpus_size=15, train_seed=1))
    if mutate is not None:
        mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_inspect_artifact_returns_typed_ok_status_for_valid_fixture() -> None:
    inspection = inspect_artifact(GOLDEN)

    assert inspection.format_slug == "aws-access-key-id"
    assert inspection.snapshot_status is SnapshotStatus.OK
    assert inspection.to_dict()["snapshot_status"] == "OK"
    assert inspection.to_dict()["safety"]["provider_valid"] is False


def test_inspect_artifact_reports_drift_without_strict_load_failure(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path / "drifted.json",
        lambda data: data["format"].__setitem__("spec_hash", "sha256:" + "0" * 64),
    )

    inspection = inspect_artifact(artifact)

    assert inspection.snapshot_status is SnapshotStatus.DRIFT
    assert inspection.to_dict()["snapshot_status"] == "DRIFT"


def test_inspect_artifact_reports_unknown_format_without_strict_load_failure(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path / "unknown.json",
        lambda data: data["format"].__setitem__("slug", "not-real"),
    )

    inspection = inspect_artifact(artifact)

    assert inspection.format_slug == "not-real"
    assert inspection.snapshot_status is SnapshotStatus.UNKNOWN_FORMAT


def test_validate_artifact_returns_typed_result_instead_of_raising(tmp_path: Path) -> None:
    bad = tmp_path / "broken.json"
    bad.write_text("{not json", encoding="utf-8")

    result = validate_artifact(bad)

    assert result.valid is False
    assert result.error
    assert result.to_dict()["valid"] is False
