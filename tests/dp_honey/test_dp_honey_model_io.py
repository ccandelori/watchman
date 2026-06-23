"""Tests for JSON model artifact export, load, and fail-closed validation (U4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from detect.dp_honey import build_model, load_model, model_to_dict, save_model
from detect.dp_honey.errors import (
    FormatSpecMismatchError,
    ModelArtifactDecodeError,
    ModelArtifactExistsError,
    ModelSchemaError,
    UnknownFormatError,
)

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "dp_honey" / "golden_model.json"


def _base_model():
    return build_model("aws-access-key-id", epsilon=1.0, clip=1.0, corpus_size=20, train_seed=5)


def _valid_dict() -> dict:
    return model_to_dict(_base_model())


def test_save_load_roundtrip_is_deterministic(tmp_path):
    model = _base_model()
    path = save_model(model, tmp_path / "m.json")
    loaded = load_model(path)
    assert model.sample(4, seed=9) == loaded.sample(4, seed=9)


def test_committed_golden_fixture_loads_and_generates():
    model = load_model(GOLDEN)
    tokens = model.sample(3, seed=1)
    assert len(tokens) == 3
    assert all(model.format_spec.validate(t) for t in tokens)
    assert model.format_slug == "aws-access-key-id"


def test_bad_json_raises_decode_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ModelArtifactDecodeError):
        load_model(bad)


def test_missing_required_field_raises_schema_error():
    data = _valid_dict()
    del data["transitions"]
    with pytest.raises(ModelSchemaError):
        load_model(data)


def test_unknown_schema_version_raises():
    data = _valid_dict()
    data["schema_version"] = "2"
    with pytest.raises(ModelSchemaError):
        load_model(data)


def test_format_snapshot_hash_drift_raises():
    data = _valid_dict()
    data["format"]["spec_hash"] = "sha256:" + "0" * 64
    with pytest.raises(FormatSpecMismatchError):
        load_model(data)


def test_snapshot_structure_drift_raises():
    data = _valid_dict()
    data["format"]["spec_snapshot"]["segments"][1]["length"] = 999
    with pytest.raises(FormatSpecMismatchError):
        load_model(data)


def test_unknown_checksum_algorithm_in_snapshot_raises_schema_error():
    data = model_to_dict(build_model("github-ghp", corpus_size=20, train_seed=1))
    for segment in data["format"]["spec_snapshot"]["segments"]:
        if segment.get("kind") == "checksum":
            segment["algorithm"] = "totally-unknown"
    with pytest.raises(ModelSchemaError):
        load_model(data)


def test_unknown_format_slug_raises():
    data = _valid_dict()
    data["format"]["slug"] = "not-a-real-format"
    with pytest.raises(UnknownFormatError):
        load_model(data)


def test_invalid_alphabet_membership_raises():
    data = _valid_dict()
    data["transitions"]["<START>"]["@"] = 0.0  # '@' is not in the format alphabet
    with pytest.raises(ModelSchemaError):
        load_model(data)


def test_non_finite_probability_raises():
    data = _valid_dict()
    row = data["transitions"]["<START>"]
    row[next(iter(row))] = float("inf")
    with pytest.raises(ModelSchemaError):
        load_model(data)


def test_negative_probability_raises():
    data = _valid_dict()
    row = data["transitions"]["<START>"]
    row[next(iter(row))] = -0.25
    with pytest.raises(ModelSchemaError):
        load_model(data)


def test_unnormalized_row_raises():
    data = _valid_dict()
    row = data["transitions"]["<START>"]
    for key in row:
        row[key] *= 2.0  # row now sums to ~2.0
    with pytest.raises(ModelSchemaError):
        load_model(data)


def test_save_refuses_overwrite_without_force(tmp_path):
    model = _base_model()
    path = tmp_path / "m.json"
    save_model(model, path)
    with pytest.raises(ModelArtifactExistsError):
        save_model(model, path)


def test_save_force_overwrites(tmp_path):
    model = _base_model()
    path = tmp_path / "m.json"
    save_model(model, path)
    save_model(model, path, force=True)  # must not raise
    # File is still valid, canonical JSON.
    json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "field, value",
    [("epsilon", 0.0), ("epsilon", -1.0), ("clip", 0.0), ("clip", -2.0), ("corpus_size", -3)],
)
def test_out_of_range_privacy_metadata_raises(field, value):
    # Load must enforce the same bounds train_model does (save/load agree).
    data = _valid_dict()
    data["privacy"][field] = value
    with pytest.raises(ModelSchemaError):
        load_model(data)
