"""Tests for the command-line interface and error mapping (U5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from detect.dp_honey import build_model, get_format, list_format_slugs, load_model, model_to_dict
from detect.dp_honey.__main__ import GENERATE_MAX, build_parser, main
from detect.dp_honey.realism import REPORT_MAX

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "dp_honey" / "golden_model.json"

REQUIRED_SLUGS = list_format_slugs()


def test_list_formats_includes_every_slug(capsys):
    assert main(["list-formats"]) == 0
    out = capsys.readouterr().out
    for slug in REQUIRED_SLUGS:
        assert slug in out


def test_preview_corpus_outputs_valid_examples(capsys):
    assert main(["preview-corpus", "--format", "github-ghp", "--count", "3", "--seed", "0"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    assert len(lines) == 3
    spec = get_format("github-ghp")
    assert all(spec.validate(ln) for ln in lines)


def test_train_writes_artifact(tmp_path, capsys):
    out = tmp_path / "m.json"
    rc = main(["train", "--format", "github-ghp", "--out", str(out), "--corpus-size", "20", "--seed", "1"])
    assert rc == 0
    assert out.exists()
    load_model(out)  # must be a loadable, valid artifact


def test_generate_from_format_emits_n_valid_tokens(capsys):
    assert main(["generate", "--format", "github-ghp", "--count", "3", "--seed", "7"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    assert len(lines) == 3
    spec = get_format("github-ghp")
    assert all(spec.validate(ln) for ln in lines)


def test_generate_from_model_emits_valid_tokens(tmp_path, capsys):
    out = tmp_path / "m.json"
    main(["train", "--format", "stripe-sk-live", "--out", str(out), "--corpus-size", "20", "--seed", "1"])
    capsys.readouterr()  # clear
    assert main(["generate", "--model", str(out), "--count", "3", "--seed", "2"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    assert len(lines) == 3
    spec = get_format("stripe-sk-live")
    assert all(spec.validate(ln) for ln in lines)


def test_generate_json_output_is_a_list(capsys):
    assert main(["generate", "--format", "github-ghp", "--count", "4", "--seed", "1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and len(payload) == 4


def test_inspect_model_shows_fields(capsys):
    assert main(["inspect-model", "--model", str(GOLDEN)]) == 0
    out = capsys.readouterr().out
    assert "aws-access-key-id" in out
    assert "epsilon=" in out
    assert "schema_version" in out
    assert "snapshot_status: OK" in out
    assert "synthetic_only=True" in out


def test_validate_exits_zero_for_valid_artifact(capsys):
    assert main(["validate", "--model", str(GOLDEN)]) == 0


def test_validate_exits_nonzero_for_invalid_artifact(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    assert main(["validate", "--model", str(bad)]) == 1
    assert "error" in capsys.readouterr().err.lower()


def test_report_emits_all_metrics(capsys):
    assert main(["report", "--format", "github-ghp", "--count", "10", "--seed", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    for field in ("validity_rate", "char_entropy_bits", "duplicate_rate", "avg_log_likelihood", "debug"):
        assert field in report


def test_unknown_format_exits_nonzero(capsys):
    assert main(["generate", "--format", "not-real", "--count", "1"]) == 1
    assert "error" in capsys.readouterr().err.lower()


def test_existing_output_without_force_exits_nonzero(tmp_path, capsys):
    out = tmp_path / "m.json"
    assert main(["train", "--format", "github-ghp", "--out", str(out), "--corpus-size", "10"]) == 0
    rc = main(["train", "--format", "github-ghp", "--out", str(out), "--corpus-size", "10"])
    assert rc == 1
    assert "error" in capsys.readouterr().err.lower()


def test_oversized_generate_count_exits_before_work(capsys):
    rc = main(["generate", "--format", "github-ghp", "--count", str(GENERATE_MAX + 1)])
    assert rc == 1
    assert str(GENERATE_MAX) in capsys.readouterr().err


def test_oversized_report_count_exits_before_work(capsys):
    rc = main(["report", "--format", "github-ghp", "--count", str(REPORT_MAX + 1)])
    assert rc == 1
    assert str(REPORT_MAX) in capsys.readouterr().err


def test_generate_emits_safety_banner_on_stderr(capsys):
    assert main(["generate", "--format", "github-ghp", "--count", "1", "--seed", "0"]) == 0
    assert "synthetic" in capsys.readouterr().err.lower()


def test_help_text_mentions_synthetic():
    assert "synthetic" in build_parser().format_help().lower()


def _write_artifact(path: Path, mutate=None) -> Path:
    data = model_to_dict(build_model("github-ghp", corpus_size=15, train_seed=1))
    if mutate is not None:
        mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["format"].__setitem__("spec_hash", "sha256:" + "0" * 64),  # drift
        lambda d: d.__setitem__("schema_version", "999"),  # unknown schema
        lambda d: d["format"].__setitem__("slug", "not-a-real-format"),  # unknown slug
    ],
)
def test_validate_exits_nonzero_for_typed_errors(tmp_path, mutate, capsys):
    art = _write_artifact(tmp_path / "a.json", mutate)
    assert main(["validate", "--model", str(art)]) == 1
    assert "error" in capsys.readouterr().err.lower()


def test_inspect_model_reports_drift(tmp_path, capsys):
    art = _write_artifact(tmp_path / "a.json", lambda d: d["format"].__setitem__("spec_hash", "sha256:" + "0" * 64))
    assert main(["inspect-model", "--model", str(art)]) == 0  # lenient: still exits 0
    assert "DRIFT" in capsys.readouterr().out


def test_inspect_model_reports_unknown_format(tmp_path, capsys):
    art = _write_artifact(tmp_path / "a.json", lambda d: d["format"].__setitem__("slug", "nope"))
    assert main(["inspect-model", "--model", str(art)]) == 0
    assert "UNKNOWN_FORMAT" in capsys.readouterr().out


def test_oversized_count_is_checked_before_model_load(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.json"
    rc = main(["generate", "--model", str(missing), "--count", str(GENERATE_MAX + 1)])
    assert rc == 1
    # The count cap fires before the (missing) artifact is read (AE6 ordering).
    assert str(GENERATE_MAX) in capsys.readouterr().err


def test_list_formats_json_includes_safety_notes(capsys):
    assert main(["list-formats", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    slugs = {item["slug"] for item in payload}
    for slug in REQUIRED_SLUGS:
        assert slug in slugs
    for item in payload:
        assert item["safety_note"]
        assert item["provider_valid"] is False


def test_preview_and_report_emit_safety_banner(capsys):
    assert main(["preview-corpus", "--format", "github-ghp", "--count", "1", "--seed", "0"]) == 0
    assert "synthetic" in capsys.readouterr().err.lower()
    assert main(["report", "--format", "github-ghp", "--count", "5", "--seed", "0"]) == 0
    assert "synthetic" in capsys.readouterr().err.lower()


def test_cli_scan_reports_findings_without_secret_values(tmp_path, capsys):
    import numpy as np

    spec = get_format("github-ghp")
    token = spec.random_example(np.random.default_rng(1))
    path = tmp_path / "in.txt"
    path.write_text(f"TOKEN={token}", encoding="utf-8")
    assert main(["scan", "--file", str(path)]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["findings"][0]["format"] == "github-ghp"
    assert token not in out


def test_cli_scan_show_matches_is_explicit_opt_in(tmp_path, capsys):
    import numpy as np

    spec = get_format("github-ghp")
    token = spec.random_example(np.random.default_rng(1))
    path = tmp_path / "in.txt"
    path.write_text(f"TOKEN={token}", encoding="utf-8")
    assert main(["scan", "--file", str(path), "--show-matches"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["matches"] == [token]


def test_cli_auto_decoy_emits_swapped_text(tmp_path, capsys):
    import numpy as np

    spec = get_format("github-ghp")
    token = spec.random_example(np.random.default_rng(2))
    path = tmp_path / "in.txt"
    path.write_text(f"TOKEN={token}", encoding="utf-8")
    assert main(["auto-decoy", "--file", str(path), "--seed", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["decoys"]) == 1
    assert token not in payload["swapped_text"]


def test_cli_scan_missing_file_exits_nonzero(tmp_path, capsys):
    missing = tmp_path / "missing.txt"
    assert main(["scan", "--file", str(missing)]) == 1
    assert "error" in capsys.readouterr().err.lower()
