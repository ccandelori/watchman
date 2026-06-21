"""Tests for the DP-HONEY web UI service layer (no HTTP)."""

from __future__ import annotations

import pytest

from detect.dp_honey import get_format, list_format_slugs
from detect.dp_honey.errors import CountLimitError, DPHoneyError, ModelArtifactExistsError
from detect.dp_honey.webui import service
from detect.dp_honey.webui.service import InvalidModelName


@pytest.mark.parametrize(
    "bad",
    ["../secret", "a/b", "a\\b", "..", ".", ".hidden", "", "name with space", "/abs", "x" * 101],
)
def test_resolve_model_ref_rejects_unsafe_names(bad, tmp_path):
    with pytest.raises(InvalidModelName):
        service.resolve_model_ref(bad, models_dir=tmp_path)


def test_resolve_model_ref_maps_safe_name_into_models_dir(tmp_path):
    ref = service.resolve_model_ref("my-model", models_dir=tmp_path)
    assert ref == tmp_path / "my-model.json"


def test_resolve_model_ref_maps_golden_label_to_fixture():
    ref = service.resolve_model_ref(service.GOLDEN_NAME)
    assert ref.name == "golden_model.json"
    assert "fixtures" in ref.parts


def test_list_formats_payload_has_all_slugs_and_safety():
    payload = service.list_formats_payload()
    slugs = {item["slug"] for item in payload}
    assert "github-ghp" in slugs and "aws-access-key-id" in slugs
    assert slugs == set(list_format_slugs())
    for item in payload:
        assert item["safety_note"]
        assert item["provider_valid"] is False


def test_preview_corpus_returns_valid_examples():
    examples = service.preview_corpus("github-ghp", count=3, seed=0)
    assert len(examples) == 3
    spec = get_format("github-ghp")
    assert all(spec.validate(e) for e in examples)


def test_preview_corpus_rejects_oversized_count():
    with pytest.raises(CountLimitError):
        service.preview_corpus("github-ghp", count=10_001, seed=0)


def test_run_generate_from_format_is_valid_and_deterministic():
    params = {"source": "format", "format": "github-ghp", "count": 4, "seed": 7, "train_seed": 3}
    a = service.run_generate(params)
    b = service.run_generate(params)
    assert a["tokens"] == b["tokens"]
    assert len(a["tokens"]) == 4
    spec = get_format("github-ghp")
    assert all(spec.validate(t) for t in a["tokens"])
    assert a["safety"]["provider_valid"] is False


def test_run_generate_oversized_count_rejected():
    with pytest.raises(CountLimitError):
        service.run_generate({"source": "format", "format": "github-ghp", "count": 10_001})


def test_run_report_has_metrics():
    report = service.run_report({"source": "format", "format": "github-ghp", "count": 20, "seed": 1})
    for field in ("validity_rate", "char_entropy_bits", "duplicate_rate", "avg_log_likelihood", "debug"):
        assert field in report


def test_run_report_oversized_count_rejected():
    with pytest.raises(CountLimitError):
        service.run_report({"source": "format", "format": "github-ghp", "count": 5001})


def test_run_generate_from_golden_fixture_model():
    out = service.run_generate({"source": "model", "model": service.GOLDEN_NAME, "count": 3, "seed": 1})
    assert len(out["tokens"]) == 3
    spec = get_format("aws-access-key-id")
    assert all(spec.validate(t) for t in out["tokens"])


def test_run_train_writes_and_list_models_shows_it(tmp_path):
    result = service.run_train(
        {"format": "github-ghp", "out_name": "demo", "corpus_size": 20, "seed": 1},
        models_dir=tmp_path,
    )
    assert result["saved"] == "demo.json"
    assert (tmp_path / "demo.json").exists()

    models = service.list_models(models_dir=tmp_path)
    names = {m["name"] for m in models}
    assert "demo" in names
    assert service.GOLDEN_NAME in names  # the fixture is always listed
    demo = next(m for m in models if m["name"] == "demo")
    assert demo["slug"] == "github-ghp"
    assert demo["source"] == "library"


def test_run_train_refuses_overwrite_without_force(tmp_path):
    args = {"format": "github-ghp", "out_name": "demo", "corpus_size": 20, "seed": 1}
    service.run_train(args, models_dir=tmp_path)
    with pytest.raises(ModelArtifactExistsError):
        service.run_train(args, models_dir=tmp_path)


def test_run_train_rejects_unsafe_out_name(tmp_path):
    with pytest.raises(InvalidModelName):
        service.run_train(
            {"format": "github-ghp", "out_name": "../evil", "corpus_size": 20},
            models_dir=tmp_path,
        )


def test_run_inspect_golden_fixture_reports_ok():
    info = service.run_inspect(service.GOLDEN_NAME)
    assert info["format"] == "aws-access-key-id"
    assert info["snapshot_status"] == "OK"
    assert info["schema_version"] == "1"
    assert info["safety"]["provider_valid"] is False


def test_run_validate_golden_fixture_is_valid():
    assert service.run_validate(service.GOLDEN_NAME) == {"valid": True, "error": None}


def test_run_validate_reports_error_for_bad_artifact(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    result = service.run_validate("broken", models_dir=tmp_path)
    assert result["valid"] is False
    assert result["error"]


def test_run_inspect_missing_file_raises(tmp_path):
    with pytest.raises(DPHoneyError):
        service.run_inspect("nonexistent", models_dir=tmp_path)


def test_run_generate_missing_model_name_raises():
    with pytest.raises(DPHoneyError):
        service.run_generate({"source": "model"})  # 'model' key absent -> 400, not 500


def test_run_generate_missing_format_raises():
    with pytest.raises(DPHoneyError):
        service.run_generate({"source": "format"})  # 'format' key absent -> 400, not 500


def test_run_report_metadata_is_synthetic_only():
    report = service.run_report({"source": "format", "format": "github-ghp", "count": 10, "seed": 1})
    assert report["safety"]["provider_valid"] is False


def test_service_run_scan_and_auto_decoy():
    tok = get_format("github-ghp").random_example(__import__("numpy").random.default_rng(1))
    text = f"TOKEN={tok}"
    findings = service.run_scan(text)["findings"]
    assert findings and findings[0]["format"] == "github-ghp"
    result = service.run_auto_decoy(text, seed=1)
    assert tok not in result["swapped_text"]
    assert len(result["decoys"]) == 1


def test_launcher_parser_defaults_to_localhost():
    from detect.dp_honey.webui.__main__ import build_parser

    args = build_parser().parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
