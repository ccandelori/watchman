from __future__ import annotations

import json

import numpy as np
import pytest

from detect.dp_honey.__main__ import main
from detect.dp_honey.bigram import build_model
from detect.dp_honey.formats import list_formats
from detect.dp_honey.realism import compute_report
from detect.dp_honey.statistical_distinguisher_eval import (
    FEATURE_NAMES,
    REFERENCE_FEATURE_CORPUS_SCHEMA_VERSION,
    REQUIRED_TESTS,
    STATISTICAL_DISTINGUISHER_EVAL_MAX_PER_FORMAT,
    STATISTICAL_DISTINGUISHER_EVAL_SCHEMA_VERSION,
    DPHoneyReferenceFeatureCorpusConfig,
    DPHoneyStatisticalDistinguisherEvalConfig,
    DPHoneyStatisticalDistinguisherEvalError,
    _numeric_profile,
    _token_features,
    build_reference_feature_corpus_report,
    build_statistical_distinguisher_eval_report,
)


def test_statistical_distinguisher_eval_reports_required_suite_without_raw_values() -> None:
    report = build_statistical_distinguisher_eval_report(
        DPHoneyStatisticalDistinguisherEvalConfig(
            train_count_per_format=3,
            test_count_per_format=3,
            seed=11,
            alpha=0.1,
            reference_feature_corpus_path=None,
        )
    )
    rendered = json.dumps(report, sort_keys=True)
    suite = report["statistical_distinguisher_suite"]

    assert report["schema_version"] == STATISTICAL_DISTINGUISHER_EVAL_SCHEMA_VERSION
    assert report["status"] == "statistical_distinguisher_suite_evaluated"
    assert report["raw_values_serialized"] is False
    assert report["required_tests"] == list(REQUIRED_TESTS)
    assert set(suite) == set(REQUIRED_TESTS)
    assert report["audit_safety"]["raw_secret_values_in_report"] is False
    assert "ghp_" not in rendered
    assert "xoxb-" not in rendered
    assert "sk_live_" not in rendered


def test_statistical_distinguisher_eval_declared_flag_matches_required_test_statuses() -> None:
    report = build_statistical_distinguisher_eval_report(
        DPHoneyStatisticalDistinguisherEvalConfig(
            train_count_per_format=2,
            test_count_per_format=2,
            seed=7,
            alpha=0.1,
            reference_feature_corpus_path=None,
        )
    )
    suite = report["statistical_distinguisher_suite"]
    passed = all(suite[test_name]["status"] == "passed" for test_name in REQUIRED_TESTS)

    assert report["all_required_tests_passed"] is passed
    assert report["synthetic_registry_statistical_distinguisher_passed"] is passed
    assert report["paper_faithful_statistical_distinguisher"] is False
    assert report["reference_source"] == "same_format_uniform_synthetic_holdout"


def test_statistical_distinguisher_eval_rejects_invalid_config() -> None:
    with pytest.raises(DPHoneyStatisticalDistinguisherEvalError, match="train_count_per_format"):
        DPHoneyStatisticalDistinguisherEvalConfig(
            train_count_per_format=0,
            test_count_per_format=1,
            seed=1,
            alpha=0.1,
            reference_feature_corpus_path=None,
        )
    with pytest.raises(DPHoneyStatisticalDistinguisherEvalError, match="test_count_per_format"):
        DPHoneyStatisticalDistinguisherEvalConfig(
            train_count_per_format=1,
            test_count_per_format=STATISTICAL_DISTINGUISHER_EVAL_MAX_PER_FORMAT + 1,
            seed=1,
            alpha=0.1,
            reference_feature_corpus_path=None,
        )
    with pytest.raises(DPHoneyStatisticalDistinguisherEvalError, match="alpha"):
        DPHoneyStatisticalDistinguisherEvalConfig(
            train_count_per_format=1,
            test_count_per_format=1,
            seed=1,
            alpha=1.0,
            reference_feature_corpus_path=None,
        )


def test_statistical_distinguisher_eval_accepts_redacted_provider_like_features(tmp_path) -> None:
    manifest_path = tmp_path / "provider-like-reference-features.json"
    _write_reference_feature_manifest(
        path=manifest_path,
        train_count_per_format=3,
        test_count_per_format=3,
        source="provider_like_sealed_holdout",
    )

    report = build_statistical_distinguisher_eval_report(
        DPHoneyStatisticalDistinguisherEvalConfig(
            train_count_per_format=3,
            test_count_per_format=3,
            seed=13,
            alpha=0.1,
            reference_feature_corpus_path=manifest_path,
        )
    )
    rendered = json.dumps(report, sort_keys=True)

    assert report["reference_source"] == "provider_like_sealed_holdout"
    assert report["reference_feature_corpus"]["schema_version"] == REFERENCE_FEATURE_CORPUS_SCHEMA_VERSION
    assert report["reference_feature_corpus"]["sha256"]
    assert report["reference_feature_corpus"]["raw_values_serialized"] is False
    assert report["paper_faithful_statistical_distinguisher"] is report["all_required_tests_passed"]
    assert "ghp_" not in rendered
    assert "xoxb-" not in rendered
    assert "sk_live_" not in rendered


def test_reference_feature_corpus_builder_emits_redacted_provider_like_features() -> None:
    report = build_reference_feature_corpus_report(
        DPHoneyReferenceFeatureCorpusConfig(
            train_count_per_format=2,
            test_count_per_format=2,
            seed=17,
            source="provider_like_sealed_holdout",
            source_description="test redacted nonfunctional provider-like feature holdout",
        )
    )
    rendered = json.dumps(report, sort_keys=True)

    assert report["schema_version"] == REFERENCE_FEATURE_CORPUS_SCHEMA_VERSION
    assert report["source"] == "provider_like_sealed_holdout"
    assert report["source_generation_method"] == "public_provider_morphology_nonfunctional_synthetic_holdout"
    assert report["raw_values_serialized"] is False
    assert report["feature_names"] == list(FEATURE_NAMES)
    assert report["train_count_per_format"] == 2
    assert report["test_count_per_format"] == 2
    assert len(report["format_features"]) == len(list_formats())
    assert "ghp_" not in rendered
    assert "xoxb-" not in rendered
    assert "sk_live_" not in rendered


def test_statistical_distinguisher_eval_rejects_raw_reference_fields(tmp_path) -> None:
    manifest_path = tmp_path / "raw-reference-features.json"
    payload = _reference_feature_manifest(
        train_count_per_format=1,
        test_count_per_format=1,
        source="provider_like_sealed_holdout",
    )
    payload["format_features"][0]["test"]["tokens"] = ["ghp_forbidden_raw_value"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DPHoneyStatisticalDistinguisherEvalError, match="raw-value field"):
        build_statistical_distinguisher_eval_report(
            DPHoneyStatisticalDistinguisherEvalConfig(
                train_count_per_format=1,
                test_count_per_format=1,
                seed=13,
                alpha=0.1,
                reference_feature_corpus_path=manifest_path,
            )
        )


def test_cli_eval_statistical_distinguishers_writes_report_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "dp-honey-statistical-distinguisher-eval.json"
    manifest_path = tmp_path / "provider-like-reference-features.json"
    _write_reference_feature_manifest(
        path=manifest_path,
        train_count_per_format=2,
        test_count_per_format=2,
        source="provider_like_sealed_holdout",
    )

    assert (
        main(
            [
                "eval-statistical-distinguishers",
                "--train-count-per-format",
                "2",
                "--test-count-per-format",
                "2",
                "--alpha",
                "0.1",
                "--seed",
                "3",
                "--reference-feature-manifest",
                str(manifest_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_payload == file_payload
    assert file_payload["schema_version"] == STATISTICAL_DISTINGUISHER_EVAL_SCHEMA_VERSION
    assert file_payload["reference_source"] == "provider_like_sealed_holdout"
    assert set(file_payload["statistical_distinguisher_suite"]) == set(REQUIRED_TESTS)


def test_cli_build_reference_feature_corpus_writes_report_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "provider-like-reference-features.json"

    assert (
        main(
            [
                "build-reference-feature-corpus",
                "--train-count-per-format",
                "2",
                "--test-count-per-format",
                "2",
                "--seed",
                "3",
                "--source",
                "provider_like_sealed_holdout",
                "--source-description",
                "test redacted nonfunctional provider-like feature holdout",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_payload == file_payload
    assert file_payload["schema_version"] == REFERENCE_FEATURE_CORPUS_SCHEMA_VERSION
    assert file_payload["source"] == "provider_like_sealed_holdout"
    assert file_payload["source_generation_method"] == "public_provider_morphology_nonfunctional_synthetic_holdout"
    assert file_payload["feature_names"] == list(FEATURE_NAMES)


def _write_reference_feature_manifest(
    path,
    train_count_per_format: int,
    test_count_per_format: int,
    source: str,
) -> None:
    path.write_text(
        json.dumps(
            _reference_feature_manifest(
                train_count_per_format=train_count_per_format,
                test_count_per_format=test_count_per_format,
                source=source,
            )
        ),
        encoding="utf-8",
    )


def _reference_feature_manifest(
    train_count_per_format: int,
    test_count_per_format: int,
    source: str,
) -> dict[str, object]:
    return {
        "schema_version": REFERENCE_FEATURE_CORPUS_SCHEMA_VERSION,
        "source": source,
        "source_description": "test redacted nonfunctional provider-like feature holdout",
        "source_generation_method": "public_provider_morphology_nonfunctional_synthetic_holdout",
        "raw_values_serialized": False,
        "feature_names": list(FEATURE_NAMES),
        "format_features": [
            _format_feature_record(spec, train_count_per_format, test_count_per_format) for spec in list_formats()
        ],
    }


def _format_feature_record(spec, train_count_per_format: int, test_count_per_format: int) -> dict[str, object]:
    slug_offset = sum((index + 1) * ord(character) for index, character in enumerate(spec.slug))
    train_rng = np.random.default_rng(90_000 + slug_offset)
    test_rng = np.random.default_rng(91_000 + slug_offset)
    train_tokens = tuple(spec.random_example(train_rng) for _ in range(train_count_per_format))
    test_tokens = tuple(spec.random_example(test_rng) for _ in range(test_count_per_format))
    model = build_model(spec)
    return {
        "format_slug": spec.slug,
        "train": _feature_split(train_tokens, model),
        "test": _feature_split(test_tokens, model),
    }


def _feature_split(tokens: tuple[str, ...], model) -> dict[str, object]:
    report = compute_report(list(tokens), model)
    numeric = _numeric_profile(tokens)
    return {
        "count": len(tokens),
        "char_entropy_bits": report["char_entropy_bits"],
        "avg_log_likelihood": report["avg_log_likelihood"],
        "numeric_profile": {
            "digit_fraction": numeric.digit_fraction,
            "numeric_run_count_per_token": numeric.numeric_run_count_per_token,
            "numeric_run_avg_length": numeric.numeric_run_avg_length,
            "numeric_run_p95_length": numeric.numeric_run_p95_length,
            "numeric_run_max_length": numeric.numeric_run_max_length,
        },
        "features": [_token_features(token, model) for token in tokens],
    }
