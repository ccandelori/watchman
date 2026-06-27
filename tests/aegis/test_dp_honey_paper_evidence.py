from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aegis.replay.dp_honey_paper_evidence import (
    DP_HONEY_PAPER_EVIDENCE_SCHEMA_VERSION,
    DPHoneyPaperEvidenceConfig,
    DPHoneyPaperEvidenceError,
    build_dp_honey_paper_evidence_report,
    main,
)
from detect.dp_honey.statistical_distinguisher_eval import FEATURE_NAMES

SCANNER_EVAL_PATH = Path("introspection/data/reports/dp_honey_scanner_eval_v1.json")
PROMOTED_GENERATION_REALISM_EVAL_PATH = Path("introspection/data/reports/dp_honey_generation_realism_eval_v2.json")
PROMOTED_STATISTICAL_DISTINGUISHER_EVAL_PATH = Path(
    "introspection/data/reports/dp_honey_statistical_distinguisher_eval_v2.json"
)
PROVIDER_LIKE_STATISTICAL_DISTINGUISHER_EVAL_PATH = Path(
    "introspection/data/reports/dp_honey_statistical_distinguisher_eval_v3.json"
)
PROMOTED_SMOKE_PATH = Path("introspection/data/reports/aegis_default_mock_provider_smoke_dp_honey_segment_v2.json")
PROMOTED_AUDIT_JSONL_PATH = Path(
    "introspection/data/reports/aegis_default_mock_provider_smoke_dp_honey_segment_audit_v2.jsonl"
)
LEGACY_GENERATION_REALISM_EVAL_PATH = Path("introspection/data/reports/dp_honey_generation_realism_eval_v1.json")
LEGACY_STATISTICAL_DISTINGUISHER_EVAL_PATH = Path(
    "introspection/data/reports/dp_honey_statistical_distinguisher_eval_v1.json"
)
LEGACY_SMOKE_PATH = Path("introspection/data/reports/aegis_default_mock_provider_smoke_nimbus_dp_honey_refresh_v2.json")
LEGACY_AUDIT_JSONL_PATH = Path(
    "introspection/data/reports/aegis_default_mock_provider_smoke_nimbus_dp_honey_refresh_audit_v2.jsonl"
)


def test_dp_honey_paper_evidence_report_keeps_synthetic_registry_evidence_non_promotable() -> None:
    report = build_dp_honey_paper_evidence_report(
        DPHoneyPaperEvidenceConfig(
            scanner_eval_path=SCANNER_EVAL_PATH,
            generation_realism_eval_path=PROMOTED_GENERATION_REALISM_EVAL_PATH,
            statistical_distinguisher_eval_path=PROMOTED_STATISTICAL_DISTINGUISHER_EVAL_PATH,
            smoke_path=PROMOTED_SMOKE_PATH,
            audit_jsonl_path=PROMOTED_AUDIT_JSONL_PATH,
        )
    )
    checklist = {str(item["requirement_id"]): item for item in _checklist(report)}

    assert report["schema_version"] == DP_HONEY_PAPER_EVIDENCE_SCHEMA_VERSION
    assert report["promotion_status"] == "paper_aligned_operational_beta"
    assert report["paper_faithful_plus"] is False
    assert report["promotion_eligible"] is False
    assert report["scanner_metrics"]["target_alpha"] == 0.01
    assert report["scanner_metrics"]["target_coverage"] == 0.99
    assert report["scanner_metrics"]["negative_example_count"] == 1000
    assert report["scanner_metrics"]["positive_example_count"] >= 1000
    assert report["scanner_metrics"]["false_positive_rate"] == 0.0
    assert report["scanner_metrics"]["false_negative_rate"] == 0.0
    assert report["generator_metadata"]["corpus_size"] == 2000
    assert report["generation_realism_metrics"]["bounded_sanity_gate_passed"] is True
    assert report["generation_realism_metrics"]["paper_faithful_statistical_distinguisher"] is False
    assert report["statistical_distinguisher_metrics"]["present"] is True
    assert report["statistical_distinguisher_metrics"]["all_required_tests_passed"] is True
    assert report["statistical_distinguisher_metrics"]["synthetic_registry_statistical_distinguisher_passed"] is True
    assert report["statistical_distinguisher_metrics"]["paper_sufficient_reference_source"] is False
    assert report["statistical_distinguisher_metrics"]["paper_faithful_statistical_distinguisher"] is False
    assert report["statistical_distinguisher_metrics"]["test_statuses"]["character_entropy_tests"] == "passed"
    assert report["statistical_distinguisher_metrics"]["test_statuses"]["bigram_likelihood_tests"] == "passed"
    assert report["statistical_distinguisher_metrics"]["test_statuses"]["numeric_substring_tests"] == "passed"
    assert report["statistical_distinguisher_metrics"]["test_statuses"]["discriminator_mlp"] == "passed"
    assert checklist["split_conformal_calibration"]["status"] == "met"
    assert checklist["scanner_fn_fp"]["status"] == "met"
    assert checklist["gateway_substitution_and_ledger"]["status"] == "met"
    assert checklist["output_leak_detection"]["status"] == "met"
    assert checklist["redacted_audit"]["status"] == "met"
    assert checklist["format_fidelity"]["status"] == "met"
    assert checklist["statistical_realism_distinguishers"]["status"] == "partial"
    assert checklist["tool_argument_leakage"]["status"] == "met"
    assert report["checklist_summary"] == {"met": 8, "missing": 0, "partial": 1, "total": 9}
    assert "same_format_uniform_synthetic_holdout" in " ".join(report["missing_before_paper_faithful_plus"])


def test_dp_honey_paper_evidence_accepts_provider_like_feature_manifest_metadata(tmp_path: Path) -> None:
    statistical_path = tmp_path / "provider-like-statistical-distinguisher.json"
    payload = json.loads(PROMOTED_STATISTICAL_DISTINGUISHER_EVAL_PATH.read_text(encoding="utf-8"))
    payload["reference_source"] = "provider_like_sealed_holdout"
    payload["paper_faithful_statistical_distinguisher"] = True
    payload["reference_feature_corpus"] = {
        "schema_version": "detect.dp_honey.reference_feature_corpus/v1",
        "source": "provider_like_sealed_holdout",
        "source_description": "test redacted nonfunctional provider-like feature holdout",
        "source_generation_method": "public_provider_morphology_nonfunctional_synthetic_holdout",
        "sha256": "a" * 64,
        "raw_values_serialized": False,
        "feature_names": list(FEATURE_NAMES),
        "format_count": payload["format_count"],
        "train_count_per_format": payload["train_count_per_format"],
        "test_count_per_format": payload["test_count_per_format"],
    }
    statistical_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_dp_honey_paper_evidence_report(
        DPHoneyPaperEvidenceConfig(
            scanner_eval_path=SCANNER_EVAL_PATH,
            generation_realism_eval_path=PROMOTED_GENERATION_REALISM_EVAL_PATH,
            statistical_distinguisher_eval_path=statistical_path,
            smoke_path=PROMOTED_SMOKE_PATH,
            audit_jsonl_path=PROMOTED_AUDIT_JSONL_PATH,
        )
    )
    checklist = {str(item["requirement_id"]): item for item in _checklist(report)}

    assert report["promotion_status"] == "paper_faithful_plus_candidate"
    assert report["paper_faithful_plus"] is True
    assert report["promotion_eligible"] is True
    assert report["statistical_distinguisher_metrics"]["paper_sufficient_reference_source"] is True
    assert checklist["statistical_realism_distinguishers"]["status"] == "met"
    assert report["checklist_summary"] == {"met": 9, "missing": 0, "partial": 0, "total": 9}


def test_dp_honey_paper_evidence_report_accepts_provider_like_artifact() -> None:
    report = build_dp_honey_paper_evidence_report(
        DPHoneyPaperEvidenceConfig(
            scanner_eval_path=SCANNER_EVAL_PATH,
            generation_realism_eval_path=PROMOTED_GENERATION_REALISM_EVAL_PATH,
            statistical_distinguisher_eval_path=PROVIDER_LIKE_STATISTICAL_DISTINGUISHER_EVAL_PATH,
            smoke_path=PROMOTED_SMOKE_PATH,
            audit_jsonl_path=PROMOTED_AUDIT_JSONL_PATH,
        )
    )

    assert report["promotion_status"] == "paper_faithful_plus_candidate"
    assert report["paper_faithful_plus"] is True
    assert report["promotion_eligible"] is True
    assert report["checklist_summary"] == {"met": 9, "missing": 0, "partial": 0, "total": 9}
    assert report["statistical_distinguisher_metrics"]["reference_source"] == "provider_like_sealed_holdout"
    assert report["statistical_distinguisher_metrics"]["paper_sufficient_reference_source"] is True


def test_dp_honey_paper_evidence_report_keeps_failed_statistical_suite_as_beta() -> None:
    report = build_dp_honey_paper_evidence_report(
        DPHoneyPaperEvidenceConfig(
            scanner_eval_path=SCANNER_EVAL_PATH,
            generation_realism_eval_path=LEGACY_GENERATION_REALISM_EVAL_PATH,
            statistical_distinguisher_eval_path=LEGACY_STATISTICAL_DISTINGUISHER_EVAL_PATH,
            smoke_path=LEGACY_SMOKE_PATH,
            audit_jsonl_path=LEGACY_AUDIT_JSONL_PATH,
        )
    )
    checklist = {str(item["requirement_id"]): item for item in _checklist(report)}

    assert report["promotion_status"] == "paper_aligned_operational_beta"
    assert report["paper_faithful_plus"] is False
    assert report["promotion_eligible"] is False
    assert report["statistical_distinguisher_metrics"]["all_required_tests_passed"] is False
    assert report["statistical_distinguisher_metrics"]["test_statuses"]["bigram_likelihood_tests"] == "failed"
    assert report["statistical_distinguisher_metrics"]["test_statuses"]["discriminator_mlp"] == "failed"
    assert checklist["statistical_realism_distinguishers"]["status"] == "partial"
    assert report["checklist_summary"] == {"met": 8, "missing": 0, "partial": 1, "total": 9}


def test_dp_honey_paper_evidence_cli_writes_json(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "dp-honey-paper-evidence.json"
    monkeypatch.setattr(
        sys,
        "argv",
        (
            "aegis-dp-honey-paper-evidence",
            "--scanner-eval",
            str(SCANNER_EVAL_PATH),
            "--generation-realism-eval",
            str(PROMOTED_GENERATION_REALISM_EVAL_PATH),
            "--statistical-distinguisher-eval",
            str(PROMOTED_STATISTICAL_DISTINGUISHER_EVAL_PATH),
            "--smoke",
            str(PROMOTED_SMOKE_PATH),
            "--audit-jsonl",
            str(PROMOTED_AUDIT_JSONL_PATH),
            "--output",
            str(output_path),
        ),
    )

    main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == DP_HONEY_PAPER_EVIDENCE_SCHEMA_VERSION
    assert payload["promotion_eligible"] is False
    assert payload["artifact_hashes"]["scanner_eval_sha256"]
    assert payload["artifact_hashes"]["generation_realism_eval_sha256"]
    assert payload["artifact_hashes"]["statistical_distinguisher_eval_sha256"]
    assert payload["statistical_distinguisher_metrics"]["present"] is True


def test_dp_honey_paper_evidence_missing_statistical_suite_stays_partial() -> None:
    report = build_dp_honey_paper_evidence_report(
        DPHoneyPaperEvidenceConfig(
            scanner_eval_path=SCANNER_EVAL_PATH,
            generation_realism_eval_path=PROMOTED_GENERATION_REALISM_EVAL_PATH,
            statistical_distinguisher_eval_path=None,
            smoke_path=PROMOTED_SMOKE_PATH,
            audit_jsonl_path=PROMOTED_AUDIT_JSONL_PATH,
        )
    )
    checklist = {str(item["requirement_id"]): item for item in _checklist(report)}

    assert report["paper_faithful_plus"] is False
    assert report["statistical_distinguisher_metrics"]["present"] is False
    assert checklist["statistical_realism_distinguishers"]["status"] == "partial"


def test_dp_honey_paper_evidence_rejects_forged_generation_realism_eval(tmp_path: Path) -> None:
    forged_path = tmp_path / "forged-generation-realism.json"
    forged_path.write_text(
        json.dumps(
            {
                "all_generated_tokens_valid": True,
                "all_reference_tokens_valid": True,
                "bounded_sanity_gate_passed": True,
                "paper_faithful_statistical_distinguisher": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DPHoneyPaperEvidenceError, match=r"generation_realism_eval\.schema_version"):
        build_dp_honey_paper_evidence_report(
            DPHoneyPaperEvidenceConfig(
                scanner_eval_path=SCANNER_EVAL_PATH,
                generation_realism_eval_path=forged_path,
                statistical_distinguisher_eval_path=PROMOTED_STATISTICAL_DISTINGUISHER_EVAL_PATH,
                smoke_path=PROMOTED_SMOKE_PATH,
                audit_jsonl_path=PROMOTED_AUDIT_JSONL_PATH,
            )
        )


def test_dp_honey_paper_evidence_rejects_unproven_paper_faithful_realism_flag(tmp_path: Path) -> None:
    forged_path = tmp_path / "forged-paper-faithful-realism.json"
    payload = json.loads(PROMOTED_GENERATION_REALISM_EVAL_PATH.read_text(encoding="utf-8"))
    payload["paper_faithful_statistical_distinguisher"] = True
    forged_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DPHoneyPaperEvidenceError, match="statistical_distinguisher_suite"):
        build_dp_honey_paper_evidence_report(
            DPHoneyPaperEvidenceConfig(
                scanner_eval_path=SCANNER_EVAL_PATH,
                generation_realism_eval_path=forged_path,
                statistical_distinguisher_eval_path=PROMOTED_STATISTICAL_DISTINGUISHER_EVAL_PATH,
                smoke_path=PROMOTED_SMOKE_PATH,
                audit_jsonl_path=PROMOTED_AUDIT_JSONL_PATH,
            )
        )


def test_dp_honey_paper_evidence_rejects_forged_statistical_distinguisher_flag(tmp_path: Path) -> None:
    forged_path = tmp_path / "forged-statistical-distinguisher.json"
    payload = json.loads(PROMOTED_STATISTICAL_DISTINGUISHER_EVAL_PATH.read_text(encoding="utf-8"))
    payload["paper_faithful_statistical_distinguisher"] = True
    forged_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DPHoneyPaperEvidenceError, match="reference source"):
        build_dp_honey_paper_evidence_report(
            DPHoneyPaperEvidenceConfig(
                scanner_eval_path=SCANNER_EVAL_PATH,
                generation_realism_eval_path=PROMOTED_GENERATION_REALISM_EVAL_PATH,
                statistical_distinguisher_eval_path=forged_path,
                smoke_path=PROMOTED_SMOKE_PATH,
                audit_jsonl_path=PROMOTED_AUDIT_JSONL_PATH,
            )
        )


def test_dp_honey_paper_evidence_rejects_forged_provider_like_source_without_manifest(tmp_path: Path) -> None:
    forged_path = tmp_path / "forged-provider-like-source.json"
    payload = json.loads(PROMOTED_STATISTICAL_DISTINGUISHER_EVAL_PATH.read_text(encoding="utf-8"))
    payload["reference_source"] = "provider_like_sealed_holdout"
    payload["paper_faithful_statistical_distinguisher"] = True
    payload["reference_feature_corpus"] = None
    forged_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DPHoneyPaperEvidenceError, match="reference_feature_corpus"):
        build_dp_honey_paper_evidence_report(
            DPHoneyPaperEvidenceConfig(
                scanner_eval_path=SCANNER_EVAL_PATH,
                generation_realism_eval_path=PROMOTED_GENERATION_REALISM_EVAL_PATH,
                statistical_distinguisher_eval_path=forged_path,
                smoke_path=PROMOTED_SMOKE_PATH,
                audit_jsonl_path=PROMOTED_AUDIT_JSONL_PATH,
            )
        )


def test_dp_honey_paper_evidence_rejects_forged_provider_like_source_with_wrong_features(tmp_path: Path) -> None:
    forged_path = tmp_path / "forged-provider-like-feature-schema.json"
    payload = json.loads(PROMOTED_STATISTICAL_DISTINGUISHER_EVAL_PATH.read_text(encoding="utf-8"))
    payload["reference_source"] = "provider_like_sealed_holdout"
    payload["paper_faithful_statistical_distinguisher"] = True
    payload["reference_feature_corpus"] = {
        "schema_version": "detect.dp_honey.reference_feature_corpus/v1",
        "source": "provider_like_sealed_holdout",
        "source_description": "test redacted nonfunctional provider-like feature holdout",
        "source_generation_method": "public_provider_morphology_nonfunctional_synthetic_holdout",
        "sha256": "a" * 64,
        "raw_values_serialized": False,
        "feature_names": ["token_length"],
        "format_count": payload["format_count"],
        "train_count_per_format": payload["train_count_per_format"],
        "test_count_per_format": payload["test_count_per_format"],
    }
    forged_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DPHoneyPaperEvidenceError, match="feature_names"):
        build_dp_honey_paper_evidence_report(
            DPHoneyPaperEvidenceConfig(
                scanner_eval_path=SCANNER_EVAL_PATH,
                generation_realism_eval_path=PROMOTED_GENERATION_REALISM_EVAL_PATH,
                statistical_distinguisher_eval_path=forged_path,
                smoke_path=PROMOTED_SMOKE_PATH,
                audit_jsonl_path=PROMOTED_AUDIT_JSONL_PATH,
            )
        )


def test_dp_honey_paper_evidence_rejects_runtime_eval_parameter_mismatch(tmp_path: Path) -> None:
    forged_audit_path = tmp_path / "forged-audit.jsonl"
    audit_text = PROMOTED_AUDIT_JSONL_PATH.read_text(encoding="utf-8")
    forged_text = audit_text.replace('"corpus_size": 2000', '"corpus_size": 200')
    forged_text = forged_text.replace('"corpus_size":2000', '"corpus_size":200')
    assert forged_text != audit_text
    forged_audit_path.write_text(forged_text, encoding="utf-8")

    with pytest.raises(DPHoneyPaperEvidenceError, match="must match runtime audit DP-HONEY metadata"):
        build_dp_honey_paper_evidence_report(
            DPHoneyPaperEvidenceConfig(
                scanner_eval_path=SCANNER_EVAL_PATH,
                generation_realism_eval_path=PROMOTED_GENERATION_REALISM_EVAL_PATH,
                statistical_distinguisher_eval_path=PROMOTED_STATISTICAL_DISTINGUISHER_EVAL_PATH,
                smoke_path=PROMOTED_SMOKE_PATH,
                audit_jsonl_path=forged_audit_path,
            )
        )


def _checklist(report: dict[str, object]) -> tuple[dict[str, object], ...]:
    checklist = report["checklist"]
    if not isinstance(checklist, list):
        raise AssertionError("checklist must be a list.")
    return tuple(item for item in checklist if isinstance(item, dict))
