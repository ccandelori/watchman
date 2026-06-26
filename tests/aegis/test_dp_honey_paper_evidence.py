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

SCANNER_EVAL_PATH = Path("introspection/data/reports/dp_honey_scanner_eval_v1.json")
GENERATION_REALISM_EVAL_PATH = Path("introspection/data/reports/dp_honey_generation_realism_eval_v1.json")
SMOKE_PATH = Path("introspection/data/reports/aegis_default_mock_provider_smoke_nimbus_dp_honey_refresh_v2.json")
AUDIT_JSONL_PATH = Path(
    "introspection/data/reports/aegis_default_mock_provider_smoke_nimbus_dp_honey_refresh_audit_v2.jsonl"
)


def test_dp_honey_paper_evidence_report_separates_met_and_partial_requirements() -> None:
    report = build_dp_honey_paper_evidence_report(
        DPHoneyPaperEvidenceConfig(
            scanner_eval_path=SCANNER_EVAL_PATH,
            generation_realism_eval_path=GENERATION_REALISM_EVAL_PATH,
            smoke_path=SMOKE_PATH,
            audit_jsonl_path=AUDIT_JSONL_PATH,
        )
    )
    checklist = {str(item["requirement_id"]): item for item in _checklist(report)}

    assert report["schema_version"] == DP_HONEY_PAPER_EVIDENCE_SCHEMA_VERSION
    assert report["promotion_status"] == "paper_aligned_operational_beta"
    assert report["paper_faithful_plus"] is False
    assert report["promotion_eligible"] is False
    assert report["scanner_metrics"]["false_positive_rate"] == 0.0
    assert report["scanner_metrics"]["false_negative_rate"] == 0.0
    assert report["generation_realism_metrics"]["bounded_sanity_gate_passed"] is True
    assert report["generation_realism_metrics"]["paper_faithful_statistical_distinguisher"] is False
    assert checklist["split_conformal_calibration"]["status"] == "met"
    assert checklist["scanner_fn_fp"]["status"] == "met"
    assert checklist["gateway_substitution_and_ledger"]["status"] == "met"
    assert checklist["output_leak_detection"]["status"] == "met"
    assert checklist["redacted_audit"]["status"] == "met"
    assert checklist["format_fidelity"]["status"] == "met"
    assert checklist["statistical_realism_distinguishers"]["status"] == "partial"
    assert checklist["tool_argument_leakage"]["status"] == "met"
    assert report["checklist_summary"] == {"met": 8, "missing": 0, "partial": 1, "total": 9}
    assert "statistical distinguisher" in " ".join(report["missing_before_paper_faithful_plus"])
    assert "tool-call arguments" not in " ".join(report["missing_before_paper_faithful_plus"])


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
            str(GENERATION_REALISM_EVAL_PATH),
            "--smoke",
            str(SMOKE_PATH),
            "--audit-jsonl",
            str(AUDIT_JSONL_PATH),
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
                smoke_path=SMOKE_PATH,
                audit_jsonl_path=AUDIT_JSONL_PATH,
            )
        )


def test_dp_honey_paper_evidence_rejects_unproven_paper_faithful_realism_flag(tmp_path: Path) -> None:
    forged_path = tmp_path / "forged-paper-faithful-realism.json"
    payload = json.loads(GENERATION_REALISM_EVAL_PATH.read_text(encoding="utf-8"))
    payload["paper_faithful_statistical_distinguisher"] = True
    forged_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DPHoneyPaperEvidenceError, match="statistical_distinguisher_suite"):
        build_dp_honey_paper_evidence_report(
            DPHoneyPaperEvidenceConfig(
                scanner_eval_path=SCANNER_EVAL_PATH,
                generation_realism_eval_path=forged_path,
                smoke_path=SMOKE_PATH,
                audit_jsonl_path=AUDIT_JSONL_PATH,
            )
        )


def _checklist(report: dict[str, object]) -> tuple[dict[str, object], ...]:
    checklist = report["checklist"]
    if not isinstance(checklist, list):
        raise AssertionError("checklist must be a list.")
    return tuple(item for item in checklist if isinstance(item, dict))
