from __future__ import annotations

import json
import sys
from pathlib import Path

from aegis.replay.dp_honey_paper_evidence import (
    DP_HONEY_PAPER_EVIDENCE_SCHEMA_VERSION,
    DPHoneyPaperEvidenceConfig,
    build_dp_honey_paper_evidence_report,
    main,
)

SCANNER_EVAL_PATH = Path("introspection/data/reports/dp_honey_scanner_eval_v1.json")
SMOKE_PATH = Path("introspection/data/reports/aegis_default_mock_provider_smoke_nimbus_dp_honey_refresh_v1.json")
AUDIT_JSONL_PATH = Path(
    "introspection/data/reports/aegis_default_mock_provider_smoke_nimbus_dp_honey_refresh_audit_v1.jsonl"
)


def test_dp_honey_paper_evidence_report_separates_met_and_partial_requirements() -> None:
    report = build_dp_honey_paper_evidence_report(
        DPHoneyPaperEvidenceConfig(
            scanner_eval_path=SCANNER_EVAL_PATH,
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
    assert checklist["split_conformal_calibration"]["status"] == "met"
    assert checklist["scanner_fn_fp"]["status"] == "met"
    assert checklist["gateway_substitution_and_ledger"]["status"] == "met"
    assert checklist["output_leak_detection"]["status"] == "met"
    assert checklist["redacted_audit"]["status"] == "met"
    assert checklist["format_fidelity"]["status"] == "partial"
    assert checklist["tool_argument_leakage"]["status"] == "partial"
    assert report["checklist_summary"] == {"met": 6, "missing": 0, "partial": 2, "total": 8}
    assert "statistical distinguisher" in " ".join(report["missing_before_paper_faithful_plus"])
    assert "tool-call arguments" in " ".join(report["missing_before_paper_faithful_plus"])


def test_dp_honey_paper_evidence_cli_writes_json(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "dp-honey-paper-evidence.json"
    monkeypatch.setattr(
        sys,
        "argv",
        (
            "aegis-dp-honey-paper-evidence",
            "--scanner-eval",
            str(SCANNER_EVAL_PATH),
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


def _checklist(report: dict[str, object]) -> tuple[dict[str, object], ...]:
    checklist = report["checklist"]
    if not isinstance(checklist, list):
        raise AssertionError("checklist must be a list.")
    return tuple(item for item in checklist if isinstance(item, dict))
