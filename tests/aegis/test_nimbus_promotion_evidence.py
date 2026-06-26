from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aegis.core.contracts import JsonValue
from aegis.replay.nimbus_promotion_evidence import (
    NIMBUS_PROMOTION_EVIDENCE_SCHEMA_VERSION,
    NimbusPromotionEvidenceConfig,
    NimbusPromotionEvidenceError,
    build_nimbus_promotion_evidence_report,
    main,
)

DETERMINISTIC_EVAL_PATH = Path("introspection/data/reports/aegis_nimbus_deterministic_beta_eval_v1.json")
CALIBRATION_MANIFEST_PATH = Path("introspection/data/reports/aegis_nimbus_training_corpus_manifest_v0.json")
SEALED_MANIFEST_PATH = Path("introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_manifest_v0.json")
INFONCE_MODEL_PATH = Path("introspection/data/reports/aegis_nimbus_infonce_model_v0.json")
GROUPED_CV_PATH = Path("introspection/data/reports/aegis_nimbus_infonce_grouped_cv_v0.json")
SEALED_HOLDOUT_PATH = Path("introspection/data/reports/aegis_nimbus_infonce_sealed_holdout_eval_v0.json")
GATEWAY_SMOKE_PATH = Path("introspection/data/reports/aegis_default_mock_provider_smoke_learned_nimbus_beta_v3.json")
RUNTIME_BETA_EVAL_PATH = Path("introspection/data/reports/aegis_nimbus_runtime_beta_eval_v0.json")


def test_nimbus_promotion_evidence_keeps_learned_scaffold_non_promotable() -> None:
    report = build_nimbus_promotion_evidence_report(_config())
    checklist = {str(item["requirement_id"]): item for item in _checklist(report)}

    assert report["schema_version"] == NIMBUS_PROMOTION_EVIDENCE_SCHEMA_VERSION
    assert report["promotion_status"] == "deterministic_beta_active_learned_not_promotable"
    assert report["promotion_eligible"] is False
    assert report["promote_learned_runtime"] is False
    assert report["promote_hybrid_runtime"] is False
    assert report["keep_deterministic_default"] is True
    assert report["reject_learned_runtime"] is True
    assert report["paper_faithful_learned_critic"] is False
    assert report["recommended_runtime_critic"] == "deterministic_canary_beta"
    assert report["promotion_decision"]["verdict"] == "reject_learned_runtime_keep_deterministic_default"
    assert report["promotion_decision"]["reject_learned_runtime"] is True
    assert report["promotion_decision"]["keep_deterministic_default"] is True
    assert report["deterministic_baseline_metrics"]["false_negative_rate"] == 0.0
    assert report["learned_sealed_holdout_metrics"]["false_negative_rate"] == 0.0
    assert report["learned_sealed_holdout_metrics"]["false_positive_rate"] == 0.005369127516778523
    assert report["learned_sealed_holdout_metrics"]["training_eval_reused"] is False
    assert report["learned_sealed_holdout_metrics"]["training_eval_allowed"] is False
    assert report["comparison"]["learned_turn_fnr_beats_deterministic"] is False
    assert report["comparison"]["learned_turn_fnr_matches_deterministic"] is True
    assert report["comparison"]["learned_runtime_beta_turn_rates_within_policy"] is True
    assert report["comparison"]["offline_learned_session_signal_observed"] is False
    assert report["comparison"]["learned_session_signal_complements_deterministic"] is False
    assert report["comparison"]["learned_runtime_adapter_evidence_present"] is True
    assert report["comparison"]["learned_runtime_gateway_evidence_present"] is True
    assert report["comparison"]["runtime_beta_paper_session_false_blocks_clean"] is True
    assert report["learned_runtime_beta_metrics"]["false_negative_rate"] == 0.0
    assert report["learned_runtime_beta_metrics"]["false_positive_rate"] == 0.005369127516778523
    assert report["learned_runtime_beta_metrics"]["session_false_negative_rate"] == 0.0
    assert report["learned_runtime_beta_metrics"]["session_false_positive_rate"] == 0.0
    assert report["learned_runtime_beta_metrics"]["session_block_false_negative_rate"] == 0.0
    assert report["learned_runtime_beta_metrics"]["session_block_false_positive_rate"] == 0.0
    assert report["learned_runtime_beta_metrics"]["paper_conversation_metrics"]["false_block_rate"] == 0.0
    assert report["gateway_runtime_evidence"]["readiness_nimbus_status"] == "learned_runtime_beta"
    assert report["gateway_runtime_evidence"]["learned_runtime_evidence_present"] is True
    assert report["gateway_runtime_evidence"]["learned_gateway_metrics"]["sample_count"] == 6
    assert report["gateway_runtime_evidence"]["learned_gateway_metrics"]["false_positive_rate"] == 0.0
    assert report["gateway_runtime_evidence"]["learned_gateway_metrics"]["false_negative_rate"] == 0.0
    assert report["gateway_runtime_evidence"]["adversarial_benign_final_action"] == "allow"
    assert report["gateway_runtime_evidence"]["adversarial_benign_nimbus_action"] == "allow"
    critic_binding = report["gateway_runtime_evidence"]["learned_gateway_critic_binding"]
    assert critic_binding["model_artifact_sha256s"] == [
        "8c5bd62b4f54d9a0758c90cca93521b9498ce4252c6f143a4d2cb2a6cd8725e8"
    ]
    assert critic_binding["negative_context_counts"] == [16]
    assert critic_binding["candidate_counts"] == [17]
    gates = _gate_map(report)
    assert gates["grouped_cv_metrics"]["status"] == "met"
    assert gates["sealed_holdout_metrics"]["status"] == "met"
    assert gates["turn_level_fn_fp"]["status"] == "met"
    assert gates["session_level_fn_fp"]["status"] == "met"
    assert gates["benign_false_block_rate"]["status"] == "met"
    assert gates["live_gateway_metrics"]["status"] == "met"
    assert gates["head_to_head_common_live_corpus"]["status"] == "missing"
    assert gates["latency"]["status"] == "missing"
    assert gates["hybrid_policy_evaluation"]["status"] == "missing"
    assert gates["promotion_manifest"]["status"] == "missing"
    blocking_gate_ids = report["promotion_decision"]["blocking_gate_ids"]
    assert isinstance(blocking_gate_ids, list)
    assert "head_to_head_common_live_corpus" in blocking_gate_ids
    assert "latency" in blocking_gate_ids
    assert "hybrid_policy_evaluation" in blocking_gate_ids
    assert report["head_to_head_evaluation"]["hybrid_policy"]["promote_hybrid_runtime"] is False
    assert checklist["session_level_corpus_coverage"]["status"] == "met"
    assert checklist["negative_contexts_for_infonce"]["status"] == "met"
    assert checklist["grouped_cross_validation"]["status"] == "met"
    assert checklist["sealed_holdout"]["status"] == "met"
    assert checklist["fn_fp_reported_separately"]["status"] == "met"
    assert checklist["learned_beats_or_complements_deterministic"]["status"] == "partial"
    assert checklist["runtime_learned_critic_adapter"]["status"] == "met"
    assert checklist["live_gateway_learned_fn_fp"]["status"] == "met"
    assert checklist["promotion_manifest"]["status"] == "missing"
    assert report["checklist_summary"] == {"met": 7, "missing": 1, "partial": 1, "total": 9}
    missing = " ".join(str(item) for item in report["missing_before_paper_faithful_learned_promotion"])
    assert "ties deterministic beta" in missing
    assert "live runtime head-to-head" in missing
    assert "promotion manifest" in missing
    assert "runtime adapter" not in missing
    assert "gateway smoke proves deterministic beta" not in missing


def test_nimbus_promotion_evidence_recognizes_opt_in_learned_gateway_metrics(tmp_path: Path) -> None:
    gateway_smoke_path = tmp_path / "learned-gateway-smoke.json"
    smoke = json.loads(GATEWAY_SMOKE_PATH.read_text(encoding="utf-8"))
    checks = smoke["checks"]
    checks["gateway_readiness"]["nimbus_status"] = "learned_runtime_beta"
    checks["gateway_readiness"]["nimbus_critic_kind"] = "learned_infonce_beta"
    checks["gateway_readiness"]["nimbus_paper_faithful_learned_critic"] = False
    checks["gateway_readiness"]["nimbus_promotion_status"] = "learned_runtime_beta_not_promotable"
    checks["capabilities"]["nimbus_status"] = "learned_runtime_beta"
    checks["capabilities"]["nimbus_critic_kind"] = "learned_infonce_beta"
    checks["capabilities"]["nimbus_paper_faithful_learned_critic"] = False
    checks["capabilities"]["nimbus_promotion_status"] = "learned_runtime_beta_not_promotable"
    checks["benign_chat"]["nimbus"] = _learned_nimbus_summary("allow")
    checks["adversarial_benign_no_block"] = {
        "final_action": "allow",
        "nimbus": _learned_nimbus_summary("allow"),
    }
    checks["tool_argument_canary_leak"]["nimbus_tool"] = _learned_nimbus_summary("block")
    checks["encoded_canary_leak"]["nimbus"] = _learned_nimbus_summary("block")
    checks["metadata_slot_canary_leak"]["nimbus"] = _learned_nimbus_summary("block")
    checks["nimbus_partial_leak"]["nimbus"] = _learned_nimbus_summary("warn")
    gateway_smoke_path.write_text(json.dumps(smoke), encoding="utf-8")

    report = build_nimbus_promotion_evidence_report(_config_with_gateway_smoke(gateway_smoke_path))
    checklist = {str(item["requirement_id"]): item for item in _checklist(report)}
    learned_gateway_metrics = report["gateway_runtime_evidence"]["learned_gateway_metrics"]

    assert report["promotion_eligible"] is False
    assert report["promote_learned_runtime"] is False
    assert report["reject_learned_runtime"] is True
    assert report["gateway_runtime_evidence"]["runtime_critic_kind"] == "learned_infonce_beta"
    assert report["gateway_runtime_evidence"]["learned_runtime_evidence_present"] is True
    assert learned_gateway_metrics["sample_count"] == 6
    assert learned_gateway_metrics["true_positive"] == 4
    assert learned_gateway_metrics["true_negative"] == 2
    assert learned_gateway_metrics["false_positive"] == 0
    assert learned_gateway_metrics["false_negative"] == 0
    assert learned_gateway_metrics["false_positive_rate"] == 0.0
    assert learned_gateway_metrics["false_negative_rate"] == 0.0
    assert report["comparison"]["learned_runtime_gateway_evidence_present"] is True
    assert checklist["live_gateway_learned_fn_fp"]["status"] == "met"
    assert checklist["promotion_manifest"]["status"] == "missing"


def test_nimbus_promotion_evidence_cli_writes_json(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "nimbus-promotion-evidence.json"
    monkeypatch.setattr(
        sys,
        "argv",
        (
            "aegis-nimbus-promotion-evidence",
            "--deterministic-eval",
            str(DETERMINISTIC_EVAL_PATH),
            "--calibration-manifest",
            str(CALIBRATION_MANIFEST_PATH),
            "--sealed-manifest",
            str(SEALED_MANIFEST_PATH),
            "--infonce-model",
            str(INFONCE_MODEL_PATH),
            "--grouped-cv",
            str(GROUPED_CV_PATH),
            "--sealed-holdout",
            str(SEALED_HOLDOUT_PATH),
            "--gateway-smoke",
            str(GATEWAY_SMOKE_PATH),
            "--runtime-beta-eval",
            str(RUNTIME_BETA_EVAL_PATH),
            "--output",
            str(output_path),
        ),
    )

    main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == NIMBUS_PROMOTION_EVIDENCE_SCHEMA_VERSION
    assert payload["promote_learned_runtime"] is False
    assert payload["reject_learned_runtime"] is True
    assert payload["keep_deterministic_default"] is True
    assert payload["artifact_hashes"]["sealed_holdout_sha256"]


def test_nimbus_promotion_evidence_rejects_zero_count_required_label(tmp_path: Path) -> None:
    sealed_manifest_path = tmp_path / "sealed-manifest.json"
    sealed_manifest = json.loads(SEALED_MANIFEST_PATH.read_text(encoding="utf-8"))
    sealed_manifest["label_counts"]["partial"] = 0
    sealed_manifest_path.write_text(json.dumps(sealed_manifest), encoding="utf-8")

    report = build_nimbus_promotion_evidence_report(_config_with_sealed_manifest(sealed_manifest_path))
    checklist = {str(item["requirement_id"]): item for item in _checklist(report)}

    assert checklist["session_level_corpus_coverage"]["status"] == "missing"
    assert checklist["sealed_holdout"]["status"] == "missing"
    assert "required leakage-label coverage" in " ".join(str(gap) for gap in checklist["sealed_holdout"]["gaps"])


def test_nimbus_promotion_evidence_requires_distinct_sealed_manifest() -> None:
    report = build_nimbus_promotion_evidence_report(_config_with_sealed_manifest(CALIBRATION_MANIFEST_PATH))
    checklist = {str(item["requirement_id"]): item for item in _checklist(report)}

    assert report["corpus_evidence"]["profiles_distinct"] is False
    assert checklist["sealed_holdout"]["status"] == "missing"
    assert "not distinct" in " ".join(str(gap) for gap in checklist["sealed_holdout"]["gaps"])


def test_nimbus_promotion_evidence_reports_malformed_json_path(tmp_path: Path) -> None:
    grouped_cv_path = tmp_path / "bad-grouped-cv.json"
    grouped_cv_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(NimbusPromotionEvidenceError, match=str(grouped_cv_path)):
        build_nimbus_promotion_evidence_report(_config_with_grouped_cv(grouped_cv_path))


def _config() -> NimbusPromotionEvidenceConfig:
    return NimbusPromotionEvidenceConfig(
        deterministic_eval_path=DETERMINISTIC_EVAL_PATH,
        calibration_manifest_path=CALIBRATION_MANIFEST_PATH,
        sealed_manifest_path=SEALED_MANIFEST_PATH,
        infonce_model_path=INFONCE_MODEL_PATH,
        grouped_cv_path=GROUPED_CV_PATH,
        sealed_holdout_path=SEALED_HOLDOUT_PATH,
        gateway_smoke_path=GATEWAY_SMOKE_PATH,
        runtime_beta_eval_path=RUNTIME_BETA_EVAL_PATH,
    )


def _config_with_sealed_manifest(sealed_manifest_path: Path) -> NimbusPromotionEvidenceConfig:
    return NimbusPromotionEvidenceConfig(
        deterministic_eval_path=DETERMINISTIC_EVAL_PATH,
        calibration_manifest_path=CALIBRATION_MANIFEST_PATH,
        sealed_manifest_path=sealed_manifest_path,
        infonce_model_path=INFONCE_MODEL_PATH,
        grouped_cv_path=GROUPED_CV_PATH,
        sealed_holdout_path=SEALED_HOLDOUT_PATH,
        gateway_smoke_path=GATEWAY_SMOKE_PATH,
        runtime_beta_eval_path=RUNTIME_BETA_EVAL_PATH,
    )


def _config_with_grouped_cv(grouped_cv_path: Path) -> NimbusPromotionEvidenceConfig:
    return NimbusPromotionEvidenceConfig(
        deterministic_eval_path=DETERMINISTIC_EVAL_PATH,
        calibration_manifest_path=CALIBRATION_MANIFEST_PATH,
        sealed_manifest_path=SEALED_MANIFEST_PATH,
        infonce_model_path=INFONCE_MODEL_PATH,
        grouped_cv_path=grouped_cv_path,
        sealed_holdout_path=SEALED_HOLDOUT_PATH,
        gateway_smoke_path=GATEWAY_SMOKE_PATH,
        runtime_beta_eval_path=RUNTIME_BETA_EVAL_PATH,
    )


def _config_with_gateway_smoke(gateway_smoke_path: Path) -> NimbusPromotionEvidenceConfig:
    return NimbusPromotionEvidenceConfig(
        deterministic_eval_path=DETERMINISTIC_EVAL_PATH,
        calibration_manifest_path=CALIBRATION_MANIFEST_PATH,
        sealed_manifest_path=SEALED_MANIFEST_PATH,
        infonce_model_path=INFONCE_MODEL_PATH,
        grouped_cv_path=GROUPED_CV_PATH,
        sealed_holdout_path=SEALED_HOLDOUT_PATH,
        gateway_smoke_path=gateway_smoke_path,
        runtime_beta_eval_path=RUNTIME_BETA_EVAL_PATH,
    )


def _learned_nimbus_summary(recommended_action: str) -> dict[str, JsonValue]:
    return {
        "present": True,
        "detector_name": "nimbus",
        "recommended_action": recommended_action,
        "critic_kind": "learned_infonce_beta",
        "critic_version": "nimbus-infonce-lexical-v0",
        "paper_faithful_learned_critic": False,
        "promotion_status": "learned_runtime_beta_not_promotable",
        "turn_estimated_leakage_bits": 1.0 if recommended_action != "allow" else 0.0,
        "budget_fraction": 1.0 if recommended_action != "allow" else 0.0,
    }


def _checklist(report: dict[str, object]) -> tuple[dict[str, object], ...]:
    checklist = report["checklist"]
    if not isinstance(checklist, list):
        raise AssertionError("checklist must be a list.")
    return tuple(item for item in checklist if isinstance(item, dict))


def _gate_map(report: dict[str, object]) -> dict[str, dict[str, object]]:
    gates = report["promotion_gates"]
    if not isinstance(gates, list):
        raise AssertionError("promotion_gates must be a list.")
    return {str(gate["gate_id"]): gate for gate in gates if isinstance(gate, dict)}
