from __future__ import annotations

import json
import sys
from pathlib import Path

from aegis.replay.nimbus_runtime_beta import (
    NIMBUS_RUNTIME_BETA_EVAL_SCHEMA_VERSION,
    NimbusRuntimeBetaEvalConfig,
    build_nimbus_runtime_beta_eval_report,
    main,
)
from aegis.replay.nimbus_training import NIMBUS_REFERENCE_CONVERSATION_COUNT, NIMBUS_REFERENCE_TURNS_PER_CONVERSATION

INFONCE_MODEL_PATH = Path("introspection/data/reports/aegis_nimbus_infonce_model_v0.json")
SEALED_HOLDOUT_PATH = Path("introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_v0.jsonl")


def test_nimbus_runtime_beta_eval_reports_fn_fp_without_raw_contexts() -> None:
    report = build_nimbus_runtime_beta_eval_report(
        NimbusRuntimeBetaEvalConfig(input_path=SEALED_HOLDOUT_PATH, model_path=INFONCE_MODEL_PATH, confidence=0.8)
    )

    assert report["schema_version"] == NIMBUS_RUNTIME_BETA_EVAL_SCHEMA_VERSION
    assert report["critic_kind"] == "learned_infonce_beta"
    assert report["runtime_adapter_present"] is True
    assert report["live_gateway_evidence"] is False
    assert report["paper_faithful_learned_critic"] is False
    assert report["promotion_status"] == "learned_runtime_beta_not_promotable"
    assert report["record_count"] == NIMBUS_REFERENCE_CONVERSATION_COUNT * NIMBUS_REFERENCE_TURNS_PER_CONVERSATION
    assert report["split_group_count"] == NIMBUS_REFERENCE_CONVERSATION_COUNT
    assert report["true_positive"] == 248
    assert report["true_negative"] == 418
    assert report["false_positive"] == 327
    assert report["false_negative"] == 7
    assert report["session_true_positive"] == 42
    assert report["session_true_negative"] == 8
    assert report["session_false_positive"] == 0
    assert report["session_false_negative"] == 0
    assert report["session_block_true_positive"] == 42
    assert report["session_block_true_negative"] == 8
    assert report["session_block_false_positive"] == 0
    assert report["session_block_false_negative"] == 0
    assert report["false_positive_rate"] is not None
    assert report["false_negative_rate"] is not None
    assert report["session_false_positive_rate"] is not None
    assert report["session_false_negative_rate"] is not None
    assert report["session_false_positive_rate"] == 0.0
    assert report["session_false_negative_rate"] == 0.0
    assert report["session_block_false_positive_rate"] == 0.0
    assert report["session_block_false_negative_rate"] == 0.0
    assert report["paper_conversation_metrics"]["detection_rate"] == 1.0
    assert report["paper_conversation_metrics"]["false_block_rate"] == 0.0
    assert report["paper_conversation_metrics"]["mean_first_block_turn_index"] is not None
    assert isinstance(report["threshold_sweep"], list)
    assert len(report["threshold_sweep"]) > 1
    assert report["threshold_sweep"][0]["threshold_bits"] == 0.0
    assert report["threshold_sweep"][0]["false_positive_rate"] == report["false_positive_rate"]
    assert report["threshold_sweep"][0]["false_negative_rate"] == report["false_negative_rate"]
    assert report["selected_operating_point"] is None
    assert report["operating_point_policy"] == {
        "max_false_positive_rate": 0.05,
        "max_false_negative_rate": 0.05,
        "requires_turn_and_session_rates": True,
    }
    assert "eval-registered candidate contexts" in " ".join(str(item) for item in report["limitations"])
    error_slices = report["error_slices"]
    assert isinstance(error_slices, list)
    labels = {
        str(slice_metric["slice_value"])
        for slice_metric in error_slices
        if slice_metric["slice_kind"] == "leakage_label"
    }
    assert {"benign", "direct", "encoded", "partial", "paraphrased", "tool_output", "delayed"} <= labels
    assert "safe-canary" not in json.dumps(report, sort_keys=True)
    assert "safe-decoy" not in json.dumps(report, sort_keys=True)


def test_nimbus_runtime_beta_eval_cli_writes_json(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "runtime-beta.json"
    monkeypatch.setattr(
        sys,
        "argv",
        (
            "aegis-nimbus-runtime-beta-eval",
            "--input",
            str(SEALED_HOLDOUT_PATH),
            "--model",
            str(INFONCE_MODEL_PATH),
            "--output",
            str(output_path),
        ),
    )

    main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == NIMBUS_RUNTIME_BETA_EVAL_SCHEMA_VERSION
    assert payload["runtime_adapter_present"] is True
