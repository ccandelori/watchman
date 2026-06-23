from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from aegis.replay.nimbus_infonce import (
    NIMBUS_INFONCE_MODEL_SCHEMA_VERSION,
    NimbusInfoNCEError,
    NimbusInfoNCERunConfig,
    evaluate_nimbus_infonce_model,
    load_nimbus_infonce_model,
    main_eval,
    main_train,
    save_nimbus_infonce_model,
    train_nimbus_infonce_model,
)
from aegis.replay.nimbus_training import (
    generate_default_nimbus_training_records,
    write_nimbus_training_records_jsonl,
)


def test_trained_nimbus_infonce_model_scores_leaks_above_benign() -> None:
    records = generate_default_nimbus_training_records()
    model = train_nimbus_infonce_model(records, NimbusInfoNCERunConfig(max_weight=4, weight_step=1))
    report = evaluate_nimbus_infonce_model(model, records)
    bits_by_label = {
        label: tuple(metric.estimated_leakage_bits for metric in report.turn_metrics if metric.leakage_label == label)
        for label in {metric.leakage_label for metric in report.turn_metrics}
    }

    assert model.schema_version == NIMBUS_INFONCE_MODEL_SCHEMA_VERSION
    assert model.negative_count == 16
    assert model.feature_names == ("output_token_overlap", "decoded_output_token_overlap", "state_token_overlap")
    assert report.attack_top1_accuracy == pytest.approx(4 / 6)
    assert report.mean_absolute_error_bits > 0.0
    assert max(bits_by_label["benign"]) == 0.0
    assert max(bits_by_label["partial"]) > max(bits_by_label["benign"])
    assert min(bits_by_label["encoded"]) > max(bits_by_label["benign"])
    assert min(bits_by_label["direct"]) > max(bits_by_label["partial"])
    label_metrics = {metric.leakage_label: metric for metric in report.label_metrics}
    assert label_metrics["benign"].count == 2
    assert label_metrics["benign"].mean_absolute_error_bits == 0.0
    assert label_metrics["partial"].count == 4
    assert label_metrics["encoded"].mean_target_turn_leakage_bits == pytest.approx(1.2)
    assert label_metrics["direct"].mean_target_turn_leakage_bits == pytest.approx(2.0)


def test_nimbus_infonce_model_artifact_round_trips_without_raw_contexts(tmp_path: Path) -> None:
    output_path = tmp_path / "nimbus-infonce-model.json"
    records = generate_default_nimbus_training_records()
    model = train_nimbus_infonce_model(records, NimbusInfoNCERunConfig(max_weight=4, weight_step=1))

    save_nimbus_infonce_model(output_path, model)

    raw_artifact = output_path.read_text(encoding="utf-8")
    loaded_model = load_nimbus_infonce_model(output_path)

    assert loaded_model == model
    assert "safe-canary-repo-alpha-7294" not in raw_artifact
    assert "safe-decoy-marker" not in raw_artifact
    assert "{{CREDENTIAL:" not in raw_artifact
    json.loads(raw_artifact)


def test_nimbus_infonce_train_and_eval_clis_write_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    model_path = tmp_path / "model.json"
    report_path = tmp_path / "report.json"
    write_nimbus_training_records_jsonl(corpus_path, generate_default_nimbus_training_records())

    monkeypatch.setattr(
        sys,
        "argv",
        ("aegis-nimbus-train-infonce", "--input", str(corpus_path), "--output", str(model_path)),
    )
    main_train()
    monkeypatch.setattr(
        sys,
        "argv",
        (
            "aegis-nimbus-eval-infonce",
            "--input",
            str(corpus_path),
            "--model",
            str(model_path),
            "--output",
            str(report_path),
        ),
    )
    main_eval()

    model = load_nimbus_infonce_model(model_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert model.training_record_count == 8
    assert report["schema_version"] == "nimbus-infonce-eval/v0"
    assert report["attack_top1_accuracy"] == pytest.approx(4 / 6)
    assert report["mean_absolute_error_bits"] > 0.0
    assert {metric["leakage_label"] for metric in report["label_metrics"]} == {
        "benign",
        "direct",
        "encoded",
        "partial",
    }


def test_nimbus_infonce_train_rejects_malformed_in_memory_record() -> None:
    malformed_record = replace(generate_default_nimbus_training_records()[0], schema_version="wrong-schema")

    with pytest.raises(NimbusInfoNCEError, match="schema_version"):
        train_nimbus_infonce_model((malformed_record,), NimbusInfoNCERunConfig(max_weight=4, weight_step=1))


def test_nimbus_infonce_eval_rejects_credential_shaped_public_identifiers() -> None:
    records = generate_default_nimbus_training_records()
    model = train_nimbus_infonce_model(records, NimbusInfoNCERunConfig(max_weight=4, weight_step=1))
    unsafe_records = (replace(records[0], example_id="github_pat_raw_identifier"), *records[1:])

    with pytest.raises(NimbusInfoNCEError, match="credential-shaped"):
        evaluate_nimbus_infonce_model(model, unsafe_records)
