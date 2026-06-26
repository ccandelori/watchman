from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from aegis.replay.nimbus_infonce import (
    NIMBUS_INFONCE_EVAL_SCHEMA_VERSION,
    NIMBUS_INFONCE_GROUPED_CV_SCHEMA_VERSION,
    NIMBUS_INFONCE_MODEL_SCHEMA_VERSION,
    NIMBUS_INFONCE_PROMOTION_STATUS,
    NimbusInfoNCEError,
    NimbusInfoNCEEvalConfig,
    NimbusInfoNCERunConfig,
    evaluate_nimbus_infonce_model,
    grouped_cross_validate_nimbus_infonce,
    load_nimbus_infonce_model,
    main_eval,
    main_train,
    render_nimbus_infonce_markdown,
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
    report = evaluate_nimbus_infonce_model(model, records, NimbusInfoNCEEvalConfig(allow_training_eval=True))
    bits_by_label = {
        label: tuple(metric.estimated_leakage_bits for metric in report.turn_metrics if metric.leakage_label == label)
        for label in {metric.leakage_label for metric in report.turn_metrics}
    }

    assert model.schema_version == NIMBUS_INFONCE_MODEL_SCHEMA_VERSION
    assert model.negative_count == 16
    assert model.training_record_count == 14
    assert model.training_split_group_count == 7
    assert model.feature_names == ("output_token_overlap", "decoded_output_token_overlap", "state_token_overlap")
    assert model.promotion_status == NIMBUS_INFONCE_PROMOTION_STATUS
    assert model.paper_faithful_learned_critic is False
    assert report.schema_version == NIMBUS_INFONCE_EVAL_SCHEMA_VERSION
    assert report.eval_corpus_sha256 == model.source_corpus_sha256
    assert report.training_eval_reused is True
    assert report.training_eval_allowed is True
    assert report.promotion_status == NIMBUS_INFONCE_PROMOTION_STATUS
    assert report.paper_faithful_learned_critic is False
    assert report.true_positive + report.true_negative + report.false_positive + report.false_negative == len(records)
    assert report.false_positive_rate is not None
    assert report.false_negative_rate is not None
    assert report.mean_absolute_error_bits > 0.0
    assert max(bits_by_label["benign"]) == 0.0
    assert bits_by_label.keys() == {
        "benign",
        "delayed",
        "direct",
        "encoded",
        "paraphrased",
        "partial",
        "tool_output",
    }
    assert max(bits_by_label["partial"]) > max(bits_by_label["benign"])
    assert max(bits_by_label["encoded"]) > max(bits_by_label["benign"])
    assert max(bits_by_label["direct"]) > max(bits_by_label["benign"])
    assert max(bits_by_label["tool_output"]) > max(bits_by_label["benign"])


def test_nimbus_infonce_eval_rejects_training_corpus_without_explicit_allowance() -> None:
    records = generate_default_nimbus_training_records()
    model = train_nimbus_infonce_model(records, NimbusInfoNCERunConfig(max_weight=4, weight_step=1))

    with pytest.raises(NimbusInfoNCEError, match=r"evaluation corpus matches model\.source_corpus_sha256"):
        evaluate_nimbus_infonce_model(model, records, NimbusInfoNCEEvalConfig(allow_training_eval=False))


def test_nimbus_infonce_grouped_cv_reports_heldout_fn_fp_separately() -> None:
    records = generate_default_nimbus_training_records()

    report = grouped_cross_validate_nimbus_infonce(records, NimbusInfoNCERunConfig(max_weight=4, weight_step=1))

    assert report.schema_version == NIMBUS_INFONCE_GROUPED_CV_SCHEMA_VERSION
    assert report.record_count == len(records)
    assert report.split_group_count == 7
    assert report.fold_count == 7
    assert report.promotion_status == NIMBUS_INFONCE_PROMOTION_STATUS
    assert report.paper_faithful_learned_critic is False
    assert report.true_positive + report.true_negative + report.false_positive + report.false_negative == len(records)
    assert report.false_positive_rate is not None
    assert report.false_negative_rate is not None
    assert all(metric.training_split_group_count == 6 for metric in report.fold_metrics)
    assert all(metric.eval_record_count >= 1 for metric in report.fold_metrics)


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
    assert "paper_faithful_learned_critic" in raw_artifact
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
            "--allow-training-eval",
        ),
    )
    main_eval()

    model = load_nimbus_infonce_model(model_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert model.training_record_count == 14
    assert report["schema_version"] == NIMBUS_INFONCE_EVAL_SCHEMA_VERSION
    assert report["eval_corpus_sha256"] == model.source_corpus_sha256
    assert report["training_eval_reused"] is True
    assert report["training_eval_allowed"] is True
    assert report["promotion_status"] == NIMBUS_INFONCE_PROMOTION_STATUS
    assert report["paper_faithful_learned_critic"] is False
    assert {metric["leakage_label"] for metric in report["label_metrics"]} == {
        "benign",
        "delayed",
        "direct",
        "encoded",
        "paraphrased",
        "partial",
        "tool_output",
    }


def test_nimbus_infonce_eval_cli_writes_grouped_cv_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    model_path = tmp_path / "model.json"
    report_path = tmp_path / "report.json"
    grouped_cv_path = tmp_path / "grouped-cv.json"
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
            "--allow-training-eval",
            "--grouped-cv-output",
            str(grouped_cv_path),
        ),
    )
    main_eval()

    grouped_cv = json.loads(grouped_cv_path.read_text(encoding="utf-8"))

    assert grouped_cv["schema_version"] == NIMBUS_INFONCE_GROUPED_CV_SCHEMA_VERSION
    assert grouped_cv["fold_count"] == 7
    assert grouped_cv["promotion_status"] == NIMBUS_INFONCE_PROMOTION_STATUS


def test_nimbus_infonce_eval_cli_writes_markdown_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    model_path = tmp_path / "model.json"
    report_path = tmp_path / "report.md"
    records = generate_default_nimbus_training_records()
    write_nimbus_training_records_jsonl(corpus_path, records)
    model = train_nimbus_infonce_model(records, NimbusInfoNCERunConfig(max_weight=4, weight_step=1))
    save_nimbus_infonce_model(model_path, model)

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
            "--format",
            "markdown",
            "--allow-training-eval",
        ),
    )
    main_eval()

    markdown = report_path.read_text(encoding="utf-8")

    assert "# NIMBUS InfoNCE Evaluation" in markdown
    assert "| partial | 4 |" in markdown
    assert f"Promotion status: `{NIMBUS_INFONCE_PROMOTION_STATUS}`" in markdown
    assert "safe-canary-repo-alpha-7294" not in markdown
    assert "safe-decoy-marker" not in markdown


def test_render_nimbus_infonce_markdown_is_stable() -> None:
    records = generate_default_nimbus_training_records()
    model = train_nimbus_infonce_model(records, NimbusInfoNCERunConfig(max_weight=4, weight_step=1))
    report = evaluate_nimbus_infonce_model(model, records, NimbusInfoNCEEvalConfig(allow_training_eval=True))

    rendered = render_nimbus_infonce_markdown(report)

    assert rendered.endswith("\n")
    assert "False positive rate" in rendered
    assert "False negative rate" in rendered
    assert "Training eval reused" in rendered


def test_nimbus_infonce_train_rejects_malformed_in_memory_record() -> None:
    malformed_record = replace(generate_default_nimbus_training_records()[0], schema_version="wrong-schema")

    with pytest.raises(NimbusInfoNCEError, match="schema_version"):
        train_nimbus_infonce_model((malformed_record,), NimbusInfoNCERunConfig(max_weight=4, weight_step=1))


def test_nimbus_infonce_eval_rejects_credential_shaped_public_identifiers() -> None:
    records = generate_default_nimbus_training_records()
    model = train_nimbus_infonce_model(records, NimbusInfoNCERunConfig(max_weight=4, weight_step=1))
    unsafe_records = (replace(records[0], example_id="github_pat_raw_identifier"), *records[1:])

    with pytest.raises(NimbusInfoNCEError, match="credential-shaped"):
        evaluate_nimbus_infonce_model(model, unsafe_records, NimbusInfoNCEEvalConfig(allow_training_eval=False))


def test_nimbus_infonce_train_rejects_duplicate_example_ids() -> None:
    records = generate_default_nimbus_training_records()
    duplicate_records = (records[0], records[0], *records[1:])

    with pytest.raises(NimbusInfoNCEError, match="duplicate example_id"):
        train_nimbus_infonce_model(duplicate_records, NimbusInfoNCERunConfig(max_weight=4, weight_step=1))


def test_nimbus_infonce_train_rejects_corpora_without_leakage_examples() -> None:
    benign_records = tuple(
        record for record in generate_default_nimbus_training_records() if record.leakage_label.value == "benign"
    )

    with pytest.raises(NimbusInfoNCEError, match="non-benign"):
        train_nimbus_infonce_model(benign_records, NimbusInfoNCERunConfig(max_weight=4, weight_step=1))
