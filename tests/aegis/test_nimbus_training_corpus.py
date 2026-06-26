from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aegis.replay.nimbus_training import (
    INFO_NCE_NEGATIVE_COUNT,
    NIMBUS_ATTACK_TURN_MAX,
    NIMBUS_ATTACK_TURN_MIN,
    NIMBUS_REFERENCE_CONVERSATION_COUNT,
    NIMBUS_REFERENCE_TURNS_PER_CONVERSATION,
    NIMBUS_TRAINING_MANIFEST_SCHEMA_VERSION,
    NIMBUS_TRAINING_SCHEMA_VERSION,
    NimbusTrainingCorpusError,
    NimbusTrainingCorpusProfile,
    NimbusTrainingTurnRecord,
    generate_default_nimbus_training_records,
    generate_sealed_holdout_nimbus_training_records,
    main,
    nimbus_training_manifest,
    read_nimbus_training_records_jsonl,
    render_nimbus_training_manifest_json,
    validate_nimbus_training_record,
    write_nimbus_training_records_jsonl,
)


def test_default_nimbus_training_records_match_session_corpus_contract() -> None:
    records = generate_default_nimbus_training_records()

    assert len(records) == NIMBUS_REFERENCE_CONVERSATION_COUNT * NIMBUS_REFERENCE_TURNS_PER_CONVERSATION
    assert {record.scenario_name for record in records} == {
        "benign",
        "exact_canary_leak",
        "partial_drip",
        "encoded_leak",
        "paraphrased_leak",
        "tool_output_leak",
        "delayed_leak",
    }
    assert {record.leakage_label.value for record in records} == {
        "benign",
        "partial",
        "encoded",
        "direct",
        "paraphrased",
        "tool_output",
        "delayed",
    }
    assert all(record.schema_version == NIMBUS_TRAINING_SCHEMA_VERSION for record in records)
    assert all(record.split_group_key == record.session_id for record in records)
    assert all(record.info_nce.negative_count == INFO_NCE_NEGATIVE_COUNT for record in records)
    assert all(len(record.negative_secret_contexts) == INFO_NCE_NEGATIVE_COUNT for record in records)
    assert all(record.info_nce.positive_context_index == 0 for record in records)
    assert all(
        record.info_nce.candidate_context_ids
        == (
            record.true_secret_context.context_id,
            *(context.context_id for context in record.negative_secret_contexts),
        )
        for record in records
    )
    assert all(record.state_messages[-1].role == "user" for record in records)
    assert all(record.output_text not in {message.content for message in record.state_messages} for record in records)

    for session_id in {record.session_id for record in records}:
        session_records = tuple(record for record in records if record.session_id == session_id)
        attack_count = sum(record.leakage_label.value != "benign" for record in session_records)
        ordered_records = tuple(sorted(session_records, key=lambda record: record.turn_index))
        cumulative_bits = tuple(record.target_cumulative_leakage_bits for record in ordered_records)
        assert len(session_records) == NIMBUS_REFERENCE_TURNS_PER_CONVERSATION
        if attack_count > 0:
            assert NIMBUS_ATTACK_TURN_MIN <= attack_count <= NIMBUS_ATTACK_TURN_MAX
        assert cumulative_bits == tuple(sorted(cumulative_bits))


def test_sealed_holdout_nimbus_training_records_use_distinct_session_groups() -> None:
    calibration_records = generate_default_nimbus_training_records()
    sealed_records = generate_sealed_holdout_nimbus_training_records()
    manifest = nimbus_training_manifest(sealed_records)

    assert len(sealed_records) == len(calibration_records)
    assert manifest["corpus_profile"] == NimbusTrainingCorpusProfile.SEALED_HOLDOUT.value
    assert manifest["record_count"] == NIMBUS_REFERENCE_CONVERSATION_COUNT * NIMBUS_REFERENCE_TURNS_PER_CONVERSATION
    assert manifest["split_group_count"] == NIMBUS_REFERENCE_CONVERSATION_COUNT
    assert {record.leakage_label.value for record in sealed_records} == {
        "benign",
        "partial",
        "encoded",
        "direct",
        "paraphrased",
        "tool_output",
        "delayed",
    }
    assert {record.session_id for record in calibration_records}.isdisjoint(
        {record.session_id for record in sealed_records}
    )
    assert all(record.session_id.startswith("nimbus-sealed-") for record in sealed_records)
    assert all(record.split_group_key == record.session_id for record in sealed_records)


def test_nimbus_training_records_round_trip_as_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "nimbus-training.jsonl"
    records = generate_default_nimbus_training_records()

    write_nimbus_training_records_jsonl(output_path, records)

    raw_output = output_path.read_text(encoding="utf-8")
    loaded_records = read_nimbus_training_records_jsonl(output_path)

    assert loaded_records == records
    assert "{{CREDENTIAL:" not in raw_output
    assert "ghp_" not in raw_output
    assert "github_pat_" not in raw_output
    assert "sk_live_" not in raw_output
    assert "AKIA" not in raw_output


def test_nimbus_training_manifest_marks_scaffold_as_not_promotable() -> None:
    records = generate_default_nimbus_training_records()

    manifest = nimbus_training_manifest(records)
    quality_gates = {str(gate["name"]): gate["passed"] for gate in _manifest_quality_gates(manifest)}

    assert manifest["schema_version"] == NIMBUS_TRAINING_MANIFEST_SCHEMA_VERSION
    assert manifest["training_schema_version"] == NIMBUS_TRAINING_SCHEMA_VERSION
    assert manifest["corpus_profile"] == NimbusTrainingCorpusProfile.CALIBRATION.value
    assert manifest["critic_status"] == "training_corpus_scaffold"
    assert manifest["paper_faithful_learned_critic"] is False
    assert manifest["promotion_status"] == "not_promotable_training_contract_only"
    assert manifest["record_count"] == NIMBUS_REFERENCE_CONVERSATION_COUNT * NIMBUS_REFERENCE_TURNS_PER_CONVERSATION
    assert manifest["split_group_count"] == NIMBUS_REFERENCE_CONVERSATION_COUNT
    assert manifest["label_counts"] == {
        "benign": 745,
        "delayed": 45,
        "direct": 45,
        "encoded": 39,
        "paraphrased": 41,
        "partial": 42,
        "tool_output": 43,
    }
    assert quality_gates == {
        "expected_negative_context_count": True,
        "label_coverage": True,
        "scenario_family_coverage": True,
        "credential_shaped_material_absent": True,
        "cumulative_bits_monotonic_by_session": True,
        "paper_reference_session_count": True,
        "paper_reference_turn_count": True,
        "paper_reference_attack_turn_range": True,
        "non_singleton_leakage_families": True,
    }
    assert "grouped_cross_validation" in manifest["required_before_paper_faithful_promotion"]
    assert "sealed_holdout" in manifest["required_before_paper_faithful_promotion"]


def test_nimbus_training_cli_writes_jsonl_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "generated" / "nimbus-training.jsonl"
    manifest_path = tmp_path / "generated" / "nimbus-training-manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        (
            "aegis-nimbus-training-corpus",
            "--output",
            str(output_path),
            "--manifest-output",
            str(manifest_path),
        ),
    )

    main()

    loaded_records = read_nimbus_training_records_jsonl(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert manifest_path.exists()
    assert len(loaded_records) == NIMBUS_REFERENCE_CONVERSATION_COUNT * NIMBUS_REFERENCE_TURNS_PER_CONVERSATION
    assert manifest["promotion_status"] == "not_promotable_training_contract_only"
    assert manifest["corpus_profile"] == NimbusTrainingCorpusProfile.CALIBRATION.value


def test_nimbus_training_cli_writes_sealed_holdout_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "generated" / "nimbus-sealed.jsonl"
    manifest_path = tmp_path / "generated" / "nimbus-sealed-manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        (
            "aegis-nimbus-training-corpus",
            "--output",
            str(output_path),
            "--manifest-output",
            str(manifest_path),
            "--profile",
            "sealed_holdout",
        ),
    )

    main()

    loaded_records = read_nimbus_training_records_jsonl(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(loaded_records) == NIMBUS_REFERENCE_CONVERSATION_COUNT * NIMBUS_REFERENCE_TURNS_PER_CONVERSATION
    assert manifest["corpus_profile"] == NimbusTrainingCorpusProfile.SEALED_HOLDOUT.value
    assert all(record.session_id.startswith("nimbus-sealed-") for record in loaded_records)


def test_nimbus_training_record_validation_rejects_wrong_negative_count() -> None:
    record = generate_default_nimbus_training_records()[0]
    malformed_record = record.with_negative_secret_contexts(record.negative_secret_contexts[:-1])

    with pytest.raises(NimbusTrainingCorpusError, match="must have exactly 16 negative contexts"):
        validate_nimbus_training_record(malformed_record)


def test_nimbus_training_record_validation_rejects_wrong_group_key() -> None:
    record = generate_default_nimbus_training_records()[0]
    malformed_record = _replace_training_record(record, split_group_key="different-session")

    with pytest.raises(NimbusTrainingCorpusError, match="split_group_key"):
        validate_nimbus_training_record(malformed_record)


def test_render_nimbus_training_manifest_json_is_stable() -> None:
    rendered = render_nimbus_training_manifest_json(generate_default_nimbus_training_records())

    assert json.loads(rendered)["schema_version"] == NIMBUS_TRAINING_MANIFEST_SCHEMA_VERSION
    assert rendered.endswith("\n")


def _replace_training_record(record: NimbusTrainingTurnRecord, split_group_key: str) -> NimbusTrainingTurnRecord:
    return NimbusTrainingTurnRecord(
        schema_version=record.schema_version,
        example_id=record.example_id,
        scenario_name=record.scenario_name,
        session_id=record.session_id,
        split_group_key=split_group_key,
        turn_index=record.turn_index,
        state_messages=record.state_messages,
        output_text=record.output_text,
        true_secret_context=record.true_secret_context,
        negative_secret_contexts=record.negative_secret_contexts,
        info_nce=record.info_nce,
        leakage_label=record.leakage_label,
        leakage_transform=record.leakage_transform,
        target_turn_leakage_bits=record.target_turn_leakage_bits,
        target_cumulative_leakage_bits=record.target_cumulative_leakage_bits,
    )


def _manifest_quality_gates(manifest: dict[str, object]) -> tuple[dict[str, object], ...]:
    quality_gates = manifest["quality_gates"]
    if not isinstance(quality_gates, list):
        raise AssertionError("quality_gates must be a list.")
    return tuple(gate for gate in quality_gates if isinstance(gate, dict))
