from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aegis.replay.nimbus_training import (
    INFO_NCE_NEGATIVE_COUNT,
    NIMBUS_TRAINING_SCHEMA_VERSION,
    NimbusTrainingCorpusError,
    generate_default_nimbus_training_records,
    main,
    read_nimbus_training_records_jsonl,
    validate_nimbus_training_record,
    write_nimbus_training_records_jsonl,
)


def test_default_nimbus_training_records_match_infonce_contract() -> None:
    records = generate_default_nimbus_training_records()

    assert len(records) >= 8
    assert {record.scenario_name for record in records} == {"benign", "partial_drip", "encoded_leak", "direct_leak"}
    assert all(record.schema_version == NIMBUS_TRAINING_SCHEMA_VERSION for record in records)
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
        cumulative_bits = tuple(record.target_cumulative_leakage_bits for record in session_records)
        assert cumulative_bits == tuple(sorted(cumulative_bits))


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


def test_nimbus_training_cli_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "generated" / "nimbus-training.jsonl"
    monkeypatch.setattr(sys, "argv", ("aegis-nimbus-training-corpus", "--output", str(output_path)))

    main()

    loaded_records = read_nimbus_training_records_jsonl(output_path)

    assert output_path.exists()
    assert len(loaded_records) >= 8


def test_nimbus_training_record_validation_rejects_wrong_negative_count() -> None:
    record = generate_default_nimbus_training_records()[0]
    malformed_record = record.with_negative_secret_contexts(record.negative_secret_contexts[:-1])

    with pytest.raises(NimbusTrainingCorpusError, match="must have exactly 16 negative contexts"):
        validate_nimbus_training_record(malformed_record)
