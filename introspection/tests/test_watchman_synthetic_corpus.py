from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aegis_introspection.grok_redteam_corpus import GrokRedteamCorpusConfig, ingest_grok_redteam_corpus
from aegis_introspection.watchman_synthetic_corpus import (
    WatchmanSyntheticCorpusConfig,
    WatchmanSyntheticCorpusError,
    materialize_watchman_synthetic_corpus,
)


class WatchmanSyntheticCorpusTest(unittest.TestCase):
    def test_generates_crossed_source_that_passes_strict_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_source_dir = root / "seed"
            generated_source_dir = root / "generated"
            _write_seed_shard(
                source_dir=seed_source_dir,
                records=(
                    _seed_record(
                        family="repository_access",
                        credential_type="github_pat",
                        credential_slot="repository_access_slot",
                    ),
                    _seed_record(
                        family="cloud_infra",
                        credential_type="aws_access_key",
                        credential_slot="cloud_infra_slot",
                    ),
                    _seed_record(
                        family="ci_cd",
                        credential_type="generic_api_key",
                        credential_slot="ci_cd_slot",
                    ),
                ),
            )
            generate_config = WatchmanSyntheticCorpusConfig(
                seed_source_dir=seed_source_dir,
                output_dir=generated_source_dir,
                manifest_output_path=root / "generated_manifest.json",
                corpus_id="watchman-synthetic-v11-test",
                created_at="2026-06-25T00:00:00Z",
                shard_count=1,
                records_per_shard=12,
                family_record_count=4,
                overwrite=False,
            )

            generate_result = materialize_watchman_synthetic_corpus(generate_config)
            ingest_config = GrokRedteamCorpusConfig(
                source_dir=generated_source_dir,
                normalized_output_path=root / "normalized.jsonl",
                calibration_output_path=root / "calibration.jsonl",
                sealed_holdout_output_path=root / "sealed.jsonl",
                manifest_output_path=root / "ingest_manifest.json",
                corpus_id="watchman-synthetic-v11-test",
                calibration_split_id="watchman-synthetic-v11-test/calibration",
                sealed_holdout_split_id="watchman-synthetic-v11-test/sealed",
                created_at="2026-06-25T00:00:00Z",
                expected_shard_count=1,
                expected_records_per_shard=12,
                expected_label_counts={
                    "secret_present_safe": 4,
                    "exfiltration_intent": 4,
                    "benign": 4,
                },
                expected_family_record_count=4,
                hard_near_neighbor_min_rate=0.1,
                tool_payload_min_rate=0.1,
                multi_turn_min_rate=0.1,
                min_unique_message_ratio=0.8,
                sealed_fraction=0.5,
                require_family_label_crossing=True,
                normalization_mode="selected_choice_ledger",
                allow_quarantine_output=False,
                overwrite=False,
            )
            ingest_result = ingest_grok_redteam_corpus(ingest_config)
            normalized_rows = _read_jsonl(ingest_config.normalized_output_path)

        self.assertEqual(12, generate_result.record_count)
        self.assertEqual(
            {"secret_present_safe": 4, "exfiltration_intent": 4, "benign": 4},
            generate_result.manifest["label_counts"],
        )
        self.assertEqual(1.0, generate_result.manifest["unique_message_ratio"])
        self.assertTrue(ingest_result.manifest["training_eligible"])
        family_label_counts = ingest_result.manifest["family_label_counts"]
        self.assertIsInstance(family_label_counts, dict)
        for family_counts in family_label_counts.values():
            self.assertEqual({"benign", "exfiltration_intent", "secret_present_safe"}, set(family_counts))
        exfil_row = next(row for row in normalized_rows if row["label"] == "exfiltration_intent")
        self.assertEqual(2, len(exfil_row["normalized_turn"]["sensitive_spans"]))

    def test_rejects_unknown_seed_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_source_dir = root / "seed"
            _write_seed_shard(
                source_dir=seed_source_dir,
                records=(
                    _seed_record(
                        family="unmapped_family",
                        credential_type="generic_api_key",
                        credential_slot="unmapped_slot",
                    ),
                ),
            )
            config = WatchmanSyntheticCorpusConfig(
                seed_source_dir=seed_source_dir,
                output_dir=root / "generated",
                manifest_output_path=root / "manifest.json",
                corpus_id="watchman-synthetic-v11-test",
                created_at="2026-06-25T00:00:00Z",
                shard_count=1,
                records_per_shard=3,
                family_record_count=3,
                overwrite=False,
            )

            with self.assertRaisesRegex(WatchmanSyntheticCorpusError, "unsupported families"):
                materialize_watchman_synthetic_corpus(config)


def _seed_record(family: str, credential_type: str, credential_slot: str) -> dict[str, object]:
    return {
        "schema_version": "trace_collection/v1",
        "batch_id": "seed",
        "record_id": f"seed-{family}",
        "family": family,
        "label": "benign",
        "task_id": f"seed-{family}",
        "variant_id": "seed",
        "credential_type": credential_type,
        "credential_slot": credential_slot,
    }


def _write_seed_shard(source_dir: Path, records: tuple[dict[str, object], ...]) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "shard_001.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True))
            output.write("\n")


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        decoded = json.loads(line)
        if not isinstance(decoded, dict):
            raise AssertionError("expected JSON object")
        rows.append(decoded)
    return tuple(rows)


if __name__ == "__main__":
    unittest.main()
