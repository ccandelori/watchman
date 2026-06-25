from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aegis_introspection.cift_selected_choice_calibration import (
    SelectedChoiceCalibrationConfig,
    SelectedChoiceCalibrationError,
    materialize_selected_choice_calibration,
)
from aegis_introspection.runtime_requests import load_runtime_requests_jsonl
from aegis_introspection.trace_record_adapter import TokenOffset

from aegis.core.contracts import CapabilityMode


class CharacterOffsetEncoder:
    def encode_offsets(self, text: str) -> tuple[TokenOffset, ...]:
        return tuple(TokenOffset(start=index, end=index + 1) for index in range(len(text)))


class SelectedChoiceCalibrationTest(unittest.TestCase):
    def test_materializes_v6_selected_choice_calibration_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root=root, overwrite=False)

            result = materialize_selected_choice_calibration(
                config=config,
                encoder=CharacterOffsetEncoder(),
            )
            structured_rows = _load_jsonl(config.structured_prompts_path)
            runtime_rows = _load_jsonl(config.runtime_turns_path)
            runtime_requests = load_runtime_requests_jsonl(config.runtime_turns_path)

        self.assertEqual(36, result.source_trace_record_count)
        self.assertEqual(24, result.selected_choice_record_count)
        self.assertEqual(24, result.runtime_turn_count)
        self.assertEqual(24, len(structured_rows))
        self.assertEqual(24, len(runtime_rows))
        self.assertEqual(24, len(runtime_requests))
        self.assertEqual(
            "aegis_introspection.selected_choice_calibration_manifest/v1",
            result.manifest["schema_version"],
        )
        self.assertEqual("synthetic-qwen3-4b-selected-choice-v6-calibration", result.manifest["corpus_id"])
        self.assertFalse(result.manifest["sealed_holdout"])
        self.assertEqual("paired_semantic_indirection_v6", result.manifest["source_profile"])
        self.assertEqual("paired_semantic_indirection_v6_calibration", result.manifest["profile"])
        self.assertEqual({"exfiltration_intent": 12, "secret_present_safe": 12}, result.manifest["label_counts"])

        sealed_v5_tokens = {
            "apricot",
            "bastion",
            "citadel",
            "dovetail",
            "equinox",
            "ferrule",
            "granite",
            "halyard",
            "inkwell",
            "javelin",
            "kiln",
            "monsoon",
            "nebula",
            "oxbow",
            "pinnacle",
            "quorum",
            "rivet",
            "saffron",
            "topaz",
            "umbra",
        }
        structured_text = " ".join(str(row["text"]) for row in structured_rows)
        self.assertTrue(sealed_v5_tokens.isdisjoint(set(structured_text.split())))
        for row in structured_rows:
            tags = row["tags"]
            self.assertIsInstance(tags, list)
            self.assertNotIn("sealed_holdout", tags)
            self.assertIn("calibration", tags)
            self.assertIn("profile:paired_semantic_indirection_v6_calibration", tags)
            self.assertIn("source_profile:paired_semantic_indirection_v6", tags)
            self.assertIsNotNone(row["selected_choice_char_span"])
            self.assertIsNotNone(row["selected_choice_token_span"])
            self.assertIsNotNone(row["selected_choice_readout_token_indices"])
            self.assertIsNone(row["fallback_reason"])

        for row in runtime_rows:
            metadata = row["metadata"]
            self.assertIsInstance(metadata, dict)
            eval_metadata = metadata["eval"]
            self.assertIsInstance(eval_metadata, dict)
            self.assertNotIn("sealed_holdout", eval_metadata["tags"])
            self.assertIn("profile:paired_semantic_indirection_v6_calibration", eval_metadata["tags"])
            self.assertIn("source_profile:paired_semantic_indirection_v6", eval_metadata["tags"])
            self.assertEqual("selected_choice", eval_metadata["expected_cift_window_family"])
            self.assertEqual("qwen3-4b/v6-selected-choice-calibration", eval_metadata["calibration_split_id"])
            cift_metadata = metadata["cift"]
            self.assertIsInstance(cift_metadata, dict)
            self.assertIsNotNone(cift_metadata["selected_choice_char_span"])
            self.assertIsNotNone(cift_metadata["selected_choice_token_span"])
            self.assertIsNotNone(cift_metadata["selected_choice_readout_token_indices"])

    def test_materializes_v7_selected_choice_calibration_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(
                _config(root=root, overwrite=False),
                corpus_id="synthetic-qwen3-4b-selected-choice-v7-calibration",
                calibration_split_id="qwen3-4b/v7-selected-choice-calibration",
                source_profile="paired_semantic_indirection_v7",
                session_id="qwen3-4b-v7-selected-choice-calibration-test",
                sensitive_source="watchman_semantic_v7_selected_choice_calibration",
            )

            result = materialize_selected_choice_calibration(
                config=config,
                encoder=CharacterOffsetEncoder(),
            )
            structured_rows = _load_jsonl(config.structured_prompts_path)

        self.assertEqual(36, result.source_trace_record_count)
        self.assertEqual(24, result.selected_choice_record_count)
        self.assertEqual("paired_semantic_indirection_v7", result.manifest["source_profile"])
        self.assertEqual("paired_semantic_indirection_v7_calibration", result.manifest["profile"])
        self.assertEqual({"exfiltration_intent": 12, "secret_present_safe": 12}, result.manifest["label_counts"])
        structured_text = " ".join(str(row["text"]) for row in structured_rows)
        self.assertIn("Open gate aurora", structured_text)
        self.assertIn("Open gate brisket", structured_text)
        self.assertIn("profile:paired_semantic_indirection_v7_calibration", str(structured_rows))
        self.assertIn("source_profile:paired_semantic_indirection_v7", str(structured_rows))

    def test_refuses_sealed_split_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(root=Path(directory), overwrite=False)
            sealed_config = SelectedChoiceCalibrationConfig(
                trace_records_path=config.trace_records_path,
                structured_prompts_path=config.structured_prompts_path,
                runtime_turns_path=config.runtime_turns_path,
                manifest_path=config.manifest_path,
                corpus_id=config.corpus_id,
                calibration_split_id="qwen3-4b/v6-selected-choice-sealed",
                source_profile=config.source_profile,
                participant_ids=config.participant_ids,
                variants_per_label=config.variants_per_label,
                model_provider=config.model_provider,
                model_id=config.model_id,
                revision=config.revision,
                selected_device=config.selected_device,
                session_id=config.session_id,
                sensitive_source=config.sensitive_source,
                readout_token_count=config.readout_token_count,
                capability_mode=config.capability_mode,
                created_at=config.created_at,
                overwrite=config.overwrite,
            )

            with self.assertRaisesRegex(SelectedChoiceCalibrationError, "must not identify a sealed split"):
                materialize_selected_choice_calibration(
                    config=sealed_config,
                    encoder=CharacterOffsetEncoder(),
                )

    def test_refuses_to_overwrite_existing_outputs_without_explicit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root=root, overwrite=False)
            config.structured_prompts_path.write_text("existing\n", encoding="utf-8")

            with self.assertRaisesRegex(SelectedChoiceCalibrationError, "already exists"):
                materialize_selected_choice_calibration(
                    config=config,
                    encoder=CharacterOffsetEncoder(),
                )


def _config(root: Path, overwrite: bool) -> SelectedChoiceCalibrationConfig:
    return SelectedChoiceCalibrationConfig(
        trace_records_path=root / "trace_records.jsonl",
        structured_prompts_path=root / "structured_prompts.jsonl",
        runtime_turns_path=root / "runtime_turns.jsonl",
        manifest_path=root / "manifest.json",
        corpus_id="synthetic-qwen3-4b-selected-choice-v6-calibration",
        calibration_split_id="qwen3-4b/v6-selected-choice-calibration",
        source_profile="paired_semantic_indirection_v6",
        participant_ids=("qwen3-4b-calibration-test",),
        variants_per_label=1,
        model_provider="huggingface",
        model_id="Qwen/Qwen3-4B",
        revision="main",
        selected_device="mps",
        session_id="qwen3-4b-v6-selected-choice-calibration-test",
        sensitive_source="watchman_semantic_v6_selected_choice_calibration",
        readout_token_count=4,
        capability_mode=CapabilityMode.OFFLINE_EVAL,
        created_at="2026-06-24T00:00:00Z",
        overwrite=overwrite,
    )


def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip() == "":
            continue
        decoded = json.loads(raw_line)
        if not isinstance(decoded, dict):
            raise AssertionError("expected JSON object rows")
        rows.append(decoded)
    return tuple(rows)


if __name__ == "__main__":
    unittest.main()
