from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aegis_introspection.cift_selected_choice_sealed_holdout import (
    SelectedChoiceSealedHoldoutConfig,
    SelectedChoiceSealedHoldoutError,
    materialize_selected_choice_sealed_holdout,
)
from aegis_introspection.runtime_requests import load_runtime_requests_jsonl
from aegis_introspection.trace_record_adapter import TokenOffset

from aegis.core.contracts import CapabilityMode


class CharacterOffsetEncoder:
    def encode_offsets(self, text: str) -> tuple[TokenOffset, ...]:
        return tuple(TokenOffset(start=index, end=index + 1) for index in range(len(text)))


class SelectedChoiceSealedHoldoutTest(unittest.TestCase):
    def test_materializes_fresh_v5_selected_choice_sealed_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root=root, overwrite=False)

            result = materialize_selected_choice_sealed_holdout(
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
            "aegis_introspection.selected_choice_sealed_holdout_manifest/v1",
            result.manifest["schema_version"],
        )
        self.assertEqual("synthetic-qwen3-4b-selected-choice-v5-sealed", result.manifest["corpus_id"])
        self.assertTrue(result.manifest["sealed_holdout"])
        self.assertEqual("paired_semantic_indirection_v5_sealed", result.manifest["profile"])
        self.assertEqual("paired_semantic_indirection_v5", result.manifest["source_profile"])
        self.assertEqual({"exfiltration_intent": 12, "secret_present_safe": 12}, result.manifest["label_counts"])

        evaluated_v4_tokens = {
            "anchor",
            "beacon",
            "column",
            "forest",
            "island",
            "kernel",
            "ladder",
            "needle",
            "orbit",
            "parcel",
            "ribbon",
            "temple",
            "unit",
            "vector",
            "amber",
            "binary",
            "canyon",
            "engine",
            "fabric",
            "garden",
        }
        structured_text = " ".join(str(row["text"]) for row in structured_rows)
        self.assertTrue(evaluated_v4_tokens.isdisjoint(set(structured_text.split())))
        for row in structured_rows:
            tags = row["tags"]
            self.assertIsInstance(tags, list)
            self.assertIn("sealed_holdout", tags)
            self.assertIn("profile:paired_semantic_indirection_v5_sealed", tags)
            self.assertIn("source_profile:paired_semantic_indirection_v5", tags)
            self.assertIsNotNone(row["selected_choice_char_span"])
            self.assertIsNotNone(row["selected_choice_token_span"])
            self.assertIsNotNone(row["selected_choice_readout_token_indices"])
            self.assertIsNone(row["fallback_reason"])

        for row in runtime_rows:
            metadata = row["metadata"]
            self.assertIsInstance(metadata, dict)
            eval_metadata = metadata["eval"]
            self.assertIsInstance(eval_metadata, dict)
            self.assertIn("sealed_holdout", eval_metadata["tags"])
            self.assertIn("profile:paired_semantic_indirection_v5_sealed", eval_metadata["tags"])
            self.assertIn("source_profile:paired_semantic_indirection_v5", eval_metadata["tags"])
            self.assertEqual("selected_choice", eval_metadata["expected_cift_window_family"])
            self.assertEqual("qwen3-4b/v5-selected-choice-sealed", eval_metadata["sealed_holdout_split_id"])
            cift_metadata = metadata["cift"]
            self.assertIsInstance(cift_metadata, dict)
            self.assertIsNotNone(cift_metadata["selected_choice_char_span"])
            self.assertIsNotNone(cift_metadata["selected_choice_token_span"])
            self.assertIsNotNone(cift_metadata["selected_choice_readout_token_indices"])

    def test_refuses_to_overwrite_existing_outputs_without_explicit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root=root, overwrite=False)
            config.structured_prompts_path.write_text("existing\n", encoding="utf-8")

            with self.assertRaisesRegex(SelectedChoiceSealedHoldoutError, "already exists"):
                materialize_selected_choice_sealed_holdout(
                    config=config,
                    encoder=CharacterOffsetEncoder(),
                )

    def test_materializes_fresh_v8_selected_choice_sealed_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(
                _config(root=root, overwrite=False),
                corpus_id="synthetic-qwen3-4b-selected-choice-v8-sealed",
                sealed_holdout_split_id="qwen3-4b/v8-selected-choice-sealed",
                source_profile="paired_semantic_indirection_v8",
                session_id="qwen3-4b-v8-selected-choice-sealed-test",
                sensitive_source="watchman_semantic_v8_selected_choice_sealed",
            )

            result = materialize_selected_choice_sealed_holdout(
                config=config,
                encoder=CharacterOffsetEncoder(),
            )
            structured_rows = _load_jsonl(config.structured_prompts_path)
            runtime_rows = _load_jsonl(config.runtime_turns_path)

        self.assertEqual(36, result.source_trace_record_count)
        self.assertEqual(24, result.selected_choice_record_count)
        self.assertEqual(24, result.runtime_turn_count)
        self.assertEqual("synthetic-qwen3-4b-selected-choice-v8-sealed", result.manifest["corpus_id"])
        self.assertEqual("paired_semantic_indirection_v8_sealed", result.manifest["profile"])
        self.assertEqual("paired_semantic_indirection_v8", result.manifest["source_profile"])
        self.assertEqual({"exfiltration_intent": 12, "secret_present_safe": 12}, result.manifest["label_counts"])

        structured_text = " ".join(str(row["text"]) for row in structured_rows)
        self.assertIn("Open door ambril", structured_text)
        self.assertIn("Open door borven", structured_text)
        for row in structured_rows:
            tags = row["tags"]
            self.assertIsInstance(tags, list)
            self.assertIn("sealed_holdout", tags)
            self.assertIn("profile:paired_semantic_indirection_v8_sealed", tags)
            self.assertIn("source_profile:paired_semantic_indirection_v8", tags)
            self.assertIsNotNone(row["selected_choice_char_span"])
            self.assertIsNotNone(row["selected_choice_token_span"])
            self.assertIsNotNone(row["selected_choice_readout_token_indices"])
            self.assertIsNone(row["fallback_reason"])

        for row in runtime_rows:
            metadata = row["metadata"]
            self.assertIsInstance(metadata, dict)
            eval_metadata = metadata["eval"]
            self.assertIsInstance(eval_metadata, dict)
            self.assertIn("sealed_holdout", eval_metadata["tags"])
            self.assertIn("profile:paired_semantic_indirection_v8_sealed", eval_metadata["tags"])
            self.assertIn("source_profile:paired_semantic_indirection_v8", eval_metadata["tags"])
            self.assertEqual("selected_choice", eval_metadata["expected_cift_window_family"])
            self.assertEqual("qwen3-4b/v8-selected-choice-sealed", eval_metadata["sealed_holdout_split_id"])


def _config(root: Path, overwrite: bool) -> SelectedChoiceSealedHoldoutConfig:
    return SelectedChoiceSealedHoldoutConfig(
        trace_records_path=root / "trace_records.jsonl",
        structured_prompts_path=root / "structured_prompts.jsonl",
        runtime_turns_path=root / "runtime_turns.jsonl",
        manifest_path=root / "manifest.json",
        corpus_id="synthetic-qwen3-4b-selected-choice-v5-sealed",
        sealed_holdout_split_id="qwen3-4b/v5-selected-choice-sealed",
        source_profile="paired_semantic_indirection_v5",
        participant_ids=("qwen3-4b-sealed-test",),
        variants_per_label=1,
        model_provider="huggingface",
        model_id="Qwen/Qwen3-4B",
        revision="main",
        selected_device="mps",
        session_id="qwen3-4b-v5-selected-choice-sealed-test",
        sensitive_source="watchman_semantic_v5_selected_choice_sealed",
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
