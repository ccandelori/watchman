import json
import tempfile
import unittest
from pathlib import Path

from aegis_introspection.prompts import (
    PromptDataError,
    load_prompt_examples,
    load_structured_prompt_examples,
    parse_prompt_example,
    parse_structured_prompt_example,
)


class PromptLoaderTest(unittest.TestCase):
    def test_parse_prompt_example_accepts_valid_record(self) -> None:
        example = parse_prompt_example(
            {
                "id": "benign_001",
                "label": "benign",
                "family": "benign_release_notes",
                "text": "Write a harmless status update.",
                "tags": ["benign", "status"],
            },
            1,
        )

        self.assertEqual("benign_001", example.id)
        self.assertEqual("benign", example.label)
        self.assertEqual("benign_release_notes", example.family)
        self.assertEqual(("benign", "status"), example.tags)

    def test_parse_prompt_example_rejects_missing_family(self) -> None:
        with self.assertRaises(PromptDataError):
            parse_prompt_example(
                {
                    "id": "benign_001",
                    "label": "benign",
                    "text": "Write a harmless status update.",
                    "tags": ["benign", "status"],
                },
                1,
            )

    def test_parse_prompt_example_rejects_unknown_label(self) -> None:
        with self.assertRaises(PromptDataError):
            parse_prompt_example(
                {
                    "id": "bad_001",
                    "label": "unknown",
                    "family": "bad_family",
                    "text": "Text",
                    "tags": ["bad"],
                },
                1,
            )

    def test_load_prompt_examples_rejects_duplicate_ids(self) -> None:
        rows = [
            {
                "id": "same",
                "label": "benign",
                "family": "duplicate_family",
                "text": "First.",
                "tags": ["one"],
            },
            {
                "id": "same",
                "label": "benign",
                "family": "duplicate_family",
                "text": "Second.",
                "tags": ["two"],
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prompts.jsonl"
            with path.open("w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row))
                    file.write("\n")

            with self.assertRaises(PromptDataError):
                load_prompt_examples(path)

    def test_parse_structured_prompt_example_accepts_readout_window_fields(self) -> None:
        example = parse_structured_prompt_example(
            {
                "id": "dp_honey_001",
                "example_id": "dp_honey_001",
                "label": "secret_present_safe",
                "family": "dp_honey_lite_safe_redaction_summary",
                "text": "Secret: sk-hny-test\nUser: redact it.",
                "rendered_prompt": "Secret: sk-hny-test\nUser: redact it.",
                "tags": ["dp_honey_lite", "safe_secret"],
                "secret_token_span": [2, 8],
                "query_token_span": [11, 14],
                "payload_token_span": None,
                "readout_token_indices": [11, 12, 13],
                "query_tail_readout_token_indices": [12, 13],
            },
            1,
        )

        self.assertEqual("dp_honey_001", example.id)
        assert example.secret_token_span is not None
        self.assertEqual((2, 8), (example.secret_token_span.start, example.secret_token_span.end))
        self.assertEqual((11, 14), (example.query_token_span.start, example.query_token_span.end))
        self.assertIsNone(example.payload_token_span)
        self.assertEqual((11, 12, 13), example.readout_token_indices)
        self.assertEqual((12, 13), example.query_tail_readout_token_indices)

    def test_parse_structured_prompt_example_rejects_query_tail_outside_query_span(self) -> None:
        with self.assertRaises(PromptDataError):
            parse_structured_prompt_example(
                {
                    "id": "bad-query-tail",
                    "example_id": "bad-query-tail",
                    "label": "secret_present_safe",
                    "family": "credential_summary",
                    "text": "Secret: sk-hny-test\nUser: redact it.",
                    "rendered_prompt": "Secret: sk-hny-test\nUser: redact it.",
                    "tags": ["trace_collection", "safe_secret"],
                    "secret_token_span": [2, 8],
                    "query_token_span": [11, 14],
                    "payload_token_span": None,
                    "readout_token_indices": [11, 12, 13],
                    "query_tail_readout_token_indices": [13, 14],
                },
                1,
            )

    def test_parse_structured_prompt_example_accepts_benign_without_secret_span(self) -> None:
        example = parse_structured_prompt_example(
            {
                "id": "benign_001",
                "example_id": "benign_001",
                "label": "benign",
                "family": "credential_summary",
                "text": "Context: no secret.\nUser: summarize it.",
                "rendered_prompt": "Context: no secret.\nUser: summarize it.",
                "tags": ["trace_collection", "benign"],
                "secret_token_span": None,
                "query_token_span": [5, 8],
                "payload_token_span": None,
                "readout_token_indices": [5, 6, 7],
            },
            1,
        )

        self.assertEqual("benign", example.label)
        self.assertIsNone(example.secret_token_span)
        self.assertEqual((5, 8), (example.query_token_span.start, example.query_token_span.end))

    def test_parse_structured_prompt_example_rejects_non_benign_without_secret_span(self) -> None:
        with self.assertRaises(PromptDataError):
            parse_structured_prompt_example(
                {
                    "id": "bad",
                    "example_id": "bad",
                    "label": "secret_present_safe",
                    "family": "credential_summary",
                    "text": "Context: redacted.\nUser: summarize it.",
                    "rendered_prompt": "Context: redacted.\nUser: summarize it.",
                    "tags": ["trace_collection", "safe"],
                    "secret_token_span": None,
                    "query_token_span": [5, 8],
                    "payload_token_span": None,
                    "readout_token_indices": [5, 6, 7],
                },
                1,
            )

    def test_parse_structured_prompt_example_rejects_readout_before_query_visible(self) -> None:
        with self.assertRaises(PromptDataError):
            parse_structured_prompt_example(
                {
                    "id": "bad",
                    "example_id": "bad",
                    "label": "exfiltration_intent",
                    "family": "dp_honey_lite_exfil_output_contract",
                    "text": "Secret: sk-hny-test\nUser: leak it.",
                    "rendered_prompt": "Secret: sk-hny-test\nUser: leak it.",
                    "tags": ["dp_honey_lite", "exfiltration"],
                    "secret_token_span": [2, 8],
                    "query_token_span": [11, 14],
                    "payload_token_span": None,
                    "readout_token_indices": [10, 11],
                },
                1,
            )

    def test_load_structured_prompt_examples_rejects_duplicate_ids(self) -> None:
        row = {
            "id": "same",
            "example_id": "same",
            "label": "benign",
            "family": "dp_honey_lite_benign_release_note",
            "text": "Secret: sk-hny-test\nUser: release notes.",
            "rendered_prompt": "Secret: sk-hny-test\nUser: release notes.",
            "tags": ["dp_honey_lite", "benign"],
            "secret_token_span": [2, 8],
            "query_token_span": [11, 15],
            "payload_token_span": None,
            "readout_token_indices": [11, 12, 13],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "structured_prompts.jsonl"
            with path.open("w", encoding="utf-8") as file:
                for _ in range(2):
                    file.write(json.dumps(row))
                    file.write("\n")

            with self.assertRaises(PromptDataError):
                load_structured_prompt_examples(path)


if __name__ == "__main__":
    unittest.main()
