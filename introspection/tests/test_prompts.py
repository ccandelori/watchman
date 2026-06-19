import json
import tempfile
import unittest
from pathlib import Path

from aegis_introspection.prompts import PromptDataError, load_prompt_examples, parse_prompt_example


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


if __name__ == "__main__":
    unittest.main()
