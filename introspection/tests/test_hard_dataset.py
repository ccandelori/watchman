import unittest
from collections import Counter, defaultdict
from pathlib import Path

from aegis_introspection.prompts import load_prompt_examples


HARD_PROMPTS_PATH = Path(__file__).resolve().parents[1] / "data" / "prompts_hard.jsonl"
EXPECTED_LABELS = ("benign", "secret_present_safe", "exfiltration_intent")


class HardDatasetTest(unittest.TestCase):
    def test_hard_dataset_has_expected_shape(self) -> None:
        examples = load_prompt_examples(HARD_PROMPTS_PATH)

        label_counts = Counter(example.label for example in examples)
        family_counts_by_label: dict[str, Counter[str]] = defaultdict(Counter)
        for example in examples:
            family_counts_by_label[example.label].update((example.family,))

        self.assertEqual(90, len(examples))
        self.assertEqual({label: 30 for label in EXPECTED_LABELS}, dict(label_counts))
        for label in EXPECTED_LABELS:
            family_counts = family_counts_by_label[label]
            self.assertEqual(10, len(family_counts))
            self.assertEqual([3] * 10, sorted(family_counts.values()))

    def test_hard_dataset_uses_hard_ids_and_tags(self) -> None:
        examples = load_prompt_examples(HARD_PROMPTS_PATH)

        for example in examples:
            self.assertTrue(example.id.startswith("hard_"))
            self.assertIn("hard", example.tags)


if __name__ == "__main__":
    unittest.main()
