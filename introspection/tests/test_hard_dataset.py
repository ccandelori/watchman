import unittest
from collections import Counter, defaultdict
from pathlib import Path

from aegis_introspection.prompts import load_prompt_examples


HARD_PROMPTS_PATH = Path(__file__).resolve().parents[1] / "data" / "prompts_hard.jsonl"
HARD_V2_PROMPTS_PATH = Path(__file__).resolve().parents[1] / "data" / "prompts_hard_v2.jsonl"
EXPECTED_LABELS = ("benign", "secret_present_safe", "exfiltration_intent")
EXPECTED_HARD_V2_TARGETS = (
    "broker",
    "output_contract",
    "policy",
    "summary",
    "tool_argument",
)


def _assert_balanced_family_shape(test_case: unittest.TestCase, path: Path) -> None:
    examples = load_prompt_examples(path)
    label_counts = Counter(example.label for example in examples)
    family_counts_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        family_counts_by_label[example.label].update((example.family,))

    test_case.assertEqual(90, len(examples))
    test_case.assertEqual({label: 30 for label in EXPECTED_LABELS}, dict(label_counts))
    for label in EXPECTED_LABELS:
        family_counts = family_counts_by_label[label]
        test_case.assertEqual(10, len(family_counts))
        test_case.assertEqual([3] * 10, sorted(family_counts.values()))


class HardDatasetTest(unittest.TestCase):
    def test_hard_dataset_has_expected_shape(self) -> None:
        _assert_balanced_family_shape(self, HARD_PROMPTS_PATH)

    def test_hard_dataset_uses_hard_ids_and_tags(self) -> None:
        examples = load_prompt_examples(HARD_PROMPTS_PATH)

        for example in examples:
            self.assertTrue(example.id.startswith("hard_"))
            self.assertIn("hard", example.tags)

    def test_hard_v2_dataset_has_expected_shape(self) -> None:
        _assert_balanced_family_shape(self, HARD_V2_PROMPTS_PATH)

    def test_hard_v2_dataset_targets_current_weak_families(self) -> None:
        examples = load_prompt_examples(HARD_V2_PROMPTS_PATH)
        families = {example.family for example in examples}

        for target in EXPECTED_HARD_V2_TARGETS:
            matching_families = {family for family in families if target in family}
            self.assertGreaterEqual(len(matching_families), 2)

        for example in examples:
            self.assertTrue(example.id.startswith("hard_v2_"))
            self.assertIn("hard", example.tags)
            self.assertIn("hard_v2", example.tags)


if __name__ == "__main__":
    unittest.main()
