import unittest
from collections import Counter
from pathlib import Path
from typing import Mapping

from introspection.scripts.generate_dp_honey_lite_prompts import (
    GenerateDpHoneyLitePromptsConfig,
    _example_specs,
    _parse_args,
)


class CharacterOffsetTokenizer:
    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        add_special_tokens: bool,
    ) -> Mapping[str, object]:
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _selected_mode(text: str) -> str:
    if "selected_mode=mode_a" in text or "'selected_mode': 'mode_a'" in text:
        return "mode_a"
    if "selected_mode=mode_b" in text or "'selected_mode': 'mode_b'" in text:
        return "mode_b"
    raise AssertionError("Rendered text does not declare a selected mode.")


class GenerateDpHoneyLitePromptsTest(unittest.TestCase):
    def test_parse_args_accepts_hard_v2_template_set(self) -> None:
        config = _parse_args(
            (
                "--template-set",
                "hard_v2",
                "--output",
                "data/prompts_dp_honey_lite_v2.jsonl",
            )
        )

        self.assertEqual("hard_v2", config.template_set)
        self.assertEqual(Path("data/prompts_dp_honey_lite_v2.jsonl"), config.output_path)

    def test_parse_args_uses_template_specific_default_output(self) -> None:
        config = _parse_args(("--template-set", "hard_v2"))

        self.assertEqual("prompts_dp_honey_lite_v2.jsonl", config.output_path.name)

    def test_parse_args_uses_hard_v3_default_output(self) -> None:
        config = _parse_args(("--template-set", "hard_v3"))

        self.assertEqual("prompts_dp_honey_lite_v3.jsonl", config.output_path.name)

    def test_parse_args_uses_hard_v4_default_output(self) -> None:
        config = _parse_args(("--template-set", "hard_v4"))

        self.assertEqual("prompts_dp_honey_lite_v4.jsonl", config.output_path.name)

    def test_parse_args_uses_hard_v4_1_default_output(self) -> None:
        config = _parse_args(("--template-set", "hard_v4_1"))

        self.assertEqual("prompts_dp_honey_lite_v4_1.jsonl", config.output_path.name)

    def test_hard_v2_example_specs_preserve_balance_dimensions(self) -> None:
        config = GenerateDpHoneyLitePromptsConfig(
            output_path=Path("unused.jsonl"),
            model_id="unused",
            revision="main",
            local_files_only=True,
            seed="capstone-hard-v2",
            examples_per_template=4,
            readout_width=6,
            template_set="hard_v2",
        )

        specs = _example_specs(tokenizer=CharacterOffsetTokenizer(), config=config)
        labels = Counter(spec[3].label for spec in specs)
        credentials_by_label = Counter((spec[3].label, spec[1]) for spec in specs)
        payloads_by_label = Counter((spec[3].label, spec[3].payload_span is not None) for spec in specs)

        self.assertEqual(240, len(specs))
        self.assertEqual(
            {"benign": 80, "secret_present_safe": 80, "exfiltration_intent": 80},
            dict(labels),
        )
        self.assertEqual(10, len({spec[3].family for spec in specs}))
        for label in ("benign", "secret_present_safe", "exfiltration_intent"):
            self.assertEqual(40, credentials_by_label[(label, "api_key")])
            self.assertEqual(40, credentials_by_label[(label, "database_uri")])
            self.assertEqual(40, payloads_by_label[(label, False)])
            self.assertEqual(40, payloads_by_label[(label, True)])

    def test_hard_v3_example_specs_preserve_balance_dimensions(self) -> None:
        config = GenerateDpHoneyLitePromptsConfig(
            output_path=Path("unused.jsonl"),
            model_id="unused",
            revision="main",
            local_files_only=True,
            seed="capstone-hard-v3",
            examples_per_template=4,
            readout_width=6,
            template_set="hard_v3",
        )

        specs = _example_specs(tokenizer=CharacterOffsetTokenizer(), config=config)
        labels = Counter(spec[3].label for spec in specs)
        credentials_by_label = Counter((spec[3].label, spec[1]) for spec in specs)
        payloads_by_label = Counter((spec[3].label, spec[3].payload_span is not None) for spec in specs)
        selected_modes_by_label = Counter((spec[3].label, _selected_mode(spec[3].text)) for spec in specs)

        self.assertEqual(240, len(specs))
        self.assertEqual(
            {"benign": 80, "secret_present_safe": 80, "exfiltration_intent": 80},
            dict(labels),
        )
        self.assertEqual(10, len({spec[3].family for spec in specs}))
        for label in ("benign", "secret_present_safe", "exfiltration_intent"):
            self.assertEqual(40, credentials_by_label[(label, "api_key")])
            self.assertEqual(40, credentials_by_label[(label, "database_uri")])
            self.assertEqual(40, payloads_by_label[(label, False)])
            self.assertEqual(40, payloads_by_label[(label, True)])
            self.assertEqual(40, selected_modes_by_label[(label, "mode_a")])
            self.assertEqual(40, selected_modes_by_label[(label, "mode_b")])

    def test_hard_v4_example_specs_focus_mode_b_payload_failure_slices(self) -> None:
        config = GenerateDpHoneyLitePromptsConfig(
            output_path=Path("unused.jsonl"),
            model_id="unused",
            revision="main",
            local_files_only=True,
            seed="capstone-hard-v4",
            examples_per_template=4,
            readout_width=6,
            template_set="hard_v4",
        )

        specs = _example_specs(tokenizer=CharacterOffsetTokenizer(), config=config)
        labels = Counter(spec[3].label for spec in specs)
        credentials_by_label = Counter((spec[3].label, spec[1]) for spec in specs)
        payloads_by_label = Counter((spec[3].label, spec[3].payload_span is not None) for spec in specs)
        selected_modes_by_label = Counter((spec[3].label, _selected_mode(spec[3].text)) for spec in specs)

        self.assertEqual(144, len(specs))
        self.assertEqual(
            {"benign": 48, "secret_present_safe": 48, "exfiltration_intent": 48},
            dict(labels),
        )
        self.assertEqual(6, len({spec[3].family for spec in specs}))
        for label in ("benign", "secret_present_safe", "exfiltration_intent"):
            self.assertEqual(24, credentials_by_label[(label, "api_key")])
            self.assertEqual(24, credentials_by_label[(label, "database_uri")])
            self.assertEqual(24, payloads_by_label[(label, False)])
            self.assertEqual(24, payloads_by_label[(label, True)])
            self.assertEqual(0, selected_modes_by_label[(label, "mode_a")])
            self.assertEqual(48, selected_modes_by_label[(label, "mode_b")])

    def test_hard_v4_1_example_specs_focus_failure_slices_without_mode_shortcut(self) -> None:
        config = GenerateDpHoneyLitePromptsConfig(
            output_path=Path("unused.jsonl"),
            model_id="unused",
            revision="main",
            local_files_only=True,
            seed="capstone-hard-v4-1",
            examples_per_template=4,
            readout_width=6,
            template_set="hard_v4_1",
        )

        specs = _example_specs(tokenizer=CharacterOffsetTokenizer(), config=config)
        labels = Counter(spec[3].label for spec in specs)
        credentials_by_label = Counter((spec[3].label, spec[1]) for spec in specs)
        payloads_by_label = Counter((spec[3].label, spec[3].payload_span is not None) for spec in specs)
        selected_modes_by_label = Counter((spec[3].label, _selected_mode(spec[3].text)) for spec in specs)

        self.assertEqual(144, len(specs))
        self.assertEqual(
            {"benign": 48, "secret_present_safe": 48, "exfiltration_intent": 48},
            dict(labels),
        )
        self.assertEqual(6, len({spec[3].family for spec in specs}))
        for label in ("benign", "secret_present_safe", "exfiltration_intent"):
            self.assertEqual(24, credentials_by_label[(label, "api_key")])
            self.assertEqual(24, credentials_by_label[(label, "database_uri")])
            self.assertEqual(24, payloads_by_label[(label, False)])
            self.assertEqual(24, payloads_by_label[(label, True)])
            self.assertEqual(24, selected_modes_by_label[(label, "mode_a")])
            self.assertEqual(24, selected_modes_by_label[(label, "mode_b")])


if __name__ == "__main__":
    unittest.main()
