import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.honeytokens import (
    DpHoneyLiteTemplate,
    generate_honeytoken,
    hard_dp_honey_lite_v3_templates,
    render_honeytoken_prompt,
)
from aegis_introspection.v3_policy_resolution import (
    V3PolicyResolutionParseError,
    evaluate_v3_policy_resolution,
    parse_v3_policy_resolution_response,
    render_v3_policy_resolution_markdown,
    v3_policy_resolution_report_to_json,
    write_v3_policy_resolution_json,
    write_v3_policy_resolution_markdown,
)


def _artifact_from_templates(templates: tuple[DpHoneyLiteTemplate, ...]) -> ActivationArtifact:
    texts: list[str] = []
    labels: list[str] = []
    families: list[str] = []
    example_ids: list[str] = []
    tags: list[tuple[str, ...]] = []
    for index, template in enumerate(templates):
        secret = generate_honeytoken(credential_type="api_key", seed="resolution", index=index)
        rendered = render_honeytoken_prompt(template=template, secret=secret)
        example_ids.append(f"{template.template_id}_000")
        labels.append(template.label)
        families.append(template.family)
        texts.append(rendered.text)
        tags.append(template.tags)

    return {
        "metadata": {
            "model_id": "synthetic",
            "revision": "main",
            "selected_device": "cpu",
            "layer_indices": (21,),
            "pooling_methods": ("readout_window",),
        },
        "example_ids": tuple(example_ids),
        "labels": tuple(labels),
        "families": tuple(families),
        "texts": tuple(texts),
        "tags": tuple(tags),
        "features": {"readout_window_layer_21": torch.zeros((len(texts), 2), dtype=torch.float32)},
    }


def _target_v3_templates() -> tuple[DpHoneyLiteTemplate, ...]:
    selected_families = tuple(sorted({template.family for template in hard_dp_honey_lite_v3_templates()}))[:4]
    return tuple(
        template
        for template in hard_dp_honey_lite_v3_templates()
        if template.family in selected_families and template.label in {"secret_present_safe", "exfiltration_intent"}
    )


def _stub_resolver(prompt: str) -> str:
    if "audit_export_safe_secret_payload_000" in prompt:
        return "I am not sure."
    if "_safe_secret_" in prompt:
        return '{"selected_action": "mask"}'
    if "_exfiltration_" in prompt:
        return "The selected action is copy."
    return "invalid"


def _copy_resolver(prompt: str) -> str:
    return "copy"


class V3PolicyResolutionTest(unittest.TestCase):
    def test_parse_v3_policy_resolution_response_accepts_json_action(self) -> None:
        self.assertEqual("mask", parse_v3_policy_resolution_response('{"selected_action": "mask"}'))

    def test_parse_v3_policy_resolution_response_accepts_single_action_word(self) -> None:
        self.assertEqual("copy", parse_v3_policy_resolution_response("The selected action is COPY."))

    def test_parse_v3_policy_resolution_response_rejects_ambiguous_actions(self) -> None:
        with self.assertRaises(V3PolicyResolutionParseError):
            parse_v3_policy_resolution_response("copy the summary but mask the credential")

    def test_evaluate_v3_policy_resolution_reports_model_and_parser_metrics(self) -> None:
        report = evaluate_v3_policy_resolution(
            artifact=_artifact_from_templates(_target_v3_templates()),
            resolver=_stub_resolver,
            resolver_model_id="stub-policy-resolver",
            max_examples=None,
        )
        decoded = v3_policy_resolution_report_to_json(report)
        markdown = render_v3_policy_resolution_markdown(report)

        self.assertEqual("safe_secret_vs_exfiltration_policy_resolution", report.task_name)
        self.assertEqual(("all", "payload", "no_payload", "mode_a", "mode_b"), tuple(item.slice_name for item in report.slices))
        self.assertEqual(16, report.example_count)
        self.assertEqual(16, len(report.predictions))
        self.assertEqual(1.0, report.slices[0].parser.macro_f1)
        self.assertEqual(1, report.slices[0].model.invalid_count)
        self.assertLess(report.slices[0].model.accuracy, 1.0)
        self.assertEqual("safe_secret_vs_exfiltration_policy_resolution", decoded["task_name"])
        self.assertIn("V3 Policy Resolution Diagnostic", markdown)

    def test_evaluate_v3_policy_resolution_handles_limited_single_label_slices(self) -> None:
        report = evaluate_v3_policy_resolution(
            artifact=_artifact_from_templates(_target_v3_templates()),
            resolver=_stub_resolver,
            resolver_model_id="stub-policy-resolver",
            max_examples=4,
        )

        for slice_report in report.slices:
            self.assertEqual(("exfiltration_intent", "secret_present_safe"), slice_report.model.label_names)

    def test_evaluate_v3_policy_resolution_handles_predictions_outside_limited_truth_labels(self) -> None:
        report = evaluate_v3_policy_resolution(
            artifact=_artifact_from_templates(_target_v3_templates()),
            resolver=_copy_resolver,
            resolver_model_id="copy-only-resolver",
            max_examples=2,
        )

        self.assertEqual(("exfiltration_intent", "secret_present_safe"), report.slices[0].model.label_names)
        self.assertEqual(0.0, report.slices[0].model.accuracy)

    def test_write_v3_policy_resolution_outputs_json_and_markdown_files(self) -> None:
        report = evaluate_v3_policy_resolution(
            artifact=_artifact_from_templates(_target_v3_templates()),
            resolver=_stub_resolver,
            resolver_model_id="stub-policy-resolver",
            max_examples=8,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / "resolution.json"
            markdown_path = root / "resolution.md"

            write_v3_policy_resolution_json(json_path, report)
            write_v3_policy_resolution_markdown(markdown_path, report)

            self.assertEqual(
                "safe_secret_vs_exfiltration_policy_resolution",
                json.loads(json_path.read_text())["task_name"],
            )
            self.assertIn("V3 Policy Resolution Diagnostic", markdown_path.read_text())


if __name__ == "__main__":
    unittest.main()
