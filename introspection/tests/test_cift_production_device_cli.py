from __future__ import annotations

import contextlib
import io
import unittest

from introspection.scripts.extract_activations import _parse_args as parse_extract_activations_args
from introspection.scripts.run_cift_extractor_sidecar import _parse_args as parse_extractor_sidecar_args
from introspection.scripts.run_cift_hidden_state_patching import _parse_args as parse_hidden_state_patching_args


class CiftProductionDeviceCliTest(unittest.TestCase):
    def test_extract_activations_requires_explicit_device(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_extract_activations_args(())

    def test_extract_activations_accepts_explicit_mps_device(self) -> None:
        config = parse_extract_activations_args(("--device", "mps"))

        self.assertEqual("mps", config.requested_device)

    def test_hidden_state_patching_requires_explicit_device(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_hidden_state_patching_args(
                (
                    "--prompts",
                    "prompts.jsonl",
                    "--output",
                    "patching.json",
                    "--report-id",
                    "patching-report",
                    "--model-id",
                    "Qwen/Qwen3-4B",
                    "--patch-layer-index",
                    "21",
                    "--observable-mode",
                    "paired_selected_choice",
                    "--minimum-margin-shift",
                    "0.1",
                )
            )

    def test_hidden_state_patching_accepts_explicit_mps_device(self) -> None:
        config = parse_hidden_state_patching_args(
            (
                "--prompts",
                "prompts.jsonl",
                "--output",
                "patching.json",
                "--report-id",
                "patching-report",
                "--model-id",
                "Qwen/Qwen3-4B",
                "--device",
                "mps",
                "--patch-layer-index",
                "21",
                "--observable-mode",
                "paired_selected_choice",
                "--minimum-margin-shift",
                "0.1",
                "--created-at",
                "2026-06-25T00:00:00Z",
            )
        )

        self.assertEqual("mps", config.requested_device)

    def test_qwen3_4b_extractor_sidecar_rejects_cpu_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --device mps"):
            parse_extractor_sidecar_args(
                (
                    "--model-id",
                    "Qwen/Qwen3-4B",
                    "--revision",
                    "main",
                    "--device",
                    "cpu",
                    "--dtype",
                    "device",
                    "--feature-key",
                    "selected_choice_window_layer_21",
                    "--selected-choice-readout-token-count",
                    "4",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9000",
                )
            )

    def test_qwen3_4b_extractor_sidecar_rejects_auto_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --device mps"):
            parse_extractor_sidecar_args(
                (
                    "--model-id",
                    "Qwen/Qwen3-4B",
                    "--revision",
                    "main",
                    "--device",
                    "auto",
                    "--dtype",
                    "device",
                    "--feature-key",
                    "selected_choice_window_layer_21",
                    "--selected-choice-readout-token-count",
                    "4",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9000",
                )
            )


if __name__ == "__main__":
    unittest.main()
