from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from aegis_introspection.device_preflight import (
    DEVICE_PREFLIGHT_SCHEMA_VERSION,
    device_preflight_report_to_json,
    run_device_preflight,
    run_device_preflight_cli,
)
from aegis_introspection.model_loader import DeviceSelection


class DevicePreflightTest(unittest.TestCase):
    def test_run_device_preflight_smokes_selected_device(self) -> None:
        tensor = Mock()
        tensor.device = torch.device("mps")
        selection = DeviceSelection(
            name="mps",
            torch_device=torch.device("mps"),
            torch_dtype=torch.float16,
        )

        with (
            patch("aegis_introspection.device_preflight.select_device", return_value=selection),
            patch("aegis_introspection.device_preflight.torch.ones", return_value=tensor),
            patch("aegis_introspection.device_preflight._mps_built", return_value="True"),
        ):
            report = run_device_preflight("mps")

        payload = device_preflight_report_to_json(report)
        self.assertEqual(DEVICE_PREFLIGHT_SCHEMA_VERSION, payload["schema_version"])
        self.assertEqual(True, payload["eligible"])
        self.assertEqual("mps", payload["requested_device"])
        self.assertEqual("mps", payload["selected_device"])
        self.assertEqual("mps", payload["torch_device"])
        self.assertEqual("mps", payload["smoke_tensor_device"])
        self.assertEqual("True", payload["mps_built"])

    def test_cli_writes_preflight_output_file(self) -> None:
        report = Mock()
        report.requested_device = "mps"
        report.selected_device = "mps"
        report.torch_device = "mps"
        report.torch_version = "test"
        report.python_machine = "arm64"
        report.macos_version = "test"
        report.mps_built = "True"
        report.smoke_tensor_device = "mps:0"

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "preflight.json"
            with (
                patch("aegis_introspection.device_preflight.run_device_preflight", return_value=report),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = run_device_preflight_cli(("--device", "mps", "--output", str(output_path)))
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertEqual(DEVICE_PREFLIGHT_SCHEMA_VERSION, payload["schema_version"])
        self.assertEqual(True, payload["eligible"])
        self.assertEqual("mps", payload["selected_device"])


if __name__ == "__main__":
    unittest.main()
