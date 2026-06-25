from __future__ import annotations

import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from aegis_introspection.model_loader import DeviceSelection, device_diagnostic, select_device

DEVICE_PREFLIGHT_SCHEMA_VERSION = "aegis_introspection.device_preflight/v1"


@dataclass(frozen=True)
class DevicePreflightReport:
    requested_device: str
    selected_device: str
    torch_device: str
    torch_version: str
    python_machine: str
    macos_version: str
    mps_built: str
    smoke_tensor_device: str


def run_device_preflight(requested_device: str) -> DevicePreflightReport:
    selection = select_device(requested_device)
    smoke_tensor_device = _smoke_tensor_device(selection)
    return DevicePreflightReport(
        requested_device=requested_device,
        selected_device=selection.name,
        torch_device=str(selection.torch_device),
        torch_version=torch.__version__,
        python_machine=platform.machine() or "unknown",
        macos_version=platform.mac_ver()[0] or "unknown",
        mps_built=_mps_built(),
        smoke_tensor_device=smoke_tensor_device,
    )


def device_preflight_report_to_json(report: DevicePreflightReport) -> dict[str, str | bool]:
    return {
        "schema_version": DEVICE_PREFLIGHT_SCHEMA_VERSION,
        "eligible": True,
        "requested_device": report.requested_device,
        "selected_device": report.selected_device,
        "torch_device": report.torch_device,
        "torch_version": report.torch_version,
        "python_machine": report.python_machine,
        "macos_version": report.macos_version,
        "mps_built": report.mps_built,
        "smoke_tensor_device": report.smoke_tensor_device,
    }


def device_preflight_error_to_json(requested_device: str, message: str) -> dict[str, str | bool]:
    return {
        "schema_version": DEVICE_PREFLIGHT_SCHEMA_VERSION,
        "eligible": False,
        "requested_device": requested_device,
        "error": message,
        "diagnostics": device_diagnostic(),
    }


def run_device_preflight_cli(argv: Sequence[str]) -> int:
    cli_config = _parse_cli_config(argv)
    try:
        report = run_device_preflight(cli_config.requested_device)
    except RuntimeError as exc:
        payload = device_preflight_error_to_json(requested_device=cli_config.requested_device, message=str(exc))
        _write_optional_output(output_path=cli_config.output_path, payload=payload)
        print(json.dumps(payload, sort_keys=True))
        return 1
    payload = device_preflight_report_to_json(report)
    _write_optional_output(output_path=cli_config.output_path, payload=payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


@dataclass(frozen=True)
class _DevicePreflightCliConfig:
    requested_device: str
    output_path: Path | None


def _parse_cli_config(argv: Sequence[str]) -> _DevicePreflightCliConfig:
    requested_device: str | None = None
    output_path: Path | None = None
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--device":
            index += 1
            if index >= len(argv):
                raise SystemExit("--device requires a value.")
            requested_device = argv[index]
        elif item == "--output":
            index += 1
            if index >= len(argv):
                raise SystemExit("--output requires a value.")
            output_path = Path(argv[index])
        else:
            raise SystemExit("usage: check_cift_device_preflight.py --device {auto|cuda|gpu|mps|cpu} [--output PATH]")
        index += 1
    if requested_device is None:
        raise SystemExit("usage: check_cift_device_preflight.py --device {auto|cuda|gpu|mps|cpu} [--output PATH]")
    if requested_device == "":
        raise SystemExit("--device must not be empty.")
    return _DevicePreflightCliConfig(requested_device=requested_device, output_path=output_path)


def _write_optional_output(output_path: Path | None, payload: dict[str, str | bool]) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _smoke_tensor_device(selection: DeviceSelection) -> str:
    try:
        tensor = torch.ones(1, device=selection.torch_device)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Device {selection.name} smoke tensor allocation failed for {selection.torch_device}: {exc}"
        ) from exc
    return str(tensor.device)


def _mps_built() -> str:
    if not hasattr(torch.backends, "mps"):
        return "missing"
    return str(torch.backends.mps.is_built())


if __name__ == "__main__":
    raise SystemExit(run_device_preflight_cli(tuple(sys.argv[1:])))
