from dataclasses import dataclass
from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


class UnsupportedDeviceError(ValueError):
    """Raised when the requested device name is not supported."""


class DeviceUnavailableError(RuntimeError):
    """Raised when the requested device is supported but unavailable locally."""


@dataclass(frozen=True)
class DeviceSelection:
    name: str
    torch_device: torch.device
    torch_dtype: torch.dtype


@dataclass(frozen=True)
class ModelLoadConfig:
    model_id: str
    revision: str
    requested_device: str
    local_files_only: bool


@dataclass(frozen=True)
class LoadedCausalLM:
    model_id: str
    revision: str
    device: DeviceSelection
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel


def _cuda_selection() -> DeviceSelection:
    return DeviceSelection(
        name="cuda",
        torch_device=torch.device("cuda"),
        torch_dtype=torch.float16,
    )


def _mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _mps_selection() -> DeviceSelection:
    return DeviceSelection(
        name="mps",
        torch_device=torch.device("mps"),
        torch_dtype=torch.float16,
    )


def _cpu_selection() -> DeviceSelection:
    return DeviceSelection(
        name="cpu",
        torch_device=torch.device("cpu"),
        torch_dtype=torch.float32,
    )


def select_device(requested_device: str) -> DeviceSelection:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return _cuda_selection()
        if _mps_is_available():
            return _mps_selection()
        return _cpu_selection()

    if requested_device in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise DeviceUnavailableError("CUDA was requested, but torch.cuda.is_available() is false.")
        return _cuda_selection()

    if requested_device == "mps":
        if not _mps_is_available():
            raise DeviceUnavailableError("MPS was requested, but torch.backends.mps.is_available() is false.")
        return _mps_selection()

    if requested_device == "cpu":
        return _cpu_selection()

    raise UnsupportedDeviceError(
        f"Unsupported device '{requested_device}'. Expected one of: auto, cuda, gpu, mps, cpu."
    )


def load_causal_lm(config: ModelLoadConfig) -> LoadedCausalLM:
    device = select_device(config.requested_device)
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            local_files_only=config.local_files_only,
        ),
    )
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            local_files_only=config.local_files_only,
            dtype=device.torch_dtype,
        ),
    )

    model.to(device.torch_device)
    model.eval()

    return LoadedCausalLM(
        model_id=config.model_id,
        revision=config.revision,
        device=device,
        tokenizer=tokenizer,
        model=model,
    )
