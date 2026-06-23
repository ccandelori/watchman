from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

ModelDTypeName: TypeAlias = Literal["auto", "device", "float32", "float16", "bfloat16"]
ModelLoadDType: TypeAlias = torch.dtype | Literal["auto"]
_VALID_MODEL_DTYPES: frozenset[str] = frozenset(("auto", "device", "float32", "float16", "bfloat16"))


class UnsupportedDeviceError(ValueError):
    """Raised when the requested device name is not supported."""


class DeviceUnavailableError(RuntimeError):
    """Raised when the requested device is supported but unavailable locally."""


class UnsupportedModelDTypeError(ValueError):
    """Raised when a model dtype selector is not supported."""


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
    dtype_name: ModelDTypeName
    trust_remote_code: bool


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


def parse_model_dtype(raw_value: str) -> ModelDTypeName:
    if raw_value not in _VALID_MODEL_DTYPES:
        valid = ", ".join(sorted(_VALID_MODEL_DTYPES))
        raise UnsupportedModelDTypeError(f"Unsupported model dtype '{raw_value}'. Expected one of: {valid}.")
    return cast(ModelDTypeName, raw_value)


def resolve_model_load_dtype(dtype_name: ModelDTypeName, device: DeviceSelection) -> ModelLoadDType:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "device":
        return device.torch_dtype
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise UnsupportedModelDTypeError(f"Unsupported model dtype '{dtype_name}'.")


def load_causal_lm(config: ModelLoadConfig) -> LoadedCausalLM:
    device = select_device(config.requested_device)
    load_dtype = resolve_model_load_dtype(dtype_name=config.dtype_name, device=device)
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        ),
    )
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
            dtype=load_dtype,
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
