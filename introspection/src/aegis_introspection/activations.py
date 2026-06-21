from dataclasses import dataclass

import torch
from transformers import BatchEncoding

from aegis_introspection.model_loader import LoadedCausalLM


class HiddenStatesUnavailableError(RuntimeError):
    """Raised when the model response does not include hidden states."""


class EncodedFieldError(TypeError):
    """Raised when tokenization output does not contain the expected tensor fields."""


class ReadoutWindowError(ValueError):
    """Raised when a readout token window cannot be pooled."""


@dataclass(frozen=True)
class HiddenStateSummary:
    layer_index: int
    shape: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class HiddenStateForwardPass:
    prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    hidden_states: tuple[torch.Tensor, ...]


def _tensor_field(encoded: BatchEncoding, field_name: str) -> torch.Tensor:
    value = encoded.data.get(field_name)
    if not isinstance(value, torch.Tensor):
        raise EncodedFieldError(f"Expected tokenizer field '{field_name}' to be a torch.Tensor.")
    return value


def _optional_tensor_field(encoded: BatchEncoding, field_name: str) -> torch.Tensor | None:
    value = encoded.data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise EncodedFieldError(f"Expected tokenizer field '{field_name}' to be a torch.Tensor.")
    return value


def encode_prompt(loaded_model: LoadedCausalLM, prompt: str) -> BatchEncoding:
    encoded = loaded_model.tokenizer(prompt, return_tensors="pt")
    if not isinstance(encoded, BatchEncoding):
        raise EncodedFieldError("Expected tokenizer output to be a transformers.BatchEncoding instance.")
    return encoded.to(loaded_model.device.torch_device)


def run_hidden_state_forward(loaded_model: LoadedCausalLM, prompt: str) -> HiddenStateForwardPass:
    encoded = encode_prompt(loaded_model, prompt)
    input_ids = _tensor_field(encoded, "input_ids")
    attention_mask = _optional_tensor_field(encoded, "attention_mask")

    with torch.no_grad():
        outputs = loaded_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise HiddenStatesUnavailableError("Model response did not include hidden states.")

    return HiddenStateForwardPass(
        prompt=prompt,
        input_ids=input_ids.detach().cpu(),
        attention_mask=None if attention_mask is None else attention_mask.detach().cpu(),
        hidden_states=tuple(hidden_state.detach().cpu() for hidden_state in hidden_states),
    )


def summarize_hidden_states(hidden_states: tuple[torch.Tensor, ...]) -> tuple[HiddenStateSummary, ...]:
    return tuple(
        HiddenStateSummary(
            layer_index=layer_index,
            shape=tuple(hidden_state.shape),
            dtype=str(hidden_state.dtype),
            device=str(hidden_state.device),
        )
        for layer_index, hidden_state in enumerate(hidden_states)
    )


def final_token_activation(forward_pass: HiddenStateForwardPass, layer_index: int) -> torch.Tensor:
    hidden_state = forward_pass.hidden_states[layer_index]
    return hidden_state[:, -1, :].clone()


def mean_pool_activation(forward_pass: HiddenStateForwardPass, layer_index: int) -> torch.Tensor:
    hidden_state = forward_pass.hidden_states[layer_index]
    if forward_pass.attention_mask is None:
        return hidden_state.mean(dim=1).clone()

    attention_mask = forward_pass.attention_mask.to(hidden_state.device).unsqueeze(-1)
    token_counts = attention_mask.sum(dim=1).clamp_min(1)
    pooled = (hidden_state * attention_mask).sum(dim=1) / token_counts
    return pooled.clone()


def _validate_readout_token_indices(token_indices: tuple[int, ...], sequence_length: int) -> None:
    if len(token_indices) == 0:
        raise ReadoutWindowError("readout_token_indices must not be empty.")
    for index, token_index in enumerate(token_indices):
        if token_index < 0:
            raise ReadoutWindowError(f"readout_token_indices item {index} must be non-negative.")
        if token_index >= sequence_length:
            raise ReadoutWindowError(
                f"readout_token_indices item {index}={token_index} is out of range for sequence length "
                f"{sequence_length}."
            )


def readout_window_activation(
    forward_pass: HiddenStateForwardPass,
    layer_index: int,
    token_indices: tuple[int, ...],
) -> torch.Tensor:
    hidden_state = forward_pass.hidden_states[layer_index]
    sequence_length = int(hidden_state.shape[1])
    _validate_readout_token_indices(token_indices=token_indices, sequence_length=sequence_length)
    selected = hidden_state[:, list(token_indices), :]
    return selected.mean(dim=1).clone()
