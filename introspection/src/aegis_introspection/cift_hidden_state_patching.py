from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import torch
import torch.nn.functional as torch_functional
from torch import nn

from aegis_introspection.activations import (
    HiddenStateForwardPass,
    HiddenStatesUnavailableError,
    encode_prompt,
    run_hidden_state_forward,
)
from aegis_introspection.model_loader import LoadedCausalLM
from aegis_introspection.prompts import StructuredPromptExample

_SCHEMA_VERSION = "aegis_introspection.cift_hidden_state_patching/v2"
_INTERVENTION_TYPE = "transformer_layer_output_replacement"
_CLAIM_SCOPE = "model_output_log_probability_margin"
_PAIRING_TAG_FIELDS = ("participant", "task", "family", "variant", "credential_type")

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
PatchObservableMode: TypeAlias = Literal["fixed_completion", "paired_selected_choice"]


class HiddenStatePatchError(ValueError):
    """Raised when a hidden-state patch cannot be applied safely."""


@dataclass(frozen=True)
class HiddenStatePatchSpec:
    layer_index: int
    token_indices: tuple[int, ...]
    replacement_values: torch.Tensor


@dataclass(frozen=True)
class ModelOutputLogProbabilityMargin:
    positive_log_probability: float
    negative_log_probability: float
    margin: float


@dataclass(frozen=True)
class HiddenStatePatchPairingKey:
    participant: str
    task: str
    family: str
    variant: str
    credential_type: str


@dataclass(frozen=True)
class HiddenStatePatchPromptPair:
    key: HiddenStatePatchPairingKey
    safe: StructuredPromptExample
    exfil: StructuredPromptExample


@dataclass(frozen=True)
class HiddenStatePatchReportConfig:
    report_id: str
    patch_layer_index: int
    positive_completion: str
    negative_completion: str
    minimum_margin_shift: float
    max_pairs: int | None
    created_at: str


@dataclass(frozen=True)
class SelectedChoiceHiddenStatePatchReportConfig:
    report_id: str
    patch_layer_index: int
    minimum_margin_shift: float
    max_pairs: int | None
    created_at: str


@dataclass(frozen=True)
class HiddenStatePatchPairResult:
    key: HiddenStatePatchPairingKey
    safe_example_id: str
    exfil_example_id: str
    safe_selected_choice_token_indices: tuple[int, ...]
    exfil_selected_choice_token_indices: tuple[int, ...]
    safe_selected_choice_token_ids: tuple[int, ...]
    exfil_selected_choice_token_ids: tuple[int, ...]
    safe_positive_target_token_ids: tuple[int, ...]
    safe_negative_target_token_ids: tuple[int, ...]
    exfil_positive_target_token_ids: tuple[int, ...]
    exfil_negative_target_token_ids: tuple[int, ...]
    safe_original_margin: float
    exfil_original_margin: float
    safe_to_exfil_patched_margin: float
    exfil_to_safe_patched_margin: float
    safe_to_exfil_margin_shift: float
    exfil_to_safe_margin_shift: float
    original_polarity_correct: bool
    patched_polarity_flipped: bool
    safe_to_exfil_success: bool
    exfil_to_safe_success: bool


@dataclass(frozen=True)
class HiddenStatePatchSkippedPair:
    key: HiddenStatePatchPairingKey
    safe_example_id: str
    exfil_example_id: str
    safe_selected_choice_token_count: int
    exfil_selected_choice_token_count: int
    reason: str


@dataclass(frozen=True)
class HiddenStatePatchReport:
    schema_version: str
    report_id: str
    source_model_id: str
    source_revision: str
    patch_layer_index: int
    patched_hidden_state_index: int
    intervention_type: str
    claim_scope: str
    transformer_hidden_state_patching: bool
    observable_mode: PatchObservableMode
    positive_completion: str
    negative_completion: str
    candidate_pair_count: int
    eligible_pair_count: int
    pair_count: int
    skipped_pair_count: int
    truncated_pair_count: int
    minimum_margin_shift: float
    safe_to_exfil_success_rate: float
    exfil_to_safe_success_rate: float
    directional_intervention_passed: bool
    coverage_complete: bool
    passed: bool
    pairs: tuple[HiddenStatePatchPairResult, ...]
    skipped_pairs: tuple[HiddenStatePatchSkippedPair, ...]
    created_at: str


@dataclass(frozen=True)
class _PatchPairTargets:
    safe_positive_target_token_ids: tuple[int, ...]
    safe_negative_target_token_ids: tuple[int, ...]
    exfil_positive_target_token_ids: tuple[int, ...]
    exfil_negative_target_token_ids: tuple[int, ...]


def transformer_layer_module(model: nn.Module, layer_index: int) -> nn.Module:
    if layer_index < 0:
        raise HiddenStatePatchError("layer_index must be non-negative.")
    candidates = (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for path in candidates:
        layer_stack = _nested_attribute(owner=model, path=path)
        if _is_indexable_layer_stack(layer_stack):
            if layer_index >= len(layer_stack):
                raise HiddenStatePatchError(
                    f"layer_index {layer_index} is out of range for transformer layer stack of size {len(layer_stack)}."
                )
            layer = layer_stack[layer_index]
            if not isinstance(layer, nn.Module):
                raise HiddenStatePatchError(f"Transformer layer {layer_index} is not a torch module.")
            return layer
    raise HiddenStatePatchError("Could not locate a supported transformer layer stack on the model.")


def patch_hidden_state_tensor(
    hidden_state: torch.Tensor,
    token_indices: tuple[int, ...],
    replacement_values: torch.Tensor,
) -> torch.Tensor:
    _validate_hidden_state_tensor(hidden_state)
    _validate_token_indices(token_indices=token_indices, sequence_length=int(hidden_state.shape[1]))
    expected_shape = (len(token_indices), int(hidden_state.shape[2]))
    if tuple(replacement_values.shape) != expected_shape:
        raise HiddenStatePatchError(
            f"replacement_values shape must be {expected_shape}, received {tuple(replacement_values.shape)}."
        )
    patched = hidden_state.clone()
    patched[:, list(token_indices), :] = replacement_values.to(device=hidden_state.device, dtype=hidden_state.dtype)
    return patched


def patch_layer_output(
    output: object,
    token_indices: tuple[int, ...],
    replacement_values: torch.Tensor,
) -> object:
    if isinstance(output, torch.Tensor):
        return patch_hidden_state_tensor(
            hidden_state=output,
            token_indices=token_indices,
            replacement_values=replacement_values,
        )
    if isinstance(output, tuple):
        if len(output) == 0:
            raise HiddenStatePatchError("Layer output tuple must not be empty.")
        hidden_state = output[0]
        if not isinstance(hidden_state, torch.Tensor):
            raise HiddenStatePatchError("Layer output tuple first item must be a hidden-state tensor.")
        patched_hidden_state = patch_hidden_state_tensor(
            hidden_state=hidden_state,
            token_indices=token_indices,
            replacement_values=replacement_values,
        )
        return (patched_hidden_state, *output[1:])
    raise HiddenStatePatchError("Layer output must be a tensor or tuple with a hidden-state tensor first item.")


def run_hidden_state_forward_with_patch(
    loaded_model: LoadedCausalLM,
    prompt: str,
    layer_index: int,
    token_indices: tuple[int, ...],
    replacement_values: torch.Tensor,
) -> HiddenStateForwardPass:
    encoded = encode_prompt(loaded_model, prompt)
    input_ids = encoded.data.get("input_ids")
    attention_mask = encoded.data.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor):
        raise HiddenStatePatchError("Expected tokenizer field 'input_ids' to be a tensor.")
    if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
        raise HiddenStatePatchError("Expected tokenizer field 'attention_mask' to be a tensor when present.")
    layer = transformer_layer_module(loaded_model.model, layer_index=layer_index)

    with (
        torch.no_grad(),
        patched_layer_output(
            layer=layer,
            token_indices=token_indices,
            replacement_values=replacement_values,
        ),
    ):
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
        source_input_device=str(input_ids.device),
        source_hidden_state_devices=tuple(str(hidden_state.device) for hidden_state in hidden_states),
        source_hidden_state_dtypes=tuple(str(hidden_state.dtype) for hidden_state in hidden_states),
        input_ids=input_ids.detach().cpu(),
        attention_mask=None if attention_mask is None else attention_mask.detach().cpu(),
        hidden_states=tuple(hidden_state.detach().cpu() for hidden_state in hidden_states),
    )


def model_output_log_probability_margin(
    loaded_model: LoadedCausalLM,
    prompt: str,
    positive_completion: str,
    negative_completion: str,
    patch_spec: HiddenStatePatchSpec | None,
) -> ModelOutputLogProbabilityMargin:
    positive_target_token_ids = _completion_token_ids(
        loaded_model=loaded_model,
        prompt=prompt,
        completion=positive_completion,
    )
    negative_target_token_ids = _completion_token_ids(
        loaded_model=loaded_model,
        prompt=prompt,
        completion=negative_completion,
    )
    return model_output_token_sequence_log_probability_margin(
        loaded_model=loaded_model,
        prompt=prompt,
        positive_target_token_ids=positive_target_token_ids,
        negative_target_token_ids=negative_target_token_ids,
        patch_spec=patch_spec,
    )


def model_output_token_sequence_log_probability_margin(
    loaded_model: LoadedCausalLM,
    prompt: str,
    positive_target_token_ids: tuple[int, ...],
    negative_target_token_ids: tuple[int, ...],
    patch_spec: HiddenStatePatchSpec | None,
) -> ModelOutputLogProbabilityMargin:
    positive_log_probability = _target_token_sequence_log_probability(
        loaded_model=loaded_model,
        prompt=prompt,
        target_token_ids=positive_target_token_ids,
        patch_spec=patch_spec,
    )
    negative_log_probability = _target_token_sequence_log_probability(
        loaded_model=loaded_model,
        prompt=prompt,
        target_token_ids=negative_target_token_ids,
        patch_spec=patch_spec,
    )
    return ModelOutputLogProbabilityMargin(
        positive_log_probability=positive_log_probability,
        negative_log_probability=negative_log_probability,
        margin=positive_log_probability - negative_log_probability,
    )


def pair_structured_prompt_examples(
    examples: tuple[StructuredPromptExample, ...],
) -> tuple[HiddenStatePatchPromptPair, ...]:
    rows_by_key: dict[HiddenStatePatchPairingKey, list[StructuredPromptExample]] = {}
    for example in examples:
        if example.label not in ("secret_present_safe", "exfiltration_intent"):
            continue
        if example.fallback_reason is not None:
            continue
        if example.selected_choice_readout_token_indices is None:
            raise HiddenStatePatchError(f"Example '{example.id}' must include selected_choice_readout_token_indices.")
        key = _pairing_key(example)
        rows_by_key.setdefault(key, []).append(example)

    pairs: list[HiddenStatePatchPromptPair] = []
    for key in sorted(rows_by_key, key=_pairing_key_sort_value):
        rows = tuple(rows_by_key[key])
        safe_rows = tuple(row for row in rows if row.label == "secret_present_safe")
        exfil_rows = tuple(row for row in rows if row.label == "exfiltration_intent")
        if len(safe_rows) != 1 or len(exfil_rows) != 1:
            raise HiddenStatePatchError(
                f"Pair key '{_pairing_key_text(key)}' must contain exactly one safe and one exfil row."
            )
        pairs.append(HiddenStatePatchPromptPair(key=key, safe=safe_rows[0], exfil=exfil_rows[0]))
    if len(pairs) == 0:
        raise HiddenStatePatchError("No hidden-state patching prompt pairs were found.")
    return tuple(pairs)


def evaluate_hidden_state_patch_report(
    loaded_model: LoadedCausalLM,
    examples: tuple[StructuredPromptExample, ...],
    config: HiddenStatePatchReportConfig,
) -> HiddenStatePatchReport:
    _validate_report_config(config)
    return _evaluate_hidden_state_patch_report(
        loaded_model=loaded_model,
        examples=examples,
        report_id=config.report_id,
        patch_layer_index=config.patch_layer_index,
        positive_completion=config.positive_completion,
        negative_completion=config.negative_completion,
        minimum_margin_shift=config.minimum_margin_shift,
        max_pairs=config.max_pairs,
        created_at=config.created_at,
        observable_mode="fixed_completion",
    )


def evaluate_selected_choice_hidden_state_patch_report(
    loaded_model: LoadedCausalLM,
    examples: tuple[StructuredPromptExample, ...],
    config: SelectedChoiceHiddenStatePatchReportConfig,
) -> HiddenStatePatchReport:
    _validate_selected_choice_report_config(config)
    return _evaluate_hidden_state_patch_report(
        loaded_model=loaded_model,
        examples=examples,
        report_id=config.report_id,
        patch_layer_index=config.patch_layer_index,
        positive_completion="paired_exfil_selected_choice_tokens",
        negative_completion="paired_safe_selected_choice_tokens",
        minimum_margin_shift=config.minimum_margin_shift,
        max_pairs=config.max_pairs,
        created_at=config.created_at,
        observable_mode="paired_selected_choice",
    )


def _evaluate_hidden_state_patch_report(
    loaded_model: LoadedCausalLM,
    examples: tuple[StructuredPromptExample, ...],
    report_id: str,
    patch_layer_index: int,
    positive_completion: str,
    negative_completion: str,
    minimum_margin_shift: float,
    max_pairs: int | None,
    created_at: str,
    observable_mode: PatchObservableMode,
) -> HiddenStatePatchReport:
    pairs = pair_structured_prompt_examples(examples)
    patchable_pairs, skipped_pairs = _patchable_pairs(pairs)
    if len(patchable_pairs) == 0:
        raise HiddenStatePatchError("No hidden-state patching prompt pairs had equal selected-choice token counts.")
    selected_pairs = patchable_pairs if max_pairs is None else patchable_pairs[:max_pairs]
    truncated_pair_count = len(patchable_pairs) - len(selected_pairs)
    pair_results = tuple(
        _evaluate_hidden_state_patch_pair(
            loaded_model=loaded_model,
            pair=pair,
            patch_layer_index=patch_layer_index,
            positive_completion=positive_completion,
            negative_completion=negative_completion,
            minimum_margin_shift=minimum_margin_shift,
            observable_mode=observable_mode,
        )
        for pair in selected_pairs
    )
    safe_to_exfil_success_rate = _success_rate(tuple(result.safe_to_exfil_success for result in pair_results))
    exfil_to_safe_success_rate = _success_rate(tuple(result.exfil_to_safe_success for result in pair_results))
    directional_intervention_passed = safe_to_exfil_success_rate == 1.0 and exfil_to_safe_success_rate == 1.0
    coverage_complete = len(skipped_pairs) == 0 and truncated_pair_count == 0
    return HiddenStatePatchReport(
        schema_version=_SCHEMA_VERSION,
        report_id=report_id,
        source_model_id=loaded_model.model_id,
        source_revision=loaded_model.revision,
        patch_layer_index=patch_layer_index,
        patched_hidden_state_index=_patched_hidden_state_index(patch_layer_index),
        intervention_type=_INTERVENTION_TYPE,
        claim_scope=_CLAIM_SCOPE,
        transformer_hidden_state_patching=True,
        observable_mode=observable_mode,
        positive_completion=positive_completion,
        negative_completion=negative_completion,
        candidate_pair_count=len(pairs),
        eligible_pair_count=len(patchable_pairs),
        pair_count=len(pair_results),
        skipped_pair_count=len(skipped_pairs),
        truncated_pair_count=truncated_pair_count,
        minimum_margin_shift=minimum_margin_shift,
        safe_to_exfil_success_rate=safe_to_exfil_success_rate,
        exfil_to_safe_success_rate=exfil_to_safe_success_rate,
        directional_intervention_passed=directional_intervention_passed,
        coverage_complete=coverage_complete,
        passed=directional_intervention_passed and coverage_complete,
        pairs=pair_results,
        skipped_pairs=skipped_pairs,
        created_at=created_at,
    )


def hidden_state_patch_report_to_json(report: HiddenStatePatchReport) -> dict[str, JsonValue]:
    return {
        "schema_version": report.schema_version,
        "report_id": report.report_id,
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "patch_layer_index": report.patch_layer_index,
        "patched_hidden_state_index": report.patched_hidden_state_index,
        "intervention_type": report.intervention_type,
        "claim_scope": report.claim_scope,
        "transformer_hidden_state_patching": report.transformer_hidden_state_patching,
        "observable_mode": report.observable_mode,
        "positive_completion": report.positive_completion,
        "negative_completion": report.negative_completion,
        "candidate_pair_count": report.candidate_pair_count,
        "eligible_pair_count": report.eligible_pair_count,
        "pair_count": report.pair_count,
        "skipped_pair_count": report.skipped_pair_count,
        "truncated_pair_count": report.truncated_pair_count,
        "minimum_margin_shift": report.minimum_margin_shift,
        "safe_to_exfil_success_rate": report.safe_to_exfil_success_rate,
        "exfil_to_safe_success_rate": report.exfil_to_safe_success_rate,
        "directional_intervention_passed": report.directional_intervention_passed,
        "coverage_complete": report.coverage_complete,
        "passed": report.passed,
        "pairs": [hidden_state_patch_pair_result_to_json(pair) for pair in report.pairs],
        "skipped_pairs": [hidden_state_patch_skipped_pair_to_json(pair) for pair in report.skipped_pairs],
        "created_at": report.created_at,
    }


def hidden_state_patch_pair_result_to_json(result: HiddenStatePatchPairResult) -> dict[str, JsonValue]:
    return {
        "key": {
            "participant": result.key.participant,
            "task": result.key.task,
            "family": result.key.family,
            "variant": result.key.variant,
            "credential_type": result.key.credential_type,
        },
        "safe_example_id": result.safe_example_id,
        "exfil_example_id": result.exfil_example_id,
        "safe_selected_choice_token_indices": list(result.safe_selected_choice_token_indices),
        "exfil_selected_choice_token_indices": list(result.exfil_selected_choice_token_indices),
        "safe_selected_choice_token_ids": list(result.safe_selected_choice_token_ids),
        "exfil_selected_choice_token_ids": list(result.exfil_selected_choice_token_ids),
        "safe_positive_target_token_ids": list(result.safe_positive_target_token_ids),
        "safe_negative_target_token_ids": list(result.safe_negative_target_token_ids),
        "exfil_positive_target_token_ids": list(result.exfil_positive_target_token_ids),
        "exfil_negative_target_token_ids": list(result.exfil_negative_target_token_ids),
        "safe_original_margin": result.safe_original_margin,
        "exfil_original_margin": result.exfil_original_margin,
        "safe_to_exfil_patched_margin": result.safe_to_exfil_patched_margin,
        "exfil_to_safe_patched_margin": result.exfil_to_safe_patched_margin,
        "safe_to_exfil_margin_shift": result.safe_to_exfil_margin_shift,
        "exfil_to_safe_margin_shift": result.exfil_to_safe_margin_shift,
        "original_polarity_correct": result.original_polarity_correct,
        "patched_polarity_flipped": result.patched_polarity_flipped,
        "safe_to_exfil_success": result.safe_to_exfil_success,
        "exfil_to_safe_success": result.exfil_to_safe_success,
    }


def hidden_state_patch_skipped_pair_to_json(result: HiddenStatePatchSkippedPair) -> dict[str, JsonValue]:
    return {
        "key": {
            "participant": result.key.participant,
            "task": result.key.task,
            "family": result.key.family,
            "variant": result.key.variant,
            "credential_type": result.key.credential_type,
        },
        "safe_example_id": result.safe_example_id,
        "exfil_example_id": result.exfil_example_id,
        "safe_selected_choice_token_count": result.safe_selected_choice_token_count,
        "exfil_selected_choice_token_count": result.exfil_selected_choice_token_count,
        "reason": result.reason,
    }


def write_hidden_state_patch_report_json(path: Path, report: HiddenStatePatchReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cast(dict[str, JsonValue], hidden_state_patch_report_to_json(report)), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


@contextmanager
def patched_layer_output(
    layer: nn.Module,
    token_indices: tuple[int, ...],
    replacement_values: torch.Tensor,
) -> Iterator[None]:
    def _hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
        return patch_layer_output(
            output=output,
            token_indices=token_indices,
            replacement_values=replacement_values,
        )

    handle = layer.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def _sequence_log_probability(
    loaded_model: LoadedCausalLM,
    prompt: str,
    completion: str,
    patch_spec: HiddenStatePatchSpec | None,
) -> float:
    prompt_encoding = encode_prompt(loaded_model, prompt)
    prompt_input_ids = _input_ids_from_encoding(prompt_encoding)
    prompt_token_count = int(prompt_input_ids.shape[1])
    full_encoding = encode_prompt(loaded_model, prompt + completion)
    full_input_ids = _input_ids_from_encoding(full_encoding)
    full_attention_mask = _optional_attention_mask_from_encoding(full_encoding)
    if int(full_input_ids.shape[1]) <= prompt_token_count:
        raise HiddenStatePatchError("completion must add at least one token.")
    if not torch.equal(full_input_ids[:, :prompt_token_count], prompt_input_ids):
        raise HiddenStatePatchError("Prompt tokenization must be a prefix of prompt+completion tokenization.")
    outputs = _forward_for_logits(
        loaded_model=loaded_model,
        input_ids=full_input_ids,
        attention_mask=full_attention_mask,
        patch_spec=patch_spec,
    )
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise HiddenStatePatchError("Model output must include logits.")
    return _completion_log_probability(
        logits=logits,
        input_ids=full_input_ids,
        completion_start_index=prompt_token_count,
    )


def _forward_for_logits(
    loaded_model: LoadedCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    patch_spec: HiddenStatePatchSpec | None,
) -> object:
    if patch_spec is None:
        with torch.no_grad():
            return loaded_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
    layer = transformer_layer_module(loaded_model.model, layer_index=patch_spec.layer_index)
    with (
        torch.no_grad(),
        patched_layer_output(
            layer=layer,
            token_indices=patch_spec.token_indices,
            replacement_values=patch_spec.replacement_values,
        ),
    ):
        return loaded_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )


def _completion_log_probability(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_start_index: int,
) -> float:
    log_probabilities = torch_functional.log_softmax(logits[:, :-1, :], dim=-1)
    total = 0.0
    for token_index in range(completion_start_index, int(input_ids.shape[1])):
        token_id = int(input_ids[0, token_index].item())
        total += float(log_probabilities[0, token_index - 1, token_id].item())
    return total


def _input_ids_from_encoding(encoded: object) -> torch.Tensor:
    input_ids = getattr(encoded, "data", {}).get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise HiddenStatePatchError("Expected tokenizer field 'input_ids' to be a tensor.")
    return input_ids


def _optional_attention_mask_from_encoding(encoded: object) -> torch.Tensor | None:
    attention_mask = getattr(encoded, "data", {}).get("attention_mask")
    if attention_mask is None:
        return None
    if not isinstance(attention_mask, torch.Tensor):
        raise HiddenStatePatchError("Expected tokenizer field 'attention_mask' to be a tensor when present.")
    return attention_mask


def _target_token_sequence_log_probability(
    loaded_model: LoadedCausalLM,
    prompt: str,
    target_token_ids: tuple[int, ...],
    patch_spec: HiddenStatePatchSpec | None,
) -> float:
    if len(target_token_ids) == 0:
        raise HiddenStatePatchError("target_token_ids must not be empty.")
    prompt_encoding = encode_prompt(loaded_model, prompt)
    prompt_input_ids = _input_ids_from_encoding(prompt_encoding)
    prompt_attention_mask = _optional_attention_mask_from_encoding(prompt_encoding)
    prompt_token_count = int(prompt_input_ids.shape[1])
    target_tensor = torch.tensor(
        (target_token_ids,),
        dtype=prompt_input_ids.dtype,
        device=prompt_input_ids.device,
    )
    input_ids = torch.cat((prompt_input_ids, target_tensor), dim=1)
    attention_mask = _extended_attention_mask(
        prompt_attention_mask=prompt_attention_mask,
        target_token_count=len(target_token_ids),
        input_device=input_ids.device,
    )
    outputs = _forward_for_logits(
        loaded_model=loaded_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        patch_spec=patch_spec,
    )
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise HiddenStatePatchError("Model output must include logits.")
    return _completion_log_probability(
        logits=logits,
        input_ids=input_ids,
        completion_start_index=prompt_token_count,
    )


def _extended_attention_mask(
    prompt_attention_mask: torch.Tensor | None,
    target_token_count: int,
    input_device: torch.device,
) -> torch.Tensor | None:
    if prompt_attention_mask is None:
        return None
    target_mask = torch.ones(
        (int(prompt_attention_mask.shape[0]), target_token_count),
        dtype=prompt_attention_mask.dtype,
        device=input_device,
    )
    return torch.cat((prompt_attention_mask, target_mask), dim=1)


def _completion_token_ids(
    loaded_model: LoadedCausalLM,
    prompt: str,
    completion: str,
) -> tuple[int, ...]:
    prompt_encoding = encode_prompt(loaded_model, prompt)
    prompt_input_ids = _input_ids_from_encoding(prompt_encoding)
    prompt_token_count = int(prompt_input_ids.shape[1])
    full_encoding = encode_prompt(loaded_model, prompt + completion)
    full_input_ids = _input_ids_from_encoding(full_encoding)
    if int(full_input_ids.shape[1]) <= prompt_token_count:
        raise HiddenStatePatchError("completion must add at least one token.")
    if not torch.equal(full_input_ids[:, :prompt_token_count], prompt_input_ids):
        raise HiddenStatePatchError("Prompt tokenization must be a prefix of prompt+completion tokenization.")
    return tuple(int(token_id.item()) for token_id in full_input_ids[0, prompt_token_count:])


def _token_ids_at_indices(
    loaded_model: LoadedCausalLM,
    prompt: str,
    token_indices: tuple[int, ...],
) -> tuple[int, ...]:
    encoded = encode_prompt(loaded_model, prompt)
    input_ids = _input_ids_from_encoding(encoded)
    _validate_token_indices(token_indices=token_indices, sequence_length=int(input_ids.shape[1]))
    return tuple(int(input_ids[0, token_index].item()) for token_index in token_indices)


def _pairing_key(example: StructuredPromptExample) -> HiddenStatePatchPairingKey:
    parsed_tags = _parse_tags(example.tags)
    missing_fields = tuple(field for field in _PAIRING_TAG_FIELDS if parsed_tags.get(field, "") == "")
    if len(missing_fields) > 0:
        raise HiddenStatePatchError(f"Example '{example.id}' is missing pairing tags: {', '.join(missing_fields)}.")
    family = parsed_tags["family"]
    if family != example.family:
        raise HiddenStatePatchError(
            f"Example '{example.id}' family tag '{family}' does not match family '{example.family}'."
        )
    return HiddenStatePatchPairingKey(
        participant=parsed_tags["participant"],
        task=parsed_tags["task"],
        family=parsed_tags["family"],
        variant=parsed_tags["variant"],
        credential_type=parsed_tags["credential_type"],
    )


def _parse_tags(tags: tuple[str, ...]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for tag in tags:
        if ":" not in tag:
            continue
        key, value = tag.split(":", maxsplit=1)
        parsed[key] = value
    return parsed


def _pairing_key_sort_value(key: HiddenStatePatchPairingKey) -> tuple[str, str, str, str, str]:
    return (key.participant, key.task, key.family, key.variant, key.credential_type)


def _pairing_key_text(key: HiddenStatePatchPairingKey) -> str:
    return "|".join(
        f"{field}={value}" for field, value in zip(_PAIRING_TAG_FIELDS, _pairing_key_sort_value(key), strict=True)
    )


def _patchable_pairs(
    pairs: tuple[HiddenStatePatchPromptPair, ...],
) -> tuple[tuple[HiddenStatePatchPromptPair, ...], tuple[HiddenStatePatchSkippedPair, ...]]:
    patchable_pairs: list[HiddenStatePatchPromptPair] = []
    skipped_pairs: list[HiddenStatePatchSkippedPair] = []
    for pair in pairs:
        safe_count = len(_selected_choice_indices(pair.safe))
        exfil_count = len(_selected_choice_indices(pair.exfil))
        if safe_count == exfil_count:
            patchable_pairs.append(pair)
            continue
        skipped_pairs.append(
            HiddenStatePatchSkippedPair(
                key=pair.key,
                safe_example_id=pair.safe.id,
                exfil_example_id=pair.exfil.id,
                safe_selected_choice_token_count=safe_count,
                exfil_selected_choice_token_count=exfil_count,
                reason="unequal_selected_choice_readout_token_count",
            )
        )
    return tuple(patchable_pairs), tuple(skipped_pairs)


def _validate_selected_choice_alignment(
    pair: HiddenStatePatchPromptPair,
    safe_indices: tuple[int, ...],
    exfil_indices: tuple[int, ...],
) -> None:
    _validate_selected_choice_indices_cover_span(
        example_id=pair.safe.id,
        token_span=pair.safe.selected_choice_token_span,
        token_indices=safe_indices,
    )
    _validate_selected_choice_indices_cover_span(
        example_id=pair.exfil.id,
        token_span=pair.exfil.selected_choice_token_span,
        token_indices=exfil_indices,
    )


def _validate_selected_choice_indices_cover_span(
    example_id: str,
    token_span: object,
    token_indices: tuple[int, ...],
) -> None:
    if token_span is None:
        raise HiddenStatePatchError(f"Example '{example_id}' must include selected_choice_token_span.")
    start = getattr(token_span, "start", None)
    end = getattr(token_span, "end", None)
    if not isinstance(start, int) or not isinstance(end, int):
        raise HiddenStatePatchError(f"Example '{example_id}' selected_choice_token_span is malformed.")
    expected_indices = tuple(range(start, end))
    if token_indices != expected_indices:
        raise HiddenStatePatchError(
            f"Example '{example_id}' selected_choice_readout_token_indices must exactly cover "
            "selected_choice_token_span."
        )


def _validate_report_config(config: HiddenStatePatchReportConfig) -> None:
    if config.report_id == "":
        raise HiddenStatePatchError("report_id must not be empty.")
    if config.patch_layer_index < 0:
        raise HiddenStatePatchError("patch_layer_index must be non-negative.")
    if config.positive_completion == "":
        raise HiddenStatePatchError("positive_completion must not be empty.")
    if config.negative_completion == "":
        raise HiddenStatePatchError("negative_completion must not be empty.")
    if config.minimum_margin_shift < 0.0:
        raise HiddenStatePatchError("minimum_margin_shift must be non-negative.")
    if config.max_pairs is not None and config.max_pairs < 1:
        raise HiddenStatePatchError("max_pairs must be positive when present.")
    if config.created_at == "":
        raise HiddenStatePatchError("created_at must not be empty.")


def _validate_selected_choice_report_config(config: SelectedChoiceHiddenStatePatchReportConfig) -> None:
    if config.report_id == "":
        raise HiddenStatePatchError("report_id must not be empty.")
    if config.patch_layer_index < 0:
        raise HiddenStatePatchError("patch_layer_index must be non-negative.")
    if config.minimum_margin_shift < 0.0:
        raise HiddenStatePatchError("minimum_margin_shift must be non-negative.")
    if config.max_pairs is not None and config.max_pairs < 1:
        raise HiddenStatePatchError("max_pairs must be positive when present.")
    if config.created_at == "":
        raise HiddenStatePatchError("created_at must not be empty.")


def _evaluate_hidden_state_patch_pair(
    loaded_model: LoadedCausalLM,
    pair: HiddenStatePatchPromptPair,
    patch_layer_index: int,
    positive_completion: str,
    negative_completion: str,
    minimum_margin_shift: float,
    observable_mode: PatchObservableMode,
) -> HiddenStatePatchPairResult:
    safe_indices = _selected_choice_indices(pair.safe)
    exfil_indices = _selected_choice_indices(pair.exfil)
    if len(safe_indices) != len(exfil_indices):
        raise HiddenStatePatchError(
            f"Pair '{_pairing_key_text(pair.key)}' must have equal selected-choice readout token counts."
        )
    _validate_selected_choice_alignment(pair=pair, safe_indices=safe_indices, exfil_indices=exfil_indices)
    safe_selected_choice_token_ids = _token_ids_at_indices(
        loaded_model=loaded_model,
        prompt=pair.safe.text,
        token_indices=safe_indices,
    )
    exfil_selected_choice_token_ids = _token_ids_at_indices(
        loaded_model=loaded_model,
        prompt=pair.exfil.text,
        token_indices=exfil_indices,
    )
    targets = _patch_pair_targets(
        loaded_model=loaded_model,
        pair=pair,
        observable_mode=observable_mode,
        positive_completion=positive_completion,
        negative_completion=negative_completion,
        safe_selected_choice_token_ids=safe_selected_choice_token_ids,
        exfil_selected_choice_token_ids=exfil_selected_choice_token_ids,
    )
    safe_original = model_output_token_sequence_log_probability_margin(
        loaded_model=loaded_model,
        prompt=pair.safe.text,
        positive_target_token_ids=targets.safe_positive_target_token_ids,
        negative_target_token_ids=targets.safe_negative_target_token_ids,
        patch_spec=None,
    )
    exfil_original = model_output_token_sequence_log_probability_margin(
        loaded_model=loaded_model,
        prompt=pair.exfil.text,
        positive_target_token_ids=targets.exfil_positive_target_token_ids,
        negative_target_token_ids=targets.exfil_negative_target_token_ids,
        patch_spec=None,
    )
    exfil_donor_values = _donor_replacement_values(
        loaded_model=loaded_model,
        prompt=pair.exfil.text,
        patch_layer_index=patch_layer_index,
        token_indices=exfil_indices,
    )
    safe_donor_values = _donor_replacement_values(
        loaded_model=loaded_model,
        prompt=pair.safe.text,
        patch_layer_index=patch_layer_index,
        token_indices=safe_indices,
    )
    safe_to_exfil = model_output_token_sequence_log_probability_margin(
        loaded_model=loaded_model,
        prompt=pair.safe.text,
        positive_target_token_ids=targets.safe_positive_target_token_ids,
        negative_target_token_ids=targets.safe_negative_target_token_ids,
        patch_spec=HiddenStatePatchSpec(
            layer_index=patch_layer_index,
            token_indices=safe_indices,
            replacement_values=exfil_donor_values,
        ),
    )
    exfil_to_safe = model_output_token_sequence_log_probability_margin(
        loaded_model=loaded_model,
        prompt=pair.exfil.text,
        positive_target_token_ids=targets.exfil_positive_target_token_ids,
        negative_target_token_ids=targets.exfil_negative_target_token_ids,
        patch_spec=HiddenStatePatchSpec(
            layer_index=patch_layer_index,
            token_indices=exfil_indices,
            replacement_values=safe_donor_values,
        ),
    )
    safe_to_exfil_shift = safe_to_exfil.margin - safe_original.margin
    exfil_to_safe_shift = exfil_original.margin - exfil_to_safe.margin
    original_polarity_correct = safe_original.margin < 0.0 and exfil_original.margin > 0.0
    patched_polarity_flipped = safe_to_exfil.margin > 0.0 and exfil_to_safe.margin < 0.0
    return HiddenStatePatchPairResult(
        key=pair.key,
        safe_example_id=pair.safe.id,
        exfil_example_id=pair.exfil.id,
        safe_selected_choice_token_indices=safe_indices,
        exfil_selected_choice_token_indices=exfil_indices,
        safe_selected_choice_token_ids=safe_selected_choice_token_ids,
        exfil_selected_choice_token_ids=exfil_selected_choice_token_ids,
        safe_positive_target_token_ids=targets.safe_positive_target_token_ids,
        safe_negative_target_token_ids=targets.safe_negative_target_token_ids,
        exfil_positive_target_token_ids=targets.exfil_positive_target_token_ids,
        exfil_negative_target_token_ids=targets.exfil_negative_target_token_ids,
        safe_original_margin=safe_original.margin,
        exfil_original_margin=exfil_original.margin,
        safe_to_exfil_patched_margin=safe_to_exfil.margin,
        exfil_to_safe_patched_margin=exfil_to_safe.margin,
        safe_to_exfil_margin_shift=safe_to_exfil_shift,
        exfil_to_safe_margin_shift=exfil_to_safe_shift,
        original_polarity_correct=original_polarity_correct,
        patched_polarity_flipped=patched_polarity_flipped,
        safe_to_exfil_success=original_polarity_correct
        and patched_polarity_flipped
        and safe_to_exfil_shift >= minimum_margin_shift,
        exfil_to_safe_success=original_polarity_correct
        and patched_polarity_flipped
        and exfil_to_safe_shift >= minimum_margin_shift,
    )


def _patch_pair_targets(
    loaded_model: LoadedCausalLM,
    pair: HiddenStatePatchPromptPair,
    observable_mode: PatchObservableMode,
    positive_completion: str,
    negative_completion: str,
    safe_selected_choice_token_ids: tuple[int, ...],
    exfil_selected_choice_token_ids: tuple[int, ...],
) -> _PatchPairTargets:
    if observable_mode == "fixed_completion":
        return _fixed_completion_targets(
            loaded_model=loaded_model,
            pair=pair,
            positive_completion=positive_completion,
            negative_completion=negative_completion,
        )
    if observable_mode == "paired_selected_choice":
        return _paired_selected_choice_targets(
            safe_selected_choice_token_ids=safe_selected_choice_token_ids,
            exfil_selected_choice_token_ids=exfil_selected_choice_token_ids,
        )
    raise HiddenStatePatchError(f"Unsupported hidden-state patch observable_mode '{observable_mode}'.")


def _fixed_completion_targets(
    loaded_model: LoadedCausalLM,
    pair: HiddenStatePatchPromptPair,
    positive_completion: str,
    negative_completion: str,
) -> _PatchPairTargets:
    return _PatchPairTargets(
        safe_positive_target_token_ids=_completion_token_ids(
            loaded_model=loaded_model,
            prompt=pair.safe.text,
            completion=positive_completion,
        ),
        safe_negative_target_token_ids=_completion_token_ids(
            loaded_model=loaded_model,
            prompt=pair.safe.text,
            completion=negative_completion,
        ),
        exfil_positive_target_token_ids=_completion_token_ids(
            loaded_model=loaded_model,
            prompt=pair.exfil.text,
            completion=positive_completion,
        ),
        exfil_negative_target_token_ids=_completion_token_ids(
            loaded_model=loaded_model,
            prompt=pair.exfil.text,
            completion=negative_completion,
        ),
    )


def _paired_selected_choice_targets(
    safe_selected_choice_token_ids: tuple[int, ...],
    exfil_selected_choice_token_ids: tuple[int, ...],
) -> _PatchPairTargets:
    return _PatchPairTargets(
        safe_positive_target_token_ids=exfil_selected_choice_token_ids,
        safe_negative_target_token_ids=safe_selected_choice_token_ids,
        exfil_positive_target_token_ids=exfil_selected_choice_token_ids,
        exfil_negative_target_token_ids=safe_selected_choice_token_ids,
    )


def _selected_choice_indices(example: StructuredPromptExample) -> tuple[int, ...]:
    indices = example.selected_choice_readout_token_indices
    if indices is None:
        raise HiddenStatePatchError(f"Example '{example.id}' must include selected_choice_readout_token_indices.")
    return indices


def _donor_replacement_values(
    loaded_model: LoadedCausalLM,
    prompt: str,
    patch_layer_index: int,
    token_indices: tuple[int, ...],
) -> torch.Tensor:
    forward_pass = run_hidden_state_forward(loaded_model=loaded_model, prompt=prompt)
    hidden_state_index = _patched_hidden_state_index(patch_layer_index)
    if hidden_state_index >= len(forward_pass.hidden_states):
        raise HiddenStatePatchError(
            f"patch_layer_index {patch_layer_index} maps to hidden state index {hidden_state_index}, "
            f"but the model returned {len(forward_pass.hidden_states)} hidden states."
        )
    hidden_state = forward_pass.hidden_states[hidden_state_index]
    return hidden_state[0, list(token_indices), :].clone()


def _patched_hidden_state_index(patch_layer_index: int) -> int:
    return patch_layer_index + 1


def _success_rate(successes: tuple[bool, ...]) -> float:
    if len(successes) == 0:
        raise HiddenStatePatchError("Cannot compute success rate for zero pairs.")
    return float(sum(1 for success in successes if success)) / float(len(successes))


def _nested_attribute(owner: object, path: tuple[str, ...]) -> object | None:
    current = owner
    for attribute_name in path:
        if not hasattr(current, attribute_name):
            return None
        current = getattr(current, attribute_name)
    return current


def _is_indexable_layer_stack(value: object) -> bool:
    return hasattr(value, "__len__") and hasattr(value, "__getitem__")


def _validate_hidden_state_tensor(hidden_state: torch.Tensor) -> None:
    if hidden_state.ndim != 3:
        raise HiddenStatePatchError(f"hidden_state must be 3D, received shape {tuple(hidden_state.shape)}.")
    if int(hidden_state.shape[0]) != 1:
        raise HiddenStatePatchError("hidden_state batch size must be 1.")
    if int(hidden_state.shape[1]) < 1:
        raise HiddenStatePatchError("hidden_state sequence length must be positive.")
    if int(hidden_state.shape[2]) < 1:
        raise HiddenStatePatchError("hidden_state width must be positive.")


def _validate_token_indices(token_indices: tuple[int, ...], sequence_length: int) -> None:
    if len(token_indices) == 0:
        raise HiddenStatePatchError("token_indices must not be empty.")
    if tuple(sorted(token_indices)) != token_indices:
        raise HiddenStatePatchError("token_indices must be sorted.")
    if len(set(token_indices)) != len(token_indices):
        raise HiddenStatePatchError("token_indices must be unique.")
    for token_index in token_indices:
        if token_index < 0 or token_index >= sequence_length:
            raise HiddenStatePatchError(
                f"token_index {token_index} is out of range for sequence length {sequence_length}."
            )
