from __future__ import annotations

import unittest

import torch
from aegis_introspection.cift_hidden_state_patching import (
    HiddenStatePatchError,
    HiddenStatePatchPairingKey,
    HiddenStatePatchReportConfig,
    SelectedChoiceHiddenStatePatchReportConfig,
    evaluate_hidden_state_patch_report,
    evaluate_selected_choice_hidden_state_patch_report,
    hidden_state_patch_report_to_json,
    model_output_log_probability_margin,
    model_output_token_sequence_log_probability_margin,
    pair_structured_prompt_examples,
    patch_hidden_state_tensor,
    patch_layer_output,
    patched_layer_output,
    run_hidden_state_forward_with_patch,
    transformer_layer_module,
)
from aegis_introspection.model_loader import DeviceSelection, LoadedCausalLM
from aegis_introspection.prompts import PromptTokenSpan, StructuredPromptExample
from torch import nn
from transformers import BatchEncoding


class CiftHiddenStatePatchingTest(unittest.TestCase):
    def test_patch_hidden_state_tensor_replaces_only_selected_token_rows(self) -> None:
        hidden_state = torch.zeros((1, 4, 3), dtype=torch.float32)
        replacement = torch.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=torch.float32)

        patched = patch_hidden_state_tensor(
            hidden_state=hidden_state,
            token_indices=(1, 3),
            replacement_values=replacement,
        )

        self.assertEqual((1, 4, 3), tuple(patched.shape))
        self.assertEqual([0.0, 0.0, 0.0], patched[0, 0, :].tolist())
        self.assertEqual([1.0, 2.0, 3.0], patched[0, 1, :].tolist())
        self.assertEqual([0.0, 0.0, 0.0], patched[0, 2, :].tolist())
        self.assertEqual([4.0, 5.0, 6.0], patched[0, 3, :].tolist())
        self.assertEqual(0.0, float(hidden_state.sum()))

    def test_patch_hidden_state_tensor_rejects_mismatched_replacement_shape(self) -> None:
        hidden_state = torch.zeros((1, 4, 3), dtype=torch.float32)
        replacement = torch.zeros((1, 3), dtype=torch.float32)

        with self.assertRaisesRegex(HiddenStatePatchError, "replacement_values shape"):
            patch_hidden_state_tensor(
                hidden_state=hidden_state,
                token_indices=(1, 3),
                replacement_values=replacement,
            )

    def test_transformer_layer_module_resolves_qwen_style_model_layers(self) -> None:
        model = _FakeQwenCausalLm()

        layer = transformer_layer_module(model, layer_index=1)

        self.assertIs(layer, model.model.layers[1])

    def test_patch_layer_output_preserves_huggingface_tuple_payload(self) -> None:
        hidden_state = torch.zeros((1, 4, 2), dtype=torch.float32)
        replacement = torch.asarray(((8.0, 9.0),), dtype=torch.float32)
        cache = object()

        patched = patch_layer_output(
            output=(hidden_state, cache),
            token_indices=(2,),
            replacement_values=replacement,
        )

        self.assertIsInstance(patched, tuple)
        assert isinstance(patched, tuple)
        self.assertIs(patched[1], cache)
        self.assertEqual([8.0, 9.0], patched[0][0, 2, :].tolist())

    def test_patched_layer_output_context_removes_hook_after_exit(self) -> None:
        layer = _IdentityLayer()
        replacement = torch.asarray(((7.0, 7.0),), dtype=torch.float32)

        with patched_layer_output(layer=layer, token_indices=(1,), replacement_values=replacement):
            patched = layer(torch.zeros((1, 3, 2), dtype=torch.float32))

        unpatched = layer(torch.zeros((1, 3, 2), dtype=torch.float32))

        self.assertEqual([7.0, 7.0], patched[0, 1, :].tolist())
        self.assertEqual(0.0, float(unpatched.sum()))

    def test_run_hidden_state_forward_with_patch_returns_patched_hidden_states(self) -> None:
        loaded_model = LoadedCausalLM(
            model_id="fake",
            revision="main",
            device=DeviceSelection(name="cpu", torch_device=torch.device("cpu"), torch_dtype=torch.float32),
            tokenizer=_FakeTokenizer(),
            model=_FakeCausalLm(),
        )
        replacement = torch.asarray(((3.0, 4.0),), dtype=torch.float32)

        forward_pass = run_hidden_state_forward_with_patch(
            loaded_model=loaded_model,
            prompt="fake prompt",
            layer_index=0,
            token_indices=(1,),
            replacement_values=replacement,
        )

        self.assertEqual("fake prompt", forward_pass.prompt)
        self.assertEqual([3.0, 4.0], forward_pass.hidden_states[1][0, 1, :].tolist())

    def test_pair_structured_prompt_examples_requires_exact_safe_exfil_pairs(self) -> None:
        pairs = pair_structured_prompt_examples(
            (
                _example(
                    example_id="safe-a",
                    label="secret_present_safe",
                    task="alpha",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
                _example(
                    example_id="exfil-a",
                    label="exfiltration_intent",
                    task="alpha",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
            )
        )

        self.assertEqual(1, len(pairs))
        self.assertEqual(
            HiddenStatePatchPairingKey(
                participant="codex",
                task="alpha",
                family="alpha_credentials",
                variant="v000",
                credential_type="synthetic",
            ),
            pairs[0].key,
        )
        self.assertEqual("safe-a", pairs[0].safe.id)
        self.assertEqual("exfil-a", pairs[0].exfil.id)

    def test_pair_structured_prompt_examples_rejects_missing_selected_choice_indices(self) -> None:
        with self.assertRaisesRegex(HiddenStatePatchError, "selected_choice_readout_token_indices"):
            pair_structured_prompt_examples(
                (
                    _example(
                        example_id="safe-a",
                        label="secret_present_safe",
                        task="alpha",
                        variant="v000",
                        selected_choice_readout_token_indices=None,
                    ),
                    _example(
                        example_id="exfil-a",
                        label="exfiltration_intent",
                        task="alpha",
                        variant="v000",
                        selected_choice_readout_token_indices=(1,),
                    ),
                )
            )

    def test_model_output_log_probability_margin_tracks_block_minus_allow_completion_logprob(self) -> None:
        loaded_model = _loaded_fake_causal_lm()

        safe_margin = model_output_log_probability_margin(
            loaded_model=loaded_model,
            prompt="safe prompt",
            positive_completion=" block",
            negative_completion=" allow",
            patch_spec=None,
        )
        exfil_margin = model_output_log_probability_margin(
            loaded_model=loaded_model,
            prompt="exfil prompt",
            positive_completion=" block",
            negative_completion=" allow",
            patch_spec=None,
        )

        self.assertLess(safe_margin.margin, 0.0)
        self.assertGreater(exfil_margin.margin, 0.0)

    def test_evaluate_hidden_state_patch_report_records_directional_margin_shifts(self) -> None:
        loaded_model = _loaded_fake_causal_lm()

        report = evaluate_hidden_state_patch_report(
            loaded_model=loaded_model,
            examples=(
                _example(
                    example_id="safe-a",
                    label="secret_present_safe",
                    task="alpha",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
                _example(
                    example_id="exfil-a",
                    label="exfiltration_intent",
                    task="alpha",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
            ),
            config=HiddenStatePatchReportConfig(
                report_id="synthetic-hidden-state-patching",
                patch_layer_index=0,
                positive_completion=" block",
                negative_completion=" allow",
                minimum_margin_shift=1.0,
                max_pairs=None,
                created_at="2026-06-24T00:00:00Z",
            ),
        )

        self.assertEqual("aegis_introspection.cift_hidden_state_patching/v2", report.schema_version)
        self.assertTrue(report.transformer_hidden_state_patching)
        self.assertEqual("fixed_completion", report.observable_mode)
        self.assertEqual("transformer_layer_output_replacement", report.intervention_type)
        self.assertEqual("model_output_log_probability_margin", report.claim_scope)
        self.assertEqual(1, report.candidate_pair_count)
        self.assertEqual(1, report.eligible_pair_count)
        self.assertEqual(1, report.pair_count)
        self.assertEqual(0, report.skipped_pair_count)
        self.assertEqual(0, report.truncated_pair_count)
        self.assertEqual(1.0, report.safe_to_exfil_success_rate)
        self.assertEqual(1.0, report.exfil_to_safe_success_rate)
        self.assertTrue(report.directional_intervention_passed)
        self.assertTrue(report.coverage_complete)
        self.assertTrue(report.passed)
        self.assertTrue(report.pairs[0].original_polarity_correct)
        self.assertTrue(report.pairs[0].patched_polarity_flipped)
        self.assertEqual((1,), report.pairs[0].safe_selected_choice_token_indices)
        self.assertEqual((1,), report.pairs[0].exfil_selected_choice_token_indices)
        self.assertEqual((100,), report.pairs[0].safe_selected_choice_token_ids)
        self.assertEqual((200,), report.pairs[0].exfil_selected_choice_token_ids)
        self.assertEqual((11,), report.pairs[0].safe_positive_target_token_ids)
        self.assertEqual((10,), report.pairs[0].safe_negative_target_token_ids)
        self.assertGreater(report.pairs[0].safe_to_exfil_margin_shift, 1.0)
        self.assertGreater(report.pairs[0].exfil_to_safe_margin_shift, 1.0)

        encoded = hidden_state_patch_report_to_json(report)

        self.assertEqual("aegis_introspection.cift_hidden_state_patching/v2", encoded["schema_version"])
        self.assertEqual("transformer_layer_output_replacement", encoded["intervention_type"])
        self.assertEqual("model_output_log_probability_margin", encoded["claim_scope"])
        self.assertEqual(True, encoded["transformer_hidden_state_patching"])
        self.assertEqual("fixed_completion", encoded["observable_mode"])
        self.assertEqual(1, encoded["candidate_pair_count"])
        self.assertEqual(1, encoded["eligible_pair_count"])
        self.assertEqual(1, encoded["pair_count"])
        self.assertEqual(0, encoded["skipped_pair_count"])
        self.assertEqual(0, encoded["truncated_pair_count"])
        self.assertEqual(True, encoded["passed"])

    def test_evaluate_hidden_state_patch_report_cannot_pass_on_truncated_slice(self) -> None:
        loaded_model = _loaded_fake_causal_lm()

        report = evaluate_hidden_state_patch_report(
            loaded_model=loaded_model,
            examples=(
                _example(
                    example_id="safe-a",
                    label="secret_present_safe",
                    task="alpha",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
                _example(
                    example_id="exfil-a",
                    label="exfiltration_intent",
                    task="alpha",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
                _example(
                    example_id="safe-b",
                    label="secret_present_safe",
                    task="beta",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
                _example(
                    example_id="exfil-b",
                    label="exfiltration_intent",
                    task="beta",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
            ),
            config=HiddenStatePatchReportConfig(
                report_id="synthetic-hidden-state-patching",
                patch_layer_index=0,
                positive_completion=" block",
                negative_completion=" allow",
                minimum_margin_shift=1.0,
                max_pairs=1,
                created_at="2026-06-24T00:00:00Z",
            ),
        )

        self.assertEqual(2, report.candidate_pair_count)
        self.assertEqual(2, report.eligible_pair_count)
        self.assertEqual(1, report.pair_count)
        self.assertEqual(1, report.truncated_pair_count)
        self.assertFalse(report.coverage_complete)
        self.assertTrue(report.directional_intervention_passed)
        self.assertFalse(report.passed)

    def test_evaluate_hidden_state_patch_report_records_unequal_token_count_skips(self) -> None:
        loaded_model = _loaded_fake_causal_lm()

        report = evaluate_hidden_state_patch_report(
            loaded_model=loaded_model,
            examples=(
                _example(
                    example_id="safe-a",
                    label="secret_present_safe",
                    task="alpha",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
                _example(
                    example_id="exfil-a",
                    label="exfiltration_intent",
                    task="alpha",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
                _example(
                    example_id="safe-b",
                    label="secret_present_safe",
                    task="beta",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
                _example(
                    example_id="exfil-b",
                    label="exfiltration_intent",
                    task="beta",
                    variant="v000",
                    selected_choice_readout_token_indices=(1, 2),
                ),
            ),
            config=HiddenStatePatchReportConfig(
                report_id="synthetic-hidden-state-patching",
                patch_layer_index=0,
                positive_completion=" block",
                negative_completion=" allow",
                minimum_margin_shift=1.0,
                max_pairs=None,
                created_at="2026-06-24T00:00:00Z",
            ),
        )

        self.assertEqual(2, report.candidate_pair_count)
        self.assertEqual(1, report.eligible_pair_count)
        self.assertEqual(1, report.pair_count)
        self.assertEqual(1, report.skipped_pair_count)
        self.assertEqual("safe-b", report.skipped_pairs[0].safe_example_id)
        self.assertEqual("exfil-b", report.skipped_pairs[0].exfil_example_id)
        self.assertEqual("unequal_selected_choice_readout_token_count", report.skipped_pairs[0].reason)
        self.assertFalse(report.coverage_complete)
        self.assertFalse(report.passed)

    def test_evaluate_hidden_state_patch_report_requires_preference_flips_not_shift_only(self) -> None:
        loaded_model = LoadedCausalLM(
            model_id="fake",
            revision="main",
            device=DeviceSelection(name="cpu", torch_device=torch.device("cpu"), torch_dtype=torch.float32),
            tokenizer=_FakeTokenizer(),
            model=_ShiftWithoutFlipCausalLm(),
        )

        report = evaluate_hidden_state_patch_report(
            loaded_model=loaded_model,
            examples=(
                _example(
                    example_id="safe-a",
                    label="secret_present_safe",
                    task="alpha",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
                _example(
                    example_id="exfil-a",
                    label="exfiltration_intent",
                    task="alpha",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
            ),
            config=HiddenStatePatchReportConfig(
                report_id="synthetic-hidden-state-patching",
                patch_layer_index=0,
                positive_completion=" block",
                negative_completion=" allow",
                minimum_margin_shift=1.0,
                max_pairs=None,
                created_at="2026-06-24T00:00:00Z",
            ),
        )

        self.assertTrue(report.pairs[0].original_polarity_correct)
        self.assertFalse(report.pairs[0].patched_polarity_flipped)
        self.assertGreater(report.pairs[0].safe_to_exfil_margin_shift, 1.0)
        self.assertGreater(report.pairs[0].exfil_to_safe_margin_shift, 1.0)
        self.assertEqual(0.0, report.safe_to_exfil_success_rate)
        self.assertEqual(0.0, report.exfil_to_safe_success_rate)
        self.assertFalse(report.directional_intervention_passed)
        self.assertFalse(report.passed)

    def test_token_sequence_margin_scores_selected_choice_token_ids(self) -> None:
        loaded_model = _loaded_fake_causal_lm()

        safe_margin = model_output_token_sequence_log_probability_margin(
            loaded_model=loaded_model,
            prompt="safe prompt",
            positive_target_token_ids=(200,),
            negative_target_token_ids=(100,),
            patch_spec=None,
        )
        exfil_margin = model_output_token_sequence_log_probability_margin(
            loaded_model=loaded_model,
            prompt="exfil prompt",
            positive_target_token_ids=(200,),
            negative_target_token_ids=(100,),
            patch_spec=None,
        )

        self.assertLess(safe_margin.margin, 0.0)
        self.assertGreater(exfil_margin.margin, 0.0)

    def test_selected_choice_hidden_state_patch_report_uses_pair_choice_tokens(self) -> None:
        loaded_model = _loaded_fake_causal_lm()

        report = evaluate_selected_choice_hidden_state_patch_report(
            loaded_model=loaded_model,
            examples=(
                _example(
                    example_id="safe-a",
                    label="secret_present_safe",
                    task="alpha",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
                _example(
                    example_id="exfil-a",
                    label="exfiltration_intent",
                    task="alpha",
                    variant="v000",
                    selected_choice_readout_token_indices=(1,),
                ),
            ),
            config=SelectedChoiceHiddenStatePatchReportConfig(
                report_id="synthetic-selected-choice-hidden-state-patching",
                patch_layer_index=0,
                minimum_margin_shift=1.0,
                max_pairs=None,
                created_at="2026-06-24T00:00:00Z",
            ),
        )

        self.assertEqual("paired_selected_choice", report.observable_mode)
        self.assertEqual("paired_exfil_selected_choice_tokens", report.positive_completion)
        self.assertEqual("paired_safe_selected_choice_tokens", report.negative_completion)
        self.assertEqual((200,), report.pairs[0].safe_positive_target_token_ids)
        self.assertEqual((100,), report.pairs[0].safe_negative_target_token_ids)
        self.assertEqual((200,), report.pairs[0].exfil_positive_target_token_ids)
        self.assertEqual((100,), report.pairs[0].exfil_negative_target_token_ids)
        self.assertTrue(report.pairs[0].original_polarity_correct)
        self.assertTrue(report.pairs[0].patched_polarity_flipped)
        self.assertEqual(1.0, report.safe_to_exfil_success_rate)
        self.assertEqual(1.0, report.exfil_to_safe_success_rate)
        self.assertTrue(report.passed)


class _FakeQwenCausalLm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeQwenBackbone()


class _FakeQwenBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList((_IdentityLayer(), _IdentityLayer()))


class _IdentityLayer(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class _FakeTokenizer:
    def __call__(self, prompt: str, return_tensors: str) -> BatchEncoding:
        if prompt == "" or return_tensors != "pt":
            raise ValueError("unexpected tokenizer input")
        token_ids: list[int]
        if prompt.startswith("safe"):
            token_ids = [1, 100, 3]
        elif prompt.startswith("exfil"):
            token_ids = [1, 200, 3]
        else:
            token_ids = [1, 2, 3]
        if prompt.endswith(" block"):
            token_ids.append(11)
        elif prompt.endswith(" allow"):
            token_ids.append(10)
        return BatchEncoding(
            {
                "input_ids": torch.asarray([token_ids], dtype=torch.int64),
                "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.int64),
            }
        )


class _FakeCausalLm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeQwenBackbone()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
    ) -> object:
        if not output_hidden_states or use_cache:
            raise ValueError("unexpected fake model flags")
        hidden_0 = torch.zeros((input_ids.shape[0], input_ids.shape[1], 2), dtype=torch.float32)
        hidden_0[:, 1, 0] = torch.where(input_ids[:, 1] == 200, 2.0, -2.0)
        hidden_1 = self.model.layers[0](hidden_0)
        hidden_2 = self.model.layers[1](hidden_1 + attention_mask.to(dtype=torch.float32).unsqueeze(-1))
        signal = hidden_2[:, 1, 0]
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 256), dtype=torch.float32)
        logits[:, :, 10] = -signal.unsqueeze(-1)
        logits[:, :, 11] = signal.unsqueeze(-1)
        logits[:, :, 100] = -signal.unsqueeze(-1)
        logits[:, :, 200] = signal.unsqueeze(-1)
        return _FakeCausalLmOutput(hidden_states=(hidden_0, hidden_1, hidden_2), logits=logits)


class _ShiftWithoutFlipCausalLm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeQwenBackbone()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
    ) -> object:
        if not output_hidden_states or use_cache:
            raise ValueError("unexpected fake model flags")
        hidden_0 = torch.zeros((input_ids.shape[0], input_ids.shape[1], 2), dtype=torch.float32)
        hidden_0[:, 1, 0] = torch.where(input_ids[:, 1] == 200, 4.0, -4.0)
        hidden_1 = self.model.layers[0](hidden_0)
        hidden_2_input = hidden_1.clone()
        original_prompt_bias = torch.where(input_ids[:, 1] == 100, -5.0, 5.0)
        hidden_2_input[:, 1, 0] = hidden_2_input[:, 1, 0] + original_prompt_bias
        hidden_2 = self.model.layers[1](hidden_2_input + attention_mask.to(dtype=torch.float32).unsqueeze(-1))
        signal = hidden_2[:, 1, 0]
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 256), dtype=torch.float32)
        logits[:, :, 10] = -signal.unsqueeze(-1)
        logits[:, :, 11] = signal.unsqueeze(-1)
        logits[:, :, 100] = -signal.unsqueeze(-1)
        logits[:, :, 200] = signal.unsqueeze(-1)
        return _FakeCausalLmOutput(hidden_states=(hidden_0, hidden_1, hidden_2), logits=logits)


class _FakeCausalLmOutput:
    def __init__(self, hidden_states: tuple[torch.Tensor, ...], logits: torch.Tensor) -> None:
        self.hidden_states = hidden_states
        self.logits = logits


def _loaded_fake_causal_lm() -> LoadedCausalLM:
    return LoadedCausalLM(
        model_id="fake",
        revision="main",
        device=DeviceSelection(name="cpu", torch_device=torch.device("cpu"), torch_dtype=torch.float32),
        tokenizer=_FakeTokenizer(),
        model=_FakeCausalLm(),
    )


def _example(
    example_id: str,
    label: str,
    task: str,
    variant: str,
    selected_choice_readout_token_indices: tuple[int, ...] | None,
) -> StructuredPromptExample:
    return StructuredPromptExample(
        id=example_id,
        label=label,
        family=f"{task}_credentials",
        text=f"{'safe' if label == 'secret_present_safe' else 'exfil'} prompt",
        tags=(
            "trace_collection",
            f"label:{label}",
            f"family:{task}_credentials",
            f"task:{task}",
            "participant:codex",
            f"variant:{variant}",
            "credential_type:synthetic",
        ),
        secret_token_span=PromptTokenSpan(start=0, end=1),
        query_token_span=PromptTokenSpan(start=1, end=3),
        payload_token_span=None,
        readout_token_indices=(1,),
        query_tail_readout_token_indices=None,
        selected_choice_token_span=PromptTokenSpan(start=1, end=2)
        if selected_choice_readout_token_indices is not None
        else None,
        selected_choice_readout_token_indices=selected_choice_readout_token_indices,
        fallback_reason=None,
    )


if __name__ == "__main__":
    unittest.main()
