import unittest

import torch
from aegis_introspection.activations import (
    HiddenStateForwardPass,
    ReadoutWindowError,
    final_token_activation,
    mean_pool_activation,
    readout_window_activation,
    summarize_hidden_states,
)


class ActivationHelpersTest(unittest.TestCase):
    def test_summarize_hidden_states_reports_layer_metadata(self) -> None:
        hidden_states = (
            torch.zeros((1, 3, 2), dtype=torch.float32),
            torch.ones((1, 3, 2), dtype=torch.float32),
        )

        summaries = summarize_hidden_states(hidden_states)

        self.assertEqual(2, len(summaries))
        self.assertEqual(0, summaries[0].layer_index)
        self.assertEqual((1, 3, 2), summaries[0].shape)
        self.assertEqual("torch.float32", summaries[0].dtype)

    def test_final_token_activation_returns_last_token_vector(self) -> None:
        forward_pass = HiddenStateForwardPass(
            prompt="prompt",
            source_input_device="cpu",
            source_hidden_state_devices=("cpu",),
            source_hidden_state_dtypes=("torch.float32",),
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 0]]),
            hidden_states=(torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),),
        )

        activation = final_token_activation(forward_pass, 0)

        torch.testing.assert_close(torch.tensor([[5.0, 6.0]]), activation)

    def test_mean_pool_activation_uses_attention_mask(self) -> None:
        forward_pass = HiddenStateForwardPass(
            prompt="prompt",
            source_input_device="cpu",
            source_hidden_state_devices=("cpu",),
            source_hidden_state_dtypes=("torch.float32",),
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 0]]),
            hidden_states=(torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),),
        )

        activation = mean_pool_activation(forward_pass, 0)

        torch.testing.assert_close(torch.tensor([[2.0, 3.0]]), activation)

    def test_mean_pool_activation_without_mask_uses_all_tokens(self) -> None:
        forward_pass = HiddenStateForwardPass(
            prompt="prompt",
            source_input_device="cpu",
            source_hidden_state_devices=("cpu",),
            source_hidden_state_dtypes=("torch.float32",),
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=None,
            hidden_states=(torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),),
        )

        activation = mean_pool_activation(forward_pass, 0)

        torch.testing.assert_close(torch.tensor([[3.0, 4.0]]), activation)

    def test_readout_window_activation_pools_selected_tokens(self) -> None:
        forward_pass = HiddenStateForwardPass(
            prompt="prompt",
            source_input_device="cpu",
            source_hidden_state_devices=("cpu",),
            source_hidden_state_dtypes=("torch.float32",),
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            attention_mask=torch.tensor([[1, 1, 1, 1]]),
            hidden_states=(torch.tensor([[[1.0, 2.0], [3.0, 6.0], [5.0, 10.0], [7.0, 14.0]]]),),
        )

        activation = readout_window_activation(forward_pass, 0, (1, 2))

        torch.testing.assert_close(torch.tensor([[4.0, 8.0]]), activation)

    def test_readout_window_activation_rejects_empty_window(self) -> None:
        forward_pass = HiddenStateForwardPass(
            prompt="prompt",
            source_input_device="cpu",
            source_hidden_state_devices=("cpu",),
            source_hidden_state_dtypes=("torch.float32",),
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=None,
            hidden_states=(torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),),
        )

        with self.assertRaises(ReadoutWindowError):
            readout_window_activation(forward_pass, 0, ())

    def test_readout_window_activation_rejects_out_of_range_index(self) -> None:
        forward_pass = HiddenStateForwardPass(
            prompt="prompt",
            source_input_device="cpu",
            source_hidden_state_devices=("cpu",),
            source_hidden_state_dtypes=("torch.float32",),
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=None,
            hidden_states=(torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),),
        )

        with self.assertRaises(ReadoutWindowError):
            readout_window_activation(forward_pass, 0, (1, 3))


if __name__ == "__main__":
    unittest.main()
