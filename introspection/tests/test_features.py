import unittest

import torch

from aegis_introspection.activations import HiddenStateForwardPass
from aegis_introspection.features import (
    FeatureConfigError,
    extract_activation_features,
    parse_layer_indices,
    parse_pooling_methods,
    stack_feature_rows,
)


class FeatureExtractionTest(unittest.TestCase):
    def test_parse_layer_indices_accepts_comma_separated_integers(self) -> None:
        self.assertEqual((0, 7, -1), parse_layer_indices("0,7,-1"))

    def test_parse_layer_indices_rejects_non_integer(self) -> None:
        with self.assertRaises(FeatureConfigError):
            parse_layer_indices("0,last")

    def test_parse_pooling_methods_accepts_supported_methods(self) -> None:
        self.assertEqual(
            ("final_token", "mean_pool", "readout_window"),
            parse_pooling_methods("final_token,mean_pool,readout_window"),
        )

    def test_parse_pooling_methods_rejects_unknown_method(self) -> None:
        with self.assertRaises(FeatureConfigError):
            parse_pooling_methods("cls_token")

    def test_extract_activation_features_returns_selected_layers_and_methods(self) -> None:
        forward_pass = HiddenStateForwardPass(
            prompt="prompt",
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 0]]),
            hidden_states=(
                torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
                torch.tensor([[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]),
            ),
        )

        features = extract_activation_features(
            forward_pass=forward_pass,
            layer_indices=(0, -1),
            pooling_methods=("final_token", "mean_pool"),
            readout_token_indices=None,
        )

        self.assertEqual(
            (
                "final_token_layer_00",
                "mean_pool_layer_00",
                "final_token_layer_01",
                "mean_pool_layer_01",
            ),
            tuple(feature.key for feature in features),
        )

    def test_extract_activation_features_supports_readout_window_pooling(self) -> None:
        forward_pass = HiddenStateForwardPass(
            prompt="prompt",
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            attention_mask=torch.tensor([[1, 1, 1, 1]]),
            hidden_states=(torch.tensor([[[1.0, 2.0], [3.0, 6.0], [5.0, 10.0], [7.0, 14.0]]]),),
        )

        features = extract_activation_features(
            forward_pass=forward_pass,
            layer_indices=(0,),
            pooling_methods=("readout_window",),
            readout_token_indices=(1, 2),
        )

        self.assertEqual(("readout_window_layer_00",), tuple(feature.key for feature in features))
        torch.testing.assert_close(torch.tensor([[4.0, 8.0]]), features[0].values)

    def test_extract_activation_features_requires_indices_for_readout_window_pooling(self) -> None:
        forward_pass = HiddenStateForwardPass(
            prompt="prompt",
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=None,
            hidden_states=(torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),),
        )

        with self.assertRaises(FeatureConfigError):
            extract_activation_features(
                forward_pass=forward_pass,
                layer_indices=(0,),
                pooling_methods=("readout_window",),
                readout_token_indices=None,
            )

    def test_stack_feature_rows_returns_feature_matrices(self) -> None:
        first_pass = HiddenStateForwardPass(
            prompt="first",
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=None,
            hidden_states=(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),),
        )
        second_pass = HiddenStateForwardPass(
            prompt="second",
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=None,
            hidden_states=(torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),),
        )

        first_features = extract_activation_features(
            forward_pass=first_pass,
            layer_indices=(0,),
            pooling_methods=("final_token", "mean_pool"),
            readout_token_indices=None,
        )
        second_features = extract_activation_features(
            forward_pass=second_pass,
            layer_indices=(0,),
            pooling_methods=("final_token", "mean_pool"),
            readout_token_indices=None,
        )

        stacked = stack_feature_rows((first_features, second_features))

        torch.testing.assert_close(
            torch.tensor([[3.0, 4.0], [7.0, 8.0]]),
            stacked["final_token_layer_00"],
        )
        torch.testing.assert_close(
            torch.tensor([[2.0, 3.0], [6.0, 7.0]]),
            stacked["mean_pool_layer_00"],
        )


if __name__ == "__main__":
    unittest.main()
