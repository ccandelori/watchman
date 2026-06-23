from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import torch

from aegis_introspection.activations import (
    HiddenStateForwardPass,
    final_token_activation,
    mean_pool_activation,
    readout_window_activation,
)

PoolingMethod: TypeAlias = Literal[
    "final_token",
    "mean_pool",
    "readout_window",
    "query_tail_window",
    "selected_choice_window",
    "combined_readout_window",
]

_VALID_POOLING_METHODS: frozenset[str] = frozenset(
    (
        "final_token",
        "mean_pool",
        "readout_window",
        "query_tail_window",
        "selected_choice_window",
        "combined_readout_window",
    )
)


class FeatureConfigError(ValueError):
    """Raised when feature extraction configuration is invalid."""


@dataclass(frozen=True)
class ActivationFeature:
    key: str
    layer_index: int
    pooling_method: PoolingMethod
    values: torch.Tensor


def parse_pooling_methods(raw_value: str) -> tuple[PoolingMethod, ...]:
    values = tuple(item.strip() for item in raw_value.split(",") if item.strip() != "")
    if len(values) == 0:
        raise FeatureConfigError("At least one pooling method is required.")

    methods: list[PoolingMethod] = []
    for value in values:
        if value not in _VALID_POOLING_METHODS:
            valid = ", ".join(sorted(_VALID_POOLING_METHODS))
            raise FeatureConfigError(f"Unknown pooling method '{value}'. Expected one of: {valid}.")
        methods.append(cast(PoolingMethod, value))
    return tuple(methods)


def parse_layer_indices(raw_value: str) -> tuple[int, ...]:
    values = tuple(item.strip() for item in raw_value.split(",") if item.strip() != "")
    if len(values) == 0:
        raise FeatureConfigError("At least one layer index is required.")

    indices: list[int] = []
    for value in values:
        try:
            indices.append(int(value))
        except ValueError as exc:
            raise FeatureConfigError(f"Layer index '{value}' is not an integer.") from exc
    return tuple(indices)


def normalize_layer_index(layer_index: int, layer_count: int) -> int:
    normalized = layer_index if layer_index >= 0 else layer_count + layer_index
    if normalized < 0 or normalized >= layer_count:
        raise FeatureConfigError(
            f"Layer index {layer_index} is out of range for hidden-state stack with {layer_count} layers."
        )
    return normalized


def build_feature_key(pooling_method: PoolingMethod, layer_index: int) -> str:
    return f"{pooling_method}_layer_{layer_index:02d}"


def extract_activation_features(
    forward_pass: HiddenStateForwardPass,
    layer_indices: tuple[int, ...],
    pooling_methods: tuple[PoolingMethod, ...],
    readout_token_indices: tuple[int, ...] | None,
    query_tail_readout_token_indices: tuple[int, ...] | None,
    selected_choice_readout_token_indices: tuple[int, ...] | None,
) -> tuple[ActivationFeature, ...]:
    layer_count = len(forward_pass.hidden_states)
    features: list[ActivationFeature] = []

    for requested_layer_index in layer_indices:
        layer_index = normalize_layer_index(requested_layer_index, layer_count)
        for pooling_method in pooling_methods:
            if pooling_method == "final_token":
                values = final_token_activation(forward_pass, layer_index)
            elif pooling_method == "mean_pool":
                values = mean_pool_activation(forward_pass, layer_index)
            elif pooling_method == "readout_window":
                if readout_token_indices is None:
                    raise FeatureConfigError("readout_token_indices are required for readout_window pooling.")
                values = readout_window_activation(forward_pass, layer_index, readout_token_indices)
            elif pooling_method == "query_tail_window":
                if query_tail_readout_token_indices is None:
                    raise FeatureConfigError(
                        "query_tail_readout_token_indices are required for query_tail_window pooling."
                    )
                values = readout_window_activation(forward_pass, layer_index, query_tail_readout_token_indices)
            elif pooling_method == "selected_choice_window":
                if selected_choice_readout_token_indices is None:
                    raise FeatureConfigError(
                        "selected_choice_readout_token_indices are required for selected_choice_window pooling."
                    )
                values = readout_window_activation(forward_pass, layer_index, selected_choice_readout_token_indices)
            elif pooling_method == "combined_readout_window":
                if readout_token_indices is None:
                    raise FeatureConfigError("readout_token_indices are required for combined_readout_window pooling.")
                if selected_choice_readout_token_indices is None:
                    raise FeatureConfigError(
                        "selected_choice_readout_token_indices are required for combined_readout_window pooling."
                    )
                combined_indices = tuple(
                    sorted(set(readout_token_indices).union(selected_choice_readout_token_indices))
                )
                values = readout_window_activation(forward_pass, layer_index, combined_indices)
            else:
                raise FeatureConfigError(f"Unsupported pooling method '{pooling_method}'.")

            features.append(
                ActivationFeature(
                    key=build_feature_key(pooling_method, layer_index),
                    layer_index=layer_index,
                    pooling_method=pooling_method,
                    values=values,
                )
            )

    return tuple(features)


def stack_feature_rows(feature_rows: tuple[tuple[ActivationFeature, ...], ...]) -> dict[str, torch.Tensor]:
    if len(feature_rows) == 0:
        raise FeatureConfigError("Cannot stack an empty feature row set.")

    expected_keys = tuple(feature.key for feature in feature_rows[0])
    if len(expected_keys) == 0:
        raise FeatureConfigError("Cannot stack feature rows with no features.")

    stacked: dict[str, torch.Tensor] = {}
    for row_index, feature_row in enumerate(feature_rows):
        row_keys = tuple(feature.key for feature in feature_row)
        if row_keys != expected_keys:
            raise FeatureConfigError(
                f"Feature row {row_index} has keys {row_keys}, but expected {expected_keys}."
            )

    for feature_index, key in enumerate(expected_keys):
        values: list[torch.Tensor] = []
        for feature_row in feature_rows:
            feature = feature_row[feature_index]
            values.append(feature.values.squeeze(0).cpu())
        stacked[key] = torch.stack(values, dim=0)

    return stacked
