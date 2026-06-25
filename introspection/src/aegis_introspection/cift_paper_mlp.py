from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatMatrix: TypeAlias = NDArray[np.float32]
FloatVector: TypeAlias = NDArray[np.float64]
IntVector: TypeAlias = NDArray[np.int64]
ProbabilityMatrix: TypeAlias = NDArray[np.float64]

_PAPER_HIDDEN_LAYER_SIZES = (128, 64)


class CiftPaperMlpError(ValueError):
    """Raised when the paper-faithful CIFT MLP cannot be trained or scored."""


@dataclass(frozen=True)
class CiftPaperMlpConfig:
    input_dim: int
    hidden_layer_sizes: tuple[int, int]
    learning_rate: float
    max_epochs: int
    batch_size: int
    l1_softplus_weight: float
    random_seed: int


@dataclass(frozen=True)
class CiftPaperMlpParameters:
    raw_layer_weights: FloatVector
    first_weights: NDArray[np.float64]
    first_bias: FloatVector
    second_weights: NDArray[np.float64]
    second_bias: FloatVector
    output_weights: NDArray[np.float64]
    output_bias: FloatVector


@dataclass(frozen=True)
class _MlpParameters:
    raw_layer_weights: FloatVector
    first_weights: NDArray[np.float64]
    first_bias: FloatVector
    second_weights: NDArray[np.float64]
    second_bias: FloatVector
    output_weights: NDArray[np.float64]
    output_bias: FloatVector


@dataclass(frozen=True)
class _ForwardPass:
    layer_weights: FloatVector
    weighted_inputs: NDArray[np.float64]
    first_logits: NDArray[np.float64]
    first_activations: NDArray[np.float64]
    second_logits: NDArray[np.float64]
    second_activations: NDArray[np.float64]
    output_logits: FloatVector
    probabilities: FloatVector


@dataclass(frozen=True)
class _MlpGradients:
    raw_layer_weights: FloatVector
    first_weights: NDArray[np.float64]
    first_bias: FloatVector
    second_weights: NDArray[np.float64]
    second_bias: FloatVector
    output_weights: NDArray[np.float64]
    output_bias: FloatVector


class CiftPaperMlpClassifier:
    classes_: IntVector

    def __init__(self, config: CiftPaperMlpConfig) -> None:
        _validate_config(config)
        self._config = config
        self._parameters = _initial_parameters(config=config)
        self.classes_ = np.asarray((0, 1), dtype=np.int64)
        self._fitted = False

    def fit(self, matrix: FloatMatrix, labels: IntVector) -> CiftPaperMlpClassifier:
        feature_matrix = _validated_matrix(matrix=matrix, input_dim=self._config.input_dim)
        label_vector = _validated_labels(labels=labels, row_count=feature_matrix.shape[0]).astype(np.float64)
        self._parameters = _initial_parameters(config=self._config)
        rng = np.random.default_rng(self._config.random_seed)
        row_count = int(feature_matrix.shape[0])
        batch_size = min(self._config.batch_size, row_count)

        for _epoch in range(self._config.max_epochs):
            permutation = rng.permutation(row_count)
            for start_index in range(0, row_count, batch_size):
                batch_indices = permutation[start_index : start_index + batch_size]
                gradients = _gradients(
                    parameters=self._parameters,
                    matrix=feature_matrix[batch_indices].astype(np.float64, copy=False),
                    labels=label_vector[batch_indices],
                    l1_softplus_weight=self._config.l1_softplus_weight,
                )
                self._parameters = _apply_gradients(
                    parameters=self._parameters,
                    gradients=gradients,
                    learning_rate=self._config.learning_rate,
                )

        self._fitted = True
        return self

    def predict_proba(self, matrix: FloatMatrix) -> ProbabilityMatrix:
        if not self._fitted:
            raise CiftPaperMlpError("CIFT paper MLP must be fitted before predict_proba.")
        feature_matrix = _validated_matrix(matrix=matrix, input_dim=self._config.input_dim)
        probabilities = _forward(parameters=self._parameters, matrix=feature_matrix.astype(np.float64)).probabilities
        return np.stack((1.0 - probabilities, probabilities), axis=1).astype(np.float64, copy=False)

    def softplus_layer_weights(self) -> FloatMatrix:
        return _softplus(self._parameters.raw_layer_weights).astype(np.float32, copy=True)

    def runtime_parameters(self) -> CiftPaperMlpParameters:
        if not self._fitted:
            raise CiftPaperMlpError("CIFT paper MLP must be fitted before exporting runtime parameters.")
        return CiftPaperMlpParameters(
            raw_layer_weights=self._parameters.raw_layer_weights.astype(np.float64, copy=True),
            first_weights=self._parameters.first_weights.astype(np.float64, copy=True),
            first_bias=self._parameters.first_bias.astype(np.float64, copy=True),
            second_weights=self._parameters.second_weights.astype(np.float64, copy=True),
            second_bias=self._parameters.second_bias.astype(np.float64, copy=True),
            output_weights=self._parameters.output_weights.astype(np.float64, copy=True),
            output_bias=self._parameters.output_bias.astype(np.float64, copy=True),
        )


def paper_mlp_parameter_count(input_dim: int, hidden_layer_sizes: tuple[int, int]) -> int:
    if input_dim < 1:
        raise CiftPaperMlpError("input_dim must be at least 1.")
    if hidden_layer_sizes[0] < 1 or hidden_layer_sizes[1] < 1:
        raise CiftPaperMlpError("hidden_layer_sizes must contain positive widths.")
    first_hidden, second_hidden = hidden_layer_sizes
    return input_dim * first_hidden + first_hidden + first_hidden * second_hidden + second_hidden + second_hidden + 1


def _initial_parameters(config: CiftPaperMlpConfig) -> _MlpParameters:
    rng = np.random.default_rng(config.random_seed)
    first_hidden, second_hidden = config.hidden_layer_sizes
    return _MlpParameters(
        raw_layer_weights=np.zeros(config.input_dim, dtype=np.float64),
        first_weights=_xavier_matrix(rng=rng, input_dim=config.input_dim, output_dim=first_hidden),
        first_bias=np.zeros(first_hidden, dtype=np.float64),
        second_weights=_xavier_matrix(rng=rng, input_dim=first_hidden, output_dim=second_hidden),
        second_bias=np.zeros(second_hidden, dtype=np.float64),
        output_weights=_xavier_matrix(rng=rng, input_dim=second_hidden, output_dim=1),
        output_bias=np.zeros(1, dtype=np.float64),
    )


def _xavier_matrix(rng: np.random.Generator, input_dim: int, output_dim: int) -> NDArray[np.float64]:
    limit = np.sqrt(6.0 / float(input_dim + output_dim))
    return rng.uniform(low=-limit, high=limit, size=(input_dim, output_dim)).astype(np.float64)


def _forward(parameters: _MlpParameters, matrix: NDArray[np.float64]) -> _ForwardPass:
    layer_weights = _softplus(parameters.raw_layer_weights)
    weighted_inputs = matrix * layer_weights
    first_logits = weighted_inputs @ parameters.first_weights + parameters.first_bias
    first_activations = _relu(first_logits)
    second_logits = first_activations @ parameters.second_weights + parameters.second_bias
    second_activations = _relu(second_logits)
    output_logits = (second_activations @ parameters.output_weights + parameters.output_bias).reshape(-1)
    probabilities = _sigmoid(output_logits)
    return _ForwardPass(
        layer_weights=layer_weights,
        weighted_inputs=weighted_inputs,
        first_logits=first_logits,
        first_activations=first_activations,
        second_logits=second_logits,
        second_activations=second_activations,
        output_logits=output_logits,
        probabilities=probabilities,
    )


def _gradients(
    parameters: _MlpParameters,
    matrix: NDArray[np.float64],
    labels: FloatVector,
    l1_softplus_weight: float,
) -> _MlpGradients:
    forward = _forward(parameters=parameters, matrix=matrix)
    row_count = float(matrix.shape[0])
    output_delta = ((forward.probabilities - labels) / row_count).reshape(-1, 1)
    output_weights = forward.second_activations.T @ output_delta
    output_bias = output_delta.sum(axis=0)

    second_delta = (output_delta @ parameters.output_weights.T) * _relu_derivative(forward.second_logits)
    second_weights = forward.first_activations.T @ second_delta
    second_bias = second_delta.sum(axis=0)

    first_delta = (second_delta @ parameters.second_weights.T) * _relu_derivative(forward.first_logits)
    first_weights = forward.weighted_inputs.T @ first_delta
    first_bias = first_delta.sum(axis=0)

    weighted_input_delta = first_delta @ parameters.first_weights.T
    softplus_derivative = _sigmoid(parameters.raw_layer_weights)
    raw_layer_weights = (weighted_input_delta * matrix).sum(axis=0) * softplus_derivative
    raw_layer_weights = raw_layer_weights + l1_softplus_weight * softplus_derivative

    return _MlpGradients(
        raw_layer_weights=raw_layer_weights,
        first_weights=first_weights,
        first_bias=first_bias,
        second_weights=second_weights,
        second_bias=second_bias,
        output_weights=output_weights,
        output_bias=output_bias,
    )


def _apply_gradients(
    parameters: _MlpParameters,
    gradients: _MlpGradients,
    learning_rate: float,
) -> _MlpParameters:
    return _MlpParameters(
        raw_layer_weights=parameters.raw_layer_weights - learning_rate * gradients.raw_layer_weights,
        first_weights=parameters.first_weights - learning_rate * gradients.first_weights,
        first_bias=parameters.first_bias - learning_rate * gradients.first_bias,
        second_weights=parameters.second_weights - learning_rate * gradients.second_weights,
        second_bias=parameters.second_bias - learning_rate * gradients.second_bias,
        output_weights=parameters.output_weights - learning_rate * gradients.output_weights,
        output_bias=parameters.output_bias - learning_rate * gradients.output_bias,
    )


def _softplus(values: FloatVector) -> FloatVector:
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def _sigmoid(values: FloatVector) -> FloatVector:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _relu(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(values, 0.0)


def _relu_derivative(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return (values > 0.0).astype(np.float64)


def _validate_config(config: CiftPaperMlpConfig) -> None:
    if config.input_dim < 1:
        raise CiftPaperMlpError("input_dim must be at least 1.")
    if config.hidden_layer_sizes != _PAPER_HIDDEN_LAYER_SIZES:
        raise CiftPaperMlpError("hidden_layer_sizes must be (128, 64) for the paper-faithful CIFT MLP.")
    if config.learning_rate <= 0.0:
        raise CiftPaperMlpError("learning_rate must be greater than 0.")
    if config.max_epochs < 1:
        raise CiftPaperMlpError("max_epochs must be at least 1.")
    if config.batch_size < 1:
        raise CiftPaperMlpError("batch_size must be at least 1.")
    if config.l1_softplus_weight < 0.0:
        raise CiftPaperMlpError("l1_softplus_weight must be nonnegative.")


def _validated_matrix(matrix: FloatMatrix, input_dim: int) -> FloatMatrix:
    feature_matrix = np.asarray(matrix, dtype=np.float32)
    if feature_matrix.ndim != 2:
        raise CiftPaperMlpError(f"matrix must be 2D, received shape {tuple(feature_matrix.shape)}.")
    if feature_matrix.shape[1] != input_dim:
        raise CiftPaperMlpError(f"matrix width must match input_dim={input_dim}.")
    return feature_matrix


def _validated_labels(labels: IntVector, row_count: int) -> IntVector:
    label_vector = np.asarray(labels, dtype=np.int64)
    if label_vector.ndim != 1:
        raise CiftPaperMlpError("labels must be a 1D vector.")
    if label_vector.shape[0] != row_count:
        raise CiftPaperMlpError(f"labels length {label_vector.shape[0]} does not match row count {row_count}.")
    label_values = set(int(value) for value in label_vector.tolist())
    if label_values != {0, 1}:
        raise CiftPaperMlpError("labels must contain both binary classes 0 and 1.")
    return label_vector
