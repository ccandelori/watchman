from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from aegis_introspection.cift_model_bundle import (
    CandidateStatus,
    CiftModelBundle,
    CiftModelBundleMetadata,
    save_cift_model_bundle,
)
from aegis_introspection.cift_paper_mlp import CiftPaperMlpClassifier, CiftPaperMlpConfig
from aegis_introspection.lineage import sha256_file
from aegis_introspection.sealed_holdout_policy import assert_unsealed_path, assert_unsealed_tag_rows

FloatMatrix: TypeAlias = NDArray[np.float32]
FloatVector: TypeAlias = NDArray[np.float64]
IntVector: TypeAlias = NDArray[np.int64]
ClassifierFamily: TypeAlias = Literal["linear_logistic_regression", "mlp_128_64_1"]


class CiftModelTrainingError(ValueError):
    """Raised when a final CIFT model bundle cannot be trained."""


@dataclass(frozen=True)
class CiftTrainingArtifactMetadata:
    model_id: str
    revision: str
    selected_device: str
    hidden_size: int
    layer_count: int
    tokenizer_fingerprint_sha256: str
    special_tokens_map_sha256: str
    chat_template_sha256: str
    layer_indices: tuple[int, ...]
    pooling_methods: tuple[str, ...]


@dataclass(frozen=True)
class CiftTrainingArtifact:
    metadata: CiftTrainingArtifactMetadata
    example_ids: tuple[str, ...]
    labels: tuple[str, ...]
    families: tuple[str, ...]
    texts: tuple[str, ...]
    tags: tuple[tuple[str, ...], ...]
    features: dict[str, FloatMatrix]


@dataclass(frozen=True)
class BinaryTaskDefinition:
    name: str
    description: str
    source_labels: tuple[str, ...]
    target_labels: tuple[str, ...]


@dataclass(frozen=True)
class LabelEncoding:
    label_names: tuple[str, ...]
    label_to_index: dict[str, int]
    encoded_labels: IntVector


@dataclass(frozen=True)
class CiftModelTrainingConfig:
    artifact_path: Path
    output_bundle_path: Path
    training_dataset_id: str
    task_name: str
    positive_label: str
    activation_feature_key: str
    decision_threshold: float
    random_seed: int
    max_iter: int
    regularization_c: float
    classifier_family: ClassifierFamily
    evaluation_report_ids: tuple[str, ...]
    score_semantics: str
    candidate_status: CandidateStatus
    created_at: str
    allow_sealed_holdout: bool


@dataclass(frozen=True)
class CiftModelTrainingReport:
    output_bundle_path: Path
    task_name: str
    positive_label: str
    activation_feature_key: str
    example_count: int
    feature_count: int
    label_names: tuple[str, ...]
    source_artifact_sha256: str


@dataclass(frozen=True)
class CiftLinearLogisticRuntimeParameters:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float


class CiftLinearLogisticClassifier:
    classes_: IntVector

    def __init__(self, input_dim: int, max_epochs: int, regularization_c: float, random_seed: int) -> None:
        if input_dim < 1:
            raise CiftModelTrainingError("input_dim must be at least 1.")
        if max_epochs < 1:
            raise CiftModelTrainingError("max_epochs must be at least 1.")
        if regularization_c <= 0.0:
            raise CiftModelTrainingError("regularization_c must be greater than 0.")
        self._input_dim = input_dim
        self._max_epochs = max_epochs
        self._regularization_c = regularization_c
        self._random_seed = random_seed
        self._mean = np.zeros(input_dim, dtype=np.float64)
        self._scale = np.ones(input_dim, dtype=np.float64)
        self._weights = np.zeros(input_dim, dtype=np.float64)
        self._bias = 0.0
        self.classes_ = np.asarray((0, 1), dtype=np.int64)
        self._fitted = False

    def fit(self, matrix: FloatMatrix, labels: IntVector) -> CiftLinearLogisticClassifier:
        feature_matrix = _validated_feature_matrix(matrix=matrix)
        if feature_matrix.shape[1] != self._input_dim:
            raise CiftModelTrainingError(f"matrix width must match input_dim={self._input_dim}.")
        label_vector = _validated_encoded_labels(labels=labels, row_count=feature_matrix.shape[0]).astype(np.float64)
        self._mean = feature_matrix.mean(axis=0).astype(np.float64)
        scale = feature_matrix.std(axis=0).astype(np.float64)
        self._scale = np.where(scale == 0.0, 1.0, scale)
        standardized = (feature_matrix.astype(np.float64) - self._mean) / self._scale
        rng = np.random.default_rng(self._random_seed)
        self._weights = rng.normal(loc=0.0, scale=0.01, size=self._input_dim).astype(np.float64)
        self._bias = 0.0
        learning_rate = 0.1
        row_count = float(standardized.shape[0])
        l2_strength = 1.0 / (self._regularization_c * row_count)

        for _epoch in range(self._max_epochs):
            probabilities = _sigmoid(standardized @ self._weights + self._bias)
            error = probabilities - label_vector
            weight_gradient = (standardized.T @ error) / row_count + l2_strength * self._weights
            bias_gradient = float(error.mean())
            self._weights = self._weights - learning_rate * weight_gradient
            self._bias = self._bias - learning_rate * bias_gradient

        self._fitted = True
        return self

    def predict_proba(self, matrix: FloatMatrix) -> NDArray[np.float64]:
        if not self._fitted:
            raise CiftModelTrainingError("CIFT linear logistic classifier must be fitted before predict_proba.")
        feature_matrix = _validated_feature_matrix(matrix=matrix)
        if feature_matrix.shape[1] != self._input_dim:
            raise CiftModelTrainingError(f"matrix width must match input_dim={self._input_dim}.")
        standardized = (feature_matrix.astype(np.float64) - self._mean) / self._scale
        probabilities = _sigmoid(standardized @ self._weights + self._bias)
        return np.stack((1.0 - probabilities, probabilities), axis=1).astype(np.float64, copy=False)

    def runtime_parameters(self) -> CiftLinearLogisticRuntimeParameters:
        if not self._fitted:
            raise CiftModelTrainingError("CIFT linear logistic classifier must be fitted before export.")
        return CiftLinearLogisticRuntimeParameters(
            mean=tuple(float(value) for value in self._mean),
            scale=tuple(float(value) for value in self._scale),
            coefficients=tuple(float(value) for value in self._weights),
            intercept=float(self._bias),
        )


def train_cift_model_bundle(config: CiftModelTrainingConfig) -> CiftModelTrainingReport:
    _validate_config(config)
    artifact = load_cift_training_artifact_with_unseal_policy(
        path=config.artifact_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="CIFT model bundle training",
    )
    definition = cift_binary_task_definition(config.task_name)
    task_rows = build_cift_binary_task_rows(artifact=artifact, definition=definition)
    matrix = cift_feature_matrix_for_rows(
        artifact=artifact,
        feature_key=config.activation_feature_key,
        selected_indices=task_rows.artifact_indices,
    )
    label_encoding = encode_labels(task_rows.target_labels)
    if config.positive_label not in label_encoding.label_to_index:
        raise CiftModelTrainingError(f"positive_label '{config.positive_label}' is not present in task labels.")
    classifier = _build_classifier(config=config, feature_count=int(matrix.shape[1]))
    classifier.fit(matrix, label_encoding.encoded_labels)
    source_artifact_sha256 = sha256_file(config.artifact_path)
    metadata = _metadata(
        artifact=artifact,
        config=config,
        feature_count=int(matrix.shape[1]),
        label_names=label_encoding.label_names,
        source_artifact_sha256=source_artifact_sha256,
    )
    bundle = CiftModelBundle(metadata=metadata, classifier=classifier, calibrator=None)
    save_cift_model_bundle(path=config.output_bundle_path, bundle=bundle)
    return CiftModelTrainingReport(
        output_bundle_path=config.output_bundle_path,
        task_name=config.task_name,
        positive_label=config.positive_label,
        activation_feature_key=config.activation_feature_key,
        example_count=int(matrix.shape[0]),
        feature_count=int(matrix.shape[1]),
        label_names=label_encoding.label_names,
        source_artifact_sha256=source_artifact_sha256,
    )


def _validate_config(config: CiftModelTrainingConfig) -> None:
    if config.training_dataset_id == "":
        raise CiftModelTrainingError("training_dataset_id must not be empty.")
    if config.task_name == "":
        raise CiftModelTrainingError("task_name must not be empty.")
    if config.positive_label == "":
        raise CiftModelTrainingError("positive_label must not be empty.")
    if config.activation_feature_key == "":
        raise CiftModelTrainingError("activation_feature_key must not be empty.")
    if config.max_iter < 1:
        raise CiftModelTrainingError("max_iter must be at least 1.")
    if config.regularization_c <= 0.0:
        raise CiftModelTrainingError("regularization_c must be greater than 0.")
    if config.classifier_family not in ("linear_logistic_regression", "mlp_128_64_1"):
        raise CiftModelTrainingError(f"Unsupported classifier_family '{config.classifier_family}'.")


def cift_binary_task_definition(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in _default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise CiftModelTrainingError(f"Unknown binary task '{task_name}'.")
    return matches[0]


@dataclass(frozen=True)
class CiftBinaryTaskRows:
    artifact_indices: tuple[int, ...]
    example_ids: tuple[str, ...]
    families: tuple[str, ...]
    source_labels: tuple[str, ...]
    target_labels: tuple[str, ...]


def _default_binary_task_definitions() -> tuple[BinaryTaskDefinition, ...]:
    return (
        BinaryTaskDefinition(
            name="benign_vs_secret_related",
            description="Classify benign prompts against any prompt involving secret-like material.",
            source_labels=("benign", "secret_present_safe", "exfiltration_intent"),
            target_labels=("benign", "secret_related", "secret_related"),
        ),
        BinaryTaskDefinition(
            name="safe_secret_vs_exfiltration",
            description="Classify safe secret handling against exfiltration-oriented secret handling.",
            source_labels=("secret_present_safe", "exfiltration_intent"),
            target_labels=("secret_present_safe", "exfiltration_intent"),
        ),
    )


def build_cift_binary_task_rows(
    artifact: CiftTrainingArtifact,
    definition: BinaryTaskDefinition,
) -> CiftBinaryTaskRows:
    if len(definition.source_labels) != len(definition.target_labels):
        raise CiftModelTrainingError(f"Task '{definition.name}' has mismatched source and target label counts.")
    label_pairs = tuple(zip(definition.source_labels, definition.target_labels, strict=True))
    artifact_indices: list[int] = []
    example_ids: list[str] = []
    families: list[str] = []
    source_labels: list[str] = []
    target_labels: list[str] = []
    for artifact_index, source_label in enumerate(artifact.labels):
        matched_targets = tuple(target for source, target in label_pairs if source == source_label)
        if len(matched_targets) == 0:
            continue
        if len(matched_targets) > 1:
            raise CiftModelTrainingError(f"Task '{definition.name}' maps source label '{source_label}' more than once.")
        artifact_indices.append(artifact_index)
        example_ids.append(artifact.example_ids[artifact_index])
        families.append(artifact.families[artifact_index])
        source_labels.append(source_label)
        target_labels.append(matched_targets[0])
    if len(set(target_labels)) != 2:
        raise CiftModelTrainingError(f"Task '{definition.name}' must produce exactly two target labels.")
    return CiftBinaryTaskRows(
        artifact_indices=tuple(artifact_indices),
        example_ids=tuple(example_ids),
        families=tuple(families),
        source_labels=tuple(source_labels),
        target_labels=tuple(target_labels),
    )


def encode_labels(labels: tuple[str, ...]) -> LabelEncoding:
    if len(labels) == 0:
        raise CiftModelTrainingError("Cannot encode an empty label set.")
    label_names = tuple(sorted(set(labels)))
    label_to_index = {label: index for index, label in enumerate(label_names)}
    encoded_labels = np.asarray([label_to_index[label] for label in labels], dtype=np.int64)
    return LabelEncoding(label_names=label_names, label_to_index=label_to_index, encoded_labels=encoded_labels)


def cift_feature_matrix_for_rows(
    artifact: CiftTrainingArtifact,
    feature_key: str,
    selected_indices: tuple[int, ...],
) -> FloatMatrix:
    feature_matrix = artifact.features.get(feature_key)
    if feature_matrix is None:
        raise CiftModelTrainingError(f"Activation feature '{feature_key}' is not present in the artifact.")
    return feature_matrix[list(selected_indices)]


def _build_classifier(config: CiftModelTrainingConfig, feature_count: int) -> object:
    if config.classifier_family == "linear_logistic_regression":
        return CiftLinearLogisticClassifier(
            input_dim=feature_count,
            max_epochs=config.max_iter,
            regularization_c=config.regularization_c,
            random_seed=config.random_seed,
        )
    return CiftPaperMlpClassifier(
        CiftPaperMlpConfig(
            input_dim=feature_count,
            hidden_layer_sizes=(128, 64),
            learning_rate=0.05,
            max_epochs=config.max_iter,
            batch_size=min(32, feature_count * 8),
            l1_softplus_weight=config.regularization_c,
            random_seed=config.random_seed,
        )
    )


def _metadata(
    artifact: CiftTrainingArtifact,
    config: CiftModelTrainingConfig,
    feature_count: int,
    label_names: tuple[str, ...],
    source_artifact_sha256: str,
) -> CiftModelBundleMetadata:
    artifact_metadata = artifact.metadata
    return CiftModelBundleMetadata(
        schema_version="cift_model_bundle/v1",
        source_model_id=artifact_metadata.model_id,
        source_revision=artifact_metadata.revision,
        source_selected_device=artifact_metadata.selected_device,
        source_hidden_size=artifact_metadata.hidden_size,
        source_layer_count=artifact_metadata.layer_count,
        tokenizer_fingerprint_sha256=artifact_metadata.tokenizer_fingerprint_sha256,
        special_tokens_map_sha256=artifact_metadata.special_tokens_map_sha256,
        chat_template_sha256=artifact_metadata.chat_template_sha256,
        training_dataset_id=config.training_dataset_id,
        source_artifact_path=str(config.artifact_path),
        source_artifact_sha256=source_artifact_sha256,
        evaluation_report_ids=config.evaluation_report_ids,
        task_name=config.task_name,
        activation_feature_key=config.activation_feature_key,
        feature_count=feature_count,
        label_names=label_names,
        positive_label=config.positive_label,
        decision_threshold=config.decision_threshold,
        score_semantics=config.score_semantics,
        created_at=config.created_at,
        candidate_status=config.candidate_status,
    )


def load_cift_training_artifact_with_unseal_policy(
    path: Path,
    allow_sealed_holdout: bool,
    context: str,
) -> CiftTrainingArtifact:
    assert_unsealed_path(path=path, allow_sealed_holdout=allow_sealed_holdout, context=context)
    artifact = _load_training_artifact(path)
    assert_unsealed_tag_rows(tag_rows=artifact.tags, allow_sealed_holdout=allow_sealed_holdout, context=context)
    return artifact


def cift_training_artifact_to_pickle_record(artifact: CiftTrainingArtifact) -> dict[str, object]:
    return {
        "metadata": {
            "model_id": artifact.metadata.model_id,
            "revision": artifact.metadata.revision,
            "selected_device": artifact.metadata.selected_device,
            "hidden_size": artifact.metadata.hidden_size,
            "layer_count": artifact.metadata.layer_count,
            "tokenizer_fingerprint_sha256": artifact.metadata.tokenizer_fingerprint_sha256,
            "special_tokens_map_sha256": artifact.metadata.special_tokens_map_sha256,
            "chat_template_sha256": artifact.metadata.chat_template_sha256,
            "layer_indices": artifact.metadata.layer_indices,
            "pooling_methods": artifact.metadata.pooling_methods,
        },
        "example_ids": artifact.example_ids,
        "labels": artifact.labels,
        "families": artifact.families,
        "texts": artifact.texts,
        "tags": artifact.tags,
        "features": {
            feature_key: matrix.astype(np.float32, copy=True) for feature_key, matrix in artifact.features.items()
        },
    }


def _load_training_artifact(path: Path) -> CiftTrainingArtifact:
    try:
        with path.open("rb") as file:
            loaded = pickle.load(file)
    except Exception as pickle_error:
        loaded = _load_torch_artifact(path=path, pickle_error=pickle_error)
    return _training_artifact_from_mapping(_required_mapping(value=loaded, field_name=str(path)))


def _load_torch_artifact(path: Path, pickle_error: Exception) -> object:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise CiftModelTrainingError(
            f"Unable to load '{path}' as a pickle artifact, and torch is not installed for legacy .pt artifacts."
        ) from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as torch_error:
        raise CiftModelTrainingError(f"Unable to load CIFT training artifact '{path}'.") from torch_error


def _training_artifact_from_mapping(record: Mapping[str, object]) -> CiftTrainingArtifact:
    metadata = _metadata_from_mapping(_required_mapping(value=record.get("metadata"), field_name="metadata"))
    example_ids = _required_string_tuple(value=record.get("example_ids"), field_name="example_ids")
    labels = _required_string_tuple(value=record.get("labels"), field_name="labels")
    families = _required_string_tuple(value=record.get("families"), field_name="families")
    texts = _required_string_tuple(value=record.get("texts"), field_name="texts")
    tags = _required_tag_rows(value=record.get("tags"))
    features = _feature_matrices(value=record.get("features"))
    row_count = len(example_ids)
    if len(labels) != row_count or len(families) != row_count or len(texts) != row_count or len(tags) != row_count:
        raise CiftModelTrainingError("Artifact row metadata fields must have the same length.")
    for feature_key, matrix in features.items():
        if matrix.shape[0] != row_count:
            raise CiftModelTrainingError(
                f"Feature '{feature_key}' has {matrix.shape[0]} rows, but artifact has {row_count} examples."
            )
    return CiftTrainingArtifact(
        metadata=metadata,
        example_ids=example_ids,
        labels=labels,
        families=families,
        texts=texts,
        tags=tags,
        features=features,
    )


def _metadata_from_mapping(record: Mapping[str, object]) -> CiftTrainingArtifactMetadata:
    layer_indices_value = record.get("layer_indices")
    pooling_methods_value = record.get("pooling_methods")
    if not isinstance(layer_indices_value, tuple) or not all(isinstance(item, int) for item in layer_indices_value):
        raise CiftModelTrainingError("metadata.layer_indices must be a tuple of integers.")
    if not isinstance(pooling_methods_value, tuple) or not all(isinstance(item, str) for item in pooling_methods_value):
        raise CiftModelTrainingError("metadata.pooling_methods must be a tuple of strings.")
    return CiftTrainingArtifactMetadata(
        model_id=_required_string(record=record, field_name="metadata.model_id"),
        revision=_required_string(record=record, field_name="metadata.revision"),
        selected_device=_required_string(record=record, field_name="metadata.selected_device"),
        hidden_size=_required_positive_int(record=record, field_name="metadata.hidden_size"),
        layer_count=_required_positive_int(record=record, field_name="metadata.layer_count"),
        tokenizer_fingerprint_sha256=_required_sha256(
            record=record,
            field_name="metadata.tokenizer_fingerprint_sha256",
        ),
        special_tokens_map_sha256=_required_sha256(record=record, field_name="metadata.special_tokens_map_sha256"),
        chat_template_sha256=_required_sha256(record=record, field_name="metadata.chat_template_sha256"),
        layer_indices=layer_indices_value,
        pooling_methods=pooling_methods_value,
    )


def _feature_matrices(value: object) -> dict[str, FloatMatrix]:
    mapping = _required_mapping(value=value, field_name="features")
    features: dict[str, FloatMatrix] = {}
    for feature_key, feature_value in mapping.items():
        if not isinstance(feature_key, str):
            raise CiftModelTrainingError("Every feature key must be a string.")
        features[feature_key] = _feature_value_to_matrix(value=feature_value, field_name=f"features.{feature_key}")
    if len(features) == 0:
        raise CiftModelTrainingError("Artifact must contain at least one feature matrix.")
    return features


def _feature_value_to_matrix(value: object, field_name: str) -> FloatMatrix:
    value = _tensor_feature_value_to_array(value=value, field_name=field_name)
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2:
        raise CiftModelTrainingError(f"{field_name} must be a 2D feature matrix.")
    return matrix


def _tensor_feature_value_to_array(value: object, field_name: str) -> object:
    detach = getattr(value, "detach", None)
    if not callable(detach):
        return value
    detached_value = detach()
    cpu = getattr(detached_value, "cpu", None)
    if not callable(cpu):
        raise CiftModelTrainingError(f"{field_name} is tensor-like but does not expose cpu().")
    cpu_value = cpu()
    float_cast = getattr(cpu_value, "float", None)
    if not callable(float_cast):
        raise CiftModelTrainingError(f"{field_name} is tensor-like but does not expose float().")
    float_value = float_cast()
    to_numpy = getattr(float_value, "numpy", None)
    if not callable(to_numpy):
        raise CiftModelTrainingError(f"{field_name} is tensor-like but does not expose numpy().")
    return to_numpy()


def _validated_feature_matrix(matrix: FloatMatrix) -> FloatMatrix:
    feature_matrix = np.asarray(matrix, dtype=np.float32)
    if feature_matrix.ndim != 2:
        raise CiftModelTrainingError(f"matrix must be 2D, received shape {tuple(feature_matrix.shape)}.")
    return feature_matrix


def _validated_encoded_labels(labels: IntVector, row_count: int) -> IntVector:
    label_vector = np.asarray(labels, dtype=np.int64)
    if label_vector.ndim != 1:
        raise CiftModelTrainingError("labels must be a 1D vector.")
    if label_vector.shape[0] != row_count:
        raise CiftModelTrainingError(f"labels length {label_vector.shape[0]} does not match row count {row_count}.")
    if set(int(value) for value in label_vector.tolist()) != {0, 1}:
        raise CiftModelTrainingError("labels must contain both binary classes 0 and 1.")
    return label_vector


def _sigmoid(values: FloatVector) -> FloatVector:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _required_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CiftModelTrainingError(f"{field_name} must be an object.")
    return cast(Mapping[str, object], value)


def _required_string(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name.rsplit(".", maxsplit=1)[-1])
    if not isinstance(value, str):
        raise CiftModelTrainingError(f"{field_name} must be a string.")
    if value == "":
        raise CiftModelTrainingError(f"{field_name} must not be empty.")
    return value


def _required_positive_int(record: Mapping[str, object], field_name: str) -> int:
    value = record.get(field_name.rsplit(".", maxsplit=1)[-1])
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftModelTrainingError(f"{field_name} must be an integer.")
    if value < 1:
        raise CiftModelTrainingError(f"{field_name} must be positive.")
    return value


def _required_sha256(record: Mapping[str, object], field_name: str) -> str:
    value = _required_string(record=record, field_name=field_name)
    if len(value) != 64:
        raise CiftModelTrainingError(f"{field_name} must be a 64-character SHA-256 hex digest.")
    for character in value:
        if character not in "0123456789abcdef":
            raise CiftModelTrainingError(f"{field_name} must be lowercase hexadecimal.")
    return value


def _required_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise CiftModelTrainingError(f"{field_name} must be a tuple.")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise CiftModelTrainingError(f"{field_name}[{index}] must be a string.")
    return value


def _required_tag_rows(value: object) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, tuple):
        raise CiftModelTrainingError("tags must be a tuple.")
    rows: list[tuple[str, ...]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, tuple):
            raise CiftModelTrainingError(f"tags[{row_index}] must be a tuple.")
        parsed_row: list[str] = []
        for tag_index, tag in enumerate(row):
            if not isinstance(tag, str):
                raise CiftModelTrainingError(f"tags[{row_index}][{tag_index}] must be a string.")
            parsed_row.append(tag)
        rows.append(tuple(parsed_row))
    return tuple(rows)
