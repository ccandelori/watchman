from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import torch

from aegis_introspection.features import PoolingMethod

_SEALED_HOLDOUT_TAG = "sealed_holdout"


class ActivationArtifactError(ValueError):
    """Raised when an activation artifact does not match the expected schema."""


class ActivationArtifactMetadata(TypedDict):
    model_id: str
    revision: str
    selected_device: str
    dtype: NotRequired[str]
    trust_remote_code: NotRequired[bool]
    layer_indices: tuple[int, ...]
    pooling_methods: tuple[PoolingMethod, ...]


class ActivationArtifact(TypedDict):
    metadata: ActivationArtifactMetadata
    example_ids: tuple[str, ...]
    labels: tuple[str, ...]
    families: tuple[str, ...]
    texts: tuple[str, ...]
    tags: tuple[tuple[str, ...], ...]
    features: dict[str, torch.Tensor]


def _as_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ActivationArtifactError(f"Expected {description} to be a mapping.")
    return cast(Mapping[str, object], value)


def _tuple_of_strings(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ActivationArtifactError(f"Expected artifact field '{field_name}' to be a tuple.")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ActivationArtifactError(f"Expected artifact field '{field_name}' item {index} to be a string.")
    return value


def _tuple_of_tag_tuples(value: object) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, tuple):
        raise ActivationArtifactError("Expected artifact field 'tags' to be a tuple.")
    for row_index, row in enumerate(value):
        if not isinstance(row, tuple):
            raise ActivationArtifactError(f"Expected tags row {row_index} to be a tuple.")
        for tag_index, tag in enumerate(row):
            if not isinstance(tag, str):
                raise ActivationArtifactError(f"Expected tags row {row_index} item {tag_index} to be a string.")
    return value


def _feature_tensors(value: object) -> dict[str, torch.Tensor]:
    mapping = _as_mapping(value, "artifact field 'features'")
    features: dict[str, torch.Tensor] = {}
    for key, tensor in mapping.items():
        if not isinstance(key, str):
            raise ActivationArtifactError("Expected every feature key to be a string.")
        if not isinstance(tensor, torch.Tensor):
            raise ActivationArtifactError(f"Expected feature '{key}' to be a torch.Tensor.")
        if tensor.ndim != 2:
            raise ActivationArtifactError(f"Expected feature '{key}' to be a 2D tensor.")
        features[key] = tensor
    if len(features) == 0:
        raise ActivationArtifactError("Expected artifact to contain at least one feature tensor.")
    return features


def _metadata(value: object) -> ActivationArtifactMetadata:
    mapping = _as_mapping(value, "artifact field 'metadata'")
    model_id = mapping.get("model_id")
    revision = mapping.get("revision")
    selected_device = mapping.get("selected_device")
    layer_indices = mapping.get("layer_indices")
    pooling_methods = mapping.get("pooling_methods")
    dtype = mapping.get("dtype")
    trust_remote_code = mapping.get("trust_remote_code")

    if not isinstance(model_id, str):
        raise ActivationArtifactError("Expected metadata field 'model_id' to be a string.")
    if not isinstance(revision, str):
        raise ActivationArtifactError("Expected metadata field 'revision' to be a string.")
    if not isinstance(selected_device, str):
        raise ActivationArtifactError("Expected metadata field 'selected_device' to be a string.")
    if not isinstance(layer_indices, tuple) or not all(isinstance(item, int) for item in layer_indices):
        raise ActivationArtifactError("Expected metadata field 'layer_indices' to be a tuple of integers.")
    if not isinstance(pooling_methods, tuple) or not all(isinstance(item, str) for item in pooling_methods):
        raise ActivationArtifactError("Expected metadata field 'pooling_methods' to be a tuple of strings.")
    if dtype is not None and not isinstance(dtype, str):
        raise ActivationArtifactError("Expected metadata field 'dtype' to be a string when present.")
    if trust_remote_code is not None and not isinstance(trust_remote_code, bool):
        raise ActivationArtifactError("Expected metadata field 'trust_remote_code' to be a boolean when present.")

    metadata: ActivationArtifactMetadata = {
        "model_id": model_id,
        "revision": revision,
        "selected_device": selected_device,
        "layer_indices": layer_indices,
        "pooling_methods": cast(tuple[PoolingMethod, ...], pooling_methods),
    }
    if dtype is not None:
        metadata["dtype"] = dtype
    if trust_remote_code is not None:
        metadata["trust_remote_code"] = trust_remote_code
    return metadata


def validate_activation_artifact(value: object) -> ActivationArtifact:
    mapping = _as_mapping(value, "activation artifact")
    metadata = _metadata(mapping.get("metadata"))
    example_ids = _tuple_of_strings(mapping.get("example_ids"), "example_ids")
    labels = _tuple_of_strings(mapping.get("labels"), "labels")
    families = _tuple_of_strings(mapping.get("families"), "families")
    texts = _tuple_of_strings(mapping.get("texts"), "texts")
    tags = _tuple_of_tag_tuples(mapping.get("tags"))
    features = _feature_tensors(mapping.get("features"))

    row_count = len(example_ids)
    if len(labels) != row_count or len(families) != row_count or len(texts) != row_count or len(tags) != row_count:
        raise ActivationArtifactError("Artifact row metadata fields must have the same length.")

    for key, tensor in features.items():
        if tensor.shape[0] != row_count:
            raise ActivationArtifactError(
                f"Feature '{key}' has {tensor.shape[0]} rows, but artifact has {row_count} examples."
            )

    return {
        "metadata": metadata,
        "example_ids": example_ids,
        "labels": labels,
        "families": families,
        "texts": texts,
        "tags": tags,
        "features": features,
    }


def load_activation_artifact(path: Path) -> ActivationArtifact:
    artifact = load_activation_artifact_allowing_sealed_holdout(path)
    _reject_sealed_artifact(artifact=artifact, path=path)
    return artifact


def load_activation_artifact_allowing_sealed_holdout(path: Path) -> ActivationArtifact:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    return validate_activation_artifact(loaded)


def _reject_sealed_artifact(artifact: ActivationArtifact, path: Path) -> None:
    if any(_SEALED_HOLDOUT_TAG in row_tags for row_tags in artifact["tags"]):
        raise ActivationArtifactError(
            f"Refusing to load sealed holdout activation artifact '{path}'. "
            "Use an explicit sealed-holdout loader only after recording the unseal decision."
        )
