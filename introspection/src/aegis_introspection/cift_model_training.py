from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aegis_introspection.artifacts import ActivationArtifact, load_activation_artifact
from aegis_introspection.binary_tasks import (
    BinaryTaskConfig,
    BinaryTaskDefinition,
    activation_feature_tensor,
    build_activation_classifier,
    build_binary_task_dataset,
    default_binary_task_definitions,
)
from aegis_introspection.cift_model_bundle import (
    CandidateStatus,
    CiftModelBundle,
    CiftModelBundleMetadata,
    save_cift_model_bundle,
)
from aegis_introspection.lineage import sha256_file
from aegis_introspection.probe import encode_labels, tensor_to_float_matrix


class CiftModelTrainingError(ValueError):
    """Raised when a final CIFT model bundle cannot be trained."""


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
    evaluation_report_ids: tuple[str, ...]
    score_semantics: str
    candidate_status: CandidateStatus
    created_at: str


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


def train_cift_model_bundle(config: CiftModelTrainingConfig) -> CiftModelTrainingReport:
    _validate_config(config)
    artifact = load_activation_artifact(config.artifact_path)
    definition = _task_definition(config.task_name)
    dataset = build_binary_task_dataset(artifact=artifact, definition=definition)
    feature_tensor = activation_feature_tensor(artifact=artifact, feature_key=config.activation_feature_key)
    selected_indices = tuple(artifact["example_ids"].index(example_id) for example_id in dataset.example_ids)
    matrix = tensor_to_float_matrix(feature_tensor)[list(selected_indices)]
    label_encoding = encode_labels(dataset.target_labels)
    if config.positive_label not in label_encoding.label_to_index:
        raise CiftModelTrainingError(f"positive_label '{config.positive_label}' is not present in task labels.")
    classifier = build_activation_classifier(_binary_task_config(config))
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


def _task_definition(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise CiftModelTrainingError(f"Unknown binary task '{task_name}'.")
    return matches[0]


def _binary_task_config(config: CiftModelTrainingConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=2,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.activation_feature_key,
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
    )


def _metadata(
    artifact: ActivationArtifact,
    config: CiftModelTrainingConfig,
    feature_count: int,
    label_names: tuple[str, ...],
    source_artifact_sha256: str,
) -> CiftModelBundleMetadata:
    artifact_metadata = artifact["metadata"]
    return CiftModelBundleMetadata(
        schema_version="cift_model_bundle/v1",
        source_model_id=artifact_metadata["model_id"],
        source_revision=artifact_metadata["revision"],
        source_selected_device=artifact_metadata["selected_device"],
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
