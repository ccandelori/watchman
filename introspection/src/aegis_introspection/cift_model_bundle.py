from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatMatrix: TypeAlias = NDArray[np.float32]
ProbabilityMatrix: TypeAlias = NDArray[np.float64]
CandidateStatus: TypeAlias = Literal["offline_research_candidate", "runtime_candidate"]


class CiftModelBundleError(ValueError):
    """Raised when a persisted CIFT model bundle is invalid or cannot score input."""


class ProbabilityEstimator(Protocol):
    classes_: object

    def predict_proba(self, matrix: FloatMatrix) -> ProbabilityMatrix:
        """Return class probabilities for each feature row."""


class ProbabilityCalibrator(Protocol):
    def predict_proba(self, matrix: ProbabilityMatrix) -> ProbabilityMatrix:
        """Return calibrated probabilities for each raw positive probability."""


@dataclass(frozen=True)
class CiftModelBundleMetadata:
    schema_version: str
    source_model_id: str
    source_revision: str
    source_selected_device: str
    training_dataset_id: str
    source_artifact_path: str
    source_artifact_sha256: str
    evaluation_report_ids: tuple[str, ...]
    task_name: str
    activation_feature_key: str
    feature_count: int
    label_names: tuple[str, ...]
    positive_label: str
    decision_threshold: float
    score_semantics: str
    created_at: str
    candidate_status: CandidateStatus


@dataclass(frozen=True)
class CiftModelBundle:
    metadata: CiftModelBundleMetadata
    classifier: ProbabilityEstimator
    calibrator: ProbabilityCalibrator | None


@dataclass(frozen=True)
class CiftModelPrediction:
    positive_label: str
    positive_probability: float
    predicted_label: str
    decision_threshold: float
    score_semantics: str


def save_cift_model_bundle(path: Path, bundle: CiftModelBundle) -> None:
    validate_cift_model_bundle(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(bundle, file)


def load_cift_model_bundle(path: Path) -> CiftModelBundle:
    with path.open("rb") as file:
        loaded = pickle.load(file)
    if not isinstance(loaded, CiftModelBundle):
        raise CiftModelBundleError(f"Expected {path} to contain a CiftModelBundle.")
    validate_cift_model_bundle(loaded)
    return loaded


def predict_cift_model_bundle(bundle: CiftModelBundle, feature_matrix: FloatMatrix) -> tuple[CiftModelPrediction, ...]:
    validate_cift_model_bundle(bundle)
    _validate_feature_matrix(feature_matrix=feature_matrix, metadata=bundle.metadata)
    probabilities = bundle.classifier.predict_proba(feature_matrix)
    positive_probabilities = _select_positive_probabilities(
        probabilities=probabilities,
        classes=bundle.classifier.classes_,
        metadata=bundle.metadata,
    )
    calibrated_probabilities = _calibrate_positive_probabilities(
        probabilities=positive_probabilities,
        calibrator=bundle.calibrator,
    )
    return _predictions_from_probabilities(metadata=bundle.metadata, probabilities=calibrated_probabilities)


def validate_cift_model_bundle(bundle: CiftModelBundle) -> None:
    _validate_metadata(bundle.metadata)
    _validate_probability_estimator(name="classifier", estimator=bundle.classifier)
    if bundle.calibrator is not None:
        _validate_calibrator(bundle.calibrator)


def _validate_metadata(metadata: CiftModelBundleMetadata) -> None:
    if metadata.schema_version != "cift_model_bundle/v1":
        raise CiftModelBundleError(f"Unsupported CIFT model bundle schema version '{metadata.schema_version}'.")
    _validate_required_string(value=metadata.source_model_id, field_name="source_model_id")
    _validate_required_string(value=metadata.source_revision, field_name="source_revision")
    _validate_required_string(value=metadata.source_selected_device, field_name="source_selected_device")
    _validate_required_string(value=metadata.training_dataset_id, field_name="training_dataset_id")
    _validate_required_string(value=metadata.source_artifact_path, field_name="source_artifact_path")
    _validate_required_string(value=metadata.source_artifact_sha256, field_name="source_artifact_sha256")
    _validate_required_string(value=metadata.task_name, field_name="task_name")
    _validate_required_string(value=metadata.activation_feature_key, field_name="activation_feature_key")
    _validate_required_string(value=metadata.positive_label, field_name="positive_label")
    _validate_required_string(value=metadata.score_semantics, field_name="score_semantics")
    _validate_required_string(value=metadata.created_at, field_name="created_at")
    if metadata.candidate_status not in ("offline_research_candidate", "runtime_candidate"):
        raise CiftModelBundleError(f"Unsupported candidate_status '{metadata.candidate_status}'.")
    if metadata.feature_count < 1:
        raise CiftModelBundleError("feature_count must be at least 1.")
    if metadata.decision_threshold < 0.0 or metadata.decision_threshold > 1.0:
        raise CiftModelBundleError("decision_threshold must be in [0.0, 1.0].")
    if len(metadata.source_artifact_sha256) != 64:
        raise CiftModelBundleError("source_artifact_sha256 must be a 64-character SHA-256 hex digest.")
    for character in metadata.source_artifact_sha256:
        if character not in "0123456789abcdefABCDEF":
            raise CiftModelBundleError("source_artifact_sha256 must be a SHA-256 hex digest.")
    _validate_string_tuple(values=metadata.evaluation_report_ids, field_name="evaluation_report_ids")
    _validate_string_tuple(values=metadata.label_names, field_name="label_names")
    if len(metadata.label_names) != 2:
        raise CiftModelBundleError("label_names must contain exactly two labels.")
    if len(set(metadata.label_names)) != len(metadata.label_names):
        raise CiftModelBundleError("label_names must not contain duplicate labels.")
    if metadata.positive_label not in metadata.label_names:
        raise CiftModelBundleError("positive_label must be present in label_names.")


def _validate_required_string(value: str, field_name: str) -> None:
    if value == "":
        raise CiftModelBundleError(f"{field_name} must not be empty.")


def _validate_string_tuple(values: tuple[str, ...], field_name: str) -> None:
    if len(values) == 0:
        raise CiftModelBundleError(f"{field_name} must not be empty.")
    for index, value in enumerate(values):
        if value == "":
            raise CiftModelBundleError(f"{field_name}[{index}] must not be empty.")


def _validate_probability_estimator(name: str, estimator: ProbabilityEstimator) -> None:
    if not callable(getattr(estimator, "predict_proba", None)):
        raise CiftModelBundleError(f"{name} must expose predict_proba.")
    if not hasattr(estimator, "classes_"):
        raise CiftModelBundleError(f"{name} must expose classes_.")


def _validate_calibrator(calibrator: ProbabilityCalibrator) -> None:
    if not callable(getattr(calibrator, "predict_proba", None)):
        raise CiftModelBundleError("calibrator must expose predict_proba.")


def _validate_feature_matrix(feature_matrix: FloatMatrix, metadata: CiftModelBundleMetadata) -> None:
    if feature_matrix.ndim != 2:
        raise CiftModelBundleError(f"feature_matrix must be 2D, received shape {tuple(feature_matrix.shape)}.")
    if feature_matrix.shape[1] != metadata.feature_count:
        raise CiftModelBundleError(
            f"feature_matrix has {feature_matrix.shape[1]} columns, but bundle expects {metadata.feature_count}."
        )


def _select_positive_probabilities(
    probabilities: ProbabilityMatrix,
    classes: object,
    metadata: CiftModelBundleMetadata,
) -> ProbabilityMatrix:
    if probabilities.ndim != 2:
        raise CiftModelBundleError("Classifier predict_proba must return a 2D probability matrix.")
    if probabilities.shape[1] != len(metadata.label_names):
        raise CiftModelBundleError(
            f"Classifier returned {probabilities.shape[1]} probability columns, "
            f"but metadata has {len(metadata.label_names)} labels."
        )
    class_array = np.asarray(classes)
    positive_index = metadata.label_names.index(metadata.positive_label)
    matching_indices = np.where(class_array == positive_index)[0]
    if matching_indices.shape[0] != 1:
        raise CiftModelBundleError(f"Classifier does not expose positive class index {positive_index}.")
    selected = probabilities[:, int(matching_indices[0])]
    return selected.astype(np.float64, copy=False)


def _calibrate_positive_probabilities(
    probabilities: ProbabilityMatrix,
    calibrator: ProbabilityCalibrator | None,
) -> ProbabilityMatrix:
    if calibrator is None:
        return probabilities
    calibrated = calibrator.predict_proba(probabilities.reshape(-1, 1))
    if calibrated.ndim != 2 or calibrated.shape[1] != 2:
        raise CiftModelBundleError("Calibrator predict_proba must return two probability columns.")
    return calibrated[:, 1].astype(np.float64, copy=False)


def _predictions_from_probabilities(
    metadata: CiftModelBundleMetadata,
    probabilities: ProbabilityMatrix,
) -> tuple[CiftModelPrediction, ...]:
    positive_index = metadata.label_names.index(metadata.positive_label)
    negative_label = metadata.label_names[1 - positive_index]
    predictions: list[CiftModelPrediction] = []
    for probability in probabilities.tolist():
        positive_probability = float(probability)
        if positive_probability < 0.0 or positive_probability > 1.0:
            raise CiftModelBundleError("positive_probability must be in [0.0, 1.0].")
        predicted_label = metadata.positive_label if positive_probability >= metadata.decision_threshold else negative_label
        predictions.append(
            CiftModelPrediction(
                positive_label=metadata.positive_label,
                positive_probability=positive_probability,
                predicted_label=predicted_label,
                decision_threshold=metadata.decision_threshold,
                score_semantics=metadata.score_semantics,
            )
        )
    return tuple(predictions)
