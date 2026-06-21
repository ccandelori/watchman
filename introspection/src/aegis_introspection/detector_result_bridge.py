from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from aegis_introspection.cift_calibration import CalibratedCiftPrediction
from aegis_introspection.cift_model_bundle import CiftModelPrediction
from aegis_introspection.error_analysis import BinaryExamplePrediction
from aegis_introspection.probe import JsonValue


DetectorResultJson: TypeAlias = dict[str, JsonValue]
RecommendedAction: TypeAlias = Literal["allow", "warn", "sanitize", "block", "escalate"]


class DetectorResultBridgeError(ValueError):
    """Raised when an introspection result cannot be adapted to a DetectorResult shape."""


@dataclass(frozen=True)
class CiftDetectorBridgeConfig:
    detector_name: str
    feature_key: str
    task_name: str
    probe_version: str
    capability_required: str
    positive_label: str
    positive_score: float
    negative_score: float
    positive_action: RecommendedAction
    negative_action: RecommendedAction
    confidence: float


@dataclass(frozen=True)
class CalibratedCiftDetectorBridgeConfig:
    detector_name: str
    feature_key: str
    task_name: str
    probe_version: str
    capability_required: str
    decision_threshold: float
    positive_action: RecommendedAction
    negative_action: RecommendedAction
    confidence: float


@dataclass(frozen=True)
class TrainedCiftDetectorBridgeConfig:
    detector_name: str
    feature_key: str
    task_name: str
    model_bundle_id: str
    capability_required: str
    positive_action: RecommendedAction
    negative_action: RecommendedAction
    confidence: float


@dataclass(frozen=True)
class CiftModelPredictionContext:
    example_id: str
    family: str
    source_label: str
    true_label: str


def cift_prediction_to_detector_result(
    prediction: BinaryExamplePrediction,
    config: CiftDetectorBridgeConfig,
) -> DetectorResultJson:
    _validate_config(config)
    predicted_positive = prediction.predicted_label == config.positive_label
    return {
        "detector_name": config.detector_name,
        "component": "cift",
        "score": config.positive_score if predicted_positive else config.negative_score,
        "confidence": config.confidence,
        "recommended_action": config.positive_action if predicted_positive else config.negative_action,
        "capability_required": config.capability_required,
        "capability_status": "active",
        "evidence": {
            "task_name": config.task_name,
            "feature_key": config.feature_key,
            "probe_version": config.probe_version,
            "example_id": prediction.example_id,
            "family": prediction.family,
            "fold_index": prediction.fold_index,
            "source_label": prediction.source_label,
            "true_label": prediction.true_label,
            "predicted_label": prediction.predicted_label,
            "is_correct": prediction.is_correct,
            "offline_eval": True,
            "score_semantics": "prediction_label_score_not_calibrated",
        },
        "latency_ms": 0.0,
    }


def trained_cift_prediction_to_detector_result(
    prediction: CiftModelPrediction,
    context: CiftModelPredictionContext,
    config: TrainedCiftDetectorBridgeConfig,
) -> DetectorResultJson:
    _validate_trained_config(config)
    _validate_prediction_context(context)
    predicted_positive = prediction.predicted_label == prediction.positive_label
    return {
        "detector_name": config.detector_name,
        "component": "cift",
        "score": prediction.positive_probability,
        "confidence": config.confidence,
        "recommended_action": config.positive_action if predicted_positive else config.negative_action,
        "capability_required": config.capability_required,
        "capability_status": "active",
        "evidence": {
            "task_name": config.task_name,
            "feature_key": config.feature_key,
            "model_bundle_id": config.model_bundle_id,
            "example_id": context.example_id,
            "family": context.family,
            "source_label": context.source_label,
            "true_label": context.true_label,
            "predicted_label": prediction.predicted_label,
            "positive_label": prediction.positive_label,
            "decision_threshold": prediction.decision_threshold,
            "offline_eval": True,
            "score_semantics": prediction.score_semantics,
        },
        "latency_ms": 0.0,
    }


def calibrated_cift_prediction_to_detector_result(
    prediction: CalibratedCiftPrediction,
    config: CalibratedCiftDetectorBridgeConfig,
) -> DetectorResultJson:
    _validate_calibrated_config(config)
    predicted_positive = prediction.positive_probability >= config.decision_threshold
    return {
        "detector_name": config.detector_name,
        "component": "cift",
        "score": prediction.positive_probability,
        "confidence": config.confidence,
        "recommended_action": config.positive_action if predicted_positive else config.negative_action,
        "capability_required": config.capability_required,
        "capability_status": "active",
        "evidence": {
            "task_name": config.task_name,
            "feature_key": config.feature_key,
            "probe_version": config.probe_version,
            "example_id": prediction.example_id,
            "family": prediction.family,
            "fold_index": prediction.fold_index,
            "source_label": prediction.source_label,
            "true_label": prediction.true_label,
            "predicted_label": prediction.predicted_label,
            "is_correct": prediction.is_correct,
            "positive_label": prediction.positive_label,
            "decision_threshold": config.decision_threshold,
            "offline_eval": True,
            "score_semantics": "inner_cv_platt_calibrated_probability",
        },
        "latency_ms": 0.0,
    }


def _validate_config(config: CiftDetectorBridgeConfig) -> None:
    for field_name, value in (
        ("detector_name", config.detector_name),
        ("feature_key", config.feature_key),
        ("task_name", config.task_name),
        ("probe_version", config.probe_version),
        ("capability_required", config.capability_required),
        ("positive_label", config.positive_label),
    ):
        if value == "":
            raise DetectorResultBridgeError(f"Config field '{field_name}' must not be empty.")
    _validate_score(config.positive_score, "positive_score")
    _validate_score(config.negative_score, "negative_score")
    _validate_score(config.confidence, "confidence")


def _validate_calibrated_config(config: CalibratedCiftDetectorBridgeConfig) -> None:
    for field_name, value in (
        ("detector_name", config.detector_name),
        ("feature_key", config.feature_key),
        ("task_name", config.task_name),
        ("probe_version", config.probe_version),
        ("capability_required", config.capability_required),
    ):
        if value == "":
            raise DetectorResultBridgeError(f"Config field '{field_name}' must not be empty.")
    _validate_score(config.decision_threshold, "decision_threshold")
    _validate_score(config.confidence, "confidence")


def _validate_trained_config(config: TrainedCiftDetectorBridgeConfig) -> None:
    for field_name, value in (
        ("detector_name", config.detector_name),
        ("feature_key", config.feature_key),
        ("task_name", config.task_name),
        ("model_bundle_id", config.model_bundle_id),
        ("capability_required", config.capability_required),
    ):
        if value == "":
            raise DetectorResultBridgeError(f"Config field '{field_name}' must not be empty.")
    _validate_score(config.confidence, "confidence")


def _validate_prediction_context(context: CiftModelPredictionContext) -> None:
    for field_name, value in (
        ("example_id", context.example_id),
        ("family", context.family),
        ("source_label", context.source_label),
        ("true_label", context.true_label),
    ):
        if value == "":
            raise DetectorResultBridgeError(f"Prediction context field '{field_name}' must not be empty.")


def _validate_score(value: float, field_name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise DetectorResultBridgeError(f"Config field '{field_name}' must be in [0.0, 1.0].")
