from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from aegis.core.contracts import (
    Action,
    CapabilityMode,
    CapabilityStatus,
    DetectorComponent,
    DetectorResult,
    JsonValue,
    NormalizedTurn,
)
from aegis.core.orchestrator import ModelResponse

_SCHEMA_VERSION = "aegis.cift_runtime_linear/v1"
_CIFT_METADATA_KEY = "cift"
_FEATURE_VECTORS_KEY = "feature_vectors"
_FALLBACK_CONFIDENCE_CAP = 0.35


class CiftRuntimeDetectorError(ValueError):
    """Raised when a runtime CIFT detector artifact or feature vector is invalid."""


@dataclass(frozen=True)
class CiftRuntimeLinearModel:
    schema_version: str
    model_bundle_id: str
    source_model_id: str
    training_dataset_id: str
    source_artifact_sha256: str
    evaluation_report_ids: tuple[str, ...]
    task_name: str
    feature_key: str
    feature_count: int
    label_names: tuple[str, str]
    positive_label: str
    positive_class_index: int
    class_indices: tuple[int, int]
    decision_threshold: float
    score_semantics: str
    confidence: float
    candidate_status: str
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    logistic_coefficients: tuple[float, ...]
    logistic_intercept: float
    negative_action: Action
    positive_action: Action


@dataclass(frozen=True)
class CiftRuntimePrediction:
    score: float
    predicted_label: str
    recommended_action: Action
    operating_band: str


@dataclass(frozen=True)
class CiftRuntimeWindowSelectorConfig:
    detector_name: str
    selected_choice_model_path: Path
    fallback_model_path: Path
    feature_extractor: CiftFeatureExtractor
    feature_source: str


@dataclass(frozen=True)
class CiftRuntimeComponents:
    turn_annotators: tuple[CiftFeatureVectorAnnotator, ...]
    pre_generation_detectors: tuple[CiftRuntimeWindowSelector, ...]


class CiftFeatureExtractor(Protocol):
    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        """Extract a CIFT feature vector for a normalized turn."""


@dataclass(frozen=True)
class CiftFeatureVectorAnnotator:
    feature_key: str
    extractor: CiftFeatureExtractor
    source: str

    def annotate(self, turn: NormalizedTurn) -> NormalizedTurn:
        if self.feature_key == "":
            raise CiftRuntimeDetectorError("feature_key must not be empty.")
        if self.source == "":
            raise CiftRuntimeDetectorError("source must not be empty.")
        if turn.capability_mode not in (CapabilityMode.SELF_HOSTED_INTROSPECTION, CapabilityMode.OFFLINE_EVAL):
            return turn

        feature_vector = self.extractor.extract_feature_vector(turn=turn, feature_key=self.feature_key)
        if feature_vector is None:
            return turn
        validated_vector = tuple(
            _float_item(value=item, field_name=f"extractor.{self.feature_key}[{index}]")
            for index, item in enumerate(feature_vector)
        )
        return normalized_turn_with_cift_feature_vector(
            turn=turn,
            feature_key=self.feature_key,
            feature_vector=validated_vector,
            source=self.source,
        )


class CiftRuntimeDetector:
    def __init__(self, detector_name: str, model: CiftRuntimeLinearModel) -> None:
        if detector_name == "":
            raise CiftRuntimeDetectorError("detector_name must not be empty.")
        validate_cift_runtime_model(model)
        self._detector_name = detector_name
        self._model = model

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        started_at = time.perf_counter()
        if turn.capability_mode not in (CapabilityMode.SELF_HOSTED_INTROSPECTION, CapabilityMode.OFFLINE_EVAL):
            return _unavailable_result(
                detector_name=self._detector_name,
                model=self._model,
                turn=turn,
                reason="activation_access_unavailable",
                capability_status=CapabilityStatus.UNAVAILABLE,
                latency_ms=_elapsed_ms(started_at),
            )

        feature_vector = cift_feature_vector_from_turn(turn=turn, feature_key=self._model.feature_key)
        if feature_vector is None:
            return _unavailable_result(
                detector_name=self._detector_name,
                model=self._model,
                turn=turn,
                reason="activation_feature_vector_missing",
                capability_status=CapabilityStatus.DEGRADED,
                latency_ms=_elapsed_ms(started_at),
            )

        prediction = predict_cift_runtime_model(model=self._model, feature_vector=feature_vector)
        return DetectorResult(
            detector_name=self._detector_name,
            component=DetectorComponent.CIFT,
            score=prediction.score,
            confidence=self._model.confidence,
            recommended_action=prediction.recommended_action,
            capability_required=CapabilityMode.SELF_HOSTED_INTROSPECTION.value,
            capability_status=CapabilityStatus.ACTIVE,
            evidence=_active_evidence(model=self._model, turn=turn, prediction=prediction),
            latency_ms=_elapsed_ms(started_at),
        )


class CiftRuntimeWindowSelector:
    def __init__(
        self,
        detector_name: str,
        selected_choice_model: CiftRuntimeLinearModel,
        fallback_model: CiftRuntimeLinearModel,
    ) -> None:
        self._selected_choice_model = selected_choice_model
        self._fallback_model = fallback_model
        self._selected_choice_detector = CiftRuntimeDetector(detector_name=detector_name, model=selected_choice_model)
        self._fallback_detector = CiftRuntimeDetector(detector_name=detector_name, model=fallback_model)

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        if _has_selected_choice_readout_indices(turn):
            result = self._selected_choice_detector.evaluate(turn=turn, model_response=model_response)
            return _result_with_window_selection_evidence(
                result=result,
                window_family="selected_choice",
                selection_reason="selected_choice_metadata_present",
                window_coverage="primary",
                selected_choice_model=self._selected_choice_model,
                fallback_model=self._fallback_model,
            )
        result = self._fallback_detector.evaluate(turn=turn, model_response=model_response)
        selected_result = _result_with_window_selection_evidence(
            result=result,
            window_family="payload_query_fallback",
            selection_reason="selected_choice_metadata_absent",
            window_coverage="degraded_fallback",
            selected_choice_model=self._selected_choice_model,
            fallback_model=self._fallback_model,
        )
        return _degraded_fallback_result(selected_result)


def load_cift_runtime_model(path: Path) -> CiftRuntimeLinearModel:
    if not path.exists():
        raise CiftRuntimeDetectorError(f"CIFT runtime model path does not exist: {path}")
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CiftRuntimeDetectorError(f"Invalid CIFT runtime model JSON in {path}: {exc.msg}.") from exc
    if not isinstance(decoded, dict):
        raise CiftRuntimeDetectorError(f"Expected {path} to contain a JSON object.")
    model = cift_runtime_model_from_mapping(cast(Mapping[str, object], decoded))
    validate_cift_runtime_model(model)
    return model


def build_cift_window_selector_runtime_components(config: CiftRuntimeWindowSelectorConfig) -> CiftRuntimeComponents:
    if config.detector_name == "":
        raise CiftRuntimeDetectorError("detector_name must not be empty.")
    if config.feature_source == "":
        raise CiftRuntimeDetectorError("feature_source must not be empty.")
    selected_choice_model = load_cift_runtime_model(config.selected_choice_model_path)
    fallback_model = load_cift_runtime_model(config.fallback_model_path)
    feature_keys = tuple(dict.fromkeys((selected_choice_model.feature_key, fallback_model.feature_key)))
    turn_annotators = tuple(
        CiftFeatureVectorAnnotator(
            feature_key=feature_key,
            extractor=config.feature_extractor,
            source=config.feature_source,
        )
        for feature_key in feature_keys
    )
    return CiftRuntimeComponents(
        turn_annotators=turn_annotators,
        pre_generation_detectors=(
            CiftRuntimeWindowSelector(
                detector_name=config.detector_name,
                selected_choice_model=selected_choice_model,
                fallback_model=fallback_model,
            ),
        ),
    )


def cift_runtime_model_from_mapping(record: Mapping[str, object]) -> CiftRuntimeLinearModel:
    feature_count = _required_int(record=record, field_name="feature_count")
    return CiftRuntimeLinearModel(
        schema_version=_required_string(record=record, field_name="schema_version"),
        model_bundle_id=_required_string(record=record, field_name="model_bundle_id"),
        source_model_id=_required_string(record=record, field_name="source_model_id"),
        training_dataset_id=_required_string(record=record, field_name="training_dataset_id"),
        source_artifact_sha256=_required_string(record=record, field_name="source_artifact_sha256"),
        evaluation_report_ids=_required_string_tuple(record=record, field_name="evaluation_report_ids"),
        task_name=_required_string(record=record, field_name="task_name"),
        feature_key=_required_string(record=record, field_name="feature_key"),
        feature_count=feature_count,
        label_names=_required_two_string_tuple(record=record, field_name="label_names"),
        positive_label=_required_string(record=record, field_name="positive_label"),
        positive_class_index=_required_int(record=record, field_name="positive_class_index"),
        class_indices=_required_two_int_tuple(record=record, field_name="class_indices"),
        decision_threshold=_required_float(record=record, field_name="decision_threshold"),
        score_semantics=_required_string(record=record, field_name="score_semantics"),
        confidence=_required_float(record=record, field_name="confidence"),
        candidate_status=_required_string(record=record, field_name="candidate_status"),
        scaler_mean=_required_float_tuple(record=record, field_name="scaler_mean", expected_length=feature_count),
        scaler_scale=_required_float_tuple(record=record, field_name="scaler_scale", expected_length=feature_count),
        logistic_coefficients=_required_float_tuple(
            record=record,
            field_name="logistic_coefficients",
            expected_length=feature_count,
        ),
        logistic_intercept=_required_float(record=record, field_name="logistic_intercept"),
        negative_action=_required_action(record=record, field_name="negative_action"),
        positive_action=_required_action(record=record, field_name="positive_action"),
    )


def cift_runtime_model_to_dict(model: CiftRuntimeLinearModel) -> dict[str, JsonValue]:
    validate_cift_runtime_model(model)
    return {
        "schema_version": model.schema_version,
        "model_bundle_id": model.model_bundle_id,
        "source_model_id": model.source_model_id,
        "training_dataset_id": model.training_dataset_id,
        "source_artifact_sha256": model.source_artifact_sha256,
        "evaluation_report_ids": list(model.evaluation_report_ids),
        "task_name": model.task_name,
        "feature_key": model.feature_key,
        "feature_count": model.feature_count,
        "label_names": list(model.label_names),
        "positive_label": model.positive_label,
        "positive_class_index": model.positive_class_index,
        "class_indices": list(model.class_indices),
        "decision_threshold": model.decision_threshold,
        "score_semantics": model.score_semantics,
        "confidence": model.confidence,
        "candidate_status": model.candidate_status,
        "scaler_mean": list(model.scaler_mean),
        "scaler_scale": list(model.scaler_scale),
        "logistic_coefficients": list(model.logistic_coefficients),
        "logistic_intercept": model.logistic_intercept,
        "negative_action": model.negative_action.value,
        "positive_action": model.positive_action.value,
    }


def predict_cift_runtime_model(
    model: CiftRuntimeLinearModel,
    feature_vector: tuple[float, ...],
) -> CiftRuntimePrediction:
    validate_cift_runtime_model(model)
    if len(feature_vector) != model.feature_count:
        raise CiftRuntimeDetectorError(
            f"feature_vector has {len(feature_vector)} values, but model expects {model.feature_count}."
        )
    class_one_probability = _class_one_probability(model=model, feature_vector=feature_vector)
    positive_probability = _positive_probability(model=model, class_one_probability=class_one_probability)
    if positive_probability >= model.decision_threshold:
        predicted_label = model.positive_label
        recommended_action = model.positive_action
        operating_band = "positive"
    else:
        predicted_label = _negative_label(model)
        recommended_action = model.negative_action
        operating_band = "negative"
    return CiftRuntimePrediction(
        score=positive_probability,
        predicted_label=predicted_label,
        recommended_action=recommended_action,
        operating_band=operating_band,
    )


def cift_feature_vector_from_turn(turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
    cift_metadata = turn.metadata.get(_CIFT_METADATA_KEY)
    if cift_metadata is None:
        return None
    if not isinstance(cift_metadata, dict):
        raise CiftRuntimeDetectorError("NormalizedTurn metadata.cift must be an object when present.")
    feature_vectors = cift_metadata.get(_FEATURE_VECTORS_KEY)
    if feature_vectors is None:
        return None
    if not isinstance(feature_vectors, dict):
        raise CiftRuntimeDetectorError("NormalizedTurn metadata.cift.feature_vectors must be an object when present.")
    value = feature_vectors.get(feature_key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise CiftRuntimeDetectorError(f"CIFT feature vector '{feature_key}' must be a list of numbers.")
    return tuple(_float_item(value=item, field_name=f"metadata.cift.feature_vectors.{feature_key}") for item in value)


def normalized_turn_with_cift_feature_vector(
    turn: NormalizedTurn,
    feature_key: str,
    feature_vector: tuple[float, ...],
    source: str,
) -> NormalizedTurn:
    if feature_key == "":
        raise CiftRuntimeDetectorError("feature_key must not be empty.")
    if source == "":
        raise CiftRuntimeDetectorError("source must not be empty.")
    encoded_feature_vector: list[JsonValue] = [
        _float_item(value=item, field_name=f"feature_vector[{index}]") for index, item in enumerate(feature_vector)
    ]
    cift_metadata = _copied_cift_metadata(turn.metadata)
    feature_vectors = _copied_feature_vectors(cift_metadata)
    feature_vectors[feature_key] = encoded_feature_vector
    cift_metadata[_FEATURE_VECTORS_KEY] = feature_vectors
    cift_metadata["feature_sources"] = _feature_sources_metadata(
        cift_metadata=cift_metadata,
        feature_key=feature_key,
        source=source,
        feature_count=len(encoded_feature_vector),
    )
    metadata = dict(turn.metadata)
    metadata[_CIFT_METADATA_KEY] = cift_metadata
    return NormalizedTurn(
        trace_id=turn.trace_id,
        session_id=turn.session_id,
        turn_index=turn.turn_index,
        capability_mode=turn.capability_mode,
        model=turn.model,
        messages=turn.messages,
        tool_calls=turn.tool_calls,
        sensitive_spans=turn.sensitive_spans,
        metadata=metadata,
    )


def validate_cift_runtime_model(model: CiftRuntimeLinearModel) -> None:
    if model.schema_version != _SCHEMA_VERSION:
        raise CiftRuntimeDetectorError(f"Unsupported CIFT runtime model schema '{model.schema_version}'.")
    _validate_required_string(value=model.model_bundle_id, field_name="model_bundle_id")
    _validate_required_string(value=model.source_model_id, field_name="source_model_id")
    _validate_required_string(value=model.training_dataset_id, field_name="training_dataset_id")
    _validate_required_string(value=model.source_artifact_sha256, field_name="source_artifact_sha256")
    _validate_required_string(value=model.task_name, field_name="task_name")
    _validate_required_string(value=model.feature_key, field_name="feature_key")
    _validate_required_string(value=model.positive_label, field_name="positive_label")
    _validate_required_string(value=model.score_semantics, field_name="score_semantics")
    _validate_required_string(value=model.candidate_status, field_name="candidate_status")
    _validate_sha256(value=model.source_artifact_sha256)
    if model.feature_count < 1:
        raise CiftRuntimeDetectorError("feature_count must be at least 1.")
    if len(model.evaluation_report_ids) == 0:
        raise CiftRuntimeDetectorError("evaluation_report_ids must not be empty.")
    for index, report_id in enumerate(model.evaluation_report_ids):
        _validate_required_string(value=report_id, field_name=f"evaluation_report_ids[{index}]")
    if model.positive_label not in model.label_names:
        raise CiftRuntimeDetectorError("positive_label must be present in label_names.")
    if model.label_names.index(model.positive_label) != model.positive_class_index:
        raise CiftRuntimeDetectorError("positive_class_index must match the positive_label index in label_names.")
    if len(set(model.label_names)) != len(model.label_names):
        raise CiftRuntimeDetectorError("label_names must not contain duplicates.")
    if len(set(model.class_indices)) != len(model.class_indices):
        raise CiftRuntimeDetectorError("class_indices must not contain duplicates.")
    if model.positive_class_index not in model.class_indices:
        raise CiftRuntimeDetectorError("positive_class_index must be present in class_indices.")
    _validate_probability(value=model.decision_threshold, field_name="decision_threshold")
    _validate_probability(value=model.confidence, field_name="confidence")
    _validate_vector_length(values=model.scaler_mean, field_name="scaler_mean", expected_length=model.feature_count)
    _validate_vector_length(values=model.scaler_scale, field_name="scaler_scale", expected_length=model.feature_count)
    _validate_vector_length(
        values=model.logistic_coefficients,
        field_name="logistic_coefficients",
        expected_length=model.feature_count,
    )
    for index, scale in enumerate(model.scaler_scale):
        if scale <= 0.0:
            raise CiftRuntimeDetectorError(f"scaler_scale[{index}] must be greater than 0.")


def _active_evidence(
    model: CiftRuntimeLinearModel,
    turn: NormalizedTurn,
    prediction: CiftRuntimePrediction,
) -> dict[str, JsonValue]:
    return {
        "model_bundle_id": model.model_bundle_id,
        "source_model_id": model.source_model_id,
        "training_dataset_id": model.training_dataset_id,
        "task_name": model.task_name,
        "feature_key": model.feature_key,
        "feature_count": model.feature_count,
        "positive_label": model.positive_label,
        "predicted_label": prediction.predicted_label,
        "decision_threshold": model.decision_threshold,
        "operating_band": prediction.operating_band,
        "score_semantics": model.score_semantics,
        "candidate_status": model.candidate_status,
        "source_artifact_sha256": model.source_artifact_sha256,
        "activation_source": "metadata.cift.feature_vectors",
        "capability_mode": turn.capability_mode.value,
        "model_id": turn.model.model_id,
        "selected_device": turn.model.selected_device,
    }


def _unavailable_result(
    detector_name: str,
    model: CiftRuntimeLinearModel,
    turn: NormalizedTurn,
    reason: str,
    capability_status: CapabilityStatus,
    latency_ms: float,
) -> DetectorResult:
    return DetectorResult(
        detector_name=detector_name,
        component=DetectorComponent.CIFT,
        score=0.0,
        confidence=1.0,
        recommended_action=Action.ALLOW,
        capability_required=CapabilityMode.SELF_HOSTED_INTROSPECTION.value,
        capability_status=capability_status,
        evidence={
            "reason": reason,
            "model_bundle_id": model.model_bundle_id,
            "required_capability": CapabilityMode.SELF_HOSTED_INTROSPECTION.value,
            "actual_capability_mode": turn.capability_mode.value,
            "feature_key": model.feature_key,
            "model_id": turn.model.model_id,
            "selected_device": turn.model.selected_device,
        },
        latency_ms=latency_ms,
    )


def _class_one_probability(model: CiftRuntimeLinearModel, feature_vector: tuple[float, ...]) -> float:
    logit = model.logistic_intercept
    for value, mean, scale, coefficient in zip(
        feature_vector,
        model.scaler_mean,
        model.scaler_scale,
        model.logistic_coefficients,
        strict=True,
    ):
        logit += coefficient * ((value - mean) / scale)
    return _sigmoid(logit)


def _positive_probability(model: CiftRuntimeLinearModel, class_one_probability: float) -> float:
    if model.positive_class_index == model.class_indices[1]:
        return class_one_probability
    if model.positive_class_index == model.class_indices[0]:
        return 1.0 - class_one_probability
    raise CiftRuntimeDetectorError("positive_class_index must match one of the class_indices.")


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        denominator = 1.0 + math.exp(-value)
        return 1.0 / denominator
    numerator = math.exp(value)
    return numerator / (1.0 + numerator)


def _negative_label(model: CiftRuntimeLinearModel) -> str:
    return next(label for label in model.label_names if label != model.positive_label)


def _has_selected_choice_readout_indices(turn: NormalizedTurn) -> bool:
    cift_metadata = turn.metadata.get(_CIFT_METADATA_KEY)
    if cift_metadata is None:
        return False
    if not isinstance(cift_metadata, dict):
        raise CiftRuntimeDetectorError("NormalizedTurn metadata.cift must be an object when present.")
    token_indices = cift_metadata.get("selected_choice_readout_token_indices")
    if token_indices is None:
        return False
    if not isinstance(token_indices, list):
        raise CiftRuntimeDetectorError(
            "NormalizedTurn metadata.cift.selected_choice_readout_token_indices must be a list when present."
        )
    if len(token_indices) == 0:
        raise CiftRuntimeDetectorError(
            "NormalizedTurn metadata.cift.selected_choice_readout_token_indices must not be empty when present."
        )
    for index, token_index in enumerate(token_indices):
        if isinstance(token_index, bool) or not isinstance(token_index, int):
            raise CiftRuntimeDetectorError(
                f"NormalizedTurn metadata.cift.selected_choice_readout_token_indices item {index} must be an integer."
            )
        if token_index < 0:
            raise CiftRuntimeDetectorError(
                f"NormalizedTurn metadata.cift.selected_choice_readout_token_indices item {index} must be non-negative."
            )
    return True


def _result_with_window_selection_evidence(
    result: DetectorResult,
    window_family: str,
    selection_reason: str,
    window_coverage: str,
    selected_choice_model: CiftRuntimeLinearModel,
    fallback_model: CiftRuntimeLinearModel,
) -> DetectorResult:
    evidence = dict(result.evidence)
    evidence["cift_window_family"] = window_family
    evidence["cift_window_selection_reason"] = selection_reason
    evidence["cift_window_coverage"] = window_coverage
    evidence["selected_choice_model_bundle_id"] = selected_choice_model.model_bundle_id
    evidence["fallback_model_bundle_id"] = fallback_model.model_bundle_id
    return DetectorResult(
        detector_name=result.detector_name,
        component=result.component,
        score=result.score,
        confidence=result.confidence,
        recommended_action=result.recommended_action,
        capability_required=result.capability_required,
        capability_status=result.capability_status,
        evidence=evidence,
        latency_ms=result.latency_ms,
    )


def _degraded_fallback_result(result: DetectorResult) -> DetectorResult:
    evidence = dict(result.evidence)
    evidence["degradation_reason"] = "selected_choice_metadata_required_for_primary_cift"
    if result.capability_status == CapabilityStatus.ACTIVE:
        capability_status = CapabilityStatus.DEGRADED
    else:
        capability_status = result.capability_status
    return DetectorResult(
        detector_name=result.detector_name,
        component=result.component,
        score=result.score,
        confidence=min(result.confidence, _FALLBACK_CONFIDENCE_CAP),
        recommended_action=result.recommended_action,
        capability_required=result.capability_required,
        capability_status=capability_status,
        evidence=evidence,
        latency_ms=result.latency_ms,
    )


def _copied_cift_metadata(metadata: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    cift_metadata = metadata.get(_CIFT_METADATA_KEY)
    if cift_metadata is None:
        return {}
    if not isinstance(cift_metadata, dict):
        raise CiftRuntimeDetectorError("NormalizedTurn metadata.cift must be an object when present.")
    return dict(cift_metadata)


def _copied_feature_vectors(cift_metadata: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    feature_vectors = cift_metadata.get(_FEATURE_VECTORS_KEY)
    if feature_vectors is None:
        return {}
    if not isinstance(feature_vectors, dict):
        raise CiftRuntimeDetectorError("NormalizedTurn metadata.cift.feature_vectors must be an object when present.")
    return dict(feature_vectors)


def _feature_sources_metadata(
    cift_metadata: Mapping[str, JsonValue],
    feature_key: str,
    source: str,
    feature_count: int,
) -> dict[str, JsonValue]:
    feature_sources = cift_metadata.get("feature_sources")
    if feature_sources is None:
        sources: dict[str, JsonValue] = {}
    elif isinstance(feature_sources, dict):
        sources = dict(feature_sources)
    else:
        raise CiftRuntimeDetectorError("NormalizedTurn metadata.cift.feature_sources must be an object when present.")
    sources[feature_key] = {"source": source, "feature_count": feature_count}
    return sources


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0


def _required_string(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise CiftRuntimeDetectorError(f"Field '{field_name}' must be a string.")
    _validate_required_string(value=value, field_name=field_name)
    return value


def _required_int(record: Mapping[str, object], field_name: str) -> int:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftRuntimeDetectorError(f"Field '{field_name}' must be an integer.")
    return value


def _required_float(record: Mapping[str, object], field_name: str) -> float:
    return _float_item(value=record.get(field_name), field_name=field_name)


def _required_action(record: Mapping[str, object], field_name: str) -> Action:
    value = _required_string(record=record, field_name=field_name)
    try:
        return Action(value)
    except ValueError as exc:
        raise CiftRuntimeDetectorError(f"Field '{field_name}' has unsupported action '{value}'.") from exc


def _required_string_tuple(record: Mapping[str, object], field_name: str) -> tuple[str, ...]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise CiftRuntimeDetectorError(f"Field '{field_name}' must be a list of strings.")
    values: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or item == "":
            raise CiftRuntimeDetectorError(f"Field '{field_name}' item {index} must be a non-empty string.")
        values.append(item)
    return tuple(values)


def _required_two_string_tuple(record: Mapping[str, object], field_name: str) -> tuple[str, str]:
    values = _required_string_tuple(record=record, field_name=field_name)
    if len(values) != 2:
        raise CiftRuntimeDetectorError(f"Field '{field_name}' must contain exactly two strings.")
    return (values[0], values[1])


def _required_two_int_tuple(record: Mapping[str, object], field_name: str) -> tuple[int, int]:
    value = record.get(field_name)
    if not isinstance(value, list) or len(value) != 2:
        raise CiftRuntimeDetectorError(f"Field '{field_name}' must contain exactly two integers.")
    first = value[0]
    second = value[1]
    if isinstance(first, bool) or not isinstance(first, int):
        raise CiftRuntimeDetectorError(f"Field '{field_name}' item 0 must be an integer.")
    if isinstance(second, bool) or not isinstance(second, int):
        raise CiftRuntimeDetectorError(f"Field '{field_name}' item 1 must be an integer.")
    return (first, second)


def _required_float_tuple(record: Mapping[str, object], field_name: str, expected_length: int) -> tuple[float, ...]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise CiftRuntimeDetectorError(f"Field '{field_name}' must be a list of numbers.")
    values = tuple(_float_item(value=item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value))
    _validate_vector_length(values=values, field_name=field_name, expected_length=expected_length)
    return values


def _float_item(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CiftRuntimeDetectorError(f"Field '{field_name}' must be a number.")
    return float(value)


def _validate_required_string(value: str, field_name: str) -> None:
    if value == "":
        raise CiftRuntimeDetectorError(f"{field_name} must not be empty.")


def _validate_sha256(value: str) -> None:
    if len(value) != 64:
        raise CiftRuntimeDetectorError("source_artifact_sha256 must contain 64 hexadecimal characters.")
    if any(character not in "0123456789abcdef" for character in value):
        raise CiftRuntimeDetectorError("source_artifact_sha256 must be lowercase hexadecimal.")


def _validate_probability(value: float, field_name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise CiftRuntimeDetectorError(f"{field_name} must be in [0.0, 1.0].")


def _validate_vector_length(values: tuple[float, ...], field_name: str, expected_length: int) -> None:
    if len(values) != expected_length:
        raise CiftRuntimeDetectorError(f"{field_name} has {len(values)} values, but expected {expected_length}.")
