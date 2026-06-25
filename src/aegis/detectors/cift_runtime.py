from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, cast, runtime_checkable

from aegis.cift_contract import is_cift_immutable_model_revision
from aegis.core.action_severity import action_severity
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
_MLP_SCHEMA_VERSION = "aegis.cift_runtime_mlp/v1"
_CIFT_METADATA_KEY = "cift"
_FEATURE_VECTORS_KEY = "feature_vectors"
_FALLBACK_CONFIDENCE_CAP = 0.35
_OFFLINE_RESEARCH_CANDIDATE = "offline_research_candidate"
_RUNTIME_CANDIDATE = "runtime_candidate"
_PROMOTION_GATES_SCHEMA_VERSION = "cift_promotion_gates/v1"
_PROMOTION_GATE_RESULT_SCHEMA_VERSION = "cift_promotion_gate_result/v1"
_PAPER_MLP_PROBE_ARCHITECTURE = "mlp_128_64_1"
_PAPER_MLP_FIRST_HIDDEN = 128
_PAPER_MLP_SECOND_HIDDEN = 64


class CiftRuntimeDetectorError(ValueError):
    """Raised when a runtime CIFT detector artifact or feature vector is invalid."""


@dataclass(frozen=True)
class CiftRuntimeLinearModel:
    schema_version: str
    model_bundle_id: str
    source_model_id: str
    source_revision: str
    source_selected_device: str
    source_hidden_size: int
    source_layer_count: int
    tokenizer_fingerprint_sha256: str
    special_tokens_map_sha256: str
    chat_template_sha256: str
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
class CiftRuntimeMlpModel:
    schema_version: str
    model_bundle_id: str
    source_model_id: str
    source_revision: str
    source_selected_device: str
    source_hidden_size: int
    source_layer_count: int
    tokenizer_fingerprint_sha256: str
    special_tokens_map_sha256: str
    chat_template_sha256: str
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
    probe_architecture: str
    raw_layer_weights: tuple[float, ...]
    first_weights: tuple[tuple[float, ...], ...]
    first_bias: tuple[float, ...]
    second_weights: tuple[tuple[float, ...], ...]
    second_bias: tuple[float, ...]
    output_weights: tuple[float, ...]
    output_bias: float
    negative_action: Action
    positive_action: Action


CiftRuntimeModel: TypeAlias = CiftRuntimeLinearModel | CiftRuntimeMlpModel
_CiftRuntimeModelValidator: TypeAlias = Callable[[CiftRuntimeModel, str], None]


@dataclass(frozen=True)
class _LoadedRuntimeModelArtifact:
    model: CiftRuntimeModel
    sha256: str


@dataclass(frozen=True)
class CiftRuntimePrediction:
    score: float
    predicted_label: str
    recommended_action: Action
    operating_band: str


@dataclass(frozen=True)
class CiftFeatureExtraction:
    feature_vector: tuple[float, ...] | None
    selected_choice_readout_token_indices: tuple[int, ...] | None
    provenance: Mapping[str, JsonValue]


@dataclass(frozen=True)
class CiftRuntimeWindowSelectorConfig:
    detector_name: str
    selected_choice_model_path: Path
    selected_choice_model_sha256: str
    fallback_model_path: Path | None
    feature_extractor: CiftFeatureExtractor
    feature_source: str
    activation_failure_action: Action


@dataclass(frozen=True)
class CiftRuntimeComponents:
    turn_annotators: tuple[CiftFeatureVectorAnnotator, ...]
    pre_generation_detectors: tuple[CiftRuntimeWindowSelector, ...]


class CiftFeatureExtractor(Protocol):
    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        """Extract a CIFT feature vector for a normalized turn."""


@runtime_checkable
class CiftFeatureExtractionExtractor(Protocol):
    def extract_feature_extraction(self, turn: NormalizedTurn, feature_key: str) -> CiftFeatureExtraction:
        """Extract a CIFT feature vector plus trusted provenance for a normalized turn."""


@runtime_checkable
class CiftSelectedChoiceWindowExtractor(Protocol):
    def extract_selected_choice_readout_token_indices(
        self,
        turn: NormalizedTurn,
        feature_key: str,
    ) -> tuple[int, ...] | None:
        """Extract trusted selected-choice readout token indices for a normalized turn."""


@dataclass(frozen=True)
class CiftFeatureVectorAnnotator:
    feature_key: str
    extractor: CiftFeatureExtractor
    source: str
    selected_choice_window: bool

    def annotate(self, turn: NormalizedTurn) -> NormalizedTurn:
        if self.feature_key == "":
            raise CiftRuntimeDetectorError("feature_key must not be empty.")
        if self.source == "":
            raise CiftRuntimeDetectorError("source must not be empty.")
        if turn.capability_mode not in (CapabilityMode.SELF_HOSTED_INTROSPECTION, CapabilityMode.OFFLINE_EVAL):
            return turn

        try:
            extraction = _feature_extraction_from_extractor(
                extractor=self.extractor,
                turn=turn,
                feature_key=self.feature_key,
            )
        except Exception as exc:
            return normalized_turn_with_cift_activation_failure(
                turn=turn,
                feature_key=self.feature_key,
                reason=f"extractor.{self.feature_key} raised {exc.__class__.__name__}: {exc}",
                source=self.source,
            )
        feature_vector = extraction.feature_vector
        if feature_vector is None:
            return turn
        try:
            validated_vector = tuple(
                _float_item(value=item, field_name=f"extractor.{self.feature_key}[{index}]")
                for index, item in enumerate(feature_vector)
            )
        except CiftRuntimeDetectorError as exc:
            return normalized_turn_with_cift_activation_failure(
                turn=turn,
                feature_key=self.feature_key,
                reason=str(exc),
                source=self.source,
            )
        annotated = normalized_turn_with_cift_feature_vector(
            turn=turn,
            feature_key=self.feature_key,
            feature_vector=validated_vector,
            source=self.source,
            provenance=extraction.provenance,
        )
        if not self.selected_choice_window:
            return annotated
        if extraction.selected_choice_readout_token_indices is not None:
            provenance_readout_count = _selected_choice_readout_token_count_from_provenance(extraction.provenance)
            actual_readout_count = len(extraction.selected_choice_readout_token_indices)
            if provenance_readout_count is None:
                return normalized_turn_with_cift_activation_failure(
                    turn=annotated,
                    feature_key=self.feature_key,
                    reason="extractor.selected_choice_readout_token_count is required for selected-choice metadata",
                    source=self.source,
                )
            if provenance_readout_count != actual_readout_count:
                return normalized_turn_with_cift_activation_failure(
                    turn=annotated,
                    feature_key=self.feature_key,
                    reason=(
                        "extractor.selected_choice_readout_token_indices length must match "
                        "selected_choice_readout_token_count"
                    ),
                    source=self.source,
                )
            return normalized_turn_with_cift_selected_choice_readout_token_indices(
                turn=annotated,
                token_indices=extraction.selected_choice_readout_token_indices,
                source=self.source,
            )
        if isinstance(self.extractor, CiftFeatureExtractionExtractor):
            return normalized_turn_with_cift_activation_failure(
                turn=annotated,
                feature_key=self.feature_key,
                reason=(
                    "extractor.selected_choice_readout_token_indices is required in the feature extraction response "
                    "for selected-choice CIFT"
                ),
                source=self.source,
            )
        try:
            if _has_selected_choice_readout_indices(annotated):
                return annotated
        except CiftRuntimeDetectorError:
            return annotated
        return _annotated_turn_with_extracted_selected_choice_readout_indices(
            turn=annotated,
            feature_key=self.feature_key,
            extractor=self.extractor,
            source=self.source,
        )


class CiftRuntimeDetector:
    def __init__(self, detector_name: str, model: CiftRuntimeModel, activation_failure_action: Action) -> None:
        if detector_name == "":
            raise CiftRuntimeDetectorError("detector_name must not be empty.")
        validate_cift_runtime_model(model)
        self._detector_name = detector_name
        self._model = model
        self._activation_failure_action = activation_failure_action

    @property
    def detector_name(self) -> str:
        return self._detector_name

    @property
    def activation_failure_action(self) -> Action:
        return self._activation_failure_action

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        started_at = time.perf_counter()
        if turn.capability_mode not in (CapabilityMode.SELF_HOSTED_INTROSPECTION, CapabilityMode.OFFLINE_EVAL):
            return _unavailable_result(
                detector_name=self._detector_name,
                model=self._model,
                turn=turn,
                reason="activation_access_unavailable",
                capability_status=CapabilityStatus.UNAVAILABLE,
                recommended_action=Action.ALLOW,
                score=0.0,
                extra_evidence={},
                latency_ms=_elapsed_ms(started_at),
            )

        try:
            feature_vector = cift_feature_vector_from_turn(turn=turn, feature_key=self._model.feature_key)
        except CiftRuntimeDetectorError as exc:
            if action_severity(self._activation_failure_action) < action_severity(Action.BLOCK):
                raise
            return _unavailable_result(
                detector_name=self._detector_name,
                model=self._model,
                turn=turn,
                reason="activation_feature_vector_malformed",
                capability_status=CapabilityStatus.DEGRADED,
                recommended_action=self._activation_failure_action,
                score=_failure_score(self._activation_failure_action),
                extra_evidence=_activation_failure_evidence(
                    action=self._activation_failure_action,
                    error=str(exc),
                ),
                latency_ms=_elapsed_ms(started_at),
            )
        if feature_vector is None:
            try:
                activation_failure = _activation_failure_from_turn(turn=turn, feature_key=self._model.feature_key)
            except CiftRuntimeDetectorError as exc:
                if action_severity(self._activation_failure_action) < action_severity(Action.BLOCK):
                    raise
                return _unavailable_result(
                    detector_name=self._detector_name,
                    model=self._model,
                    turn=turn,
                    reason="activation_feature_vector_malformed",
                    capability_status=CapabilityStatus.DEGRADED,
                    recommended_action=self._activation_failure_action,
                    score=_failure_score(self._activation_failure_action),
                    extra_evidence=_activation_failure_evidence(
                        action=self._activation_failure_action,
                        error=str(exc),
                    ),
                    latency_ms=_elapsed_ms(started_at),
                )
            if activation_failure is not None:
                return _unavailable_result(
                    detector_name=self._detector_name,
                    model=self._model,
                    turn=turn,
                    reason="activation_feature_vector_malformed",
                    capability_status=CapabilityStatus.DEGRADED,
                    recommended_action=self._activation_failure_action,
                    score=_failure_score(self._activation_failure_action),
                    extra_evidence=_activation_failure_evidence(
                        action=self._activation_failure_action,
                        error=activation_failure,
                    ),
                    latency_ms=_elapsed_ms(started_at),
                )
            return _unavailable_result(
                detector_name=self._detector_name,
                model=self._model,
                turn=turn,
                reason="activation_feature_vector_missing",
                capability_status=CapabilityStatus.DEGRADED,
                recommended_action=self._activation_failure_action,
                score=_failure_score(self._activation_failure_action),
                extra_evidence=_activation_failure_evidence(action=self._activation_failure_action, error=None),
                latency_ms=_elapsed_ms(started_at),
            )

        try:
            prediction = predict_cift_runtime_model(model=self._model, feature_vector=feature_vector)
        except CiftRuntimeDetectorError as exc:
            if action_severity(self._activation_failure_action) < action_severity(Action.BLOCK):
                raise
            return _unavailable_result(
                detector_name=self._detector_name,
                model=self._model,
                turn=turn,
                reason="activation_feature_vector_malformed",
                capability_status=CapabilityStatus.DEGRADED,
                recommended_action=self._activation_failure_action,
                score=_failure_score(self._activation_failure_action),
                extra_evidence=_activation_failure_evidence(
                    action=self._activation_failure_action,
                    error=str(exc),
                ),
                latency_ms=_elapsed_ms(started_at),
            )
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
        selected_choice_model: CiftRuntimeModel,
        fallback_model: CiftRuntimeModel | None,
        activation_failure_action: Action,
    ) -> None:
        self._selected_choice_model = selected_choice_model
        self._fallback_model = fallback_model
        self._selected_choice_detector = CiftRuntimeDetector(
            detector_name=detector_name,
            model=selected_choice_model,
            activation_failure_action=activation_failure_action,
        )
        self._fallback_detector = (
            None
            if fallback_model is None
            else CiftRuntimeDetector(
                detector_name=detector_name,
                model=fallback_model,
                activation_failure_action=activation_failure_action,
            )
        )

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        try:
            has_selected_choice_readout_indices = _has_selected_choice_readout_indices(turn)
        except CiftRuntimeDetectorError as exc:
            if action_severity(self._selected_choice_detector.activation_failure_action) < action_severity(
                Action.BLOCK
            ):
                raise
            result = _unavailable_result(
                detector_name=self._selected_choice_detector.detector_name,
                model=self._selected_choice_model,
                turn=turn,
                reason="activation_feature_vector_malformed",
                capability_status=CapabilityStatus.DEGRADED,
                recommended_action=self._selected_choice_detector.activation_failure_action,
                score=_failure_score(self._selected_choice_detector.activation_failure_action),
                extra_evidence=_activation_failure_evidence(
                    action=self._selected_choice_detector.activation_failure_action,
                    error=str(exc),
                ),
                latency_ms=0.0,
            )
            return _result_with_window_selection_evidence(
                result=result,
                window_family="selected_choice",
                selection_reason="selected_choice_metadata_malformed",
                window_coverage="unavailable",
                selected_choice_model=self._selected_choice_model,
                fallback_model=self._fallback_model,
            )
        if has_selected_choice_readout_indices:
            result = self._selected_choice_detector.evaluate(turn=turn, model_response=model_response)
            return _result_with_window_selection_evidence(
                result=result,
                window_family="selected_choice",
                selection_reason="selected_choice_metadata_present",
                window_coverage="primary",
                selected_choice_model=self._selected_choice_model,
                fallback_model=self._fallback_model,
            )
        feature_vector_activation_failure = _feature_vector_activation_failure_from_turn(
            turn=turn,
            feature_key=self._selected_choice_model.feature_key,
        )
        if feature_vector_activation_failure is not None:
            return _selected_choice_feature_vector_activation_failure_result(
                selected_choice_detector=self._selected_choice_detector,
                selected_choice_model=self._selected_choice_model,
                fallback_model=self._fallback_model,
                turn=turn,
                error=feature_vector_activation_failure,
            )
        if action_severity(self._selected_choice_detector.activation_failure_action) >= action_severity(Action.BLOCK):
            return _selected_choice_metadata_absent_result(
                selected_choice_detector=self._selected_choice_detector,
                selected_choice_model=self._selected_choice_model,
                fallback_model=self._fallback_model,
                turn=turn,
            )
        if self._fallback_detector is None:
            return _selected_choice_metadata_absent_result(
                selected_choice_detector=self._selected_choice_detector,
                selected_choice_model=self._selected_choice_model,
                fallback_model=self._fallback_model,
                turn=turn,
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


def _selected_choice_feature_vector_activation_failure_result(
    selected_choice_detector: CiftRuntimeDetector,
    selected_choice_model: CiftRuntimeModel,
    fallback_model: CiftRuntimeModel | None,
    turn: NormalizedTurn,
    error: str,
) -> DetectorResult:
    result = _unavailable_result(
        detector_name=selected_choice_detector.detector_name,
        model=selected_choice_model,
        turn=turn,
        reason="activation_feature_vector_malformed",
        capability_status=CapabilityStatus.DEGRADED,
        recommended_action=selected_choice_detector.activation_failure_action,
        score=_failure_score(selected_choice_detector.activation_failure_action),
        extra_evidence=_activation_failure_evidence(
            action=selected_choice_detector.activation_failure_action,
            error=error,
        ),
        latency_ms=0.0,
    )
    return _result_with_window_selection_evidence(
        result=result,
        window_family="selected_choice",
        selection_reason="selected_choice_feature_vector_activation_failure",
        window_coverage="unavailable",
        selected_choice_model=selected_choice_model,
        fallback_model=fallback_model,
    )


def _selected_choice_metadata_absent_result(
    selected_choice_detector: CiftRuntimeDetector,
    selected_choice_model: CiftRuntimeModel,
    fallback_model: CiftRuntimeModel | None,
    turn: NormalizedTurn,
) -> DetectorResult:
    result = _unavailable_result(
        detector_name=selected_choice_detector.detector_name,
        model=selected_choice_model,
        turn=turn,
        reason="selected_choice_metadata_absent",
        capability_status=CapabilityStatus.DEGRADED,
        recommended_action=selected_choice_detector.activation_failure_action,
        score=_failure_score(selected_choice_detector.activation_failure_action),
        extra_evidence=_activation_failure_evidence(
            action=selected_choice_detector.activation_failure_action,
            error="metadata.cift.selected_choice_readout_token_indices is required for primary CIFT.",
        ),
        latency_ms=0.0,
    )
    return _result_with_window_selection_evidence(
        result=result,
        window_family="selected_choice",
        selection_reason="selected_choice_metadata_absent",
        window_coverage="unavailable",
        selected_choice_model=selected_choice_model,
        fallback_model=fallback_model,
    )


def _feature_vector_activation_failure_from_turn(turn: NormalizedTurn, feature_key: str) -> str | None:
    activation_failure = _activation_failure_from_turn(turn=turn, feature_key=feature_key)
    if activation_failure is None:
        return None
    if activation_failure.startswith(f"extractor.{feature_key} raised "):
        return activation_failure
    if activation_failure.startswith(f"extractor.{feature_key}["):
        return activation_failure
    if activation_failure.startswith("extractor.selected_choice_readout_token_indices raised "):
        return activation_failure
    if activation_failure.startswith("extractor.selected_choice_readout_token_indices "):
        return activation_failure
    if activation_failure.startswith("extractor selected_choice_readout_token_indices "):
        return activation_failure
    return None


def load_cift_runtime_model(path: Path) -> CiftRuntimeModel:
    return _load_cift_runtime_model_artifact(path=path).model


def load_cift_runtime_model_with_sha256(path: Path, expected_sha256: str) -> CiftRuntimeModel:
    if expected_sha256 == "":
        raise CiftRuntimeDetectorError("expected_sha256 must not be empty.")
    artifact = _load_cift_runtime_model_artifact(path=path)
    if artifact.sha256 != expected_sha256:
        raise CiftRuntimeDetectorError(
            f"CIFT runtime model sha256 mismatch for {path}: expected {expected_sha256}, got {artifact.sha256}."
        )
    return artifact.model


def _load_cift_runtime_model_artifact(path: Path) -> _LoadedRuntimeModelArtifact:
    if not path.is_file():
        raise CiftRuntimeDetectorError(f"CIFT runtime model path does not exist: {path}")
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        decoded: object = json.loads(raw_bytes.decode("utf-8"), parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise CiftRuntimeDetectorError(f"Invalid CIFT runtime model JSON in {path}: {exc.msg}.") from exc
    except ValueError as exc:
        raise CiftRuntimeDetectorError(f"Invalid CIFT runtime model JSON in {path}: {exc}.") from exc
    except UnicodeDecodeError as exc:
        raise CiftRuntimeDetectorError(f"Invalid CIFT runtime model encoding in {path}: expected UTF-8.") from exc
    if not isinstance(decoded, dict):
        raise CiftRuntimeDetectorError(f"Expected {path} to contain a JSON object.")
    record = cast(Mapping[str, object], decoded)
    model = cift_runtime_model_from_mapping(record)
    validate_cift_runtime_model(model)
    _validate_cift_runtime_promotion_gates(record=record, model=model)
    return _LoadedRuntimeModelArtifact(model=model, sha256=sha256)


def build_cift_window_selector_runtime_components(config: CiftRuntimeWindowSelectorConfig) -> CiftRuntimeComponents:
    return _build_cift_window_selector_runtime_components(
        config=config,
        runtime_model_validator=_validate_self_hosted_runtime_candidate,
    )


def build_cift_window_selector_gateway_smoke_bootstrap_components(
    config: CiftRuntimeWindowSelectorConfig,
    required_device: str,
) -> CiftRuntimeComponents:
    if required_device == "":
        raise CiftRuntimeDetectorError("required_device must not be empty.")

    def validate_gateway_smoke_bootstrap_candidate(model: CiftRuntimeModel, model_role: str) -> None:
        validate_cift_gateway_smoke_bootstrap_runtime_model(
            model=model,
            model_role=model_role,
            required_device=required_device,
        )

    return _build_cift_window_selector_runtime_components(
        config=config,
        runtime_model_validator=validate_gateway_smoke_bootstrap_candidate,
    )


def validate_cift_gateway_smoke_bootstrap_runtime_model(
    model: CiftRuntimeModel,
    model_role: str,
    required_device: str,
) -> None:
    if model.candidate_status not in (_OFFLINE_RESEARCH_CANDIDATE, _RUNTIME_CANDIDATE):
        raise CiftRuntimeDetectorError(f"{model_role} candidate_status is unsupported.")
    if action_severity(model.positive_action) < action_severity(Action.BLOCK):
        raise CiftRuntimeDetectorError(f"{model_role} positive_action must be block or escalate.")
    if not is_cift_immutable_model_revision(model.source_revision):
        raise CiftRuntimeDetectorError(f"{model_role} source_revision must be an immutable model revision.")
    if model.source_selected_device != required_device:
        raise CiftRuntimeDetectorError(
            f"{model_role} source_selected_device must match required_device, "
            f"got {model.source_selected_device} and {required_device}."
        )
    if model.source_model_id == "Qwen/Qwen3-4B" and required_device != "mps":
        raise CiftRuntimeDetectorError("Qwen/Qwen3-4B gateway smoke bootstrap requires required_device mps.")


def _build_cift_window_selector_runtime_components(
    config: CiftRuntimeWindowSelectorConfig,
    runtime_model_validator: _CiftRuntimeModelValidator,
) -> CiftRuntimeComponents:
    if config.detector_name == "":
        raise CiftRuntimeDetectorError("detector_name must not be empty.")
    if config.feature_source == "":
        raise CiftRuntimeDetectorError("feature_source must not be empty.")
    selected_choice_model = load_cift_runtime_model_with_sha256(
        path=config.selected_choice_model_path,
        expected_sha256=config.selected_choice_model_sha256,
    )
    fallback_model = (
        load_cift_runtime_model(config.fallback_model_path) if config.fallback_model_path is not None else None
    )
    runtime_model_validator(selected_choice_model, "selected_choice_model")
    if fallback_model is not None:
        runtime_model_validator(fallback_model, "fallback_model")
    feature_key_candidates = (
        (selected_choice_model.feature_key,)
        if fallback_model is None
        else (selected_choice_model.feature_key, fallback_model.feature_key)
    )
    feature_keys = tuple(dict.fromkeys(feature_key_candidates))
    turn_annotators = tuple(
        CiftFeatureVectorAnnotator(
            feature_key=feature_key,
            extractor=config.feature_extractor,
            source=config.feature_source,
            selected_choice_window=feature_key == selected_choice_model.feature_key,
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
                activation_failure_action=config.activation_failure_action,
            ),
        ),
    )


def cift_runtime_model_from_mapping(record: Mapping[str, object]) -> CiftRuntimeModel:
    feature_count = _required_int(record=record, field_name="feature_count")
    schema_version = _required_string(record=record, field_name="schema_version")
    if schema_version == _MLP_SCHEMA_VERSION:
        return CiftRuntimeMlpModel(
            schema_version=schema_version,
            model_bundle_id=_required_string(record=record, field_name="model_bundle_id"),
            source_model_id=_required_string(record=record, field_name="source_model_id"),
            source_revision=_required_string(record=record, field_name="source_revision"),
            source_selected_device=_required_string(record=record, field_name="source_selected_device"),
            source_hidden_size=_required_int(record=record, field_name="source_hidden_size"),
            source_layer_count=_required_int(record=record, field_name="source_layer_count"),
            tokenizer_fingerprint_sha256=_required_string(
                record=record,
                field_name="tokenizer_fingerprint_sha256",
            ),
            special_tokens_map_sha256=_required_string(record=record, field_name="special_tokens_map_sha256"),
            chat_template_sha256=_required_string(record=record, field_name="chat_template_sha256"),
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
            probe_architecture=_required_string(record=record, field_name="probe_architecture"),
            raw_layer_weights=_required_float_tuple(
                record=record,
                field_name="raw_layer_weights",
                expected_length=feature_count,
            ),
            first_weights=_required_float_matrix(
                record=record,
                field_name="first_weights",
                expected_rows=feature_count,
                expected_columns=_PAPER_MLP_FIRST_HIDDEN,
            ),
            first_bias=_required_float_tuple(
                record=record,
                field_name="first_bias",
                expected_length=_PAPER_MLP_FIRST_HIDDEN,
            ),
            second_weights=_required_float_matrix(
                record=record,
                field_name="second_weights",
                expected_rows=_PAPER_MLP_FIRST_HIDDEN,
                expected_columns=_PAPER_MLP_SECOND_HIDDEN,
            ),
            second_bias=_required_float_tuple(
                record=record,
                field_name="second_bias",
                expected_length=_PAPER_MLP_SECOND_HIDDEN,
            ),
            output_weights=_required_float_tuple(
                record=record,
                field_name="output_weights",
                expected_length=_PAPER_MLP_SECOND_HIDDEN,
            ),
            output_bias=_required_float(record=record, field_name="output_bias"),
            negative_action=_required_action(record=record, field_name="negative_action"),
            positive_action=_required_action(record=record, field_name="positive_action"),
        )
    return CiftRuntimeLinearModel(
        schema_version=schema_version,
        model_bundle_id=_required_string(record=record, field_name="model_bundle_id"),
        source_model_id=_required_string(record=record, field_name="source_model_id"),
        source_revision=_required_string(record=record, field_name="source_revision"),
        source_selected_device=_required_string(record=record, field_name="source_selected_device"),
        source_hidden_size=_required_int(record=record, field_name="source_hidden_size"),
        source_layer_count=_required_int(record=record, field_name="source_layer_count"),
        tokenizer_fingerprint_sha256=_required_string(record=record, field_name="tokenizer_fingerprint_sha256"),
        special_tokens_map_sha256=_required_string(record=record, field_name="special_tokens_map_sha256"),
        chat_template_sha256=_required_string(record=record, field_name="chat_template_sha256"),
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


def cift_runtime_model_to_dict(model: CiftRuntimeModel) -> dict[str, JsonValue]:
    validate_cift_runtime_model(model)
    common: dict[str, JsonValue] = {
        "schema_version": model.schema_version,
        "model_bundle_id": model.model_bundle_id,
        "source_model_id": model.source_model_id,
        "source_revision": model.source_revision,
        "source_selected_device": model.source_selected_device,
        "source_hidden_size": model.source_hidden_size,
        "source_layer_count": model.source_layer_count,
        "tokenizer_fingerprint_sha256": model.tokenizer_fingerprint_sha256,
        "special_tokens_map_sha256": model.special_tokens_map_sha256,
        "chat_template_sha256": model.chat_template_sha256,
        "training_dataset_id": model.training_dataset_id,
        "source_artifact_sha256": model.source_artifact_sha256,
        "evaluation_report_ids": cast(list[JsonValue], list(model.evaluation_report_ids)),
        "task_name": model.task_name,
        "feature_key": model.feature_key,
        "feature_count": model.feature_count,
        "label_names": cast(list[JsonValue], list(model.label_names)),
        "positive_label": model.positive_label,
        "positive_class_index": model.positive_class_index,
        "class_indices": cast(list[JsonValue], list(model.class_indices)),
        "decision_threshold": model.decision_threshold,
        "score_semantics": model.score_semantics,
        "confidence": model.confidence,
        "candidate_status": model.candidate_status,
        "negative_action": model.negative_action.value,
        "positive_action": model.positive_action.value,
    }
    if isinstance(model, CiftRuntimeLinearModel):
        result = dict(common)
        result["scaler_mean"] = cast(list[JsonValue], list(model.scaler_mean))
        result["scaler_scale"] = cast(list[JsonValue], list(model.scaler_scale))
        result["logistic_coefficients"] = cast(list[JsonValue], list(model.logistic_coefficients))
        result["logistic_intercept"] = model.logistic_intercept
        return result
    result = dict(common)
    result["probe_architecture"] = model.probe_architecture
    result["raw_layer_weights"] = cast(list[JsonValue], list(model.raw_layer_weights))
    result["first_weights"] = cast(list[JsonValue], [list(row) for row in model.first_weights])
    result["first_bias"] = cast(list[JsonValue], list(model.first_bias))
    result["second_weights"] = cast(list[JsonValue], [list(row) for row in model.second_weights])
    result["second_bias"] = cast(list[JsonValue], list(model.second_bias))
    result["output_weights"] = cast(list[JsonValue], list(model.output_weights))
    result["output_bias"] = model.output_bias
    return result


def predict_cift_runtime_model(
    model: CiftRuntimeModel,
    feature_vector: tuple[float, ...],
) -> CiftRuntimePrediction:
    validate_cift_runtime_model(model)
    if len(feature_vector) != model.feature_count:
        raise CiftRuntimeDetectorError(
            f"feature_vector has {len(feature_vector)} values, but model expects {model.feature_count}."
        )
    _validate_feature_vector_finite(feature_vector=feature_vector)
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
    provenance: Mapping[str, JsonValue],
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
        provenance=provenance,
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


def _feature_extraction_from_extractor(
    extractor: CiftFeatureExtractor,
    turn: NormalizedTurn,
    feature_key: str,
) -> CiftFeatureExtraction:
    if isinstance(extractor, CiftFeatureExtractionExtractor):
        return extractor.extract_feature_extraction(turn=turn, feature_key=feature_key)
    return CiftFeatureExtraction(
        feature_vector=extractor.extract_feature_vector(turn=turn, feature_key=feature_key),
        selected_choice_readout_token_indices=None,
        provenance={},
    )


def normalized_turn_with_cift_selected_choice_readout_token_indices(
    turn: NormalizedTurn,
    token_indices: tuple[int, ...],
    source: str,
) -> NormalizedTurn:
    if source == "":
        raise CiftRuntimeDetectorError("source must not be empty.")
    encoded_token_indices = _validated_extractor_selected_choice_token_indices(token_indices)
    cift_metadata = _copied_cift_metadata(turn.metadata)
    cift_metadata["selected_choice_readout_token_indices"] = list(encoded_token_indices)
    cift_metadata["selected_choice_readout_source"] = {
        "source": source,
        "token_count": len(encoded_token_indices),
    }
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


def normalized_turn_with_cift_activation_failure(
    turn: NormalizedTurn,
    feature_key: str,
    reason: str,
    source: str,
) -> NormalizedTurn:
    if feature_key == "":
        raise CiftRuntimeDetectorError("feature_key must not be empty.")
    if reason == "":
        raise CiftRuntimeDetectorError("reason must not be empty.")
    if source == "":
        raise CiftRuntimeDetectorError("source must not be empty.")
    cift_metadata = _copied_cift_metadata(turn.metadata)
    activation_failures = _copied_activation_failures(cift_metadata)
    activation_failures[feature_key] = {"reason": reason, "source": source}
    cift_metadata["activation_failures"] = activation_failures
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


def _annotated_turn_with_extracted_selected_choice_readout_indices(
    turn: NormalizedTurn,
    feature_key: str,
    extractor: CiftFeatureExtractor,
    source: str,
) -> NormalizedTurn:
    if not isinstance(extractor, CiftSelectedChoiceWindowExtractor):
        return normalized_turn_with_cift_activation_failure(
            turn=turn,
            feature_key=feature_key,
            reason="extractor does not provide selected_choice_readout_token_indices",
            source=source,
        )
    try:
        token_indices = extractor.extract_selected_choice_readout_token_indices(turn=turn, feature_key=feature_key)
    except Exception as exc:
        return normalized_turn_with_cift_activation_failure(
            turn=turn,
            feature_key=feature_key,
            reason=(f"extractor.selected_choice_readout_token_indices raised {exc.__class__.__name__}: {exc}"),
            source=source,
        )
    if token_indices is None:
        return normalized_turn_with_cift_activation_failure(
            turn=turn,
            feature_key=feature_key,
            reason="extractor returned no selected_choice_readout_token_indices",
            source=source,
        )
    try:
        return normalized_turn_with_cift_selected_choice_readout_token_indices(
            turn=turn,
            token_indices=token_indices,
            source=source,
        )
    except CiftRuntimeDetectorError as exc:
        return normalized_turn_with_cift_activation_failure(
            turn=turn,
            feature_key=feature_key,
            reason=str(exc),
            source=source,
        )


def validate_cift_runtime_model(model: CiftRuntimeModel) -> None:
    if isinstance(model, CiftRuntimeLinearModel):
        _validate_linear_model(model)
        return
    _validate_mlp_model(model)


def _validate_self_hosted_runtime_candidate(model: CiftRuntimeModel, model_role: str) -> None:
    if model.candidate_status != _RUNTIME_CANDIDATE:
        raise CiftRuntimeDetectorError(f"{model_role} must be a runtime_candidate artifact.")
    if action_severity(model.positive_action) < action_severity(Action.BLOCK):
        raise CiftRuntimeDetectorError(f"{model_role} positive_action must be block or escalate.")


def _validate_common_model_fields(model: CiftRuntimeModel) -> None:
    _validate_required_string(value=model.model_bundle_id, field_name="model_bundle_id")
    _validate_required_string(value=model.source_model_id, field_name="source_model_id")
    _validate_required_string(value=model.source_revision, field_name="source_revision")
    _validate_required_string(value=model.source_selected_device, field_name="source_selected_device")
    _validate_positive_int(value=model.source_hidden_size, field_name="source_hidden_size")
    _validate_positive_int(value=model.source_layer_count, field_name="source_layer_count")
    _validate_sha256_field(value=model.tokenizer_fingerprint_sha256, field_name="tokenizer_fingerprint_sha256")
    _validate_sha256_field(value=model.special_tokens_map_sha256, field_name="special_tokens_map_sha256")
    _validate_sha256_field(value=model.chat_template_sha256, field_name="chat_template_sha256")
    _validate_required_string(value=model.training_dataset_id, field_name="training_dataset_id")
    _validate_required_string(value=model.source_artifact_sha256, field_name="source_artifact_sha256")
    _validate_required_string(value=model.task_name, field_name="task_name")
    _validate_required_string(value=model.feature_key, field_name="feature_key")
    _validate_required_string(value=model.positive_label, field_name="positive_label")
    _validate_required_string(value=model.score_semantics, field_name="score_semantics")
    _validate_required_string(value=model.candidate_status, field_name="candidate_status")
    if model.candidate_status not in (_OFFLINE_RESEARCH_CANDIDATE, _RUNTIME_CANDIDATE):
        raise CiftRuntimeDetectorError(f"Unsupported CIFT candidate_status '{model.candidate_status}'.")
    _validate_runtime_candidate_positive_action(model)
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


def _validate_linear_model(model: CiftRuntimeLinearModel) -> None:
    if model.schema_version != _SCHEMA_VERSION:
        raise CiftRuntimeDetectorError(f"Unsupported CIFT runtime model schema '{model.schema_version}'.")
    _validate_common_model_fields(model)
    _validate_vector_length(values=model.scaler_mean, field_name="scaler_mean", expected_length=model.feature_count)
    _validate_vector_length(values=model.scaler_scale, field_name="scaler_scale", expected_length=model.feature_count)
    _validate_vector_length(
        values=model.logistic_coefficients,
        field_name="logistic_coefficients",
        expected_length=model.feature_count,
    )
    _validate_vector_finite(values=model.scaler_mean, field_name="scaler_mean")
    _validate_vector_finite(values=model.scaler_scale, field_name="scaler_scale")
    _validate_vector_finite(values=model.logistic_coefficients, field_name="logistic_coefficients")
    _validate_finite(value=model.logistic_intercept, field_name="logistic_intercept")
    for index, scale in enumerate(model.scaler_scale):
        if scale <= 0.0:
            raise CiftRuntimeDetectorError(f"scaler_scale[{index}] must be greater than 0.")


def _validate_mlp_model(model: CiftRuntimeMlpModel) -> None:
    if model.schema_version != _MLP_SCHEMA_VERSION:
        raise CiftRuntimeDetectorError(f"Unsupported CIFT runtime model schema '{model.schema_version}'.")
    _validate_common_model_fields(model)
    if model.probe_architecture != _PAPER_MLP_PROBE_ARCHITECTURE:
        raise CiftRuntimeDetectorError("probe_architecture must be mlp_128_64_1.")
    _validate_vector_length(
        values=model.raw_layer_weights,
        field_name="raw_layer_weights",
        expected_length=model.feature_count,
    )
    _validate_matrix_shape(
        values=model.first_weights,
        field_name="first_weights",
        expected_rows=model.feature_count,
        expected_columns=_PAPER_MLP_FIRST_HIDDEN,
    )
    _validate_vector_length(
        values=model.first_bias,
        field_name="first_bias",
        expected_length=_PAPER_MLP_FIRST_HIDDEN,
    )
    _validate_matrix_shape(
        values=model.second_weights,
        field_name="second_weights",
        expected_rows=_PAPER_MLP_FIRST_HIDDEN,
        expected_columns=_PAPER_MLP_SECOND_HIDDEN,
    )
    _validate_vector_length(
        values=model.second_bias,
        field_name="second_bias",
        expected_length=_PAPER_MLP_SECOND_HIDDEN,
    )
    _validate_vector_length(
        values=model.output_weights,
        field_name="output_weights",
        expected_length=_PAPER_MLP_SECOND_HIDDEN,
    )


def _active_evidence(
    model: CiftRuntimeModel,
    turn: NormalizedTurn,
    prediction: CiftRuntimePrediction,
) -> dict[str, JsonValue]:
    evidence: dict[str, JsonValue] = {
        "model_bundle_id": model.model_bundle_id,
        "source_model_id": model.source_model_id,
        "source_revision": model.source_revision,
        "source_selected_device": model.source_selected_device,
        "source_hidden_size": model.source_hidden_size,
        "source_layer_count": model.source_layer_count,
        "tokenizer_fingerprint_sha256": model.tokenizer_fingerprint_sha256,
        "special_tokens_map_sha256": model.special_tokens_map_sha256,
        "chat_template_sha256": model.chat_template_sha256,
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
    feature_source = _feature_source(turn=turn, feature_key=model.feature_key)
    if feature_source is not None:
        evidence["feature_source"] = feature_source
    feature_provenance = _feature_provenance(turn=turn, feature_key=model.feature_key)
    if feature_provenance is not None:
        evidence.update(_feature_provenance_evidence(feature_provenance))
    if isinstance(model, CiftRuntimeMlpModel):
        evidence["probe_architecture"] = model.probe_architecture
    return evidence


def _unavailable_result(
    detector_name: str,
    model: CiftRuntimeModel,
    turn: NormalizedTurn,
    reason: str,
    capability_status: CapabilityStatus,
    recommended_action: Action,
    score: float,
    extra_evidence: dict[str, JsonValue],
    latency_ms: float,
) -> DetectorResult:
    evidence: dict[str, JsonValue] = {
        "reason": reason,
        "model_bundle_id": model.model_bundle_id,
        "required_capability": CapabilityMode.SELF_HOSTED_INTROSPECTION.value,
        "actual_capability_mode": turn.capability_mode.value,
        "feature_key": model.feature_key,
        "source_selected_device": model.source_selected_device,
        "model_id": turn.model.model_id,
        "selected_device": turn.model.selected_device,
    }
    evidence.update(extra_evidence)
    return DetectorResult(
        detector_name=detector_name,
        component=DetectorComponent.CIFT,
        score=score,
        confidence=1.0,
        recommended_action=recommended_action,
        capability_required=CapabilityMode.SELF_HOSTED_INTROSPECTION.value,
        capability_status=capability_status,
        evidence=evidence,
        latency_ms=latency_ms,
    )


def _failure_score(action: Action) -> float:
    if action_severity(action) >= action_severity(Action.BLOCK):
        return 1.0
    return 0.0


def _activation_failure_evidence(action: Action, error: str | None) -> dict[str, JsonValue]:
    evidence: dict[str, JsonValue] = {}
    if error is not None:
        evidence["error"] = error
    fail_closed_reason = _fail_closed_reason(action)
    if fail_closed_reason is not None:
        evidence["fail_closed_reason"] = fail_closed_reason
    return evidence


def _fail_closed_reason(action: Action) -> str | None:
    if action_severity(action) >= action_severity(Action.BLOCK):
        return "runtime_candidate_missing_feature_fail_closed"
    return None


def _class_one_probability(model: CiftRuntimeModel, feature_vector: tuple[float, ...]) -> float:
    if isinstance(model, CiftRuntimeMlpModel):
        return _mlp_class_one_probability(model=model, feature_vector=feature_vector)
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


def _mlp_class_one_probability(model: CiftRuntimeMlpModel, feature_vector: tuple[float, ...]) -> float:
    layer_weights = tuple(_softplus(value) for value in model.raw_layer_weights)
    weighted_inputs = tuple(value * weight for value, weight in zip(feature_vector, layer_weights, strict=True))
    first_activations = tuple(
        max(
            0.0,
            model.first_bias[column]
            + sum(weighted_inputs[row] * model.first_weights[row][column] for row in range(model.feature_count)),
        )
        for column in range(_PAPER_MLP_FIRST_HIDDEN)
    )
    second_activations = tuple(
        max(
            0.0,
            model.second_bias[column]
            + sum(first_activations[row] * model.second_weights[row][column] for row in range(_PAPER_MLP_FIRST_HIDDEN)),
        )
        for column in range(_PAPER_MLP_SECOND_HIDDEN)
    )
    output_logit = model.output_bias + sum(
        activation * weight for activation, weight in zip(second_activations, model.output_weights, strict=True)
    )
    return _sigmoid(output_logit)


def _positive_probability(model: CiftRuntimeModel, class_one_probability: float) -> float:
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


def _softplus(value: float) -> float:
    return math.log1p(math.exp(-abs(value))) + max(value, 0.0)


def _negative_label(model: CiftRuntimeModel) -> str:
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


def _validated_extractor_selected_choice_token_indices(token_indices: object) -> tuple[int, ...]:
    if not isinstance(token_indices, tuple):
        raise CiftRuntimeDetectorError("extractor selected_choice_readout_token_indices must be a tuple.")
    if len(token_indices) == 0:
        raise CiftRuntimeDetectorError("extractor selected_choice_readout_token_indices must not be empty.")
    validated: list[int] = []
    for index, token_index in enumerate(token_indices):
        if isinstance(token_index, bool) or not isinstance(token_index, int):
            raise CiftRuntimeDetectorError(
                f"extractor selected_choice_readout_token_indices item {index} must be an integer."
            )
        if token_index < 0:
            raise CiftRuntimeDetectorError(
                f"extractor selected_choice_readout_token_indices item {index} must be non-negative."
            )
        validated.append(token_index)
    return tuple(validated)


def _selected_choice_readout_token_count_from_provenance(provenance: Mapping[str, JsonValue]) -> int | None:
    value = provenance.get("selected_choice_readout_token_count")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 1:
        return None
    return value


def _result_with_window_selection_evidence(
    result: DetectorResult,
    window_family: str,
    selection_reason: str,
    window_coverage: str,
    selected_choice_model: CiftRuntimeModel,
    fallback_model: CiftRuntimeModel | None,
) -> DetectorResult:
    evidence = dict(result.evidence)
    evidence["cift_window_family"] = window_family
    evidence["cift_window_selection_reason"] = selection_reason
    evidence["cift_window_coverage"] = window_coverage
    evidence["selected_choice_model_bundle_id"] = selected_choice_model.model_bundle_id
    if fallback_model is not None:
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


def _copied_activation_failures(cift_metadata: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    activation_failures = cift_metadata.get("activation_failures")
    if activation_failures is None:
        return {}
    if not isinstance(activation_failures, dict):
        raise CiftRuntimeDetectorError(
            "NormalizedTurn metadata.cift.activation_failures must be an object when present."
        )
    return dict(activation_failures)


def _activation_failure_from_turn(turn: NormalizedTurn, feature_key: str) -> str | None:
    cift_metadata = turn.metadata.get(_CIFT_METADATA_KEY)
    if cift_metadata is None:
        return None
    if not isinstance(cift_metadata, dict):
        raise CiftRuntimeDetectorError("NormalizedTurn metadata.cift must be an object when present.")
    activation_failures = cift_metadata.get("activation_failures")
    if activation_failures is None:
        return None
    if not isinstance(activation_failures, dict):
        raise CiftRuntimeDetectorError(
            "NormalizedTurn metadata.cift.activation_failures must be an object when present."
        )
    failure = activation_failures.get(feature_key)
    if failure is None:
        return None
    if not isinstance(failure, dict):
        raise CiftRuntimeDetectorError(
            f"NormalizedTurn metadata.cift.activation_failures.{feature_key} must be an object when present."
        )
    reason = failure.get("reason")
    if not isinstance(reason, str) or reason == "":
        raise CiftRuntimeDetectorError(
            f"NormalizedTurn metadata.cift.activation_failures.{feature_key}.reason must be a non-empty string."
        )
    return reason


def _feature_sources_metadata(
    cift_metadata: Mapping[str, JsonValue],
    feature_key: str,
    source: str,
    feature_count: int,
    provenance: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    feature_sources = cift_metadata.get("feature_sources")
    if feature_sources is None:
        sources: dict[str, JsonValue] = {}
    elif isinstance(feature_sources, dict):
        sources = dict(feature_sources)
    else:
        raise CiftRuntimeDetectorError("NormalizedTurn metadata.cift.feature_sources must be an object when present.")
    source_metadata: dict[str, JsonValue] = {"source": source, "feature_count": feature_count}
    if len(provenance) > 0:
        source_metadata["provenance"] = dict(provenance)
    sources[feature_key] = source_metadata
    return sources


def _feature_source(turn: NormalizedTurn, feature_key: str) -> str | None:
    cift_metadata = turn.metadata.get(_CIFT_METADATA_KEY)
    if not isinstance(cift_metadata, dict):
        return None
    feature_sources = cift_metadata.get("feature_sources")
    if not isinstance(feature_sources, dict):
        return None
    feature_source = feature_sources.get(feature_key)
    if not isinstance(feature_source, dict):
        return None
    source = feature_source.get("source")
    if not isinstance(source, str) or source == "":
        return None
    return source


def _feature_provenance(turn: NormalizedTurn, feature_key: str) -> dict[str, JsonValue] | None:
    cift_metadata = turn.metadata.get(_CIFT_METADATA_KEY)
    if not isinstance(cift_metadata, dict):
        return None
    feature_sources = cift_metadata.get("feature_sources")
    if not isinstance(feature_sources, dict):
        return None
    feature_source = feature_sources.get(feature_key)
    if not isinstance(feature_source, dict):
        return None
    provenance = feature_source.get("provenance")
    if not isinstance(provenance, dict):
        return None
    return dict(provenance)


def _feature_provenance_evidence(provenance: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    evidence: dict[str, JsonValue] = {}
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="extractor_id",
        destination_field="extractor_id",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="model_attestation_schema_version",
        destination_field="extractor_model_attestation_schema_version",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="model_id",
        destination_field="extractor_model_id",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="revision",
        destination_field="extractor_revision",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="selected_device",
        destination_field="extractor_selected_device",
    )
    _copy_int_provenance(
        source=provenance,
        destination=evidence,
        source_field="hidden_size",
        destination_field="extractor_hidden_size",
    )
    _copy_int_provenance(
        source=provenance,
        destination=evidence,
        source_field="layer_count",
        destination_field="extractor_layer_count",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="tokenizer_fingerprint_sha256",
        destination_field="extractor_tokenizer_fingerprint_sha256",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="special_tokens_map_sha256",
        destination_field="extractor_special_tokens_map_sha256",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="chat_template_sha256",
        destination_field="extractor_chat_template_sha256",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="prompt_renderer",
        destination_field="extractor_prompt_renderer",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="selected_choice_geometry",
        destination_field="extractor_selected_choice_geometry",
    )
    _copy_int_provenance(
        source=provenance,
        destination=evidence,
        source_field="selected_choice_readout_token_count",
        destination_field="extractor_selected_choice_readout_token_count",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="extraction_receipt_schema_version",
        destination_field="extractor_extraction_receipt_schema_version",
    )
    _copy_int_provenance(
        source=provenance,
        destination=evidence,
        source_field="feature_vector_length",
        destination_field="extractor_feature_vector_length",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="feature_vector_sha256",
        destination_field="extractor_feature_vector_sha256",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="rendered_prompt_sha256",
        destination_field="extractor_rendered_prompt_sha256",
    )
    _copy_int_list_provenance(
        source=provenance,
        destination=evidence,
        source_field="selected_choice_readout_token_indices",
        destination_field="extractor_selected_choice_readout_token_indices",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="selected_choice_readout_token_indices_sha256",
        destination_field="extractor_selected_choice_readout_token_indices_sha256",
    )
    _copy_int_provenance(
        source=provenance,
        destination=evidence,
        source_field="hidden_state_layer_count",
        destination_field="extractor_hidden_state_layer_count",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="hidden_state_device_observed",
        destination_field="extractor_hidden_state_device_observed",
    )
    _copy_string_provenance(
        source=provenance,
        destination=evidence,
        source_field="input_device_observed",
        destination_field="extractor_input_device_observed",
    )
    return evidence


def _copy_string_provenance(
    source: Mapping[str, JsonValue],
    destination: dict[str, JsonValue],
    source_field: str,
    destination_field: str,
) -> None:
    value = source.get(source_field)
    if isinstance(value, str) and value != "":
        destination[destination_field] = value


def _copy_int_provenance(
    source: Mapping[str, JsonValue],
    destination: dict[str, JsonValue],
    source_field: str,
    destination_field: str,
) -> None:
    value = source.get(source_field)
    if isinstance(value, bool) or not isinstance(value, int):
        return
    destination[destination_field] = value


def _copy_int_list_provenance(
    source: Mapping[str, JsonValue],
    destination: dict[str, JsonValue],
    source_field: str,
    destination_field: str,
) -> None:
    value = source.get(source_field)
    if not isinstance(value, list):
        return
    copied: list[JsonValue] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return
        copied.append(item)
    destination[destination_field] = copied


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


def _required_float_matrix(
    record: Mapping[str, object],
    field_name: str,
    expected_rows: int,
    expected_columns: int,
) -> tuple[tuple[float, ...], ...]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise CiftRuntimeDetectorError(f"Field '{field_name}' must be a list of number lists.")
    rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list):
            raise CiftRuntimeDetectorError(f"Field '{field_name}[{row_index}]' must be a list of numbers.")
        rows.append(
            tuple(
                _float_item(value=item, field_name=f"{field_name}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            )
        )
    matrix = tuple(rows)
    _validate_matrix_shape(
        values=matrix,
        field_name=field_name,
        expected_rows=expected_rows,
        expected_columns=expected_columns,
    )
    return matrix


def _float_item(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CiftRuntimeDetectorError(f"Field '{field_name}' must be a number.")
    numeric_value = float(value)
    _validate_finite(value=numeric_value, field_name=field_name)
    return numeric_value


def _validate_finite(value: float, field_name: str) -> None:
    if not math.isfinite(value):
        raise CiftRuntimeDetectorError(f"{field_name} must be finite.")


def _validate_vector_finite(values: tuple[float, ...], field_name: str) -> None:
    for index, value in enumerate(values):
        _validate_finite(value=value, field_name=f"{field_name}[{index}]")


def _validate_required_string(value: str, field_name: str) -> None:
    if value == "":
        raise CiftRuntimeDetectorError(f"{field_name} must not be empty.")


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftRuntimeDetectorError(f"{field_name} must be an integer.")
    if value < 1:
        raise CiftRuntimeDetectorError(f"{field_name} must be positive.")


def _validate_sha256(value: str) -> None:
    _validate_sha256_field(value=value, field_name="source_artifact_sha256")


def _validate_sha256_field(value: str, field_name: str) -> None:
    if len(value) != 64:
        raise CiftRuntimeDetectorError(f"{field_name} must contain 64 hexadecimal characters.")
    if any(character not in "0123456789abcdef" for character in value):
        raise CiftRuntimeDetectorError(f"{field_name} must be lowercase hexadecimal.")


def _validate_probability(value: float, field_name: str) -> None:
    _validate_finite(value=value, field_name=field_name)
    if value < 0.0 or value > 1.0:
        raise CiftRuntimeDetectorError(f"{field_name} must be in [0.0, 1.0].")


def _validate_vector_length(values: tuple[float, ...], field_name: str, expected_length: int) -> None:
    if len(values) != expected_length:
        raise CiftRuntimeDetectorError(f"{field_name} has {len(values)} values, but expected {expected_length}.")


def _validate_feature_vector_finite(feature_vector: tuple[float, ...]) -> None:
    for index, value in enumerate(feature_vector):
        if not math.isfinite(value):
            raise CiftRuntimeDetectorError(f"feature_vector[{index}] must be finite.")


def _validate_matrix_shape(
    values: tuple[tuple[float, ...], ...],
    field_name: str,
    expected_rows: int,
    expected_columns: int,
) -> None:
    if len(values) != expected_rows:
        raise CiftRuntimeDetectorError(f"{field_name} has {len(values)} rows, but expected {expected_rows}.")
    for row_index, row in enumerate(values):
        if len(row) != expected_columns:
            raise CiftRuntimeDetectorError(
                f"{field_name}[{row_index}] has {len(row)} values, but expected {expected_columns}."
            )


def _validate_runtime_candidate_positive_action(model: CiftRuntimeModel) -> None:
    if model.candidate_status != _RUNTIME_CANDIDATE:
        return
    if action_severity(model.positive_action) < action_severity(Action.BLOCK):
        raise CiftRuntimeDetectorError("runtime_candidate positive_action must be block or escalate.")


def _validate_cift_runtime_promotion_gates(
    record: Mapping[str, object],
    model: CiftRuntimeModel,
) -> None:
    raw_gates = record.get("promotion_gates")
    if model.candidate_status != _RUNTIME_CANDIDATE:
        if raw_gates is not None:
            raise CiftRuntimeDetectorError("promotion_gates are only valid for runtime_candidate artifacts.")
        return
    gates = _required_mapping_value(value=raw_gates, field_name="promotion_gates")
    schema_version = _required_string(record=gates, field_name="schema_version")
    if schema_version != _PROMOTION_GATES_SCHEMA_VERSION:
        raise CiftRuntimeDetectorError(f"promotion_gates.schema_version must be {_PROMOTION_GATES_SCHEMA_VERSION}.")
    runtime_candidate = _required_mapping_value(
        value=gates.get("runtime_candidate"),
        field_name="promotion_gates.runtime_candidate",
    )
    _validate_runtime_candidate_gate(gate=runtime_candidate, model=model)


def _validate_runtime_candidate_gate(gate: Mapping[str, object], model: CiftRuntimeModel) -> None:
    schema_version = _required_string(record=gate, field_name="schema_version")
    if schema_version != _PROMOTION_GATE_RESULT_SCHEMA_VERSION:
        raise CiftRuntimeDetectorError(
            f"promotion_gates.runtime_candidate.schema_version must be {_PROMOTION_GATE_RESULT_SCHEMA_VERSION}."
        )
    gate_candidate_status = _required_string(record=gate, field_name="candidate_status")
    if gate_candidate_status != _RUNTIME_CANDIDATE:
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.candidate_status must be runtime_candidate.")
    failed_requirements = _required_string_tuple(record=gate, field_name="failed_requirements")
    if len(failed_requirements) > 0:
        joined = ", ".join(failed_requirements)
        raise CiftRuntimeDetectorError(
            f"promotion_gates.runtime_candidate.failed_requirements must be empty: {joined}."
        )
    missing_report_ids = _required_string_tuple(record=gate, field_name="missing_report_ids")
    if len(missing_report_ids) > 0:
        joined = ", ".join(missing_report_ids)
        raise CiftRuntimeDetectorError(f"promotion_gates.runtime_candidate.missing_report_ids must be empty: {joined}.")
    if _required_bool(record=gate, field_name="eligible") is not True:
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.eligible must be true.")
    training_dataset_id = _required_string(record=gate, field_name="training_dataset_id")
    if training_dataset_id != model.training_dataset_id:
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.training_dataset_id must match model.")
    required_report_ids = _required_string_tuple(record=gate, field_name="required_report_ids")
    if len(required_report_ids) == 0:
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.required_report_ids must not be empty.")
    model_report_ids = set(model.evaluation_report_ids)
    missing_from_model = tuple(report_id for report_id in required_report_ids if report_id not in model_report_ids)
    if len(missing_from_model) > 0:
        joined = ", ".join(missing_from_model)
        raise CiftRuntimeDetectorError(
            f"promotion_gates.runtime_candidate.required_report_ids missing from model evaluation_report_ids: {joined}."
        )
    _validate_promotion_report_artifacts(gate=gate, required_report_ids=required_report_ids)
    _validate_promotion_metric(gate=gate)
    _validate_promotion_ablation(gate=gate)
    _validate_promotion_splits(gate=gate)
    _validate_promotion_report_references(gate=gate, required_report_ids=required_report_ids)
    paper_method = _required_mapping_value(
        value=gate.get("paper_method"),
        field_name="promotion_gates.runtime_candidate.paper_method",
    )
    _validate_promotion_paper_method(method=paper_method)
    _validate_promotion_model_probe_consistency(method=paper_method, model=model)
    _validate_promotion_head_to_head_report_id(
        method=paper_method,
        required_report_ids=required_report_ids,
        model_report_ids=model_report_ids,
    )


def _validate_promotion_report_artifacts(
    gate: Mapping[str, object],
    required_report_ids: tuple[str, ...],
) -> None:
    raw_artifacts = gate.get("report_artifacts")
    if not isinstance(raw_artifacts, list):
        raise CiftRuntimeDetectorError("Field 'promotion_gates.runtime_candidate.report_artifacts' must be a list.")
    if len(raw_artifacts) == 0:
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.report_artifacts must cover required_report_ids."
        )
    report_ids: list[str] = []
    for index, raw_artifact in enumerate(raw_artifacts):
        artifact = _required_mapping_value(
            value=raw_artifact,
            field_name=f"promotion_gates.runtime_candidate.report_artifacts[{index}]",
        )
        report_id = _required_string(record=artifact, field_name="report_id")
        _required_string(record=artifact, field_name="path")
        sha256 = _required_string(record=artifact, field_name="sha256")
        _validate_sha256_field(
            value=sha256,
            field_name=f"promotion_gates.runtime_candidate.report_artifacts[{index}].sha256",
        )
        _required_string(record=artifact, field_name="schema_version")
        report_ids.append(report_id)
    if len(set(report_ids)) != len(report_ids):
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.report_artifacts report_id values must be unique."
        )
    if set(report_ids) != set(required_report_ids):
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.report_artifacts must cover required_report_ids."
        )


def _validate_promotion_metric(gate: Mapping[str, object]) -> None:
    metric = _required_mapping_value(value=gate.get("metric"), field_name="promotion_gates.runtime_candidate.metric")
    value = _required_float(record=metric, field_name="value")
    threshold = _required_float(record=metric, field_name="threshold")
    if not math.isfinite(value) or not math.isfinite(threshold):
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.metric values must be finite.")
    if value < threshold:
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.metric.value must meet threshold.")


def _validate_promotion_ablation(gate: Mapping[str, object]) -> None:
    ablation = _required_mapping_value(
        value=gate.get("ablation"),
        field_name="promotion_gates.runtime_candidate.ablation",
    )
    delta = _required_float(record=ablation, field_name="delta")
    threshold = _required_float(record=ablation, field_name="delta_threshold")
    if not math.isfinite(delta) or not math.isfinite(threshold):
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.ablation values must be finite.")
    if delta < threshold:
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.ablation.delta must meet threshold.")


def _validate_promotion_splits(gate: Mapping[str, object]) -> None:
    splits = _required_mapping_value(value=gate.get("splits"), field_name="promotion_gates.runtime_candidate.splits")
    split_ids = (
        _required_string(record=splits, field_name="train"),
        _required_string(record=splits, field_name="calibration"),
        _required_string(record=splits, field_name="heldout"),
        _required_string(record=splits, field_name="sealed_holdout"),
    )
    if len(set(split_ids)) != len(split_ids):
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate split ids must be distinct.")


def _validate_promotion_report_references(
    gate: Mapping[str, object],
    required_report_ids: tuple[str, ...],
) -> None:
    reports = _required_mapping_value(value=gate.get("reports"), field_name="promotion_gates.runtime_candidate.reports")
    required_report_fields = (
        "sealed_holdout",
        "metric",
        "calibration",
        "ablation",
        "patching",
        "failure_cases",
        "runtime_prevention",
        "lineage",
    )
    for field_name in required_report_fields:
        report_id = _required_string(record=reports, field_name=field_name)
        if report_id not in required_report_ids:
            raise CiftRuntimeDetectorError(
                f"promotion_gates.runtime_candidate.reports.{field_name} must be listed in required_report_ids."
            )


def _validate_promotion_paper_method(method: Mapping[str, object]) -> None:
    expected_values = (
        ("readout_position_contract", "post_secret_post_query_causal_readout"),
        ("monitored_layer_policy", "last_quarter_transformer_layers"),
    )
    for field_name, expected_value in expected_values:
        actual_value = _required_string(record=method, field_name=field_name)
        if actual_value != expected_value:
            raise CiftRuntimeDetectorError(f"promotion_gates.runtime_candidate.paper_method.{field_name} invalid.")
    if _required_bool(record=method, field_name="pre_output") is not True:
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.paper_method.pre_output must be true.")
    if _required_bool(record=method, field_name="uses_static_secret_token_positions") is not False:
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.paper_method.uses_static_secret_token_positions must be false."
        )
    feature_representation = _required_string(record=method, field_name="feature_representation")
    probe_architecture = _required_string(record=method, field_name="probe_architecture")
    training_loss = _required_string(record=method, field_name="training_loss")
    if probe_architecture == "" or training_loss == "":
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.paper_method probe fields must be nonempty.")
    if feature_representation == "diagonal_mahalanobis_cci":
        _validate_promotion_cci_paper_method(method=method)
        if probe_architecture == "mlp_128_64_1":
            if training_loss != "bce_with_l1_softplus_weight_sparsity":
                raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.paper_method.training_loss invalid.")
            return
        _validate_promotion_challenger_probe_meets_or_exceeds(method=method)
        return
    if feature_representation == "raw_activation":
        _validate_promotion_raw_activation_fields(method=method)
        if probe_architecture == "mlp_128_64_1":
            if training_loss != "bce_with_l1_softplus_weight_sparsity":
                raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.paper_method.training_loss invalid.")
            _validate_promotion_paper_mlp_meets_or_exceeds_challenger(method=method)
            return
        _validate_promotion_raw_activation_exception(method=method)
        _validate_promotion_raw_activation_challenger_win(method=method)
        return
    raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.paper_method.feature_representation invalid.")


def _validate_promotion_cci_paper_method(method: Mapping[str, object]) -> None:
    covariance_estimator = _required_string(record=method, field_name="covariance_estimator")
    if covariance_estimator != "diagonal_covariance":
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.paper_method.covariance_estimator invalid.")
    layer_weighting = _required_string(record=method, field_name="layer_weighting")
    if layer_weighting != "softplus_nonnegative_cfs":
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.paper_method.layer_weighting invalid.")
    ridge = _required_float(record=method, field_name="ridge")
    if not math.isfinite(ridge) or ridge != 0.001:
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.paper_method.ridge must be 0.001.")


def _validate_promotion_raw_activation_fields(method: Mapping[str, object]) -> None:
    covariance_estimator = _required_string(record=method, field_name="covariance_estimator")
    if covariance_estimator != "not_applicable":
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.paper_method.covariance_estimator must be not_applicable."
        )
    layer_weighting = _required_string(record=method, field_name="layer_weighting")
    if layer_weighting != "not_applicable":
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.paper_method.layer_weighting must be not_applicable."
        )
    ridge = _required_float(record=method, field_name="ridge")
    if not math.isfinite(ridge) or ridge != 0.0:
        raise CiftRuntimeDetectorError("promotion_gates.runtime_candidate.paper_method.ridge must be 0.0.")


def _validate_promotion_raw_activation_exception(method: Mapping[str, object]) -> None:
    paper_faithfulness_exception = _optional_string(record=method, field_name="paper_faithfulness_exception")
    if paper_faithfulness_exception is None or paper_faithfulness_exception.strip() == "":
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.paper_method raw_activation requires paper_faithfulness_exception."
        )


def _validate_promotion_challenger_metrics(method: Mapping[str, object]) -> tuple[float, float]:
    head_to_head_report_id = _optional_string(record=method, field_name="head_to_head_report_id")
    if head_to_head_report_id is None or head_to_head_report_id == "":
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.paper_method alternative probe requires head_to_head_report_id."
        )
    paper_metric = _optional_float(record=method, field_name="paper_probe_metric_value")
    candidate_metric = _optional_float(record=method, field_name="candidate_probe_metric_value")
    if paper_metric is None or candidate_metric is None:
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.paper_method alternative probe requires metric comparison."
        )
    if not math.isfinite(paper_metric) or not math.isfinite(candidate_metric):
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.paper_method alternative probe metrics must be finite."
        )
    return paper_metric, candidate_metric


def _validate_promotion_challenger_probe_meets_or_exceeds(method: Mapping[str, object]) -> None:
    paper_metric, candidate_metric = _validate_promotion_challenger_metrics(method=method)
    if candidate_metric < paper_metric:
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.paper_method candidate metric must meet or exceed paper metric."
        )


def _validate_promotion_raw_activation_challenger_win(method: Mapping[str, object]) -> None:
    paper_metric, candidate_metric = _validate_promotion_challenger_metrics(method=method)
    if candidate_metric <= paper_metric:
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.paper_method candidate metric must exceed paper metric."
        )


def _validate_promotion_paper_mlp_meets_or_exceeds_challenger(method: Mapping[str, object]) -> None:
    paper_metric, candidate_metric = _validate_promotion_challenger_metrics(method=method)
    if paper_metric < candidate_metric:
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.paper_method paper metric must meet or exceed candidate metric."
        )


def _validate_promotion_model_probe_consistency(method: Mapping[str, object], model: CiftRuntimeModel) -> None:
    probe_architecture = _required_string(record=method, field_name="probe_architecture")
    if probe_architecture == _PAPER_MLP_PROBE_ARCHITECTURE:
        if not isinstance(model, CiftRuntimeMlpModel):
            raise CiftRuntimeDetectorError("linear runtime model cannot claim mlp_128_64_1 promotion evidence.")
        return
    if isinstance(model, CiftRuntimeMlpModel):
        raise CiftRuntimeDetectorError("mlp_128_64_1 runtime model cannot claim alternative probe promotion evidence.")


def _validate_promotion_head_to_head_report_id(
    method: Mapping[str, object],
    required_report_ids: tuple[str, ...],
    model_report_ids: set[str],
) -> None:
    head_to_head_report_id = _optional_string(record=method, field_name="head_to_head_report_id")
    if head_to_head_report_id is None or head_to_head_report_id == "":
        return
    if head_to_head_report_id not in required_report_ids:
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.paper_method.head_to_head_report_id must be listed in "
            "required_report_ids."
        )
    if head_to_head_report_id not in model_report_ids:
        raise CiftRuntimeDetectorError(
            "promotion_gates.runtime_candidate.paper_method.head_to_head_report_id must be present in model "
            "evaluation_report_ids."
        )


def _required_mapping_value(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CiftRuntimeDetectorError(f"Field '{field_name}' must be an object.")
    return cast(Mapping[str, object], value)


def _required_bool(record: Mapping[str, object], field_name: str) -> bool:
    value = record.get(field_name)
    if not isinstance(value, bool):
        raise CiftRuntimeDetectorError(f"Field '{field_name}' must be a boolean.")
    return value


def _optional_string(record: Mapping[str, object], field_name: str) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CiftRuntimeDetectorError(f"Field '{field_name}' must be a string when present.")
    return value


def _optional_float(record: Mapping[str, object], field_name: str) -> float | None:
    value = record.get(field_name)
    if value is None:
        return None
    return _float_item(value=value, field_name=field_name)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value}")
