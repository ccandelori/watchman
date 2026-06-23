from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from aegis.audit.memory import InMemoryAuditSink
from aegis.core.contracts import JsonValue, NormalizedTurn
from aegis.core.orchestrator import AegisRuntime, RuntimeRequest
from aegis.detectors.cift_runtime import (
    CiftFeatureExtractor,
    CiftFeatureVectorAnnotator,
    CiftRuntimeDetector,
    CiftRuntimeWindowSelector,
    load_cift_runtime_model,
)
from aegis.policy.engine import SeverityPolicyEngine
from aegis.providers.mock import MockModelProvider
from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import activation_feature_tensor
from aegis_introspection.probe import tensor_to_float_matrix
from aegis_introspection.runtime_requests import RuntimeRequestJsonlError
from aegis_introspection.runtime_requests import load_runtime_requests_jsonl as _load_runtime_requests_jsonl
from aegis_introspection.sealed_holdout import (
    assert_unsealed_jsonl_tags,
    assert_unsealed_paths,
    load_activation_artifact_with_unseal_policy,
)


class CiftRuntimeEvalError(ValueError):
    """Raised when offline CIFT runtime evaluation cannot be completed."""


@dataclass(frozen=True)
class CiftRuntimeRequestEvalConfig:
    runtime_turns_path: Path
    runtime_model_path: Path
    output_path: Path
    detector_name: str
    feature_source: str
    mock_response: str
    allow_sealed_holdout: bool


@dataclass(frozen=True)
class CiftRuntimeEvalConfig:
    runtime_turns_path: Path
    activation_artifact_path: Path
    runtime_model_path: Path
    output_path: Path
    detector_name: str
    feature_source: str
    mock_response: str
    allow_sealed_holdout: bool


@dataclass(frozen=True)
class CiftWindowSelectorRequestEvalConfig:
    runtime_turns_path: Path
    selected_choice_runtime_model_path: Path
    fallback_runtime_model_path: Path
    output_path: Path
    detector_name: str
    feature_source: str
    mock_response: str
    allow_sealed_holdout: bool


@dataclass(frozen=True)
class CiftWindowSelectorRuntimeEvalConfig:
    runtime_turns_path: Path
    activation_artifact_path: Path
    selected_choice_runtime_model_path: Path
    fallback_runtime_model_path: Path
    output_path: Path
    detector_name: str
    feature_source: str
    mock_response: str
    allow_sealed_holdout: bool


@dataclass(frozen=True)
class CiftRuntimeEvalSummary:
    request_count: int
    output_path: Path
    detector_action_counts: dict[str, int]
    policy_action_counts: dict[str, int]
    capability_status_counts: dict[str, int]


class ActivationArtifactFeatureExtractor:
    def __init__(self, artifact: ActivationArtifact, feature_key: str) -> None:
        if feature_key == "":
            raise CiftRuntimeEvalError("feature_key must not be empty.")
        self._feature_key = feature_key
        self._example_indices = _example_indices(artifact["example_ids"])
        self._matrix = tensor_to_float_matrix(activation_feature_tensor(artifact=artifact, feature_key=feature_key))

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        if feature_key != self._feature_key:
            raise CiftRuntimeEvalError(f"Extractor was initialized for '{self._feature_key}', not '{feature_key}'.")
        example_id = _example_id_from_turn(turn)
        if example_id is None:
            return None
        row_index = self._example_indices.get(example_id)
        if row_index is None:
            return None
        return tuple(float(value) for value in self._matrix[row_index].tolist())


class ActivationArtifactFeatureSetExtractor:
    def __init__(self, artifact: ActivationArtifact, feature_keys: tuple[str, ...]) -> None:
        if len(feature_keys) == 0:
            raise CiftRuntimeEvalError("feature_keys must not be empty.")
        if len(set(feature_keys)) != len(feature_keys):
            raise CiftRuntimeEvalError("feature_keys must be unique.")
        self._example_indices = _example_indices(artifact["example_ids"])
        self._matrices = {
            feature_key: tensor_to_float_matrix(activation_feature_tensor(artifact=artifact, feature_key=feature_key))
            for feature_key in feature_keys
        }

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        matrix = self._matrices.get(feature_key)
        if matrix is None:
            raise CiftRuntimeEvalError(f"Extractor was not initialized for '{feature_key}'.")
        example_id = _example_id_from_turn(turn)
        if example_id is None:
            return None
        row_index = self._example_indices.get(example_id)
        if row_index is None:
            return None
        return tuple(float(value) for value in matrix[row_index].tolist())


def run_cift_runtime_eval(config: CiftRuntimeEvalConfig) -> CiftRuntimeEvalSummary:
    assert_unsealed_paths(
        paths=(config.activation_artifact_path,),
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="CIFT runtime eval",
    )
    artifact = load_activation_artifact_with_unseal_policy(
        path=config.activation_artifact_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="CIFT runtime eval",
    )
    runtime_model = load_cift_runtime_model(config.runtime_model_path)
    extractor = ActivationArtifactFeatureExtractor(artifact=artifact, feature_key=runtime_model.feature_key)
    return run_cift_runtime_eval_with_extractor(
        config=_request_eval_config(config),
        extractor=extractor,
    )


def run_cift_window_selector_runtime_eval(config: CiftWindowSelectorRuntimeEvalConfig) -> CiftRuntimeEvalSummary:
    _validate_window_selector_eval_config(config)
    assert_unsealed_paths(
        paths=(config.activation_artifact_path,),
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="CIFT window selector runtime eval",
    )
    artifact = load_activation_artifact_with_unseal_policy(
        path=config.activation_artifact_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="CIFT window selector runtime eval",
    )
    selected_choice_model = load_cift_runtime_model(config.selected_choice_runtime_model_path)
    fallback_model = load_cift_runtime_model(config.fallback_runtime_model_path)
    extractor = ActivationArtifactFeatureSetExtractor(
        artifact=artifact,
        feature_keys=(selected_choice_model.feature_key, fallback_model.feature_key),
    )
    return run_cift_window_selector_runtime_eval_with_extractor(
        config=_window_selector_request_eval_config(config),
        extractor=extractor,
    )


def run_cift_window_selector_runtime_eval_with_extractor(
    config: CiftWindowSelectorRequestEvalConfig,
    extractor: CiftFeatureExtractor,
) -> CiftRuntimeEvalSummary:
    _validate_window_selector_request_eval_config(config)
    assert_unsealed_paths(
        paths=(
            config.runtime_turns_path,
            config.selected_choice_runtime_model_path,
            config.fallback_runtime_model_path,
            config.output_path,
        ),
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="CIFT window selector runtime eval",
    )
    assert_unsealed_jsonl_tags(
        path=config.runtime_turns_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="CIFT window selector runtime eval",
    )
    selected_choice_model = load_cift_runtime_model(config.selected_choice_runtime_model_path)
    fallback_model = load_cift_runtime_model(config.fallback_runtime_model_path)
    requests = load_runtime_requests_jsonl(config.runtime_turns_path)
    runtime = AegisRuntime(
        turn_annotators=(
            CiftFeatureVectorAnnotator(
                feature_key=selected_choice_model.feature_key,
                extractor=extractor,
                source=config.feature_source,
            ),
            CiftFeatureVectorAnnotator(
                feature_key=fallback_model.feature_key,
                extractor=extractor,
                source=config.feature_source,
            ),
        ),
        pre_generation_detectors=(
            CiftRuntimeWindowSelector(
                detector_name=config.detector_name,
                selected_choice_model=selected_choice_model,
                fallback_model=fallback_model,
            ),
        ),
        post_generation_detectors=(),
        session_detectors=(),
        policy_engine=SeverityPolicyEngine(),
        audit_sink=InMemoryAuditSink(),
        model_provider=MockModelProvider(default_content=config.mock_response),
    )
    return _write_eval_rows(config=_window_selector_write_eval_config(config), runtime=runtime, requests=requests)


def run_cift_runtime_eval_with_extractor(
    config: CiftRuntimeRequestEvalConfig,
    extractor: CiftFeatureExtractor,
) -> CiftRuntimeEvalSummary:
    _validate_eval_config(config)
    assert_unsealed_paths(
        paths=(config.runtime_turns_path, config.runtime_model_path, config.output_path),
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="CIFT runtime eval",
    )
    assert_unsealed_jsonl_tags(
        path=config.runtime_turns_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="CIFT runtime eval",
    )
    requests = load_runtime_requests_jsonl(config.runtime_turns_path)
    runtime_model = load_cift_runtime_model(config.runtime_model_path)
    runtime = AegisRuntime(
        turn_annotators=(
            CiftFeatureVectorAnnotator(
                feature_key=runtime_model.feature_key,
                extractor=extractor,
                source=config.feature_source,
            ),
        ),
        pre_generation_detectors=(CiftRuntimeDetector(detector_name=config.detector_name, model=runtime_model),),
        post_generation_detectors=(),
        session_detectors=(),
        policy_engine=SeverityPolicyEngine(),
        audit_sink=InMemoryAuditSink(),
        model_provider=MockModelProvider(default_content=config.mock_response),
    )
    return _write_eval_rows(config=config, runtime=runtime, requests=requests)


def _request_eval_config(config: CiftRuntimeEvalConfig) -> CiftRuntimeRequestEvalConfig:
    return CiftRuntimeRequestEvalConfig(
        runtime_turns_path=config.runtime_turns_path,
        runtime_model_path=config.runtime_model_path,
        output_path=config.output_path,
        detector_name=config.detector_name,
        feature_source=config.feature_source,
        mock_response=config.mock_response,
        allow_sealed_holdout=config.allow_sealed_holdout,
    )


def _window_selector_request_eval_config(
    config: CiftWindowSelectorRuntimeEvalConfig,
) -> CiftWindowSelectorRequestEvalConfig:
    return CiftWindowSelectorRequestEvalConfig(
        runtime_turns_path=config.runtime_turns_path,
        selected_choice_runtime_model_path=config.selected_choice_runtime_model_path,
        fallback_runtime_model_path=config.fallback_runtime_model_path,
        output_path=config.output_path,
        detector_name=config.detector_name,
        feature_source=config.feature_source,
        mock_response=config.mock_response,
        allow_sealed_holdout=config.allow_sealed_holdout,
    )


def _window_selector_write_eval_config(config: CiftWindowSelectorRequestEvalConfig) -> CiftRuntimeRequestEvalConfig:
    return CiftRuntimeRequestEvalConfig(
        runtime_turns_path=config.runtime_turns_path,
        runtime_model_path=config.selected_choice_runtime_model_path,
        output_path=config.output_path,
        detector_name=config.detector_name,
        feature_source=config.feature_source,
        mock_response=config.mock_response,
        allow_sealed_holdout=config.allow_sealed_holdout,
    )


def load_runtime_requests_jsonl(path: Path) -> tuple[RuntimeRequest, ...]:
    try:
        return _load_runtime_requests_jsonl(path)
    except RuntimeRequestJsonlError as exc:
        raise CiftRuntimeEvalError(str(exc)) from exc


def _write_eval_rows(
    config: CiftRuntimeRequestEvalConfig,
    runtime: AegisRuntime,
    requests: tuple[RuntimeRequest, ...],
) -> CiftRuntimeEvalSummary:
    detector_actions: Counter[str] = Counter()
    policy_actions: Counter[str] = Counter()
    capability_statuses: Counter[str] = Counter()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as file:
        for request in requests:
            response = runtime.evaluate_turn(request)
            if len(response.detector_results) != 1:
                raise CiftRuntimeEvalError("CIFT runtime eval expects exactly one detector result per request.")
            detector_result = response.detector_results[0]
            detector_actions[detector_result.recommended_action.value] += 1
            policy_actions[response.policy_decision.final_action.value] += 1
            capability_statuses[detector_result.capability_status.value] += 1
            row = {
                "trace_id": request.trace_id,
                "session_id": request.session_id,
                "turn_index": request.turn_index,
                "example_id": _example_id_from_metadata(request.metadata),
                "detector_result": detector_result.to_dict(),
                "policy_decision": response.policy_decision.to_dict(),
            }
            json.dump(row, file, ensure_ascii=False, sort_keys=True)
            file.write("\n")
    return CiftRuntimeEvalSummary(
        request_count=len(requests),
        output_path=config.output_path,
        detector_action_counts=dict(detector_actions),
        policy_action_counts=dict(policy_actions),
        capability_status_counts=dict(capability_statuses),
    )


def _example_indices(example_ids: tuple[str, ...]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for index, example_id in enumerate(example_ids):
        if example_id in indices:
            raise CiftRuntimeEvalError(f"Activation artifact contains duplicate example_id '{example_id}'.")
        indices[example_id] = index
    return indices


def _example_id_from_turn(turn: NormalizedTurn) -> str | None:
    return _optional_example_id_from_metadata(turn.metadata)


def _example_id_from_metadata(metadata: Mapping[str, JsonValue]) -> str | None:
    return _optional_example_id_from_metadata(metadata)


def _optional_example_id_from_metadata(metadata: Mapping[str, JsonValue]) -> str | None:
    value = metadata.get("example_id")
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise CiftRuntimeEvalError("metadata.example_id must be a non-empty string when present.")
    return value


def _validate_eval_config(config: CiftRuntimeRequestEvalConfig) -> None:
    if config.detector_name == "":
        raise CiftRuntimeEvalError("detector_name must not be empty.")
    if config.feature_source == "":
        raise CiftRuntimeEvalError("feature_source must not be empty.")
    if config.mock_response == "":
        raise CiftRuntimeEvalError("mock_response must not be empty.")


def _validate_window_selector_eval_config(config: CiftWindowSelectorRuntimeEvalConfig) -> None:
    _validate_window_selector_request_eval_config(_window_selector_request_eval_config(config))


def _validate_window_selector_request_eval_config(config: CiftWindowSelectorRequestEvalConfig) -> None:
    _validate_eval_config(_window_selector_write_eval_config(config))
