from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aegis.detectors.cift_runtime import load_cift_runtime_model
from aegis_introspection.cift_live_extractor import (
    HiddenStateRunner,
    LiveCiftFeatureExtractor,
    LiveCiftFeatureSetExtractor,
    LoadedModelHiddenStateRunner,
)
from aegis_introspection.cift_runtime_eval import (
    CiftRuntimeEvalSummary,
    CiftRuntimeRequestEvalConfig,
    CiftWindowSelectorRequestEvalConfig,
    run_cift_runtime_eval_with_extractor,
    run_cift_window_selector_runtime_eval_with_extractor,
)
from aegis_introspection.model_loader import ModelDTypeName, ModelLoadConfig, load_causal_lm
from aegis_introspection.sealed_holdout import assert_unsealed_jsonl_tags, assert_unsealed_paths


class CiftLiveRuntimeEvalError(ValueError):
    """Raised when live CIFT runtime evaluation cannot be configured."""


@dataclass(frozen=True)
class CiftLiveRuntimeEvalConfig:
    runtime_turns_path: Path
    runtime_model_path: Path
    output_path: Path
    detector_name: str
    feature_source: str
    mock_response: str
    model_id: str
    revision: str
    requested_device: str
    local_files_only: bool
    dtype_name: ModelDTypeName
    trust_remote_code: bool
    allow_sealed_holdout: bool


@dataclass(frozen=True)
class CiftLiveWindowSelectorRequestEvalConfig:
    runtime_turns_path: Path
    selected_choice_runtime_model_path: Path
    fallback_runtime_model_path: Path
    output_path: Path
    detector_name: str
    feature_source: str
    mock_response: str
    allow_sealed_holdout: bool


@dataclass(frozen=True)
class CiftLiveWindowSelectorRuntimeEvalConfig:
    runtime_turns_path: Path
    selected_choice_runtime_model_path: Path
    fallback_runtime_model_path: Path
    output_path: Path
    detector_name: str
    feature_source: str
    mock_response: str
    model_id: str
    revision: str
    requested_device: str
    local_files_only: bool
    dtype_name: ModelDTypeName
    trust_remote_code: bool
    allow_sealed_holdout: bool


def run_cift_live_runtime_eval(config: CiftLiveRuntimeEvalConfig) -> CiftRuntimeEvalSummary:
    _validate_config(config)
    assert_unsealed_paths(
        paths=(config.runtime_turns_path, config.runtime_model_path, config.output_path),
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="live CIFT runtime eval",
    )
    assert_unsealed_jsonl_tags(
        path=config.runtime_turns_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="live CIFT runtime eval",
    )
    runtime_model = load_cift_runtime_model(config.runtime_model_path)
    loaded_model = load_causal_lm(
        ModelLoadConfig(
            model_id=config.model_id,
            revision=config.revision,
            requested_device=config.requested_device,
            local_files_only=config.local_files_only,
            dtype_name=config.dtype_name,
            trust_remote_code=config.trust_remote_code,
        )
    )
    extractor = LiveCiftFeatureExtractor(
        runner=LoadedModelHiddenStateRunner(loaded_model=loaded_model),
        feature_key=runtime_model.feature_key,
    )
    return run_cift_runtime_eval_with_extractor(
        config=CiftRuntimeRequestEvalConfig(
            runtime_turns_path=config.runtime_turns_path,
            runtime_model_path=config.runtime_model_path,
            output_path=config.output_path,
            detector_name=config.detector_name,
            feature_source=config.feature_source,
            mock_response=config.mock_response,
            allow_sealed_holdout=config.allow_sealed_holdout,
        ),
        extractor=extractor,
    )


def run_cift_live_window_selector_runtime_eval(
    config: CiftLiveWindowSelectorRuntimeEvalConfig,
) -> CiftRuntimeEvalSummary:
    _validate_window_selector_config(config)
    loaded_model = load_causal_lm(
        ModelLoadConfig(
            model_id=config.model_id,
            revision=config.revision,
            requested_device=config.requested_device,
            local_files_only=config.local_files_only,
            dtype_name=config.dtype_name,
            trust_remote_code=config.trust_remote_code,
        )
    )
    return run_cift_live_window_selector_runtime_eval_with_runner(
        config=_window_selector_request_eval_config(config),
        runner=LoadedModelHiddenStateRunner(loaded_model=loaded_model),
    )


def run_cift_live_window_selector_runtime_eval_with_runner(
    config: CiftLiveWindowSelectorRequestEvalConfig,
    runner: HiddenStateRunner,
) -> CiftRuntimeEvalSummary:
    selected_choice_model = load_cift_runtime_model(config.selected_choice_runtime_model_path)
    fallback_model = load_cift_runtime_model(config.fallback_runtime_model_path)
    extractor = LiveCiftFeatureSetExtractor(
        runner=runner,
        feature_keys=(selected_choice_model.feature_key, fallback_model.feature_key),
    )
    return run_cift_window_selector_runtime_eval_with_extractor(
        config=_runtime_window_selector_request_eval_config(config),
        extractor=extractor,
    )


def _validate_config(config: CiftLiveRuntimeEvalConfig) -> None:
    if config.model_id == "":
        raise CiftLiveRuntimeEvalError("model_id must not be empty.")
    if config.revision == "":
        raise CiftLiveRuntimeEvalError("revision must not be empty.")


def _validate_window_selector_config(config: CiftLiveWindowSelectorRuntimeEvalConfig) -> None:
    if config.model_id == "":
        raise CiftLiveRuntimeEvalError("model_id must not be empty.")
    if config.revision == "":
        raise CiftLiveRuntimeEvalError("revision must not be empty.")


def _window_selector_request_eval_config(
    config: CiftLiveWindowSelectorRuntimeEvalConfig,
) -> CiftLiveWindowSelectorRequestEvalConfig:
    return CiftLiveWindowSelectorRequestEvalConfig(
        runtime_turns_path=config.runtime_turns_path,
        selected_choice_runtime_model_path=config.selected_choice_runtime_model_path,
        fallback_runtime_model_path=config.fallback_runtime_model_path,
        output_path=config.output_path,
        detector_name=config.detector_name,
        feature_source=config.feature_source,
        mock_response=config.mock_response,
        allow_sealed_holdout=config.allow_sealed_holdout,
    )


def _runtime_window_selector_request_eval_config(
    config: CiftLiveWindowSelectorRequestEvalConfig,
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
