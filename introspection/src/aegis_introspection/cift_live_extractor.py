from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, cast

from aegis.cift_contract import CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION
from aegis.core.contracts import JsonValue, NormalizedTurn
from aegis.detectors.cift_runtime import CiftFeatureExtraction
from aegis_introspection.activations import HiddenStateForwardPass, run_hidden_state_forward
from aegis_introspection.features import PoolingMethod, extract_activation_features
from aegis_introspection.model_loader import LoadedCausalLM


class CiftLiveExtractorError(ValueError):
    """Raised when live CIFT feature extraction cannot be performed."""


class HiddenStateRunner(Protocol):
    def run(self, prompt: str) -> HiddenStateForwardPass:
        """Return hidden states for the exact prompt seen by the runtime model host."""


@dataclass(frozen=True)
class LoadedModelHiddenStateRunner:
    loaded_model: LoadedCausalLM

    def run(self, prompt: str) -> HiddenStateForwardPass:
        return run_hidden_state_forward(self.loaded_model, prompt)


@dataclass(frozen=True)
class CiftSourceFeatureSpec:
    key: str
    layer_index: int
    pooling_method: PoolingMethod


@dataclass(frozen=True)
class CiftFeatureSetCacheKey:
    trace_id: str
    session_id: str
    turn_index: int
    prompt: str


class LiveCiftFeatureExtractor:
    def __init__(self, runner: HiddenStateRunner, feature_key: str) -> None:
        if feature_key == "":
            raise CiftLiveExtractorError("feature_key must not be empty.")
        self._runner = runner
        self._feature_key = feature_key
        self._source_specs = parse_live_cift_feature_key(feature_key)

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        return self.extract_feature_extraction(turn=turn, feature_key=feature_key).feature_vector

    def extract_feature_extraction(self, turn: NormalizedTurn, feature_key: str) -> CiftFeatureExtraction:
        if feature_key != self._feature_key:
            raise CiftLiveExtractorError(f"Extractor was initialized for '{self._feature_key}', not '{feature_key}'.")
        readout_token_indices = readout_token_indices_from_turn(turn)
        if readout_token_indices is None and _requires_readout_window(self._source_specs):
            return CiftFeatureExtraction(feature_vector=None, selected_choice_readout_token_indices=None, provenance={})
        query_tail_readout_token_indices = query_tail_readout_token_indices_from_turn(turn)
        if query_tail_readout_token_indices is None and _requires_query_tail_window(self._source_specs):
            return CiftFeatureExtraction(feature_vector=None, selected_choice_readout_token_indices=None, provenance={})
        selected_choice_readout_token_indices = selected_choice_readout_token_indices_from_turn(turn)
        if selected_choice_readout_token_indices is None and _requires_selected_choice_window(self._source_specs):
            return CiftFeatureExtraction(feature_vector=None, selected_choice_readout_token_indices=None, provenance={})
        prompt = rendered_prompt_from_turn(turn)
        forward_pass = self._runner.run(prompt)
        feature_vector = _feature_vector_from_forward_pass(
            forward_pass=forward_pass,
            source_specs=self._source_specs,
            readout_token_indices=readout_token_indices,
            query_tail_readout_token_indices=query_tail_readout_token_indices,
            selected_choice_readout_token_indices=selected_choice_readout_token_indices,
        )
        return CiftFeatureExtraction(
            feature_vector=feature_vector,
            selected_choice_readout_token_indices=selected_choice_readout_token_indices,
            provenance=_extraction_receipt_provenance(
                feature_key=feature_key,
                feature_vector=feature_vector,
                readout_token_indices=readout_token_indices,
                query_tail_readout_token_indices=query_tail_readout_token_indices,
                selected_choice_readout_token_indices=selected_choice_readout_token_indices,
                source_specs=self._source_specs,
                turn=turn,
                prompt=prompt,
                forward_pass=forward_pass,
            ),
        )


class LiveCiftFeatureSetExtractor:
    def __init__(self, runner: HiddenStateRunner, feature_keys: tuple[str, ...]) -> None:
        if len(feature_keys) == 0:
            raise CiftLiveExtractorError("feature_keys must not be empty.")
        if len(set(feature_keys)) != len(feature_keys):
            raise CiftLiveExtractorError("feature_keys must be unique.")
        self._runner = runner
        self._source_specs_by_feature_key = {
            feature_key: parse_live_cift_feature_key(feature_key) for feature_key in feature_keys
        }
        self._cached_key: CiftFeatureSetCacheKey | None = None
        self._cached_forward_pass: HiddenStateForwardPass | None = None

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        return self.extract_feature_extraction(turn=turn, feature_key=feature_key).feature_vector

    def extract_feature_extraction(self, turn: NormalizedTurn, feature_key: str) -> CiftFeatureExtraction:
        source_specs = self._source_specs_by_feature_key.get(feature_key)
        if source_specs is None:
            raise CiftLiveExtractorError(f"Extractor was not initialized for '{feature_key}'.")
        readout_token_indices = readout_token_indices_from_turn(turn)
        if readout_token_indices is None and _requires_readout_window(source_specs):
            return CiftFeatureExtraction(feature_vector=None, selected_choice_readout_token_indices=None, provenance={})
        query_tail_readout_token_indices = query_tail_readout_token_indices_from_turn(turn)
        if query_tail_readout_token_indices is None and _requires_query_tail_window(source_specs):
            return CiftFeatureExtraction(feature_vector=None, selected_choice_readout_token_indices=None, provenance={})
        selected_choice_readout_token_indices = selected_choice_readout_token_indices_from_turn(turn)
        if selected_choice_readout_token_indices is None and _requires_selected_choice_window(source_specs):
            return CiftFeatureExtraction(feature_vector=None, selected_choice_readout_token_indices=None, provenance={})
        prompt = rendered_prompt_from_turn(turn)
        forward_pass = self._forward_pass(turn=turn, prompt=prompt)
        feature_vector = _feature_vector_from_forward_pass(
            forward_pass=forward_pass,
            source_specs=source_specs,
            readout_token_indices=readout_token_indices,
            query_tail_readout_token_indices=query_tail_readout_token_indices,
            selected_choice_readout_token_indices=selected_choice_readout_token_indices,
        )
        return CiftFeatureExtraction(
            feature_vector=feature_vector,
            selected_choice_readout_token_indices=selected_choice_readout_token_indices,
            provenance=_extraction_receipt_provenance(
                feature_key=feature_key,
                feature_vector=feature_vector,
                readout_token_indices=readout_token_indices,
                query_tail_readout_token_indices=query_tail_readout_token_indices,
                selected_choice_readout_token_indices=selected_choice_readout_token_indices,
                source_specs=source_specs,
                turn=turn,
                prompt=prompt,
                forward_pass=forward_pass,
            ),
        )

    def _forward_pass(self, turn: NormalizedTurn, prompt: str) -> HiddenStateForwardPass:
        cache_key = CiftFeatureSetCacheKey(
            trace_id=turn.trace_id,
            session_id=turn.session_id,
            turn_index=turn.turn_index,
            prompt=prompt,
        )
        if self._cached_key == cache_key and self._cached_forward_pass is not None:
            return self._cached_forward_pass
        forward_pass = self._runner.run(prompt)
        self._cached_key = cache_key
        self._cached_forward_pass = forward_pass
        return forward_pass


def parse_live_cift_feature_key(feature_key: str) -> tuple[CiftSourceFeatureSpec, ...]:
    source_feature_keys = _source_feature_keys(feature_key)
    return tuple(_source_feature_spec(source_feature_key) for source_feature_key in source_feature_keys)


def rendered_prompt_from_turn(turn: NormalizedTurn) -> str:
    if len(turn.messages) != 1:
        raise CiftLiveExtractorError("Live CIFT extraction requires exactly one rendered prompt message.")
    prompt = turn.messages[0].content
    if prompt == "":
        raise CiftLiveExtractorError("Live CIFT extraction requires a non-empty rendered prompt.")
    return prompt


def readout_token_indices_from_turn(turn: NormalizedTurn) -> tuple[int, ...] | None:
    return _cift_token_indices_from_turn(turn=turn, field_name="readout_token_indices")


def query_tail_readout_token_indices_from_turn(turn: NormalizedTurn) -> tuple[int, ...] | None:
    return _cift_token_indices_from_turn(turn=turn, field_name="query_tail_readout_token_indices")


def selected_choice_readout_token_indices_from_turn(turn: NormalizedTurn) -> tuple[int, ...] | None:
    return _cift_token_indices_from_turn(turn=turn, field_name="selected_choice_readout_token_indices")


def _cift_token_indices_from_turn(turn: NormalizedTurn, field_name: str) -> tuple[int, ...] | None:
    cift_metadata = turn.metadata.get("cift")
    if cift_metadata is None:
        return None
    if not isinstance(cift_metadata, dict):
        raise CiftLiveExtractorError("NormalizedTurn metadata.cift must be an object when present.")
    token_indices = cift_metadata.get(field_name)
    if token_indices is None:
        return None
    if not isinstance(token_indices, list):
        raise CiftLiveExtractorError(f"metadata.cift.{field_name} must be a list of integers.")
    if len(token_indices) == 0:
        raise CiftLiveExtractorError(f"metadata.cift.{field_name} must not be empty.")
    return tuple(
        _int_item(value=value, field_name=f"metadata.cift.{field_name}[{index}]")
        for index, value in enumerate(token_indices)
    )


def _source_feature_keys(feature_key: str) -> tuple[str, ...]:
    prefix = "concat("
    suffix = ")"
    if feature_key.startswith(prefix):
        if not feature_key.endswith(suffix):
            raise CiftLiveExtractorError(f"Feature expression '{feature_key}' is missing a closing parenthesis.")
        inner_value = feature_key[len(prefix) : -len(suffix)]
        source_feature_keys = tuple(item.strip() for item in inner_value.split(",") if item.strip() != "")
        if len(source_feature_keys) < 2:
            raise CiftLiveExtractorError(
                f"Feature expression '{feature_key}' must concatenate at least two source features."
            )
        return source_feature_keys
    return (feature_key,)


def _source_feature_spec(feature_key: str) -> CiftSourceFeatureSpec:
    for pooling_method in (
        "final_token",
        "mean_pool",
        "readout_window",
        "query_tail_window",
        "selected_choice_window",
        "combined_readout_window",
    ):
        typed_pooling_method = cast(PoolingMethod, pooling_method)
        prefix = f"{pooling_method}_layer_"
        if feature_key.startswith(prefix):
            layer_index = _layer_index(raw_value=feature_key[len(prefix) :], feature_key=feature_key)
            return CiftSourceFeatureSpec(
                key=feature_key,
                layer_index=layer_index,
                pooling_method=typed_pooling_method,
            )
    raise CiftLiveExtractorError(f"Unsupported live CIFT source feature '{feature_key}'.")


def _layer_index(raw_value: str, feature_key: str) -> int:
    if raw_value == "":
        raise CiftLiveExtractorError(f"Feature '{feature_key}' is missing a layer index.")
    try:
        layer_index = int(raw_value)
    except ValueError as exc:
        raise CiftLiveExtractorError(f"Feature '{feature_key}' has non-integer layer index '{raw_value}'.") from exc
    if layer_index < 0:
        raise CiftLiveExtractorError(f"Feature '{feature_key}' must use a non-negative layer index.")
    return layer_index


def _requires_readout_window(source_specs: tuple[CiftSourceFeatureSpec, ...]) -> bool:
    return any(
        source_spec.pooling_method in ("readout_window", "combined_readout_window") for source_spec in source_specs
    )


def _requires_query_tail_window(source_specs: tuple[CiftSourceFeatureSpec, ...]) -> bool:
    return any(source_spec.pooling_method == "query_tail_window" for source_spec in source_specs)


def _requires_selected_choice_window(source_specs: tuple[CiftSourceFeatureSpec, ...]) -> bool:
    return any(
        source_spec.pooling_method in ("selected_choice_window", "combined_readout_window")
        for source_spec in source_specs
    )


def _feature_vector_from_forward_pass(
    forward_pass: HiddenStateForwardPass,
    source_specs: tuple[CiftSourceFeatureSpec, ...],
    readout_token_indices: tuple[int, ...] | None,
    query_tail_readout_token_indices: tuple[int, ...] | None,
    selected_choice_readout_token_indices: tuple[int, ...] | None,
) -> tuple[float, ...]:
    values: list[float] = []
    for source_spec in source_specs:
        source_readout_indices = (
            readout_token_indices
            if source_spec.pooling_method in ("readout_window", "combined_readout_window")
            else None
        )
        source_query_tail_indices = (
            query_tail_readout_token_indices if source_spec.pooling_method == "query_tail_window" else None
        )
        source_selected_choice_indices = (
            selected_choice_readout_token_indices
            if source_spec.pooling_method in ("selected_choice_window", "combined_readout_window")
            else None
        )
        features = extract_activation_features(
            forward_pass=forward_pass,
            layer_indices=(source_spec.layer_index,),
            pooling_methods=(source_spec.pooling_method,),
            readout_token_indices=source_readout_indices,
            query_tail_readout_token_indices=source_query_tail_indices,
            selected_choice_readout_token_indices=source_selected_choice_indices,
        )
        if len(features) != 1:
            raise CiftLiveExtractorError(f"Expected one extracted feature for '{source_spec.key}'.")
        feature = features[0]
        if feature.key != source_spec.key:
            raise CiftLiveExtractorError(f"Extracted feature '{feature.key}', but expected '{source_spec.key}'.")
        values.extend(float(value) for value in feature.values.squeeze(0).cpu().float().reshape(-1).tolist())
    return tuple(values)


def _extraction_receipt_provenance(
    feature_key: str,
    feature_vector: tuple[float, ...],
    readout_token_indices: tuple[int, ...] | None,
    query_tail_readout_token_indices: tuple[int, ...] | None,
    selected_choice_readout_token_indices: tuple[int, ...] | None,
    source_specs: tuple[CiftSourceFeatureSpec, ...],
    turn: NormalizedTurn,
    prompt: str,
    forward_pass: HiddenStateForwardPass,
) -> dict[str, JsonValue]:
    _validate_forward_pass_source_metadata(forward_pass)
    provenance_readout_token_indices = (
        _final_token_readout_indices(forward_pass) if _uses_only_final_token(source_specs) else readout_token_indices
    )
    provenance: dict[str, JsonValue] = {
        "extraction_receipt_schema_version": CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
        "feature_key": feature_key,
        "feature_vector_length": len(feature_vector),
        "feature_vector_sha256": _json_sha256([float(value) for value in feature_vector]),
        "rendered_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "input_device_observed": forward_pass.source_input_device,
        "hidden_state_layer_count": len(forward_pass.hidden_states),
        "hidden_state_device_observed": _single_observed_device(forward_pass.source_hidden_state_devices),
        "hidden_state_devices": [str(device) for device in forward_pass.source_hidden_state_devices],
        "hidden_state_dtypes": [str(dtype) for dtype in forward_pass.source_hidden_state_dtypes],
    }
    if provenance_readout_token_indices is not None:
        token_indices = [int(token_index) for token_index in provenance_readout_token_indices]
        provenance["readout_token_indices"] = token_indices
        provenance["readout_token_count"] = len(token_indices)
        provenance["readout_token_indices_sha256"] = _json_sha256(token_indices)
    if query_tail_readout_token_indices is not None:
        token_indices = [int(token_index) for token_index in query_tail_readout_token_indices]
        provenance["query_tail_readout_token_indices"] = token_indices
        provenance["query_tail_readout_token_count"] = len(token_indices)
        provenance["query_tail_readout_token_indices_sha256"] = _json_sha256(token_indices)
    _copy_cift_readout_provenance(provenance=provenance, turn=turn)
    _infer_missing_readout_provenance(
        provenance=provenance,
        source_specs=source_specs,
        readout_token_indices=provenance_readout_token_indices,
        query_tail_readout_token_indices=query_tail_readout_token_indices,
    )
    if selected_choice_readout_token_indices is not None:
        token_indices = [int(token_index) for token_index in selected_choice_readout_token_indices]
        provenance["selected_choice_readout_token_indices"] = token_indices
        provenance["selected_choice_readout_token_count"] = len(token_indices)
        provenance["selected_choice_readout_token_indices_sha256"] = _json_sha256(token_indices)
    return provenance


def _copy_cift_readout_provenance(provenance: dict[str, JsonValue], turn: NormalizedTurn) -> None:
    cift_metadata = turn.metadata.get("cift")
    if not isinstance(cift_metadata, dict):
        return
    for field_name in ("readout_window_source", "readout_source"):
        value = cift_metadata.get(field_name)
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            provenance[field_name] = value
        elif isinstance(value, list):
            provenance[field_name] = list(value)
        elif isinstance(value, dict):
            provenance[field_name] = dict(value)


def _infer_missing_readout_provenance(
    provenance: dict[str, JsonValue],
    source_specs: tuple[CiftSourceFeatureSpec, ...],
    readout_token_indices: tuple[int, ...] | None,
    query_tail_readout_token_indices: tuple[int, ...] | None,
) -> None:
    if "readout_window_source" in provenance and "readout_source" in provenance:
        return
    inferred_window = _inferred_readout_window_source(source_specs)
    if inferred_window is None:
        return
    if "readout_window_source" not in provenance:
        provenance["readout_window_source"] = inferred_window
    if "readout_source" not in provenance:
        token_count = _inferred_readout_token_count(
            inferred_window=inferred_window,
            readout_token_indices=readout_token_indices,
            query_tail_readout_token_indices=query_tail_readout_token_indices,
        )
        provenance["readout_source"] = {
            "source": "live_cift_extractor",
            "readout_window": inferred_window,
            "readout_token_count": token_count,
        }


def _inferred_readout_window_source(source_specs: tuple[CiftSourceFeatureSpec, ...]) -> str | None:
    pooling_methods = tuple(source_spec.pooling_method for source_spec in source_specs)
    if "query_tail_window" in pooling_methods:
        return "query_tail"
    if "readout_window" in pooling_methods or "combined_readout_window" in pooling_methods:
        return "metadata_readout"
    if _uses_only_final_token(source_specs):
        return "final_token"
    return None


def _inferred_readout_token_count(
    inferred_window: str,
    readout_token_indices: tuple[int, ...] | None,
    query_tail_readout_token_indices: tuple[int, ...] | None,
) -> int:
    if inferred_window == "query_tail" and query_tail_readout_token_indices is not None:
        return len(query_tail_readout_token_indices)
    if readout_token_indices is not None:
        return len(readout_token_indices)
    return 0


def _uses_only_final_token(source_specs: tuple[CiftSourceFeatureSpec, ...]) -> bool:
    return len(source_specs) > 0 and all(source_spec.pooling_method == "final_token" for source_spec in source_specs)


def _final_token_readout_indices(forward_pass: HiddenStateForwardPass) -> tuple[int, ...]:
    if len(forward_pass.input_ids.shape) != 2:
        raise CiftLiveExtractorError("Hidden-state forward pass input_ids must have shape [batch, sequence].")
    sequence_length = int(forward_pass.input_ids.shape[1])
    if sequence_length < 1:
        raise CiftLiveExtractorError("Hidden-state forward pass input_ids must contain at least one token.")
    return (sequence_length - 1,)


def _validate_forward_pass_source_metadata(forward_pass: HiddenStateForwardPass) -> None:
    if forward_pass.source_input_device == "":
        raise CiftLiveExtractorError("Hidden-state forward pass source_input_device must not be empty.")
    if len(forward_pass.hidden_states) == 0:
        raise CiftLiveExtractorError("Hidden-state forward pass must include at least one hidden-state tensor.")
    if len(forward_pass.source_hidden_state_devices) != len(forward_pass.hidden_states):
        raise CiftLiveExtractorError("Hidden-state source device count must match hidden-state tensor count.")
    if len(forward_pass.source_hidden_state_dtypes) != len(forward_pass.hidden_states):
        raise CiftLiveExtractorError("Hidden-state source dtype count must match hidden-state tensor count.")
    if any(device == "" for device in forward_pass.source_hidden_state_devices):
        raise CiftLiveExtractorError("Hidden-state source devices must not contain empty values.")
    if any(dtype == "" for dtype in forward_pass.source_hidden_state_dtypes):
        raise CiftLiveExtractorError("Hidden-state source dtypes must not contain empty values.")


def _single_observed_device(devices: tuple[str, ...]) -> str:
    unique_devices = tuple(dict.fromkeys(devices))
    if len(unique_devices) == 1:
        return unique_devices[0]
    return "mixed"


def _json_sha256(value: JsonValue) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _int_item(value: JsonValue, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftLiveExtractorError(f"{field_name} must be an integer.")
    if value < 0:
        raise CiftLiveExtractorError(f"{field_name} must be non-negative.")
    return value
