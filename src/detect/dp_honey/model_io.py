"""Transparent JSON export/load for trained bigram models.

A saved model is a versioned, inspectable audit object -- never a pickle. Loading
**fails closed**: every invariant is proven before a model is constructed, so a
corrupt, tampered, or drifted artifact can never reach generation. The format
snapshot and its hash are embedded so a saved model is rejected if the registry
spec it was trained against has changed (:class:`FormatSpecMismatchError`); v1
deliberately provides no bypass.

The serialized form is canonical (sorted keys, 2-space indent) and carries no
timestamp, so an artifact's bytes are reproducible from its training inputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

from .bigram import START, BigramHoneytokenModel, transition_states_for_format
from .errors import (
    ModelArtifactDecodeError,
    ModelArtifactExistsError,
    ModelSchemaError,
)
from .formats import get_format

SCHEMA_VERSION = "1"
ROW_SUM_TOL = 1e-6

SAFETY_NOTE = (
    "Synthetic, shape-only honeytoken model. Outputs are NOT real, valid, signed, "
    "decryptable, authenticated, or usable credentials, and the model was never "
    "trained on real secrets."
)

ArtifactDict = dict[str, Any]
ArtifactSource = str | Path | ArtifactDict


def model_to_dict(model: BigramHoneytokenModel) -> ArtifactDict:
    """Serialize *model* into the canonical artifact dictionary."""
    from . import __version__  # local import avoids a package-init cycle

    spec = model.format_spec
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": {"package": "detect.dp_honey", "version": __version__},
        "format": {
            "slug": spec.slug,
            "registry_version": model.registry_version,
            "spec_hash": model.spec_hash,
            "spec_snapshot": spec.to_snapshot(),
        },
        "privacy": {
            "epsilon": model.epsilon,
            "clip": model.clip,
            "corpus_size": model.corpus_size,
            "train_seed": model.train_seed,
            "trained_at": None,  # omitted for reproducible artifact bytes
            "mechanism": "laplace-on-clipped-bigram-counts",
        },
        "alphabet": {"start_token": START, "symbols": list(model.alphabet)},
        "transitions": {state: dict(row) for state, row in model.transitions.items()},
        "safety": {"synthetic_only": True, "provider_valid": False, "note": SAFETY_NOTE},
    }


def save_model(model: BigramHoneytokenModel, path: str | Path, *, force: bool = False) -> Path:
    """Write *model* to *path* as canonical JSON.

    Refuses to overwrite an existing file unless *force* is set
    (:class:`ModelArtifactExistsError`).
    """
    path = Path(path)
    if path.exists() and not force:
        raise ModelArtifactExistsError(f"refusing to overwrite existing artifact: {path} (pass force=True)")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(model_to_dict(model), indent=2, sort_keys=True, ensure_ascii=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def read_artifact_dict(source: ArtifactSource) -> ArtifactDict:
    """Return the artifact as a dict, decoding from a path if needed.

    Raises :class:`ModelArtifactDecodeError` for unreadable files or invalid JSON.
    Used by both :func:`load_model` and the CLI's lenient ``inspect-model``.
    """
    if isinstance(source, dict):
        return source
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelArtifactDecodeError(f"could not read artifact {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelArtifactDecodeError(f"artifact is not valid JSON ({path}): {exc}") from exc
    if not isinstance(data, dict):
        raise ModelSchemaError(f"artifact root must be a JSON object, got {type(data).__name__}")
    return data


def load_model(source: ArtifactSource) -> BigramHoneytokenModel:
    """Load and fully validate a model artifact, or raise a typed error.

    Validates (in order): schema version, required fields, format registration,
    snapshot/hash drift, alphabet consistency, and every transition row (finite,
    nonnegative, normalized). Only then is a :class:`BigramHoneytokenModel` built.
    """
    data = read_artifact_dict(source)

    _require(
        data.get("schema_version") == SCHEMA_VERSION,
        f"unsupported or missing schema_version (expected {SCHEMA_VERSION!r})",
    )
    fmt = _field(data, "format")
    privacy = _field(data, "privacy")
    alphabet_meta = _field(data, "alphabet")
    transitions = _field(data, "transitions")
    _field(data, "generator")
    _field(data, "safety")

    slug = _field(fmt, "slug")
    live = get_format(slug)  # UnknownFormatError if not registered

    stored_hash = _field(fmt, "spec_hash")
    stored_snapshot = _field(fmt, "spec_snapshot")
    _validate_checksum_algorithms(stored_snapshot)
    if stored_hash != live.spec_hash() or stored_snapshot != live.to_snapshot():
        from .errors import FormatSpecMismatchError

        raise FormatSpecMismatchError(
            f"format {slug!r} snapshot/hash has drifted from the registry; "
            "the artifact was trained against a different spec"
        )

    start_token = _field(alphabet_meta, "start_token")
    _require(start_token == START, f"unexpected start token {start_token!r}")
    symbols = _field(alphabet_meta, "symbols")
    _require(
        isinstance(symbols, list) and all(isinstance(s, str) and len(s) == 1 for s in symbols),
        "alphabet.symbols must be a list of single-character strings",
    )
    _require(
        sorted(set(symbols)) == sorted(set(live.variable_alphabet())),
        "alphabet symbols do not match the format alphabet",
    )
    _validate_transitions(transitions, set(symbols), start_token, transition_states_for_format(live))

    # Privacy metadata must satisfy the same bounds train_model enforces, so the
    # save and load paths agree on what a valid model is (fail closed, KTD4).
    epsilon = _num(privacy, "epsilon")
    clip = _num(privacy, "clip")
    corpus_size = _int(privacy, "corpus_size")
    train_seed = _int(privacy, "train_seed")
    _require(epsilon > 0.0, f"epsilon must be > 0, got {epsilon}")
    _require(clip > 0.0, f"clip must be > 0, got {clip}")
    _require(corpus_size >= 0, f"corpus_size must be >= 0, got {corpus_size}")

    return BigramHoneytokenModel(
        format_spec=live,
        alphabet=live.variable_alphabet(),
        transitions={state: {char: float(prob) for char, prob in row.items()} for state, row in transitions.items()},
        epsilon=epsilon,
        clip=clip,
        corpus_size=corpus_size,
        train_seed=train_seed,
        registry_version=str(_field(fmt, "registry_version")),
        spec_hash=stored_hash,
    )


# --- small validation helpers (each raises a typed ModelSchemaError) -----------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelSchemaError(message)


def _field(mapping: object, key: str) -> Any:
    _require(isinstance(mapping, dict) and key in mapping, f"missing or malformed field: {key!r}")
    return mapping[key]  # type: ignore[index]


def _validate_checksum_algorithms(stored_snapshot: object) -> None:
    from .checksums import CHECKSUM_ALGORITHMS

    if not isinstance(stored_snapshot, dict):
        return
    for segment in stored_snapshot.get("segments", []):
        if isinstance(segment, dict) and segment.get("kind") == "checksum":
            algorithm = segment.get("algorithm")
            _require(
                isinstance(algorithm, str) and algorithm in CHECKSUM_ALGORITHMS,
                f"unknown checksum algorithm: {algorithm!r}",
            )


def _num(mapping: ArtifactDict, key: str) -> float:
    value = _field(mapping, key)
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"field {key!r} must be a number")
    number = float(value)
    _require(math.isfinite(number), f"field {key!r} must be finite")
    return number


def _int(mapping: ArtifactDict, key: str) -> int:
    value = _field(mapping, key)
    _require(isinstance(value, int) and not isinstance(value, bool), f"field {key!r} must be an integer")
    return int(value)


def _validate_transitions(
    transitions: object,
    symbols: set[str],
    start_token: str,
    segment_states: set[str],
) -> None:
    _require(isinstance(transitions, dict), "transitions must be a JSON object")
    transition_rows = cast(dict[str, object], transitions)
    valid_states = symbols | {start_token} | segment_states
    for state, row in transition_rows.items():
        _require(state in valid_states, f"transition state not in alphabet: {state!r}")
        _require(isinstance(row, dict), f"transition row must be an object: {state!r}")
        probabilities = cast(dict[str, object], row)
        total = 0.0
        for char, prob in probabilities.items():
            _require(char in symbols, f"transition target not in alphabet: {char!r}")
            if not isinstance(prob, (int, float)) or isinstance(prob, bool):
                raise ModelSchemaError(f"probability must be numeric: {state!r}->{char!r}")
            value = float(prob)
            _require(math.isfinite(value), f"probability must be finite: {state!r}->{char!r}")
            _require(value >= 0.0, f"probability must be nonnegative: {state!r}->{char!r}")
            total += value
        _require(abs(total - 1.0) <= ROW_SUM_TOL, f"transition row is not normalized (sum={total!r}): {state!r}")
