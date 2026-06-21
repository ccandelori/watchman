"""Service layer for the DP-HONEY web UI.

Pure-ish functions that wrap the core :mod:`detect.dp_honey` library: they enforce
count caps, sanitize model names, and raise :class:`DPHoneyError` subclasses. No
FastAPI/HTTP imports live here, so every function is unit-testable without a server.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .. import scanner
from ..__main__ import GENERATE_MAX
from ..bigram import (
    DEFAULT_CLIP,
    DEFAULT_CORPUS_SIZE,
    DEFAULT_EPSILON,
    DEFAULT_MAX_REPAIR_ATTEMPTS,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_TRAIN_SEED,
    BigramHoneytokenModel,
    build_model,
)
from ..errors import DPHoneyError, UnknownFormatError
from ..formats import get_format, list_formats
from ..model_io import load_model, read_artifact_dict, save_model
from ..realism import REPORT_MAX, compute_report, enforce_count_limit

# Repo root (this file is src/detect/dp_honey/webui/service.py -> parents[4])
# so the golden fixture and default models dir resolve regardless of the process CWD.
_PKG_ROOT = Path(__file__).resolve().parents[4]

# Reserved label the UI uses for the committed synthetic golden fixture.
GOLDEN_NAME = "golden-fixture"
GOLDEN_PATH = _PKG_ROOT / "tests" / "dp_honey" / "fixtures" / "dp_honey" / "golden_model.json"

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_MAX_NAME_LEN = 100

_SAFETY = {
    "synthetic_only": True,
    "provider_valid": False,
    "note": "Synthetic, shape-only honeytokens. Not real, valid, or usable credentials.",
}


class InvalidModelName(DPHoneyError):
    """A model name contains unsafe characters or path components."""


def _models_dir(models_dir: Path | None) -> Path:
    return Path(models_dir) if models_dir is not None else _PKG_ROOT / "models"


def resolve_model_ref(name: str, models_dir: Path | None = None) -> Path:
    """Resolve a client-supplied model *name* to a safe on-disk path.

    Accepts the reserved golden-fixture label, or a name matching
    ``[A-Za-z0-9._-]+`` resolved inside ``models_dir``. Rejects empty names,
    ``..``, and any path separators (blocks traversal).
    """
    if name == GOLDEN_NAME:
        return GOLDEN_PATH
    if not name or name.startswith(".") or ".." in name or len(name) > _MAX_NAME_LEN or not _SAFE_NAME.match(name):
        raise InvalidModelName(f"unsafe or unknown model name: {name!r}")
    filename = name if name.endswith(".json") else f"{name}.json"
    return _models_dir(models_dir) / filename


def list_formats_payload() -> list[dict]:
    """All registered formats as JSON-friendly dicts (for the UI)."""
    return [
        {
            "slug": spec.slug,
            "name": spec.name,
            "category": spec.category,
            "description": spec.description,
            "safety_note": spec.safety_note,
            "provider_valid": spec.provider_valid,
        }
        for spec in list_formats()
    ]


def preview_corpus(fmt: str, count: int, seed: int) -> list[str]:
    """Return *count* synthetic, spec-valid corpus examples for *fmt*."""
    enforce_count_limit(count, maximum=GENERATE_MAX, label="count")
    spec = get_format(fmt)
    rng = np.random.default_rng(seed)
    return [spec.random_example(rng) for _ in range(count)]


def _model_from_params(params: dict, models_dir: Path | None = None) -> BigramHoneytokenModel:
    if params.get("source") == "model":
        model_name = params.get("model")
        if not model_name:
            raise InvalidModelName("'model' is required when source='model'")
        return load_model(resolve_model_ref(model_name, models_dir))
    fmt = params.get("format")
    if not fmt:
        raise DPHoneyError("'format' is required when source='format'")
    return build_model(
        fmt,
        epsilon=float(params.get("epsilon", DEFAULT_EPSILON)),
        clip=float(params.get("clip", DEFAULT_CLIP)),
        corpus_size=int(params.get("corpus_size", DEFAULT_CORPUS_SIZE)),
        train_seed=int(params.get("train_seed", DEFAULT_TRAIN_SEED)),
    )


def run_generate(params: dict, models_dir: Path | None = None) -> dict:
    """Generate a batch of synthetic tokens from format params or a saved model."""
    count = int(params.get("count", 1))
    enforce_count_limit(count, maximum=GENERATE_MAX, label="count")
    model = _model_from_params(params, models_dir)
    tokens = model.sample(
        count,
        seed=int(params.get("seed", DEFAULT_SAMPLE_SEED)),
        max_repair_attempts=int(params.get("max_attempts", DEFAULT_MAX_REPAIR_ATTEMPTS)),
    )
    return {"tokens": tokens, "format": model.format_slug, "safety": dict(_SAFETY)}


def run_report(params: dict, models_dir: Path | None = None) -> dict:
    """Generate a batch (<= REPORT_MAX) and compute realism metrics."""
    count = int(params.get("count", 1))
    enforce_count_limit(count, maximum=REPORT_MAX, label="count")
    model = _model_from_params(params, models_dir)
    tokens = model.sample(
        count,
        seed=int(params.get("seed", DEFAULT_SAMPLE_SEED)),
        max_repair_attempts=int(params.get("max_attempts", DEFAULT_MAX_REPAIR_ATTEMPTS)),
    )
    return compute_report(tokens, model)


def run_scan(text: str) -> dict:
    """Scan text and return SAFE-1 findings without matched values."""
    return {"findings": scanner.scan(text)}


def run_auto_decoy(text: str, *, seed: int = 0) -> dict:
    """Scan text and return matching decoys plus swapped text."""
    return scanner.auto_decoy(text, seed=seed)


def run_train(params: dict, models_dir: Path | None = None) -> dict:
    """Train a model from format params and save it into the models dir."""
    out_name = params.get("out_name", "")
    if (
        out_name == GOLDEN_NAME
        or not out_name
        or out_name.startswith(".")
        or ".." in out_name
        or len(out_name) > _MAX_NAME_LEN
        or not _SAFE_NAME.match(out_name)
    ):
        raise InvalidModelName(f"unsafe output name: {out_name!r}")
    fmt = params.get("format")
    if not fmt:
        raise DPHoneyError("'format' is required for training")
    directory = _models_dir(models_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = out_name if out_name.endswith(".json") else f"{out_name}.json"
    model = build_model(
        fmt,
        epsilon=float(params.get("epsilon", DEFAULT_EPSILON)),
        clip=float(params.get("clip", DEFAULT_CLIP)),
        corpus_size=int(params.get("corpus_size", DEFAULT_CORPUS_SIZE)),
        train_seed=int(params.get("seed", DEFAULT_TRAIN_SEED)),
    )
    save_model(model, directory / filename, force=bool(params.get("force", False)))
    return {
        "saved": filename,
        "format": model.format_slug,
        "epsilon": model.epsilon,
        "clip": model.clip,
        "corpus_size": model.corpus_size,
        "train_seed": model.train_seed,
    }


def _describe_model(name: str, path: Path, source: str) -> dict:
    info = {"name": name, "source": source, "slug": None}
    try:
        data = read_artifact_dict(path)
        info["slug"] = data.get("format", {}).get("slug")
        info["schema_version"] = data.get("schema_version")
    except DPHoneyError:
        info["error"] = "unreadable"
    return info


def list_models(models_dir: Path | None = None) -> list[dict]:
    """List the committed golden fixture plus any saved models in the models dir."""
    entries: list[dict] = []
    if GOLDEN_PATH.exists():
        entries.append(_describe_model(GOLDEN_NAME, GOLDEN_PATH, "fixture"))
    directory = _models_dir(models_dir)
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            entries.append(_describe_model(path.stem, path, "library"))
    return entries


def _snapshot_status(slug: str, stored_hash) -> str:
    # Mirrors detect.dp_honey.__main__._snapshot_status (kept local to avoid
    # importing a private CLI helper).
    try:
        live = get_format(slug)
    except UnknownFormatError:
        return "UNKNOWN_FORMAT"
    return "OK" if stored_hash == live.spec_hash() else "DRIFT"


def run_inspect(model_name: str, models_dir: Path | None = None) -> dict:
    """Lenient inspection of an artifact (reports drift; never raises on drift)."""
    data = read_artifact_dict(resolve_model_ref(model_name, models_dir))
    fmt = data.get("format", {})
    privacy = data.get("privacy", {})
    alphabet = data.get("alphabet", {})
    slug = fmt.get("slug", "?")
    return {
        "schema_version": data.get("schema_version"),
        "format": slug,
        "registry_version": fmt.get("registry_version"),
        "epsilon": privacy.get("epsilon"),
        "clip": privacy.get("clip"),
        "corpus_size": privacy.get("corpus_size"),
        "train_seed": privacy.get("train_seed"),
        "alphabet_size": len(alphabet.get("symbols", [])),
        "snapshot_status": _snapshot_status(slug, fmt.get("spec_hash")),
        "safety": data.get("safety", {}),
    }


def run_validate(model_name: str, models_dir: Path | None = None) -> dict:
    """Strictly validate an artifact; never raises — returns a result dict."""
    try:
        load_model(resolve_model_ref(model_name, models_dir))
        return {"valid": True, "error": None}
    except DPHoneyError as exc:
        return {"valid": False, "error": str(exc)}
