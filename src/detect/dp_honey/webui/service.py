"""Service layer for the DP-HONEY web UI.

Pure-ish functions that wrap the core :mod:`detect.dp_honey` library: they enforce
count caps, sanitize model names, and raise :class:`DPHoneyError` subclasses. No
FastAPI/HTTP imports live here, so every function is unit-testable without a server.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .. import scanner
from ..artifact_status import inspect_artifact, validate_artifact
from ..bigram import (
    DEFAULT_CLIP,
    DEFAULT_CORPUS_SIZE,
    DEFAULT_EPSILON,
    DEFAULT_MAX_REPAIR_ATTEMPTS,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_TRAIN_SEED,
)
from ..errors import DPHoneyError
from ..formats import get_format, list_formats
from ..model_io import read_artifact_dict
from ..operations import (
    GENERATE_MAX,
    FormatModelSource,
    GenerateRequest,
    ModelArtifactSource,
    ReportRequest,
    TrainRequest,
    generate_tokens,
    run_report_request,
    train_to_artifact,
)
from ..realism import enforce_count_limit

# Repo root (this file is src/detect/dp_honey/webui/service.py -> parents[4])
# so the golden fixture and default models dir resolve regardless of the process CWD.
_PKG_ROOT = Path(__file__).resolve().parents[4]

# Reserved label the UI uses for the committed synthetic golden fixture.
GOLDEN_NAME = "golden-fixture"
GOLDEN_PATH = _PKG_ROOT / "tests" / "dp_honey" / "fixtures" / "dp_honey" / "golden_model.json"

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_MAX_NAME_LEN = 100


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


JsonDict = dict[str, Any]


def list_formats_payload() -> list[JsonDict]:
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


def _source_from_params(params: JsonDict, models_dir: Path | None) -> FormatModelSource | ModelArtifactSource:
    if params.get("source") == "model":
        model_name = params.get("model")
        if not model_name:
            raise InvalidModelName("'model' is required when source='model'")
        return ModelArtifactSource(path=resolve_model_ref(str(model_name), models_dir))
    fmt = params.get("format")
    if not fmt:
        raise DPHoneyError("'format' is required when source='format'")
    return FormatModelSource(
        format_slug=str(fmt),
        epsilon=_float_param(params, "epsilon", DEFAULT_EPSILON),
        clip=_float_param(params, "clip", DEFAULT_CLIP),
        corpus_size=_int_param(params, "corpus_size", DEFAULT_CORPUS_SIZE),
        train_seed=_int_param(params, "train_seed", DEFAULT_TRAIN_SEED),
    )


def run_generate(params: JsonDict, models_dir: Path | None = None) -> JsonDict:
    """Generate a batch of synthetic tokens from format params or a saved model."""
    request = GenerateRequest(
        source=_source_from_params(params, models_dir),
        count=_int_param(params, "count", 1),
        sample_seed=_int_param(params, "seed", DEFAULT_SAMPLE_SEED),
        max_repair_attempts=_int_param(params, "max_attempts", DEFAULT_MAX_REPAIR_ATTEMPTS),
    )
    return generate_tokens(request).to_dict()


def run_report(params: JsonDict, models_dir: Path | None = None) -> JsonDict:
    """Generate a batch (<= REPORT_MAX) and compute realism metrics."""
    request = ReportRequest(
        source=_source_from_params(params, models_dir),
        count=_int_param(params, "count", 1),
        sample_seed=_int_param(params, "seed", DEFAULT_SAMPLE_SEED),
        max_repair_attempts=_int_param(params, "max_attempts", DEFAULT_MAX_REPAIR_ATTEMPTS),
    )
    return run_report_request(request)


def run_scan(text: str) -> JsonDict:
    """Scan text and return SAFE-1 findings without matched values."""
    return {"findings": scanner.scan(text)}


def run_auto_decoy(text: str, *, seed: int = 0) -> JsonDict:
    """Scan text and return matching decoys plus swapped text."""
    return scanner.auto_decoy(text, seed=seed)


def run_train(params: JsonDict, models_dir: Path | None = None) -> JsonDict:
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
    request = TrainRequest(
        format_slug=str(fmt),
        output_path=directory / filename,
        epsilon=_float_param(params, "epsilon", DEFAULT_EPSILON),
        clip=_float_param(params, "clip", DEFAULT_CLIP),
        corpus_size=_int_param(params, "corpus_size", DEFAULT_CORPUS_SIZE),
        train_seed=_int_param(params, "seed", DEFAULT_TRAIN_SEED),
        force=bool(params.get("force", False)),
    )
    return train_to_artifact(request).to_dict()


def _describe_model(name: str, path: Path, source: str) -> JsonDict:
    info = {"name": name, "source": source, "slug": None}
    try:
        data = read_artifact_dict(path)
        info["slug"] = data.get("format", {}).get("slug")
        info["schema_version"] = data.get("schema_version")
    except DPHoneyError:
        info["error"] = "unreadable"
    return info


def list_models(models_dir: Path | None = None) -> list[JsonDict]:
    """List the committed golden fixture plus any saved models in the models dir."""
    entries: list[JsonDict] = []
    if GOLDEN_PATH.exists():
        entries.append(_describe_model(GOLDEN_NAME, GOLDEN_PATH, "fixture"))
    directory = _models_dir(models_dir)
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            entries.append(_describe_model(path.stem, path, "library"))
    return entries


def run_inspect(model_name: str, models_dir: Path | None = None) -> JsonDict:
    """Lenient inspection of an artifact (reports drift; never raises on drift)."""
    return inspect_artifact(resolve_model_ref(model_name, models_dir)).to_dict()


def run_validate(model_name: str, models_dir: Path | None = None) -> JsonDict:
    """Strictly validate an artifact; never raises — returns a result dict."""
    return validate_artifact(resolve_model_ref(model_name, models_dir)).to_dict()


def _int_param(params: JsonDict, key: str, default_value: int) -> int:
    value = params.get(key, default_value)
    return int(value)


def _float_param(params: JsonDict, key: str, default_value: float) -> float:
    value = params.get(key, default_value)
    return float(value)
