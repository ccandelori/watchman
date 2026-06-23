# DP-HONEY Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local web UI that runs all seven DP-HONEY commands from the browser by calling the existing `detect.dp_honey` library through a thin FastAPI layer.

**Architecture:** Three layers — a testable `service.py` (calls the core library, enforces caps, sanitizes model names, raises `DPHoneyError`), a thin FastAPI `app.py` (routes + a `DPHoneyError → HTTP 400` handler + static mount), and a vanilla-JS single-page frontend. The UI never shells out to the CLI; it imports the library directly.

**Tech Stack:** Python 3.11+, FastAPI + Uvicorn (optional `ui` extra; core stays numpy-only), stdlib `argparse`, vanilla HTML/CSS/JS, pytest.

**Spec:** `docs/superpowers/specs/2026-06-20-dp-honey-ui-design.md` (requirements UI-R1..UI-R8).

**Branch:** `feat/dp-honey-webui` (already created off `feat/dp-honey-generator`).

---

## File structure

```text
detect/dp_honey/webui/
  __init__.py        # package marker + docstring
  service.py         # testable service layer (NO fastapi/http imports)
  app.py             # FastAPI app: thin routes, error handler, static mount
  __main__.py        # uvicorn launcher (python -m detect.dp_honey.webui)
  static/
    index.html       # single-page shell
    styles.css       # layout + safety theme
    app.js           # fetch-based client (forms + result rendering)
tests/
  test_dp_honey_webui.py   # service-layer tests (no HTTP)
pyproject.toml       # add `ui` extra + `dp-honey-ui` script (modify)
```

Responsibilities: `service.py` owns all behavior and validation (so it is unit-testable without a server); `app.py` is HTTP glue only; the frontend is presentation only. Files that change together (the service and its tests) are planned together.

---

## Task 1: Package skeleton, dependency extra, and console script

**Files:**
- Create: `detect/dp_honey/webui/__init__.py`
- Create: `detect/dp_honey/webui/static/.gitkeep`
- Modify: `pyproject.toml` (optional-dependencies + scripts)

- [ ] **Step 1: Create the webui package marker**

Create `detect/dp_honey/webui/__init__.py`:

```python
"""Local web UI for DP-HONEY.

A thin FastAPI app over :mod:`detect.dp_honey.webui.service`, which wraps the
core :mod:`detect.dp_honey` library. Every output stays synthetic and shape-only;
the server binds to localhost only. Install with the optional extra::

    pip install -e ".[ui]"
    python -m detect.dp_honey.webui
"""
```

- [ ] **Step 2: Keep the static dir present in git**

Create `detect/dp_honey/webui/static/.gitkeep` (empty file) so the directory exists before the static files land.

- [ ] **Step 3: Add the `ui` extra and console script**

In `pyproject.toml`, replace this block:

```toml
[project.optional-dependencies]
dev = ["pytest>=7"]

[project.scripts]
dp-honey = "detect.dp_honey.__main__:main"
```

with:

```toml
[project.optional-dependencies]
dev = ["pytest>=7"]
ui = ["fastapi>=0.110", "uvicorn>=0.27"]

[project.scripts]
dp-honey = "detect.dp_honey.__main__:main"
dp-honey-ui = "detect.dp_honey.webui.__main__:main"
```

- [ ] **Step 4: Verify the package imports and TOML parses**

Run: `python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['optional-dependencies']['ui']); import detect.dp_honey.webui; print('ok')"`
Expected: prints `['fastapi>=0.110', 'uvicorn>=0.27']` then `ok`.

- [ ] **Step 5: Install the UI extra (so FastAPI is available for later tasks)**

Run: `pip install -e ".[ui]"`
Expected: installs fastapi + uvicorn (and starlette/pydantic) without errors.

- [ ] **Step 6: Commit**

```bash
git add detect/dp_honey/webui/__init__.py detect/dp_honey/webui/static/.gitkeep pyproject.toml
git commit -m "feat(webui): scaffold web UI package, ui extra, and console script"
```

---

## Task 2: Service — model-name sanitization and `resolve_model_ref`

**Files:**
- Create: `detect/dp_honey/webui/service.py`
- Test: `tests/test_dp_honey_webui.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dp_honey_webui.py`:

```python
"""Tests for the DP-HONEY web UI service layer (no HTTP)."""

from __future__ import annotations

import pytest

from detect.dp_honey.webui import service
from detect.dp_honey.webui.service import InvalidModelName


@pytest.mark.parametrize("bad", ["../secret", "a/b", "a\\b", "..", "", "name with space", "/abs"])
def test_resolve_model_ref_rejects_unsafe_names(bad, tmp_path):
    with pytest.raises(InvalidModelName):
        service.resolve_model_ref(bad, models_dir=tmp_path)


def test_resolve_model_ref_maps_safe_name_into_models_dir(tmp_path):
    ref = service.resolve_model_ref("my-model", models_dir=tmp_path)
    assert ref == tmp_path / "my-model.json"


def test_resolve_model_ref_maps_golden_label_to_fixture():
    ref = service.resolve_model_ref(service.GOLDEN_NAME)
    assert ref.name == "golden_model.json"
    assert "fixtures" in ref.parts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_webui.py -q`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` for `detect.dp_honey.webui.service`.

- [ ] **Step 3: Write minimal implementation**

Create `detect/dp_honey/webui/service.py`:

```python
"""Service layer for the DP-HONEY web UI.

Pure-ish functions that wrap the core :mod:`detect.dp_honey` library: they enforce
count caps, sanitize model names, and raise :class:`DPHoneyError` subclasses. No
FastAPI/HTTP imports live here, so every function is unit-testable without a server.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..errors import DPHoneyError

# Reserved label the UI uses for the committed synthetic golden fixture.
GOLDEN_NAME = "golden-fixture"
GOLDEN_PATH = Path("tests/fixtures/dp_honey/golden_model.json")

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class InvalidModelName(DPHoneyError):
    """A model name contains unsafe characters or path components."""


def _models_dir(models_dir: Optional[Path]) -> Path:
    return Path(models_dir) if models_dir is not None else Path("models")


def resolve_model_ref(name: str, models_dir: Optional[Path] = None) -> Path:
    """Resolve a client-supplied model *name* to a safe on-disk path.

    Accepts the reserved golden-fixture label, or a name matching
    ``[A-Za-z0-9._-]+`` resolved inside ``models_dir``. Rejects empty names,
    ``..``, and any path separators (blocks traversal).
    """
    if name == GOLDEN_NAME:
        return GOLDEN_PATH
    if not name or ".." in name or not _SAFE_NAME.match(name):
        raise InvalidModelName(f"unsafe or unknown model name: {name!r}")
    filename = name if name.endswith(".json") else f"{name}.json"
    return _models_dir(models_dir) / filename
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_webui.py -q`
Expected: PASS (all parametrized cases + the two mapping tests).

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/webui/service.py tests/test_dp_honey_webui.py
git commit -m "feat(webui): path-safe model-name resolution in the service layer"
```

---

## Task 3: Service — `list_formats_payload` and `preview_corpus`

**Files:**
- Modify: `detect/dp_honey/webui/service.py`
- Test: `tests/test_dp_honey_webui.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dp_honey_webui.py`:

```python
from detect.dp_honey import get_format
from detect.dp_honey.errors import CountLimitError


def test_list_formats_payload_has_all_slugs_and_safety():
    payload = service.list_formats_payload()
    slugs = {item["slug"] for item in payload}
    assert "github-ghp" in slugs and "aws-access-key-id" in slugs
    assert len(slugs) == 9
    for item in payload:
        assert item["safety_note"]
        assert item["provider_valid"] is False


def test_preview_corpus_returns_valid_examples():
    examples = service.preview_corpus("github-ghp", count=3, seed=0)
    assert len(examples) == 3
    spec = get_format("github-ghp")
    assert all(spec.validate(e) for e in examples)


def test_preview_corpus_rejects_oversized_count():
    with pytest.raises(CountLimitError):
        service.preview_corpus("github-ghp", count=10_001, seed=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_webui.py -k "list_formats or preview" -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'list_formats_payload'`.

- [ ] **Step 3: Write minimal implementation**

Add imports near the top of `service.py` (after the existing imports):

```python
import numpy as np

from ..__main__ import GENERATE_MAX
from ..formats import get_format, list_formats
from ..realism import REPORT_MAX, enforce_count_limit
```

Then append these functions to `service.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_webui.py -q`
Expected: PASS (all tests so far).

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/webui/service.py tests/test_dp_honey_webui.py
git commit -m "feat(webui): list-formats and preview-corpus service functions"
```

---

## Task 4: Service — generate and report (with model resolution + caps)

**Files:**
- Modify: `detect/dp_honey/webui/service.py`
- Test: `tests/test_dp_honey_webui.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dp_honey_webui.py`:

```python
def test_run_generate_from_format_is_valid_and_deterministic():
    params = {"source": "format", "format": "github-ghp", "count": 4, "seed": 7, "train_seed": 3}
    a = service.run_generate(params)
    b = service.run_generate(params)
    assert a["tokens"] == b["tokens"]
    assert len(a["tokens"]) == 4
    spec = get_format("github-ghp")
    assert all(spec.validate(t) for t in a["tokens"])
    assert a["safety"]["provider_valid"] is False


def test_run_generate_oversized_count_rejected():
    with pytest.raises(CountLimitError):
        service.run_generate({"source": "format", "format": "github-ghp", "count": 10_001})


def test_run_report_has_metrics():
    report = service.run_report({"source": "format", "format": "github-ghp", "count": 20, "seed": 1})
    for field in ("validity_rate", "char_entropy_bits", "duplicate_rate", "avg_log_likelihood", "debug"):
        assert field in report


def test_run_report_oversized_count_rejected():
    with pytest.raises(CountLimitError):
        service.run_report({"source": "format", "format": "github-ghp", "count": 5001})


def test_run_generate_from_golden_fixture_model():
    out = service.run_generate({"source": "model", "model": service.GOLDEN_NAME, "count": 3, "seed": 1})
    assert len(out["tokens"]) == 3
    spec = get_format("aws-access-key-id")
    assert all(spec.validate(t) for t in out["tokens"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_webui.py -k "generate or report" -q`
Expected: FAIL — `AttributeError: ... 'run_generate'`.

- [ ] **Step 3: Write minimal implementation**

Add imports to `service.py` (with the other core imports):

```python
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
from ..model_io import load_model
from ..realism import compute_report
```

Add a safety constant near the top of `service.py` (after `_SAFE_NAME`):

```python
_SAFETY = {
    "synthetic_only": True,
    "provider_valid": False,
    "note": "Synthetic, shape-only honeytokens. Not real, valid, or usable credentials.",
}
```

Append these functions to `service.py`:

```python
def _model_from_params(params: dict, models_dir: Optional[Path] = None) -> BigramHoneytokenModel:
    if params.get("source") == "model":
        return load_model(resolve_model_ref(params["model"], models_dir))
    return build_model(
        params["format"],
        epsilon=float(params.get("epsilon", DEFAULT_EPSILON)),
        clip=float(params.get("clip", DEFAULT_CLIP)),
        corpus_size=int(params.get("corpus_size", DEFAULT_CORPUS_SIZE)),
        train_seed=int(params.get("train_seed", DEFAULT_TRAIN_SEED)),
    )


def run_generate(params: dict, models_dir: Optional[Path] = None) -> dict:
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


def run_report(params: dict, models_dir: Optional[Path] = None) -> dict:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_webui.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/webui/service.py tests/test_dp_honey_webui.py
git commit -m "feat(webui): generate and report service functions with caps and model resolution"
```

---

## Task 5: Service — train and list-models

**Files:**
- Modify: `detect/dp_honey/webui/service.py`
- Test: `tests/test_dp_honey_webui.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dp_honey_webui.py`:

```python
from detect.dp_honey.errors import ModelArtifactExistsError


def test_run_train_writes_and_list_models_shows_it(tmp_path):
    result = service.run_train(
        {"format": "github-ghp", "out_name": "demo", "corpus_size": 20, "seed": 1},
        models_dir=tmp_path,
    )
    assert result["saved"] == "demo.json"
    assert (tmp_path / "demo.json").exists()

    models = service.list_models(models_dir=tmp_path)
    names = {m["name"] for m in models}
    assert "demo" in names
    assert service.GOLDEN_NAME in names  # the fixture is always listed
    demo = next(m for m in models if m["name"] == "demo")
    assert demo["slug"] == "github-ghp"
    assert demo["source"] == "library"


def test_run_train_refuses_overwrite_without_force(tmp_path):
    args = {"format": "github-ghp", "out_name": "demo", "corpus_size": 20, "seed": 1}
    service.run_train(args, models_dir=tmp_path)
    with pytest.raises(ModelArtifactExistsError):
        service.run_train(args, models_dir=tmp_path)


def test_run_train_rejects_unsafe_out_name(tmp_path):
    with pytest.raises(InvalidModelName):
        service.run_train(
            {"format": "github-ghp", "out_name": "../evil", "corpus_size": 20},
            models_dir=tmp_path,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_webui.py -k "train or list_models" -q`
Expected: FAIL — `AttributeError: ... 'run_train'`.

- [ ] **Step 3: Write minimal implementation**

Add import to `service.py`:

```python
from ..model_io import read_artifact_dict, save_model
```

(Combine with the existing `from ..model_io import load_model` line into
`from ..model_io import load_model, read_artifact_dict, save_model`.)

Append to `service.py`:

```python
def run_train(params: dict, models_dir: Optional[Path] = None) -> dict:
    """Train a model from format params and save it into the models dir."""
    out_name = params.get("out_name", "")
    if out_name == GOLDEN_NAME or not out_name or ".." in out_name or not _SAFE_NAME.match(out_name):
        raise InvalidModelName(f"unsafe output name: {out_name!r}")
    directory = _models_dir(models_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = out_name if out_name.endswith(".json") else f"{out_name}.json"
    model = build_model(
        params["format"],
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


def list_models(models_dir: Optional[Path] = None) -> list[dict]:
    """List the committed golden fixture plus any saved models in the models dir."""
    entries: list[dict] = []
    if GOLDEN_PATH.exists():
        entries.append(_describe_model(GOLDEN_NAME, GOLDEN_PATH, "fixture"))
    directory = _models_dir(models_dir)
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            entries.append(_describe_model(path.stem, path, "library"))
    return entries
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_webui.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/webui/service.py tests/test_dp_honey_webui.py
git commit -m "feat(webui): train and list-models service functions"
```

---

## Task 6: Service — inspect and validate

**Files:**
- Modify: `detect/dp_honey/webui/service.py`
- Test: `tests/test_dp_honey_webui.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dp_honey_webui.py`:

```python
def test_run_inspect_golden_fixture_reports_ok():
    info = service.run_inspect(service.GOLDEN_NAME)
    assert info["format"] == "aws-access-key-id"
    assert info["snapshot_status"] == "OK"
    assert info["schema_version"] == "1"
    assert info["safety"]["provider_valid"] is False


def test_run_validate_golden_fixture_is_valid():
    assert service.run_validate(service.GOLDEN_NAME) == {"valid": True, "error": None}


def test_run_validate_reports_error_for_bad_artifact(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    result = service.run_validate("broken", models_dir=tmp_path)
    assert result["valid"] is False
    assert result["error"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_webui.py -k "inspect or validate" -q`
Expected: FAIL — `AttributeError: ... 'run_inspect'`.

- [ ] **Step 3: Write minimal implementation**

Add import to `service.py`:

```python
from ..errors import UnknownFormatError
```

(Combine with the existing `from ..errors import DPHoneyError` into
`from ..errors import DPHoneyError, UnknownFormatError`.)

Append to `service.py`:

```python
def _snapshot_status(slug: str, stored_hash) -> str:
    # Mirrors detect.dp_honey.__main__._snapshot_status (kept local to avoid
    # importing a private CLI helper).
    try:
        live = get_format(slug)
    except UnknownFormatError:
        return "UNKNOWN_FORMAT"
    return "OK" if stored_hash == live.spec_hash() else "DRIFT"


def run_inspect(model_name: str, models_dir: Optional[Path] = None) -> dict:
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


def run_validate(model_name: str, models_dir: Optional[Path] = None) -> dict:
    """Strictly validate an artifact; never raises — returns a result dict."""
    try:
        load_model(resolve_model_ref(model_name, models_dir))
        return {"valid": True, "error": None}
    except DPHoneyError as exc:
        return {"valid": False, "error": str(exc)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_webui.py -q`
Expected: PASS (full service-layer suite).

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/webui/service.py tests/test_dp_honey_webui.py
git commit -m "feat(webui): inspect and validate service functions"
```

---

## Task 7: FastAPI app and error handler

**Files:**
- Create: `detect/dp_honey/webui/app.py`
- Test: `tests/test_dp_honey_webui_http.py` (a separate file, so it skips wholesale when httpx is absent — without skipping the service-layer tests)

- [ ] **Step 1: Write the failing test (own file; skipped entirely if httpx is absent)**

Create `tests/test_dp_honey_webui_http.py`:

```python
"""HTTP smoke tests for the web UI app.

Kept in its own file with a module-level importorskip: if httpx (TestClient's
dependency) is not installed, ONLY these HTTP tests skip — the service-layer
tests in test_dp_honey_webui.py still run.
"""

from __future__ import annotations

import pytest

pytest.importorskip("httpx")


def _client():
    from fastapi.testclient import TestClient

    from detect.dp_honey.webui.app import create_app

    return TestClient(create_app())


def test_http_formats_endpoint_lists_slugs():
    resp = _client().get("/api/formats")
    assert resp.status_code == 200
    assert any(item["slug"] == "github-ghp" for item in resp.json())


def test_http_generate_returns_valid_tokens():
    resp = _client().post(
        "/api/generate",
        json={"source": "format", "format": "github-ghp", "count": 3, "seed": 1},
    )
    assert resp.status_code == 200
    assert len(resp.json()["tokens"]) == 3


def test_http_dphoney_error_maps_to_400():
    resp = _client().post("/api/generate", json={"source": "format", "format": "nope", "count": 1})
    assert resp.status_code == 400
    assert "error" in resp.json()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_webui_http.py -q`
Expected: FAIL — `ModuleNotFoundError` for `detect.dp_honey.webui.app` (or the whole file is skipped if httpx is absent; `pip install httpx` to run these).

- [ ] **Step 3: Write minimal implementation**

Create `detect/dp_honey/webui/app.py`:

```python
"""FastAPI app for the DP-HONEY web UI: thin routes over the service layer."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..errors import DPHoneyError
from . import service

_STATIC = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="DP-HONEY UI",
        description="Synthetic, shape-only honeytoken generator. Outputs are never real credentials.",
    )

    @app.exception_handler(DPHoneyError)
    async def _on_dphoney_error(request: Request, exc: DPHoneyError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.get("/api/formats")
    def api_formats() -> list:
        return service.list_formats_payload()

    @app.post("/api/preview-corpus")
    def api_preview(body: dict) -> dict:
        examples = service.preview_corpus(body["format"], int(body.get("count", 10)), int(body.get("seed", 0)))
        return {"examples": examples}

    @app.post("/api/generate")
    def api_generate(body: dict) -> dict:
        return service.run_generate(body)

    @app.post("/api/report")
    def api_report(body: dict) -> dict:
        return service.run_report(body)

    @app.post("/api/train")
    def api_train(body: dict) -> dict:
        return service.run_train(body)

    @app.get("/api/models")
    def api_models() -> list:
        return service.list_models()

    @app.post("/api/inspect")
    def api_inspect(body: dict) -> dict:
        return service.run_inspect(body["model"])

    @app.post("/api/validate")
    def api_validate(body: dict) -> dict:
        return service.run_validate(body["model"])

    @app.get("/api/models/{name}/download")
    def api_download(name: str) -> FileResponse:
        ref = service.resolve_model_ref(name)
        return FileResponse(ref, media_type="application/json", filename=ref.name)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
    return app


app = create_app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install httpx` (if needed) then `python -m pytest tests/test_dp_honey_webui_http.py -q`
Expected: PASS (3 HTTP smoke tests).

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/webui/app.py tests/test_dp_honey_webui_http.py
git commit -m "feat(webui): FastAPI app with routes and DPHoneyError->400 handler"
```

---

## Task 8: Uvicorn launcher (`python -m detect.dp_honey.webui`)

**Files:**
- Create: `detect/dp_honey/webui/__main__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dp_honey_webui.py`:

```python
def test_launcher_parser_defaults_to_localhost():
    from detect.dp_honey.webui.__main__ import build_parser

    args = build_parser().parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_webui.py -k launcher -q`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` for `detect.dp_honey.webui.__main__`.

- [ ] **Step 3: Write minimal implementation**

Create `detect/dp_honey/webui/__main__.py`:

```python
"""Launch the DP-HONEY web UI with uvicorn (localhost only by default)."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dp-honey-ui",
        description="Run the DP-HONEY web UI. Binds to localhost by default; "
        "every output is synthetic and shape-only.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    import uvicorn

    from .app import app

    print(f"DP-HONEY UI on http://{args.host}:{args.port}  (synthetic, shape-only — not real credentials)")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_webui.py -k launcher -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/webui/__main__.py tests/test_dp_honey_webui.py
git commit -m "feat(webui): uvicorn launcher binding localhost by default"
```

---

## Task 9: Frontend (single-page client)

**Files:**
- Create: `detect/dp_honey/webui/static/index.html`
- Create: `detect/dp_honey/webui/static/styles.css`
- Create: `detect/dp_honey/webui/static/app.js`

No unit tests here (presentation layer); verified manually in Task 10. Keep behavior thin — all logic stays in the service layer.

- [ ] **Step 1: Create `index.html`**

Create `detect/dp_honey/webui/static/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DP-HONEY</title>
  <link rel="stylesheet" href="/static/styles.css" />
</head>
<body>
  <header class="banner">
    <strong>DP-HONEY</strong>
    <span class="safety">Synthetic, shape-only honeytokens — NOT real, valid, or usable credentials.</span>
  </header>
  <main>
    <nav id="nav"></nav>
    <section class="panel">
      <h2 id="cmd-title">Pick a command</h2>
      <form id="form"></form>
      <div id="error" class="error" hidden></div>
      <div id="output" class="output"></div>
    </section>
  </main>
  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `styles.css`**

Create `detect/dp_honey/webui/static/styles.css`:

```css
:root { --bg:#0f1419; --panel:#1a2230; --ink:#e6edf3; --muted:#9fb0c3; --accent:#3fb950; --warn:#d29922; --err:#f85149; }
* { box-sizing: border-box; }
body { margin:0; font:15px/1.5 system-ui, sans-serif; background:var(--bg); color:var(--ink); }
.banner { display:flex; gap:1rem; align-items:baseline; padding:.75rem 1.25rem; background:#111820; border-bottom:1px solid #2a3444; }
.banner .safety { color:var(--warn); font-size:.85rem; }
main { display:grid; grid-template-columns:200px 1fr; gap:1.25rem; padding:1.25rem; max-width:1100px; }
nav { display:flex; flex-direction:column; gap:.25rem; }
nav button { text-align:left; padding:.5rem .75rem; background:transparent; color:var(--ink); border:1px solid transparent; border-radius:6px; cursor:pointer; }
nav button:hover { background:var(--panel); }
nav button.active { background:var(--panel); border-color:#2a3444; }
.panel { background:var(--panel); border:1px solid #2a3444; border-radius:10px; padding:1.25rem; min-height:60vh; }
form { display:flex; flex-wrap:wrap; gap:.75rem 1rem; align-items:end; margin-bottom:1rem; }
label { display:flex; flex-direction:column; gap:.25rem; font-size:.8rem; color:var(--muted); }
input, select { background:#0d1117; color:var(--ink); border:1px solid #2a3444; border-radius:6px; padding:.4rem .5rem; }
button.run { background:var(--accent); color:#04150a; border:none; border-radius:6px; padding:.5rem .9rem; font-weight:600; cursor:pointer; }
.output { white-space:pre-wrap; font-family:ui-monospace, monospace; font-size:.85rem; }
.tokens div { padding:.1rem 0; border-bottom:1px solid #20293650; }
.tag { color:var(--muted); font-size:.7rem; margin-left:.5rem; }
.card { display:grid; grid-template-columns:auto 1fr; gap:.25rem 1rem; font-family:ui-monospace, monospace; }
.card b { color:var(--muted); font-weight:500; }
.badge-ok { color:var(--accent); } .badge-bad { color:var(--err); } .badge-drift { color:var(--warn); }
.error { color:var(--err); background:#2a1416; border:1px solid #5a2226; border-radius:6px; padding:.5rem .75rem; margin-bottom:1rem; }
.copy { margin-left:.5rem; font-size:.75rem; cursor:pointer; color:var(--muted); background:none; border:1px solid #2a3444; border-radius:4px; }
```

- [ ] **Step 3: Create `app.js`**

Create `detect/dp_honey/webui/static/app.js`:

```javascript
"use strict";

const COMMANDS = [
  { id: "list-formats", label: "Formats" },
  { id: "preview-corpus", label: "Preview corpus" },
  { id: "generate", label: "Generate" },
  { id: "report", label: "Report" },
  { id: "train", label: "Train" },
  { id: "inspect", label: "Inspect model" },
  { id: "validate", label: "Validate" },
];

let FORMATS = [];
let MODELS = [];

async function api(path, body) {
  const opts = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const resp = await fetch(path, opts);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => (k === "class" ? (e.className = v) : e.setAttribute(k, v)));
  kids.forEach((k) => e.append(k));
  return e;
}

function formatOptions() {
  return FORMATS.map((f) => el("option", { value: f.slug }, `${f.slug} — ${f.name}`));
}
function modelOptions() {
  return MODELS.map((m) => el("option", { value: m.name }, `${m.name} (${m.source}${m.slug ? ", " + m.slug : ""})`));
}
function field(labelText, control) {
  const l = el("label", {}, labelText);
  l.append(control);
  return l;
}
function numInput(name, value) {
  return el("input", { name, type: "number", value: String(value) });
}

function sourcePicker() {
  const sel = el("select", { name: "source" }, el("option", { value: "format" }, "train on the fly"), el("option", { value: "model" }, "saved model"));
  return sel;
}

function showError(msg) {
  const e = document.getElementById("error");
  e.textContent = msg;
  e.hidden = false;
}
function clearError() {
  document.getElementById("error").hidden = true;
}
function setOutput(node) {
  const o = document.getElementById("output");
  o.replaceChildren(node);
}

function tokenList(tokens) {
  const wrap = el("div", { class: "tokens" });
  tokens.forEach((t) => wrap.append(el("div", {}, t, el("span", { class: "tag" }, "synthetic"))));
  const copy = el("button", { class: "copy" }, "copy all");
  copy.onclick = () => navigator.clipboard.writeText(tokens.join("\n"));
  const head = el("div", {}, `${tokens.length} synthetic, shape-only tokens`, copy);
  return el("div", {}, head, wrap);
}

function jsonBlock(obj) {
  return el("pre", {}, JSON.stringify(obj, null, 2));
}

function buildForm(cmd) {
  const form = document.getElementById("form");
  form.replaceChildren();
  document.getElementById("cmd-title").textContent = COMMANDS.find((c) => c.id === cmd).label;
  clearError();
  document.getElementById("output").replaceChildren();

  const needsSource = cmd === "generate" || cmd === "report";
  const needsModelOnly = cmd === "inspect" || cmd === "validate";

  if (cmd === "list-formats") {
    // no inputs
  } else if (cmd === "preview-corpus") {
    form.append(field("format", el("select", { name: "format" }, ...formatOptions())));
    form.append(field("count", numInput("count", 5)));
    form.append(field("seed", numInput("seed", 0)));
  } else if (needsSource) {
    const src = sourcePicker();
    form.append(field("source", src));
    const fmt = field("format", el("select", { name: "format" }, ...formatOptions()));
    const mdl = field("model", el("select", { name: "model" }, ...modelOptions()));
    form.append(fmt, mdl);
    const sync = () => { fmt.hidden = src.value !== "format"; mdl.hidden = src.value !== "model"; };
    src.onchange = sync; sync();
    form.append(field("count", numInput("count", cmd === "report" ? 100 : 10)));
    form.append(field("seed", numInput("seed", 1)));
    form.append(field("epsilon", numInput("epsilon", 1)));
    form.append(field("clip", numInput("clip", 1)));
    form.append(field("corpus_size", numInput("corpus_size", 200)));
    form.append(field("train_seed", numInput("train_seed", 0)));
  } else if (cmd === "train") {
    form.append(field("format", el("select", { name: "format" }, ...formatOptions())));
    form.append(field("out_name", el("input", { name: "out_name", value: "my-model" })));
    form.append(field("epsilon", numInput("epsilon", 1)));
    form.append(field("clip", numInput("clip", 1)));
    form.append(field("corpus_size", numInput("corpus_size", 200)));
    form.append(field("seed", numInput("seed", 0)));
    const force = el("input", { name: "force", type: "checkbox" });
    form.append(field("overwrite", force));
  } else if (needsModelOnly) {
    form.append(field("model", el("select", { name: "model" }, ...modelOptions())));
  }

  form.append(el("button", { class: "run", type: "submit" }, "Run"));
  form.onsubmit = (ev) => { ev.preventDefault(); runCommand(cmd, form); };
}

function readForm(form) {
  const data = {};
  new FormData(form).forEach((v, k) => (data[k] = v));
  form.querySelectorAll("input[type=checkbox]").forEach((c) => (data[c.name] = c.checked));
  ["count", "seed", "epsilon", "clip", "corpus_size", "train_seed"].forEach((k) => {
    if (data[k] !== undefined && data[k] !== "") data[k] = Number(data[k]);
  });
  return data;
}

async function runCommand(cmd, form) {
  clearError();
  try {
    const body = readForm(form);
    if (cmd === "list-formats") {
      const formats = await api("/api/formats");
      setOutput(jsonBlock(formats));
    } else if (cmd === "preview-corpus") {
      const { examples } = await api("/api/preview-corpus", body);
      setOutput(tokenList(examples));
    } else if (cmd === "generate") {
      const { tokens } = await api("/api/generate", body);
      setOutput(tokenList(tokens));
    } else if (cmd === "report") {
      setOutput(jsonBlock(await api("/api/report", body)));
    } else if (cmd === "train") {
      const res = await api("/api/train", body);
      await refreshModels();
      setOutput(jsonBlock(res));
    } else if (cmd === "inspect") {
      setOutput(jsonBlock(await api("/api/inspect", body)));
    } else if (cmd === "validate") {
      const res = await api("/api/validate", body);
      const cls = res.valid ? "badge-ok" : "badge-bad";
      setOutput(el("div", { class: cls }, res.valid ? "VALID" : `INVALID: ${res.error}`));
    }
  } catch (err) {
    showError(err.message);
  }
}

async function refreshModels() {
  MODELS = await api("/api/models");
}

function buildNav(active) {
  const nav = document.getElementById("nav");
  nav.replaceChildren();
  COMMANDS.forEach((c) => {
    const b = el("button", c.id === active ? { class: "active" } : {}, c.label);
    b.onclick = () => { buildNav(c.id); buildForm(c.id); };
    nav.append(b);
  });
}

async function init() {
  try {
    FORMATS = await api("/api/formats");
    await refreshModels();
  } catch (err) {
    showError(err.message);
  }
  buildNav("generate");
  buildForm("generate");
}

init();
```

- [ ] **Step 4: Commit**

```bash
git add detect/dp_honey/webui/static/index.html detect/dp_honey/webui/static/styles.css detect/dp_honey/webui/static/app.js
git commit -m "feat(webui): single-page frontend (forms + result rendering)"
```

---

## Task 10: Manual verification, docs, and final commit

**Files:**
- Modify: `detect/dp_honey/README.md` (add a Web UI section)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS — the generator suite plus the new web UI service tests (and HTTP smoke tests if httpx is installed).

- [ ] **Step 2: Lint**

Run: `python -m ruff check detect tests conftest.py`
Expected: `All checks passed!` (fix any unused imports with `python -m ruff check --fix ...`).

- [ ] **Step 3: Launch and manually verify in the browser**

Run: `python -m detect.dp_honey.webui`
Then open `http://127.0.0.1:8000` and verify:
- The safety banner is visible at the top.
- **Generate** (source = train on the fly, format = `github-ghp`, count 5) prints 5 `ghp_…` tokens labeled synthetic.
- **Train** (out_name `demo`) succeeds; **Inspect** and **Validate** then list `demo` and the `golden-fixture` in the model dropdown; inspect shows `snapshot_status: OK`; validate shows VALID.
- **Report** (source = saved model, `golden-fixture`, count 100) renders the metrics JSON.
- An invalid request (e.g., **Generate** with count `99999`) shows a red error, not a crash.
Stop the server with Ctrl+C.

- [ ] **Step 4: Add a Web UI section to the README**

In `detect/dp_honey/README.md`, immediately after the `## Quickstart (CLI)` section's closing paragraph (the one ending "...exit nonzero *before* any generation work begins."), insert:

```markdown

## Web UI

A local browser UI runs every command without the terminal:

```bash
pip install -e ".[ui]"
python -m detect.dp_honey.webui      # serves http://127.0.0.1:8000 (localhost only)
```

It calls the library directly (no subprocess), keeps the synthetic/shape-only
banner on every screen, and offers a model library (saved models in `models/`
plus the committed golden fixture). The API is a thin FastAPI layer; see
`/docs` for the auto-generated endpoint reference.
```

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/README.md
git commit -m "docs(webui): document the local web UI in the package README"
```

---

## Self-review notes

- **Spec coverage:** UI-R1 (library-backed routes) → Tasks 4/7; UI-R2 (optional `ui` extra) → Task 1; UI-R3 (localhost bind) → Task 8; UI-R4 (safety banner everywhere) → Tasks 4/9; UI-R5 (model library + sanitized names) → Tasks 2/5; UI-R6 (server-side caps) → Task 4; UI-R7 (`DPHoneyError`→400, service unit-tested) → Tasks 2-7; UI-R8 (lifecycle end to end) → Task 10 manual check + Tasks 5/6 tests.
- **Caps single-sourced:** `GENERATE_MAX` is imported from `..__main__` and `REPORT_MAX` from `..realism` — no duplicated constants.
- **Determinism:** generate/report reuse the library's seeded sampling; the service adds no randomness.
- **No new core deps:** FastAPI/uvicorn live only in the `ui` extra; `httpx` is only needed for the optional HTTP smoke tests (guarded by `importorskip`).
```
