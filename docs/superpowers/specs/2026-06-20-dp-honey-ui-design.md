---
title: "DP-HONEY Web UI design"
type: design
status: draft
date: 2026-06-20
depends_on: detect.dp_honey generator package (branch feat/dp-honey-generator, PR #1)
---

# DP-HONEY Web UI design

## Summary

A small local **web UI** that lets a user run all seven DP-HONEY commands without
the terminal. It is a thin **FastAPI** app over a **testable service layer** over
the *existing, already-tested* `detect.dp_honey` package. The UI never shells out
to the CLI — it calls the Python API directly, so the core stays the single source
of behavior and the web layer stays thin.

The UI is opt-in (an optional `ui` dependency extra) and binds to `127.0.0.1`
only. Like the package, every surface keeps the **synthetic, shape-only,
non-functional** boundary visible.

## Goals

- Run all 7 commands from a browser: `list-formats`, `preview-corpus`, `generate`,
  `report`, `train`, `inspect-model`, `validate`.
- Demonstrate the full artifact lifecycle: train a model → see it in a library →
  inspect / validate / generate / report against it.
- Reuse the tested core; add no logic that duplicates the package.
- Keep the core package dependency footprint unchanged (numpy only); the UI deps
  are an optional extra.
- Keep the safety boundary (synthetic / shape-only / not real credentials) visible
  on every screen.

## Non-goals (this build)

- No authentication / multi-user support (single local user).
- No network exposure (localhost bind only).
- No arbitrary-filesystem model paths from the client (library + golden fixture
  only; names are sanitized).
- No artifact **upload** in v1 (the model library + the committed golden fixture
  cover the demo; upload can be a later increment).
- No true token streaming over HTTP (request/response with count caps is
  sufficient for a UI; the CLI keeps its streaming behavior).

## Architecture

Three layers, each independently testable:

1. **Service layer** — `detect/dp_honey/webui/service.py`. Plain functions that
   call the core library, enforce count caps, sanitize model names, and raise the
   existing `DPHoneyError` subclasses. No FastAPI/HTTP imports here, so it is unit
   testable without a running server. Functions:
   - `list_formats_payload() -> list[dict]`
   - `preview_corpus(format, count, seed) -> list[str]`
   - `run_generate(params) -> dict`  (source = format params OR saved model)
   - `run_report(params) -> dict`
   - `run_train(format, epsilon, clip, corpus_size, seed, out_name, force) -> dict`
   - `list_models(models_dir) -> list[dict]`  (saved models + golden fixture)
   - `run_inspect(model_ref) -> dict`  (lenient; reports snapshot status)
   - `run_validate(model_ref) -> dict`  (`{"valid": bool, "error": str|None}`)
   - `resolve_model_ref(name, models_dir) -> Path`  (path-safe resolver)
2. **HTTP layer** — `detect/dp_honey/webui/app.py`. A FastAPI app whose routes are
   thin wrappers over the service. A single exception handler maps any
   `DPHoneyError` to HTTP 400 with the message (mirroring the CLI's exit-1
   contract). Serves the static page and assets.
3. **Frontend** — `detect/dp_honey/webui/static/{index.html,app.js,styles.css}`.
   A single page using vanilla `fetch`; no build step.

### Why call the library, not the CLI

Shelling out to `python -m detect.dp_honey` would re-parse argv, fork a process
per request, and make error handling string-based. Calling the library directly
reuses the typed errors, is faster, and keeps one behavior path that the existing
97 tests already cover.

## Project layout

```text
detect/dp_honey/webui/
  __init__.py
  __main__.py        # uvicorn launcher: python -m detect.dp_honey.webui
  app.py             # FastAPI app + routes + DPHoneyError handler + static mount
  service.py         # testable service layer (no HTTP imports)
  static/
    index.html
    app.js
    styles.css
tests/
  test_dp_honey_webui.py   # service-layer tests (no HTTP)
```

`pyproject.toml` gains an optional extra and a console script:

```toml
[project.optional-dependencies]
dev = ["pytest>=7"]
ui = ["fastapi>=0.110", "uvicorn>=0.27"]

[project.scripts]
dp-honey = "detect.dp_honey.__main__:main"
dp-honey-ui = "detect.dp_honey.webui.__main__:main"
```

## API endpoints

All POST bodies are JSON. All responses are JSON. Any `DPHoneyError` →
`HTTP 400 {"error": "<message>"}`.

| Method & path | Body / params | Maps to | Returns |
| --- | --- | --- | --- |
| `GET /api/formats` | — | list-formats | `[{slug,name,category,description,safety_note,provider_valid}]` |
| `POST /api/preview-corpus` | `{format,count,seed}` | preview-corpus | `{examples:[...]}` |
| `POST /api/generate` | `{source,format?,model?,count,seed,epsilon,clip,corpus_size,train_seed,max_attempts}` | generate | `{tokens:[...],safety:{...}}` |
| `POST /api/report` | same as generate (count ≤ REPORT_MAX) | report | the realism report dict |
| `POST /api/train` | `{format,epsilon,clip,corpus_size,seed,out_name,force}` | train | `{saved:"<name>.json", metadata:{...}}` |
| `GET /api/models` | — | (library) | `[{name,source:"library"|"fixture",slug,...}]` |
| `POST /api/inspect` | `{model}` | inspect-model | inspect dict incl. `snapshot_status` |
| `POST /api/validate` | `{model}` | validate | `{valid:bool, error:str|null}` |
| `GET /api/models/{name}/download` | name in path | (download) | the artifact JSON as a file |
| `GET /` and `/static/*` | — | — | the SPA + assets |

`source` is `"format"` (train on the fly from params) or `"model"` (use a saved
model / the golden fixture by name).

## Model library & path safety

- Saved models live in a server-side `models/` directory (already gitignored).
- `GET /api/models` lists `models/*.json` plus the committed golden fixture
  (`tests/fixtures/dp_honey/golden_model.json`, labeled `source: "fixture"`).
- The client only ever sends a **model name**, never a path. `resolve_model_ref`
  accepts names matching `^[A-Za-z0-9._-]+$` (rejecting `/`, `\`, `..`, absolute
  paths) and resolves them inside `models/` (or to the known fixture path). Any
  other input raises a `DPHoneyError` → HTTP 400. This blocks path traversal.
- `run_train` validates `out_name` the same way and refuses overwrite unless
  `force` is set (reusing `save_model`'s `ModelArtifactExistsError`).

## Frontend

- One page. Left: a nav list of the 7 commands. Right: a form for the selected
  command and a results pane.
- A persistent header banner: **"Synthetic, shape-only — not real credentials."**
- `generate` / `preview-corpus`: monospace token list + copy button, each block
  labeled synthetic.
- `report`: a readable metrics card (validity rate, entropy, duplicate rate, avg
  log-likelihood) with a raw-JSON toggle.
- `train`: form → on success, the new model appears in the model dropdowns.
- `inspect` / `validate`: pick a model from the library dropdown; show
  fields / pass-fail and snapshot status.
- Clean, professional aesthetic. The `frontend-design` skill may be applied during
  implementation for additional polish; it must not change the safety messaging.

## Safety & security

- Bind `127.0.0.1` only; never `0.0.0.0`.
- Synthetic / non-functional language on every screen and in token output (KTD6).
- Count caps enforced server-side before generation (`generate ≤ 10000`,
  `report ≤ 5000`) via the existing `enforce_count_limit`.
- Path-safe model names (above). No `eval`/`pickle`/shell; artifacts are JSON via
  the existing fail-closed `load_model`.
- No real secrets are ever produced, ingested, or stored.

## Testing

`tests/test_dp_honey_webui.py` targets the **service layer** (deterministic, uses a
tmp models dir via fixtures, no HTTP, so no new test dependency):

- `list_formats_payload` returns all 9 slugs with safety notes.
- `preview_corpus` / `run_generate` produce shape-valid tokens; counts honored.
- Count caps: over-limit `run_generate` / `run_report` raise `CountLimitError`.
- Name sanitization: `..`, `/`, `\`, absolute paths are rejected.
- Lifecycle: `run_train` → `list_models` shows it → `run_inspect` / `run_validate`
  pass; the golden fixture also lists, inspects, and validates.
- Error mapping: invalid format / drifted model surface `DPHoneyError`.
- `run_report` includes all metric fields and the synthetic safety block.

Optional: a couple of `fastapi.testclient.TestClient` smoke tests, guarded to skip
if `httpx` (TestClient's dependency) is not installed, so the core suite never
gains a hard dependency.

## How to run

```bash
pip install -e ".[ui]"
python -m detect.dp_honey.webui        # serves http://127.0.0.1:8000
# or, after install:
dp-honey-ui
```

## Open / deferred

- **Branch & PR:** built on `feat/dp-honey-webui` (off `feat/dp-honey-generator`).
  It depends on the generator package; the PR can target `main` once PR #1 merges,
  or stack on the generator branch.
- **Deferred increments:** artifact upload, true HTTP streaming for `generate`,
  and authentication — none needed for the capstone demo.

## Requirements

- UI-R1. The UI runs all 7 commands from the browser by calling the
  `detect.dp_honey` library (not the CLI/subprocess).
- UI-R2. The core package dependency set is unchanged; FastAPI/uvicorn are an
  optional `ui` extra.
- UI-R3. The server binds `127.0.0.1` only.
- UI-R4. The synthetic / shape-only / non-functional boundary is visible on every
  screen and in token output.
- UI-R5. Saved models live in `models/`; the library lists them plus the golden
  fixture; the client references models by sanitized name only (no paths).
- UI-R6. Count caps are enforced server-side before generation.
- UI-R7. `DPHoneyError` maps to HTTP 400 with the message; the service layer is
  unit-tested without HTTP.
- UI-R8. The artifact lifecycle (train → list → inspect → validate → generate)
  works end to end through the UI.
