# Aegis

Aegis is a runtime credential-defense spine for LLM agents. It is designed to
sit between agent traffic and model/tool execution, normalize each turn, run
independent detector stages, apply policy in one place, and record auditable
events.

The repository is intentionally small right now. The current code establishes
the shared contracts and quality gates that future DP-HONEY, CIFT, NIMBUS,
proxy, SDK, dashboard, and evaluation work will plug into.

## Current Status

The first runtime spine is implemented and CI-enforced:

- Typed request, detector, policy, capability, and audit contracts.
- A minimal orchestrator that normalizes a turn, runs detector stages, calls a
  mock model provider, applies policy, and writes an audit event.
- A honeytoken ledger that replaces credential placeholders with registered
  canaries and emits `SensitiveSpan` metadata without exposing real secrets.
- First detector seams:
  - `ActivationUnavailableDetector` for explicit CIFT capability reporting in
    black-box/mock mode.
  - `NoopCanaryDetector` as the unconfigured DP-HONEY/text-canary boundary.
  - `TextCanaryDetector` for exact post-generation detection of registered
    honeytoken leaks.
  - `EncodedCanaryDetector` for base64, hex, ROT13, leet, reverse,
    fragmentation, and partial-overlap canary leakage.
- A fixture-backed `cift_selector_probe_v0` candidate monitor replay path that
  consumes promoted calibrated CIFT scores without importing research code.
- A runtime-native `aegis.detectors.cift_runtime.CiftRuntimeDetector` that loads
  an exported JSON linear CIFT model artifact, consumes feature vectors from
  `NormalizedTurn.metadata`, and emits active, degraded, or unavailable CIFT
  evidence without importing the introspection package.
- A `TurnAnnotator` seam that can attach derived metadata before pre-generation
  detectors run, plus a CIFT feature-vector annotator that preserves the
  runtime/import boundary while preparing self-hosted activation features.
- A CIFT window selector that treats selected-choice readout geometry as
  primary coverage and payload/query readout as degraded fallback evidence.
- `NimbusLeakageDetector`, a session detector that accumulates exact or encoded
  canary leakage signals into a per-session leakage budget.
- A mock OpenAI-compatible proxy surface for `/health`,
  `/v1/chat/completions`, and `/audit/recent`.
- Mandatory CI gates for linting, formatting, strict typing, import-boundary
  checks, and tests with coverage.

This is not yet a production security system. It is the enforced skeleton that
keeps future detector and proxy work compatible.

## Runtime Shape

```text
chat request
  -> NormalizedTurn
  -> turn annotators
  -> pre-generation detectors
  -> model provider
  -> post-generation detectors
  -> session detectors
  -> policy engine
  -> audit sink
  -> response
```

The core invariant is:

```text
Detectors produce evidence. Policy makes decisions. Audit records both.
```

Detectors return `DetectorResult` values. They do not call each other and do not
emit final enforcement decisions. The policy layer is the only layer that emits
`PolicyDecision`.

See [docs/aegis-runtime-spine.md](docs/aegis-runtime-spine.md) for the runtime
contract details.

## Quickstart

Install the project and development dependencies with `uv`:

```bash
uv sync --extra dev
```

Run the full local quality gate:

```bash
make quality
```

Run the built-in demo scenarios:

```bash
uv run python scripts/run_demo.py
```

Generate controlled trace-collection assignments for human operators:

```bash
uv run aegis-trace-assignments \
  --participant alice \
  --participant bob \
  --output data/trace_collection/assignments.jsonl
```

Build normalized records from completed collection inputs:

```bash
uv run aegis-trace-seed-inputs \
  --assignments data/trace_collection/assignments.jsonl \
  --variants-per-label 20 \
  --output data/trace_collection/collection_inputs.generated.jsonl

uv run aegis-trace-build-records \
  --assignments data/trace_collection/assignments.jsonl \
  --inputs data/trace_collection/collection_inputs.generated.jsonl \
  --output data/trace_collection/records.generated.jsonl \
  --model-provider mock \
  --model-id mock-model \
  --capability-mode offline_eval
```

The trace-collection harness emits proxy-shaped `NormalizedTurn` records with
DP-HONEY canaries, labels, families, `SensitiveSpan` metadata, and pending CIFT
tokenization markers. It is for controlled fake-secret data collection, not for
recording production credentials. See
[docs/trace-collection-harness.md](docs/trace-collection-harness.md) for the
input schema and workflow.

Exercise the mock proxy in Python:

```python
from aegis.proxy.mock_app import create_default_proxy

proxy = create_default_proxy()
status, payload = proxy.handle(
    method="POST",
    path="/v1/chat/completions",
    body={
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"trace_id": "trace-1", "session_id": "session-1"},
    },
)

print(status)
print(payload["choices"][0]["message"]["content"])
print(payload["aegis"]["policy_decision"])
```

## Quality Gates

The repository treats the runtime spine as an enforced contract. Pull requests
to `main` must pass CI on Python 3.11 and Python 3.12.

The local gate is:

```bash
make quality
```

It runs:

```bash
uv run --extra dev ruff check src/aegis src/detect tests/aegis tests/dp_honey scripts
uv run --extra dev ruff format --check src/aegis src/detect tests/aegis tests/dp_honey scripts
uv run --extra dev mypy src/aegis src/detect scripts
uv run python scripts/check_import_boundaries.py
uv run python scripts/check_artifact_boundaries.py
uv run --extra dev pytest
```

Coverage is enforced at 90 percent for the runtime package.

## Repository Layout

```text
src/aegis/core/        Runtime contracts and orchestrator
src/aegis/canaries/    Honeytoken registration and injection helpers
src/aegis/demo/        Built-in runtime demo scenarios
src/aegis/detectors/   Detector stage implementations
src/aegis/policy/      Policy decision logic
src/aegis/audit/       Audit sinks
src/aegis/providers/   Model provider adapters
src/aegis/proxy/       Proxy adapters and mock proxy surface
src/aegis/replay/      Offline replay harnesses for fixtures and demos
src/aegis/sdk/         SDK entrypoint for embedding the runtime
tests/aegis/           Runtime spine tests
scripts/               Repository quality and architecture checks
introspection/         Research notebooks, activation experiments, and CIFT reports
docs/                  Project and setup documentation
```

## Contribution Rules

New runtime work must preserve the spine boundaries:

- Adapters create `NormalizedTurn`; detectors consume normalized data.
- Detectors return `DetectorResult`; they do not emit `PolicyDecision`.
- Policy is the only layer that emits `PolicyDecision`.
- Audit records normalized input summary, detector outputs, policy decision,
  and latency.
- CIFT must emit either activation risk or explicit unavailable evidence.
- Runtime CIFT detectors consume promoted JSON artifacts and feature-vector
  metadata. Feature-vector annotators may attach derived activation features
  before detectors run, but training pickles and research modules remain in
  `introspection/`.
- Selected-choice CIFT metadata is the primary runtime route. Broader
  payload/query readout is degraded fallback coverage and must be labeled as
  such in detector evidence.
- DP-HONEY injection/registration and canary detection remain separate stages.
- NIMBUS-style detectors emit cumulative session risk as ordinary
  `DetectorResult` values; policy still owns the final action.
- Real credentials cross runtime boundaries as handles, spans, hashes, or
  evidence, not raw production secrets.
- Generated trace data, local worktrees, OCR assets, pycache files, and raw
  model or activation artifacts should not be committed into runtime paths.
- Parallel branches should name the lane they own: runtime spine, DP-HONEY,
  CIFT, NIMBUS, proxy/SDK, audit, dashboard, or eval. Cross-lane changes need
  explicit adapter tests and should be called out in the pull request.
- Research artifacts become runtime inputs only after promotion into small,
  typed, reviewable artifacts or fixtures. Runtime code should not import
  `aegis_introspection` or load training pickles directly.

See [CONTRIBUTING.md](CONTRIBUTING.md) before adding detector, policy, proxy, or
adapter code.

## Related Docs

- [Runtime spine](docs/aegis-runtime-spine.md)
- [Trace collection harness](docs/trace-collection-harness.md)
- [Contributing](CONTRIBUTING.md)
- [Asana MCP setup](docs/ASANA_MCP_SETUP.md)
