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
- First detector seams:
  - `ActivationUnavailableDetector` for explicit CIFT capability reporting in
    black-box/mock mode.
  - `NoopCanaryDetector` as the unconfigured DP-HONEY/text-canary boundary.
  - `TextCanaryDetector` for exact post-generation detection of registered
    honeytoken leaks.
- A fixture-backed `cift_selector_probe_v0` candidate monitor replay path that
  consumes promoted calibrated CIFT scores without importing research code.
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
uv run --extra dev ruff check src/aegis tests/aegis scripts
uv run --extra dev ruff format --check src/aegis tests/aegis scripts
uv run --extra dev mypy src/aegis scripts
uv run python scripts/check_import_boundaries.py
uv run --extra dev pytest
```

Coverage is enforced at 90 percent for the runtime package.

## Repository Layout

```text
src/aegis/core/        Runtime contracts and orchestrator
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
- DP-HONEY injection/registration and canary detection remain separate stages.
- Real credentials cross runtime boundaries as handles, spans, hashes, or
  evidence, not raw production secrets.

See [CONTRIBUTING.md](CONTRIBUTING.md) before adding detector, policy, proxy, or
adapter code.

## Related Docs

- [Runtime spine](docs/aegis-runtime-spine.md)
- [Contributing](CONTRIBUTING.md)
- [Asana MCP setup](docs/ASANA_MCP_SETUP.md)
