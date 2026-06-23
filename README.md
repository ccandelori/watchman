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
- A runtime-native `NimbusDetector` with a canary-aware critic that estimates
  exact, encoded, and partial canary leakage bits into a per-session budget.
- A development OpenAI-compatible proxy surface for `/health`,
  `/v1/chat/completions`, and `/audit/recent`, with a mock provider by
  default and an OpenAI-compatible HTTP provider adapter behind env config.
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
  -> provider egress guard
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

Run the development HTTP proxy:

```bash
uv run aegis-proxy --host 127.0.0.1 --port 8000
```

The development proxy exposes the current OpenAI-compatible mock surface:

```text
GET  /health
GET  /aegis/capabilities
POST /v1/chat/completions
GET  /audit/recent
POST /test/reset
```

`GET /aegis/capabilities` returns the machine-readable development contract:
provider kind, whether mock controls are enabled, supported mock response modes,
route list, and schema versions. Redteam tooling should call this route before
assuming that `mock_response_mode` is accepted.

By default, `aegis-proxy` runs with the deterministic mock provider. To point
the development proxy at an OpenAI-compatible model endpoint, configure the
provider explicitly:

```bash
AEGIS_PROVIDER=openai_compatible \
AEGIS_OPENAI_BASE_URL=https://api.openai.com \
AEGIS_OPENAI_API_KEY="$OPENAI_API_KEY" \
AEGIS_OPENAI_MODEL=gpt-4.1-mini \
uv run aegis-proxy --host 127.0.0.1 --port 8000
```

`AEGIS_OPENAI_BASE_URL` may be the service root, `/v1`, or the full
`/v1/chat/completions` endpoint. Mock controls such as `mock_response` and
`mock_response_mode` are accepted only when `AEGIS_PROVIDER=mock`; the proxy
rejects them for real providers so redteam-only controls cannot cross the
provider boundary.

Proxy-owned request errors use a stable envelope:

```json
{
  "error": {
    "schema_version": "aegis.proxy_error/v1",
    "code": "invalid_request",
    "message": "...",
    "details": {}
  }
}
```

Provider transport failures use the same envelope with status `502` and
`code=provider_error`. Aegis policy blocks still return HTTP `200`; the
enforcement decision lives in `aegis.policy_decision.final_action`.

Run the proxy smoke test from another terminal after the development proxy is
listening:

```bash
uv run aegis-proxy-smoke --url http://127.0.0.1:8000 --timeout 5
```

The smoke command checks health, a benign chat turn, an encoded DP-HONEY leak
turn, and audit readback. It exits nonzero if the gateway contract is broken.

Send one local chat request:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mock-model",
    "messages": [{"role": "user", "content": "hello"}],
    "metadata": {"trace_id": "trace-local-1", "session_id": "session-local-1"}
  }'
```

Redteam scenarios should target the HTTP surface first. For pre-server tests,
use the mock proxy only inside Aegis unit tests. The separate redteam repo
should treat Aegis as a black-box HTTP target and inspect the assistant response
plus the `aegis` detector/policy/trace block.

The development proxy supports deterministic mock response controls through
request metadata:

```json
{
  "metadata": {
    "mock_response_mode": "leak_first_honeytoken"
  }
}
```

Supported modes are:

- `default` - return the default benign mock response.
- `echo_last_user` - return the last user message.
- `leak_first_honeytoken` - return the first fake honeytoken visible in the prompt.
- `base64_first_honeytoken` - return the first fake honeytoken as base64.
- `partial_first_honeytoken` - return a prefix of the first fake honeytoken.

Leak modes require a planted honeytoken in the current request. Include a
credential placeholder in the scenario turn when using these modes.

To exercise DP-HONEY/canary detection through the proxy, include credential
placeholders in chat messages:

```text
{{CREDENTIAL:repo_pat:github_pat}}
```

The development proxy replaces placeholders with fake honeytokens, attaches
`SensitiveSpan` metadata, runs canary detectors when canaries exist, feeds the
same planted-canary registry to the NIMBUS critic, and returns the detector
outputs in the `aegis` block.

Use `POST /test/reset` between redteam runs. An empty JSON object clears all
development proxy audit and session state:

```bash
curl -s http://127.0.0.1:8000/test/reset \
  -H 'Content-Type: application/json' \
  -d '{}'
```

To clear only one NIMBUS session and its audit events, pass a session id:

```bash
curl -s http://127.0.0.1:8000/test/reset \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "session-local-1"}'
```

This route is a local development/testing affordance, not a production API.

Every chat response includes an additive `aegis.runtime_trace` object with
schema version `aegis.runtime_trace/v1`. The trace summarizes the ordered stages:
normalize, DP-HONEY, CIFT, provider egress guard, provider, canary, NIMBUS,
policy, and audit. It contains detector names, statuses, provider/model
identifiers, counts, and actions only; it does not include raw secrets, canary
values, feature vectors, or model output.

Fetch recent audit events:

```bash
curl -s http://127.0.0.1:8000/audit/recent
```

Filter recent audit events by session and limit:

```bash
curl -s 'http://127.0.0.1:8000/audit/recent?session_id=session-local-1&limit=5'
```

Summarize NIMBUS behavior from external redteam JSONL results:

```bash
uv run aegis-nimbus-report --input ../watchman-redteam/results/aegis-local.jsonl
```

Generate a local in-process NIMBUS fixture JSONL when the external redteam
runner is not available:

```bash
uv run aegis-nimbus-fixtures --output /tmp/aegis-nimbus-fixtures.jsonl
uv run aegis-nimbus-report --input /tmp/aegis-nimbus-fixtures.jsonl
```

The report reads detector and policy metadata only. It distinguishes immediate
public canary detectors from NIMBUS critic evidence, so partial leakage can be
understood even when `encoded_canary` does not trigger on a single turn. See
[docs/nimbus-redteam-eval.md](docs/nimbus-redteam-eval.md) for the full
black-box redteam evaluation loop.

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
- Tool-call sensitive spans should carry `metadata.tool_call_name` and
  `metadata.argument_path` using the `arguments.payload.items[1]` path dialect.
  The provider egress guard may report those handles as evidence, but never raw
  argument values.
- NIMBUS-style detectors emit cumulative session risk as ordinary
  `DetectorResult` values; policy still owns the final action.
- The canonical NIMBUS runtime path is `NimbusDetector` plus a `NimbusCritic`.
  `CanaryNimbusCritic` is the deterministic local critic for proxy and redteam
  positive controls; paper-faithful critics should satisfy the same interface.
- Real credentials cross runtime boundaries as handles, spans, hashes, or
  evidence, not raw production secrets.
- Generated trace data, local worktrees, OCR assets, pycache files, and raw
  model or activation artifacts should not be committed into runtime paths.

See [CONTRIBUTING.md](CONTRIBUTING.md) before adding detector, policy, proxy, or
adapter code.

## Related Docs

- [Runtime spine](docs/aegis-runtime-spine.md)
- [Trace collection harness](docs/trace-collection-harness.md)
- [Contributing](CONTRIBUTING.md)
- [Asana MCP setup](docs/ASANA_MCP_SETUP.md)
