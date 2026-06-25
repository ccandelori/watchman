# Aegis Runtime Spine

The runtime spine is the shared boundary between proxy, SDK, detectors, policy,
audit, dashboard, and evaluation work. The initial implementation is deliberately
small: it proves the typed pipeline with a configurable provider boundary and
leaves detector substance to follow-up branches.

## Boundary Contracts

All adapters normalize external inputs into `NormalizedTurn`. Detectors consume
that normalized shape and return `DetectorResult` values. Detectors do not call
each other and do not make final enforcement decisions.

Turn annotators may attach derived metadata to a normalized turn before detector
stages run. They are the sanctioned place for self-hosted model hooks to add
activation-derived features, tool normalization metadata, or other computed
runtime context. Annotators return a new `NormalizedTurn`; they do not emit
policy decisions or audit events.

The policy layer is the only layer that emits `PolicyDecision`. The audit layer
records `AuditEvent` objects containing the normalized turn, detector results,
policy decision, and runtime metadata.

## Initial Pipeline

The default development pipeline uses the deterministic mock provider:

```text
chat request
  -> NormalizedTurn
  -> turn annotators
  -> ActivationUnavailableDetector
  -> ProviderEgressGuardDetector
  -> MockModelProvider
  -> NoopCanaryDetector
  -> SeverityPolicyEngine
  -> InMemoryAuditSink
  -> OpenAI-compatible mock response
```

`ActivationUnavailableDetector` is the first CIFT boundary. It records that
activation monitoring is unavailable in black-box/mock mode instead of silently
omitting the signal.

`NoopCanaryDetector` is the first DP-HONEY boundary. It records that no canary
registry is configured yet and keeps future canary detection separate from
honeytoken injection.

`ProviderEgressGuardDetector` is the first pre-provider safety invariant. It
runs before the model provider and blocks generation when a non-honeytoken
`SensitiveSpan` is still present in the outbound turn. Evidence contains only
span kind, source, identifier, optional hash, message role, tool-call name,
argument path, and counts. It does not copy raw sensitive values into detector
evidence. DP-HONEY honeytokens are allowed to cross the model boundary because
they are deliberate decoys used to detect downstream leakage; raw production
credentials are not.

Tool-call egress manifests use `SensitiveSpan.metadata.tool_call_name` and
`SensitiveSpan.metadata.argument_path`. The argument path dialect starts at the
tool-call argument object and supports nested dictionaries and list indexes, for
example:

```text
arguments.token
arguments.payload.items[1]
arguments.payload.items[1].token
```

This lets the guard prove which tool argument carried a blocked sensitive value
without copying the argument value into evidence or audit output.

If any pre-generation detector recommends `block` or stronger, `AegisRuntime`
skips provider generation, applies policy to the pre-generation evidence, and
still writes an audit event. This makes pre-output detectors enforceable instead
of merely advisory.

The development HTTP proxy includes a narrow raw-credential scanner for common
credential prefixes (`ghp_`, `sk_live_`, `sk-`, `AKIA`, `ya29.`, `hny_`). Values
that were minted by DP-HONEY are recognized by hash and remain allowed
honeytokens. Other credential-shaped values are marked as non-honeytoken
`SensitiveSpan`s so the provider egress guard blocks before generation.
Serialized audit output redacts non-honeytoken sensitive spans from message
content. Individual spans may also request audit redaction with
`metadata.audit_redact=true`; the mock seed-canary route uses that flag for
internal test-seed messages.

User-supplied metadata is also validated at ingress. Credential-shaped strings
in metadata, including development controls such as `mock_response`, are
rejected before runtime construction because metadata is both provider-adjacent
and audit-visible.

Client metadata may not use Aegis-owned prefixes such as `aegis_`, `cift_`,
`dp_honey_`, or `nimbus_`. Those keys are reserved for runtime-derived state.
Rejecting them at ingress prevents a black-box client from spoofing trace or
detector state such as `dp_honey_canary_count`.

## Provider Configuration

`aegis.proxy.config.provider_config_from_env` is the proxy-owned provider
boundary. It currently supports:

- `AEGIS_PROVIDER=mock` for the deterministic local provider. This is the
  default and is the only mode that accepts development mock controls in request
  metadata.
- `AEGIS_PROVIDER=openai_compatible` for an HTTP chat-completions provider. It
  requires `AEGIS_OPENAI_BASE_URL` and `AEGIS_OPENAI_API_KEY`; it may also use
  `AEGIS_OPENAI_MODEL` and `AEGIS_OPENAI_TIMEOUT_SECONDS`.

The OpenAI-compatible adapter posts normalized messages to
`/v1/chat/completions` and returns only provider metadata that is safe for the
runtime trace: provider kind, redacted base URL, and model ID. It rejects
mock-only request metadata before building the runtime request so redteam
controls cannot accidentally reach a real model provider.

Run the no-network provider preflight before attempting real-provider smoke:

```bash
AEGIS_PROVIDER=openai_compatible \
AEGIS_OPENAI_BASE_URL=https://api.openai.com \
AEGIS_OPENAI_API_KEY="$OPENAI_API_KEY" \
AEGIS_OPENAI_MODEL=gpt-4.1-mini \
uv run aegis-provider-preflight \
  --require-real-provider \
  --output introspection/data/reports/aegis_real_provider_preflight_v1.json
```

## HTTP Redteam Contract V1

The development proxy exposes a small black-box contract for external redteam
tools:

```text
GET  /health
GET  /aegis/capabilities
POST /v1/chat/completions
GET  /audit/recent
POST /test/reset
POST /test/seed-canary   # mock provider only
```

`/aegis/capabilities` returns schema version
`aegis.proxy_capabilities/v1`. It reports the provider name, whether mock
controls are enabled, supported mock response modes, route names, detector
names, NIMBUS calibration, and the current response/error schema versions.
Redteam tools should use this route to decide whether mock-only probes and test
controls are available.

The `nimbus` capabilities object reports the active cumulative budget profile:

```json
{
  "critic_version": "canary-v0",
  "budget_bits": 1.0,
  "max_turns": 20,
  "thresholds": {"warn": 0.3, "sanitize": 0.6, "block": 0.9},
  "critic": {
    "exact_match_leakage_bits": 1.0,
    "encoded_match_leakage_bits": 1.0,
    "partial_match_leakage_bits": 0.8,
    "partial_match_threshold": 0.4,
    "confidence": 0.8
  }
}
```

The default policy intentionally accumulates partial fragments over turns before
blocking. Stricter local runs can lower `AEGIS_NIMBUS_BLOCK_THRESHOLD` and
related action thresholds without changing detector code.

`aegis-proxy-smoke` validates this distinction. The default smoke profile
expects a single partial seeded-canary leak to stay below block. The strict
profile expects the same probe to block. Against a self-hosted CIFT gateway,
the same smoke command can also require the CIFT exfiltration-intent turn to
block before provider generation:

```bash
uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8000 \
  --timeout 5 \
  --nimbus-profile strict-partial-block \
  --output introspection/data/reports/aegis_strict_nimbus_smoke_v1.json

uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8000 \
  --timeout 120 \
  --require-cift-pre-generation-block \
  --output introspection/data/reports/aegis_self_hosted_cift_smoke_v1.json
```

Proxy-owned request and provider errors use schema version
`aegis.proxy_error/v1`:

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

Transport and validation errors use non-2xx HTTP status codes. Aegis policy
decisions, including `block` and `escalate`, remain successful chat responses
with HTTP `200`; clients must read `aegis.policy_decision.final_action`. Policy
actions `sanitize`, `block`, and `escalate` gate assistant delivery: the proxy
returns sanitized output when available or the fixed `[aegis output withheld]`
message instead of raw model output.

`GET /audit/recent` accepts optional `session_id` and `limit` query parameters.
It returns newest-first events and schema version `aegis.audit_recent/v1`.

`POST /test/reset` is development-only. An empty JSON object clears all
in-memory audit and NIMBUS/canary session state. A body with `session_id`
clears only that NIMBUS/canary session and that session's audit events. The
response uses schema version `aegis.test_reset/v1`.

`POST /test/seed-canary` is mock-provider-only. It creates one server-generated,
session-scoped canary for external redteam setup:

```json
{
  "session_id": "session-redteam-1",
  "slot_name": "repo_pat",
  "credential_type": "github_pat",
  "turn_index": 0
}
```

The response uses schema version `aegis.test_seed_canary/v1` and returns a safe
summary containing `canary_id`, `slot_name`, `credential_type`, `sha256`, and
`source`; it does not return the generated canary value. The route is
idempotent for the same `session_id` and `slot_name` and rejects a repeated slot
with a different credential type. `turn_index` is optional and defaults to `0`.
Seeded test canaries are merged into matching mock chat turns as internal
honeytoken spans so the existing mock leak modes, canary detectors, and NIMBUS
critic can run without requiring the current user turn to contain a placeholder.
These seeds are redteam fixtures, not DP-HONEY injection: seeded-only turns do
not set `dp_honey_canary_count`, and
`/test/reset` clears the seed state.

Protected workflows fail closed when no deterministic credential path can be
resolved. The versioned proxy error details report
`credential_slot_status=ambiguous_protected_workflow`, `fail_closed=true`, and
zero credential/honeytoken counts. Resolved credential paths continue to report
the normal runtime statuses: `no_credential_path`, `credential_needed`,
`honeytoken_substituted`, or `real_secret_present`.

## DP-HONEY-Lite Honeytoken Registration

`HoneytokenLedger` is the first concrete DP-HONEY registration boundary. It
replaces credential placeholders such as `{{CREDENTIAL:api_key:openai_key}}`
with model-visible honeytokens, registers matching `CanaryRecord` values for
post-generation detectors, and emits `SensitiveSpan` metadata for the normalized
turn.

The ledger may also scrub registered real secret values from model-visible
messages. In both placeholder and scrub flows, runtime boundaries carry canary
IDs, hashes, spans, credential types, and non-secret metadata. Raw production
secret values remain out of detector evidence and audit events.

## DP-HONEY-Lite Text Canary Detection

`TextCanaryDetector` is the first concrete post-generation canary detector. It
scans model output for exact matches against canary values held in an in-memory
registry. A match emits a `DetectorResult` with component `text_canary`,
recommended action `escalate`, and audit-safe evidence:

- `canary_id`
- `credential_type`
- `sha256`
- source
- output character span
- non-secret metadata

The raw canary value remains in the registry and is not copied into detector
evidence or audit events. This keeps DP-HONEY injection/registration separate
from post-output canary detection while giving policy a concrete leakage signal.

`EncodedCanaryDetector` extends the post-generation canary path to encoded or
fragmented leaks. It scans for registered canary values after base64, hex,
ROT13, leet, and reverse transforms, attempts to decode larger base64/hex blobs,
and can emit a `sanitize` recommendation for configured partial-overlap matches.
Exact encoded matches emit `escalate`. Evidence includes canary IDs, hashes,
encoding names, output spans when available, and overlap ratios, but not raw
canary values.

## Candidate CIFT Monitor V0

`cift_selector_probe_v0` is the first promoted lab-to-runtime CIFT checkpoint.
It is a fixture-backed candidate monitor, not a production probe loader. The
runtime consumes calibrated selector-window scores and applies the runtime
profile locally:

```text
score < 0.25        -> allow
0.25 <= score < 0.5 -> warn, review band
score >= 0.5        -> warn, balanced band
```

The profile records:

- score semantics: `inner_cv_platt_calibrated_probability`
- feature key: `readout_window_layer_15`
- task: `safe_secret_vs_exfiltration`
- positive label: `exfiltration_intent`
- required capability: `self_hosted_introspection`

The replay harness under `aegis.replay` exists for integration tests and demos.
It loads small `NormalizedTurn` and calibrated CIFT score fixtures, runs them
through the normal `AegisRuntime`, applies policy, and writes audit events.
It does not import `aegis_introspection`; research code crosses into runtime
only through versioned fixtures or future promoted artifacts.

## Runtime CIFT Model Adapter

`CiftRuntimeDetector` is the first bundle-backed CIFT runtime adapter. It does
not load research pickles and does not import `aegis_introspection`. Instead,
the introspection side exports a promoted JSON artifact containing scaler
parameters, logistic-regression coefficients, class ordering, decision
thresholds, and metadata. The runtime loads that JSON artifact and scores a
feature vector supplied on the normalized turn:

```text
NormalizedTurn.metadata["cift"]["feature_vectors"][feature_key]
```

When the runtime mode is black-box or SDK-only, the detector emits
`capability_status=unavailable` with audit-safe evidence. When the mode is
self-hosted but no feature vector has been attached, it emits
`capability_status=degraded`. When the feature vector is present, it emits an
active CIFT `DetectorResult` with the model score, predicted label, threshold,
feature key, and artifact IDs. It never copies the feature vector into audit
evidence.

`CiftFeatureVectorAnnotator` is the sanctioned self-hosted feature hook. It runs
before pre-generation detectors, calls a trusted extractor, and attaches the
returned feature vector under:

```text
NormalizedTurn.metadata["cift"]["feature_vectors"][feature_key]
```

When configured for selected-choice CIFT, the extractor may also provide trusted
`selected_choice_readout_token_indices`; the annotator writes those under
proxy-owned `metadata.cift`. Client-supplied request metadata remains untrusted
and may not include CIFT-reserved keys. The default proxy can register an HTTP
sidecar from `AEGIS_CIFT_EXTRACTOR_BASE_URL`; the sidecar endpoint is
`POST /v1/cift/features` and implements the
`aegis.cift_feature_extract_request/v1` and
`aegis.cift_feature_extract_response/v1` schemas. Successful feature responses
must include `model_attestation` with
`schema_version=aegis.cift_model_attestation/v1`; the gateway validates the
attested model id, revision, selected device, hidden size, layer count,
tokenizer fingerprint, special-token map hash, chat-template hash, prompt
renderer, selected-choice geometry method, and selected-choice readout token
count before accepting the returned activations. The gateway's
`AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT` must match the sidecar's
`--selected-choice-readout-token-count`, and the attested tokenizer/template
identity must match the promoted runtime artifact.
Startup also validates the runtime model, certification manifest, workflow run,
all release-required manifest artifacts, gateway smoke identity, and
evidence-chain identity under `AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT` before
enabling self-hosted CIFT. The configured detector name must match the certified
gateway smoke report.

The runnable live sidecar is
`introspection/scripts/run_cift_extractor_sidecar.py`. It loads a Transformers
causal LM, serves `/v1/cift/features`, and uses the same
`LiveCiftFeatureSetExtractor` path as the live benchmark. The sidecar renders
structured proxy turns into the calibration bridge's prompt format, then derives
selected-choice readout token indices for the semantic-indirection prompt family
with the loaded tokenizer offsets and configured readout count. It does not
infer arbitrary selected-choice spans from untrusted client metadata, and it
scrubs inbound `metadata.cift` before extracting features. If the
indices cannot be derived for a selected-choice feature key, the sidecar returns
a successful response with `feature_vector=null` and the gateway fails closed
under the production `BLOCK` activation-failure policy. If the sidecar attests a
model or device that does not match the certification binding, the gateway also
fails closed before provider generation.

Before live Qwen3-4B sidecar, activation extraction, or benchmark runs, execute
`introspection/scripts/check_cift_device_preflight.py --device mps` with the
same Python environment that will load the model. The preflight must report
`eligible=true`, `selected_device=mps`, and a smoke tensor on `mps:0`. The
runtime artifact's `source_selected_device` must also match the required
production device. Explicit MPS requests in the model loader fail closed when
the Torch environment cannot allocate an MPS tensor, so CPU fallback cannot
become MPS-certified evidence.

The annotator does not call the extractor in black-box or SDK-only modes. If a
self-hosted extractor cannot provide a vector, the turn remains unchanged and
`CiftRuntimeDetector` reports degraded CIFT capability. This keeps live
activation capture as a connector concern while preserving the runtime spine
contract.

`CiftRuntimeWindowSelector` is the runtime route for the current selected-choice
CIFT candidate. When `metadata.cift.selected_choice_readout_token_indices` is
present, the selector scores the selected-choice feature family and records
`cift_window_coverage=primary`. Under production `BLOCK` activation-failure
policy, absent or malformed selected-choice geometry fails closed before
provider generation. Degraded payload/query fallback remains available only for
non-blocking offline/research evaluation and must be labeled as fallback
coverage.

`build_cift_window_selector_runtime_components` is the spine-native assembly
helper for optional CIFT deployment. It loads selected-choice and fallback JSON
artifacts, creates the required feature-vector annotators, and returns
pre-generation detector components that plug directly into `AegisRuntime`.
Callers still own the self-hosted extractor implementation; the runtime only
sees feature vectors and detector results.

## NIMBUS Session Detectors

`NimbusDetector` is the runtime-native cumulative leakage contract. It resolves
a secret context handle from sensitive spans or metadata, asks a pluggable
critic for per-turn estimated leakage bits, stores per-session state, and emits
component `nimbus` as an ordinary `DetectorResult`. The result carries budget
fraction, cumulative leakage bits, threshold metadata, and active/degraded/
unavailable capability status without copying the secret handle into evidence.

`CanaryNimbusCritic` is the deterministic runtime critic used by the development
proxy. It keeps planted canary records in memory, scans model output for exact,
encoded, and partial leakage, and returns estimated leakage bits without writing
raw canary values into normalized turns or audit evidence.
The proxy reports this path as `deterministic_beta` with
`paper_faithful_learned_critic=false`; a learned, paper-faithful NIMBUS critic
still needs separate corpus, grouped CV, sealed holdout, live FN/FP metrics, and
promotion evidence before it can replace the deterministic critic.

`BaselineNimbusCritic` and `InMemoryNimbusStateStore` are intentionally small
implementations for tests and demos. They establish the contract that a future
paper-faithful critic can satisfy without changing the runtime spine. The
in-memory store validates non-negative, monotonic leakage state, and NIMBUS
accumulates by session even when a session contains multiple planted canary
slots.

`NimbusLeakageDetector` is legacy compatibility and demo code. It reuses the
exact and encoded canary detectors as per-turn signals, then updates a
per-session leakage score:

```text
new_score = min(1.0, previous_score * decay + turn_signal_score)
```

New runtime work should extend `NimbusDetector` through a `NimbusCritic` rather
than adding another session-detector class. The boundary stays the same:
NIMBUS emits cumulative session risk as detector evidence, and policy still
owns the final action.

`aegis-nimbus-report` summarizes NIMBUS behavior from external redteam JSONL
results without importing the redteam package. It reads detector and policy
metadata, extracts per-turn leakage bits and budget fractions, and renders
scenario-level action progressions. See `docs/nimbus-redteam-eval.md` for the
runtime/redteam evaluation loop and the distinction between public canary
detectors and NIMBUS critic evidence.

`aegis-nimbus-eval` adds a required label manifest and emits session-level
`false_negative_rate` and `false_positive_rate` for the deterministic beta
critic. The report declares `paper_faithful_learned_critic=false`; it is not a
learned-NIMBUS promotion artifact.

`aegis-nimbus-training-corpus` generates the bootstrap
`nimbus-training-turn/v0` corpus contract for the future learned critic. Each
record has session state, current output, true secret context, 16 negative
contexts, explicit leakage label, target leakage bits, and `split_group_key`
for grouped CV. The companion manifest declares
`not_promotable_training_contract_only`; it is data-contract evidence, not
runtime or promotion evidence.

`aegis-nimbus-train-infonce` and `aegis-nimbus-eval-infonce` train and evaluate
an offline lexical InfoNCE scaffold over `nimbus-training-turn/v0` rows. The
model artifact stores weights, schema metadata, corpus digest, label
distribution, and aggregate metrics only. It does not store raw contexts or
outputs, reports `promotion_status=not_promotable_offline_scaffold`, and keeps
`paper_faithful_learned_critic=false`.

`aegis-nimbus-fixtures` runs a small in-process mock-proxy fixture suite and
writes redteam-shaped JSONL for fast local NIMBUS regression checks. The output
is intentionally metadata-only and is not a substitute for external black-box
redteam runs.

## Runtime Trace

The development proxy returns a compact ordered trace in every chat response:

```json
{
  "schema_version": "aegis.runtime_trace/v1",
  "stages": [
    {"stage": "normalize", "status": "ok"},
    {"stage": "dp_honey", "status": "active", "canary_count": 1},
    {"stage": "cift", "status": "unavailable", "detectors": ["activation_unavailable"]},
    {"stage": "provider_egress_guard", "status": "active", "detectors": ["provider_egress_guard"]},
    {"stage": "provider", "status": "completed", "provider": "mock", "model_id": "mock-model"},
    {"stage": "canary", "status": "active", "detectors": ["text_canary", "encoded_canary"]},
    {"stage": "nimbus", "status": "active", "detectors": ["nimbus"]},
    {"stage": "policy", "status": "decided", "final_action": "allow"},
    {"stage": "audit", "status": "written"}
  ]
}
```

The trace is a summary contract for humans, redteam tooling, and dashboards. It
is intentionally additive to `DetectorResult` and `PolicyDecision`; those remain
the authoritative detector and enforcement contracts. Trace stages must not
include raw secrets, raw canaries, model output, activation vectors, or restored
credential material.

When `AEGIS_AUDIT_JSONL_PATH` is set, the proxy writes redacted audit records to
local JSONL in addition to serving recent records. `GET /audit/explain` rebuilds
a safe stage timeline for one `trace_id` or `session_id` from the audit store,
including provider skipped/completed state, detector actions, policy outcome,
latency, and model-bound CIFT hashes where present.
`aegis-audit-explain --input <audit.jsonl> --trace-id <trace>` reconstructs the
same explanation from saved JSONL after the proxy process exits, which makes the
local smoke evidence independently inspectable.

Audit records include `runtime_evidence` with schema version
`aegis.audit_runtime_evidence/v1`. This is the durable machine-readable release
evidence block for one turn: policy mode, final action, provider state,
credential-slot status, detector versions, detector latencies, whitelisted
artifact hashes, CIFT certification/runtime summary fields when present,
fail-closed events, and total latency. The same redaction boundary applies as
for the runtime trace: no raw secrets, raw canaries, model output, activation
vectors, or restored credential material.

## Gateway Smoke

`aegis-proxy-smoke` is the first runnable gateway affordance for contributors
and external redteam tooling. Against a running development proxy it checks:

- `GET /health`
- `GET /ready`
- `GET /aegis/capabilities`
- a protected credential-slot benign chat request that should allow
- an optional self-hosted CIFT exfiltration-intent request that should block before provider generation
- a raw credential-shaped tool payload that should be blocked by the provider egress guard
- an encoded DP-HONEY leak request that should block or escalate
- a metadata-declared credential-slot canary leak request that should block or escalate
- deterministic NIMBUS partial-leak accounting
- `GET /audit/recent`
- `GET /audit/explain` for the provider-egress-block trace

The command writes a JSON summary and exits nonzero on contract failure:

```bash
uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8000 \
  --timeout 5 \
  --output introspection/data/reports/aegis_default_mock_provider_smoke_ambiguous_protected_v1.json
```

Use `--provider-mode real-provider` only when the gateway is explicitly running
with `AEGIS_PROVIDER=openai_compatible`. Real-provider smoke skips mock-only leak
probes but still verifies readiness with provider identity, benign provider
completion, pre-generation egress blocking, audit readback, and trace
explanation.

For local real-provider-mode evidence without external network access, run the
loopback OpenAI-compatible provider and point the gateway at
`http://127.0.0.1:<port>/v1`:

```bash
uv run aegis-loopback-openai-provider \
  --host 127.0.0.1 \
  --port 8772 \
  --response-content "Loopback provider completed." \
  --request-log introspection/data/reports/aegis_loopback_openai_provider_request_log_v2.jsonl \
  --expected-bearer-token loopback-test-token \
  --forbidden-substring ghp_realLookingToolSecret1234567890 \
  --forbidden-substring fake-
```

That loopback path proves the gateway uses the OpenAI-compatible adapter with
mock controls disabled and can block raw sensitive egress before the provider
receives it. It does not replace credentialed external-provider smoke for
release signoff.

After collecting loopback preflight, smoke, provider request-log, and audit
JSONL artifacts, run the evidence verifier:

```bash
uv run aegis-provider-smoke-verify \
  --preflight introspection/data/reports/aegis_loopback_real_provider_preflight_v2.json \
  --smoke introspection/data/reports/aegis_loopback_real_provider_smoke_v2.json \
  --provider-request-log introspection/data/reports/aegis_loopback_openai_provider_request_log_v2.jsonl \
  --audit-jsonl introspection/data/reports/aegis_loopback_real_provider_smoke_audit_v2.jsonl \
  --forbidden-marker ghp_realLookingToolSecret1234567890 \
  --forbidden-marker fake- \
  --output introspection/data/reports/aegis_loopback_real_provider_smoke_evidence_verification_v1.json
```

The verifier emits `aegis.provider_smoke_evidence/v1` and exits nonzero when
the receipts disagree, the attack reaches the provider, mock controls appear in
real-provider mode, `/ready` evidence is missing, or forbidden markers are
present in saved artifacts.

## Follow-Up Integration

Future branches should add real detectors behind the existing contract:

- Redteam harness: keep scenario runners, scoring, and generated campaign
  artifacts in a separate repository that targets Aegis through the HTTP proxy.
  The Aegis repo owns the development target contract, deterministic mock
  controls, audit readback, and reset affordances.
- CIFT provider implementation: run the trusted extractor sidecar that converts
  self-hosted activation capture into feature vectors and selected-choice
  readout geometry consumed by `CiftFeatureVectorAnnotator`.
- DP-HONEY runtime: register honeytokens and populate `sensitive_spans`.
- Canary scanners: extend exact model-output scanning to tool arguments and
  streaming outputs.
- Paper-faithful NIMBUS: replace or augment the baseline critic with the
  paper's learned multi-turn leakage critic and calibration.
- Tool scanner: inspect normalized tool arguments before dispatch.
