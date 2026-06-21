# Aegis Runtime Spine

The runtime spine is the shared boundary between proxy, SDK, detectors, policy,
audit, dashboard, and evaluation work. The initial implementation is deliberately
small: it proves the typed pipeline with a mock provider and leaves detector
substance to follow-up branches.

## Boundary Contracts

All adapters normalize external inputs into `NormalizedTurn`. Detectors consume
that normalized shape and return `DetectorResult` values. Detectors do not call
each other and do not make final enforcement decisions.

The policy layer is the only layer that emits `PolicyDecision`. The audit layer
records `AuditEvent` objects containing the normalized turn, detector results,
policy decision, and runtime metadata.

## Initial Pipeline

```text
chat request
  -> NormalizedTurn
  -> ActivationUnavailableDetector
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

## Follow-Up Integration

Future branches should add real detectors behind the existing contract:

- CIFT provider hook: attach `metadata["cift"]["feature_vectors"]` from
  self-hosted activation capture before `CiftRuntimeDetector` runs.
- DP-HONEY runtime: register honeytokens and populate `sensitive_spans`.
- Canary scanners: extend exact model-output scanning to tool arguments and
  streaming outputs.
- NIMBUS ledger: emit cumulative session risk as a detector result.
- Tool scanner: inspect normalized tool arguments before dispatch.
