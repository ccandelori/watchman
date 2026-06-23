# Aegis Runtime Spine

The runtime spine is the shared boundary between proxy, SDK, detectors, policy,
audit, dashboard, and evaluation work. The initial implementation is deliberately
small: it proves the typed pipeline with a mock provider and leaves detector
substance to follow-up branches.

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

```text
chat request
  -> NormalizedTurn
  -> turn annotators
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

`CiftFeatureVectorAnnotator` is the first sanctioned self-hosted feature hook.
It runs before pre-generation detectors, calls a caller-supplied extractor, and
attaches the returned feature vector under:

```text
NormalizedTurn.metadata["cift"]["feature_vectors"][feature_key]
```

The annotator does not call the extractor in black-box or SDK-only modes. If a
self-hosted extractor cannot provide a vector, the turn remains unchanged and
`CiftRuntimeDetector` reports degraded CIFT capability. This keeps live
activation capture as a connector concern while preserving the runtime spine
contract.

`CiftRuntimeWindowSelector` is the runtime route for the current selected-choice
CIFT candidate. When `metadata.cift.selected_choice_readout_token_indices` is
present, the selector scores the selected-choice feature family and records
`cift_window_coverage=primary`. When selected-choice geometry is absent, it
scores the broader payload/query readout model as degraded fallback evidence,
caps confidence, and records `cift_window_coverage=degraded_fallback`.

`build_cift_window_selector_runtime_components` is the spine-native assembly
helper for optional CIFT deployment. It loads selected-choice and fallback JSON
artifacts, creates the required feature-vector annotators, and returns
pre-generation detector components that plug directly into `AegisRuntime`.
Callers still own the self-hosted extractor implementation; the runtime only
sees feature vectors and detector results.

## NIMBUS-Lite Session Detector

`NimbusLeakageDetector` is the first runtime session detector for cumulative
leakage. It reuses the exact and encoded canary detectors as per-turn signals,
then updates a per-session leakage score:

```text
new_score = min(1.0, previous_score * decay + turn_signal_score)
```

The detector emits component `nimbus`, active/degraded capability status, a
recommended action, and audit-safe evidence containing prior score, turn signal,
current score, thresholds, and signal summaries. It does not replace text or
encoded canary scanners; it composes with them as the cumulative leakage-budget
stage.

## Follow-Up Integration

Future branches should add real detectors behind the existing contract:

- CIFT provider implementation: implement the caller-supplied extractor that
  converts self-hosted activation capture into the feature vectors consumed by
  `CiftFeatureVectorAnnotator`.
- DP-HONEY runtime: register honeytokens and populate `sensitive_spans`.
- Canary scanners: extend exact model-output scanning to tool arguments and
  streaming outputs.
- Paper-faithful NIMBUS: replace or augment NIMBUS-lite with the paper's
  learned multi-turn leakage critic and calibration.
- Tool scanner: inspect normalized tool arguments before dispatch.
