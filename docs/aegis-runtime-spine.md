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

## Follow-Up Integration

Future branches should add real detectors behind the existing contract:

- CIFT adapter: load a promoted probe/artifact and emit activation-risk or
  capability-unavailable results.
- DP-HONEY runtime: register honeytokens and populate `sensitive_spans`.
- Canary scanners: extend exact model-output scanning to tool arguments and
  streaming outputs.
- NIMBUS ledger: emit cumulative session risk as a detector result.
- Tool scanner: inspect normalized tool arguments before dispatch.
