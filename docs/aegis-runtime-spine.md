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

## Follow-Up Integration

Future branches should add real detectors behind the existing contract:

- CIFT adapter: load a promoted probe/artifact and emit activation-risk or
  capability-unavailable results.
- DP-HONEY runtime: register honeytokens and populate `sensitive_spans`.
- Canary scanners: inspect model output and tool arguments for registered
  canaries.
- NIMBUS ledger: emit cumulative session risk as a detector result.
- Tool scanner: inspect normalized tool arguments before dispatch.
