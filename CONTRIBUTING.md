# Contributing

Aegis uses the runtime spine as an enforced contract, not a suggestion.
Contributors should keep research, adapters, detectors, policy, proxy, SDK, and
audit code separated unless a reviewed adapter explicitly crosses a seam.

## Required Local Gates

Run the full gate before opening a pull request:

```bash
make quality
```

The full gate runs:

```bash
uv run --extra dev ruff check src/aegis src/detect tests/aegis tests/dp_honey scripts
uv run --extra dev ruff format --check src/aegis src/detect tests/aegis tests/dp_honey scripts
uv run --extra dev mypy src/aegis src/detect scripts
uv run python scripts/check_import_boundaries.py
uv run python scripts/check_artifact_boundaries.py
uv run --extra dev pytest
```

Pull requests must pass the same commands in CI before merge.
The root CI gate covers the runtime spine, DP-HONEY package, root scripts, and
small committed fixtures. Research work under `introspection/` remains separate
until it is promoted through an adapter into the runtime.

## Runtime Contract

Detectors produce evidence. Policy makes decisions. Audit records both.

New runtime work must preserve these rules:

- Adapters create `NormalizedTurn`; detectors consume normalized data.
- Detectors return `DetectorResult`; they do not emit `PolicyDecision`.
- Policy is the only layer that emits `PolicyDecision`.
- Audit records normalized input summary, detector outputs, policy decision, and latency.
- CIFT must emit either activation risk or explicit unavailable evidence.
- Runtime CIFT consumes promoted JSON artifacts and feature-vector metadata,
  not training pickles, activation tensors, or research package imports.
- Selected-choice CIFT metadata is primary coverage. Payload/query readout is a
  degraded fallback and must be labeled that way in evidence.
- DP-HONEY injection/registration and canary detection remain separate stages.
- NIMBUS-style cumulative risk emits an ordinary `DetectorResult`; policy still
  owns final action selection.
- Real credentials cross runtime seams as handles, spans, hashes, or evidence, not raw production secrets.

## Promotion From Research To Runtime

Research code, including `introspection/`, should not be imported directly by
runtime packages. Promote research work through a narrow adapter that satisfies
the runtime contracts and includes tests for unsupported capability modes.

The normal CIFT promotion path is:

1. Record the corpus and evaluation in `introspection/data/reports/`.
2. Export a runtime-safe JSON artifact with scaler, coefficients, thresholds,
   metadata, and artifact identifiers.
3. Add or update a small runtime fixture under `tests/aegis/fixtures/` only when
   a stable integration test needs it.
4. Load the artifact through `src/aegis/detectors/cift_runtime.py` or a
   runtime-native adapter without importing `aegis_introspection`.

Generated trace data, local worktrees, OCR assets, pycache files, and raw model
or activation artifacts are not valid runtime PR contents. The artifact-boundary
gate enforces the high-risk cases for tracked files.

## Parallel Work Lanes

Parallel work is welcome only when each lane owns a clear boundary. A pull
request should identify which of these surfaces it touches:

- runtime contracts and orchestration
- DP-HONEY injection, canary registry, or canary detection
- CIFT feature extraction, artifact promotion, or runtime scoring
- NIMBUS critic, session state, or cumulative leakage scoring
- proxy, SDK, audit, dashboard, or evaluation harness

Contributor branches should avoid mixing unrelated lanes. If a branch must
cross lanes, the PR description should name the crossing explicitly and include
tests at the boundary. For example, a CIFT promotion may update an
introspection report, a runtime-safe artifact fixture, and a detector adapter
test, but it should not also change policy semantics unless the policy change
is the point of the PR.

Research outputs are not runtime dependencies. CIFT `.pt` files, training
pickles, generated trace JSONL files, and large local corpora stay untracked.
Promoted runtime state should be small, typed, reviewable, and loaded through a
runtime adapter without importing `aegis_introspection`.

Every detector lane must cover all capability modes it claims to support:

- active evidence when the required capability is present
- degraded evidence when configured but missing recoverable runtime input
- unavailable evidence when the runtime mode or missing required context cannot
  support the detector

No detector should disappear silently.
