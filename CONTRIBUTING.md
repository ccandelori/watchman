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
uv run --extra dev ruff check src/aegis tests/aegis scripts
uv run --extra dev ruff format --check src/aegis tests/aegis scripts
uv run --extra dev mypy src/aegis scripts
uv run python scripts/check_import_boundaries.py
uv run --extra dev pytest
```

Pull requests must pass the same commands in CI before merge.
The root CI gate currently covers the runtime spine: `src/aegis`,
`tests/aegis`, and root gate scripts. Research work under `introspection/`
remains separate until it is promoted through an adapter into the runtime.

## Runtime Contract

Detectors produce evidence. Policy makes decisions. Audit records both.

New runtime work must preserve these rules:

- Adapters create `NormalizedTurn`; detectors consume normalized data.
- Detectors return `DetectorResult`; they do not emit `PolicyDecision`.
- Policy is the only layer that emits `PolicyDecision`.
- Audit records normalized input summary, detector outputs, policy decision, and latency.
- CIFT must emit either activation risk or explicit unavailable evidence.
- DP-HONEY injection/registration and canary detection remain separate stages.
- Real credentials cross runtime seams as handles, spans, hashes, or evidence, not raw production secrets.

## Promotion From Research To Runtime

Research code, including `introspection/`, should not be imported directly by
runtime packages. Promote research work through a narrow adapter that satisfies
the runtime contracts and includes tests for unsupported capability modes.
