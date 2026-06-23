# Watchman Artifact Triage

This document records local artifacts observed while establishing Watchman as
the active project repo. It is a guardrail against accidental staging. A file
listed here is not automatically approved for promotion.

## Generated Or Local-Only Artifacts

These paths should remain untracked and are covered by `.gitignore`:

- `.worktrees/` — local branch worktrees and their virtual environments.
- `introspection/data/trace_collection/` — generated trace or structured prompt
  slices from local CIFT data experiments.
- `eng.traineddata` — local OCR language data used by document tooling.

## Research-History Candidates

These files may be useful project history, but they need deliberate review
before being added:

- `Research/2304.14997v4.pdf` — untracked supporting research PDF.
- `introspection/AUDIT_CIFT_PAPER_ALIGNMENT_2026-06-21.md` — untracked CIFT
  paper-alignment audit.

Promotion rule: research-history files should stay under `Research/`,
`introspection/`, or `docs/` and should not introduce runtime imports or CI
dependencies.

## Promoted Runtime Contract Work

These files are eligible for this branch because they add runtime-safe leakage
trace primitives and contract tests:

- `src/aegis/audit/leakage_trace.py`
- `src/aegis/core/leakage.py`
- `tests/aegis/test_leakage.py`
- `tests/aegis/test_leakage_adapter.py`
- `tests/aegis/test_nimbus_critic_contract.py`
- `tests/aegis/test_nimbus_session_destruction.py`

Promotion rule: runtime files must be staged intentionally with their contract
tests. They should not be swept into unrelated CIFT or documentation-only
branches by accident.

## Generated Experiment Outputs

The current selected-choice CIFT evaluation generated local JSONL, `.pt`, and
model bundle artifacts under ignored paths. Only compact reports under
`introspection/data/reports/` are eligible for review.

## Current Decision

The roadmap plan, boundary docs, leakage/NIMBUS contract files, focused CIFT
metadata tests, and compact CIFT reports are eligible for this branch. Research
PDFs, local corpora, and raw activation or model artifacts remain outside the
staged change set until they receive their own plan or PR.
