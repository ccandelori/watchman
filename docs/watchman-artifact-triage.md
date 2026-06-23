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

## Runtime-Candidate Work

These files look like runtime or test work, but they are not promoted by this
triage pass:

- `src/aegis/audit/leakage_trace.py`
- `src/aegis/core/leakage.py`
- `tests/aegis/test_leakage.py`
- `tests/aegis/test_leakage_adapter.py`
- `tests/aegis/test_nimbus_critic_contract.py`
- `tests/aegis/test_nimbus_session_destruction.py`

Promotion rule: runtime-candidate files need a focused follow-up PR with
contract tests, import-boundary checks, and audit-safety review. They should
not be swept into roadmap or CIFT branches by accident.

## Current Decision

Only the roadmap plan, boundary docs, and quality-gate code are eligible for
this branch. Research data, local corpora, and runtime-candidate experiments
remain outside the staged change set until they receive their own plan or PR.
