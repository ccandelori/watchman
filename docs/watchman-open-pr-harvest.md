# Watchman Open PR Harvest

Current guidance for open PRs that have passing checks but are dirty against
`watchman/main`.

## Rule

Do not merge dirty broad PRs wholesale. Treat them as source branches and pull
forward narrow, current-main slices with their tests.

## PR #8: NIMBUS Training Corpus Contract

- URL: https://github.com/ccandelori/watchman/pull/8
- Branch: `codex/nimbus-training-records`
- Status: checks passing, merge state dirty
- Main value:
  - typed `nimbus-training-turn/v0` record contract
  - synthetic NIMBUS training corpus generator
  - offline InfoNCE trainer and metrics
  - learned critic adapter boundary
  - generated corpus safety documentation
- Harvest first:
  - `docs/nimbus-training-corpus.md`
  - `src/aegis/replay/nimbus_training.py`
  - focused tests from `tests/aegis/test_nimbus_training_corpus.py`
- Harvest later:
  - `src/aegis/replay/nimbus_infonce.py`
  - `src/aegis/detectors/nimbus_learned.py`
  - learned critic runtime bridge
- Do not pull blindly:
  - README/runtime-spine edits, because current main has newer runtime,
    profile, smoke, and CIFT capability wording.
  - artifact-boundary edits without checking current `scripts/check_artifact_boundaries.py`.

## PR #3: Parallel Runtime and CIFT Lanes

- URL: https://github.com/ccandelori/watchman/pull/3
- Branch: `codex/watchman-parallel-runtime-cift-work`
- Status: checks passing, merge state dirty
- Main value:
  - audit-safe leakage trace primitives
  - shared secret context helper
  - CIFT selected-choice report artifacts
  - contributor and PR governance updates
  - NIMBUS critic contract tests
- Harvest first:
  - `src/aegis/core/sensitive_context.py`
  - the safe portions of `src/aegis/core/leakage.py`
  - tests that prove no raw output or secret-shaped metadata crosses trace boundaries
- Harvest later:
  - selected CIFT report artifacts, after checking whether they still match the
    current CIFT runtime artifact and promotion path.
  - governance text from `CONTRIBUTING.md` and `.github/pull_request_template.md`.
- Do not pull blindly:
  - `src/aegis/detectors/nimbus.py` edits, because current main has newer
    NIMBUS profile and canary-aware critic work.
  - broad README edits without reconciling current proxy, redteam, NIMBUS, and
    CIFT sections.

## Suggested Next Branches

1. `codex/harvest-nimbus-training-contract`
2. `codex/harvest-leakage-trace-primitives`
3. `codex/harvest-contributor-governance`

Each branch should be small, current-main based, and should stage only the
slice being harvested.
