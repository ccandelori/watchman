# NIMBUS Training Corpus

This runbook defines the repo-native bootstrap corpus contract for a future
paper-faithful learned NIMBUS critic. It is separate from the deterministic
runtime critic, the proxy smoke loop, and the external redteam report parser.

## Purpose

The active runtime path uses `CanaryNimbusCritic`, a deterministic canary-aware
critic that estimates cumulative leakage from exact, encoded, and partial
canary matches. That is useful beta runtime evidence, but it is not the learned
session leakage critic required for a paper-faithful NIMBUS release.

`nimbus-training-turn/v0` exists to make the learned-critic data boundary
explicit before larger human, redteam, or model-generated corpora are collected.

## Contract

Each JSONL row is one current output to score:

- `state_messages`: session state before scoring the current output.
- `output_text`: current model output for this turn.
- `true_secret_context`: the positive secret context.
- `negative_secret_contexts`: 16 decoy contexts for InfoNCE-style training.
- `info_nce`: positive index, negative count, and ordered candidate ids.
- `leakage_label`: one of `benign`, `partial`, `encoded`, `direct`,
  `paraphrased`, `tool_output`, or `delayed`.
- `target_turn_leakage_bits`: synthetic target for this turn.
- `target_cumulative_leakage_bits`: synthetic cumulative target by session.
- `split_group_key`: the grouped-CV unit. In v0 it equals `session_id`, so
  turns from the same leakage session cannot be split across folds.

The v0 generator keeps the positive context at candidate index `0`, followed by
16 negatives, giving a one-turn InfoNCE ceiling of `log2(17)` bits.

## Generate

For scratch runs, write to `/tmp` or another disposable location:

```bash
uv run aegis-nimbus-training-corpus \
  --output /tmp/aegis-nimbus-training.jsonl \
  --manifest-output /tmp/aegis-nimbus-training-manifest.json
```

The manifest uses schema version `aegis.nimbus_training_manifest/v1` and records
label counts, scenario counts, split-group counts, quality gates, and the
explicit status `not_promotable_training_contract_only`.

For curated local evidence, use:

```bash
uv run aegis-nimbus-training-corpus \
  --output introspection/data/reports/aegis_nimbus_training_corpus_v0.jsonl \
  --manifest-output introspection/data/reports/aegis_nimbus_training_corpus_manifest_v0.json

uv run aegis-nimbus-training-corpus \
  --profile sealed_holdout \
  --output introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_v0.jsonl \
  --manifest-output introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_manifest_v0.json
```

The default `calibration` profile and the `sealed_holdout` profile have matching
scenario coverage, but distinct synthetic secret contexts and split-group keys.
The holdout profile is for evidence only; do not train on it.

## Train And Evaluate

Train the offline lexical InfoNCE scaffold:

```bash
uv run aegis-nimbus-train-infonce \
  --input introspection/data/reports/aegis_nimbus_training_corpus_v0.jsonl \
  --output introspection/data/reports/aegis_nimbus_infonce_model_v0.json
```

Evaluate that scaffold:

```bash
uv run aegis-nimbus-eval-infonce \
  --input introspection/data/reports/aegis_nimbus_training_corpus_v0.jsonl \
  --model introspection/data/reports/aegis_nimbus_infonce_model_v0.json \
  --output introspection/data/reports/aegis_nimbus_infonce_eval_v0.json \
  --allow-training-eval \
  --grouped-cv-output introspection/data/reports/aegis_nimbus_infonce_grouped_cv_v0.json

uv run aegis-nimbus-eval-infonce \
  --input introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_v0.jsonl \
  --model introspection/data/reports/aegis_nimbus_infonce_model_v0.json \
  --output introspection/data/reports/aegis_nimbus_infonce_sealed_holdout_eval_v0.json
```

Use markdown for a compact human-readable summary:

```bash
uv run aegis-nimbus-eval-infonce \
  --input introspection/data/reports/aegis_nimbus_training_corpus_v0.jsonl \
  --model introspection/data/reports/aegis_nimbus_infonce_model_v0.json \
  --output introspection/data/reports/aegis_nimbus_infonce_eval_v0.md \
  --format markdown \
  --allow-training-eval

uv run aegis-nimbus-eval-infonce \
  --input introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_v0.jsonl \
  --model introspection/data/reports/aegis_nimbus_infonce_model_v0.json \
  --output introspection/data/reports/aegis_nimbus_infonce_sealed_holdout_eval_v0.md \
  --format markdown
```

Run the learned critic through the in-process runtime beta adapter. This records
whether the `NimbusCritic` interface can execute the learned score, but it is
not live gateway evidence and does not make the artifact promotable:

```bash
uv run aegis-nimbus-runtime-beta-eval \
  --input introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_v0.jsonl \
  --model introspection/data/reports/aegis_nimbus_infonce_model_v0.json \
  --output introspection/data/reports/aegis_nimbus_runtime_beta_eval_v0.json
```

Build the promotion evidence binder:

```bash
uv run aegis-nimbus-promotion-evidence \
  --deterministic-eval introspection/data/reports/aegis_nimbus_deterministic_beta_eval_v1.json \
  --calibration-manifest introspection/data/reports/aegis_nimbus_training_corpus_manifest_v0.json \
  --sealed-manifest introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_manifest_v0.json \
  --infonce-model introspection/data/reports/aegis_nimbus_infonce_model_v0.json \
  --grouped-cv introspection/data/reports/aegis_nimbus_infonce_grouped_cv_v0.json \
  --sealed-holdout introspection/data/reports/aegis_nimbus_infonce_sealed_holdout_eval_v0.json \
  --gateway-smoke introspection/data/reports/aegis_default_mock_provider_smoke_learned_nimbus_beta_v2.json \
  --runtime-beta-eval introspection/data/reports/aegis_nimbus_runtime_beta_eval_v0.json \
  --output introspection/data/reports/aegis_nimbus_promotion_evidence_v1.json
```

The v0 evaluator reports retrieval/calibration metrics plus false positives and
false negatives separately. Same-corpus evaluation is rejected unless
`--allow-training-eval` is passed, and that report is labeled with
`training_eval_reused=true`. Current curated grouped-CV and sealed-holdout
scaffold evidence both have attack top-1 `0.992157`, turn FP rate `0.005369`,
turn FN rate `0.0`, session FP rate `0.0`, and session FN rate `0.0`. The
lexical model treats `state_token_overlap` as diagnostic/context-selection
evidence only, with zero current-turn leakage weight. The runtime beta adapter
registers each sealed record's positive secret context and 16 negative contexts
instead of synthesizing runtime negatives, so its in-process runtime metrics
match the sealed scaffold. It also reports paper-shaped conversation block
metrics: 42/42 attack sessions detected, 0/8 benign-only sessions false-blocked,
false-block rate `0.0`, and mean first block turn index `3.928571`. Its
diagnostic threshold sweep selects `0.0` bits under the 5% turn/session FP/FN
operating policy. The artifact remains a non-promotable beta rather than a
promotion artifact because a common live head-to-head corpus and promoted
runtime manifest are still missing.
The promotion evidence binder records that distinction as
`promotion_status=deterministic_beta_active_learned_not_promotable`,
`promote_learned_runtime=false`, and
`recommended_runtime_critic=deterministic_canary_beta`.

The current binder uses a live local learned-gateway smoke artifact rather than
offline replay: `aegis_default_mock_provider_smoke_learned_nimbus_beta_v2.json`
reports learned runtime beta readiness, benign allow, four positive learned
gateway detections, zero learned gateway false positives, zero learned gateway
false negatives, the loaded model artifact SHA-256, selected context hash,
negative-context count, and live `/audit/explain` evidence for provider-egress
and partial-leak traces. This closes the live-gateway FN/FP evidence item, but
it does not prove promotion-grade complement over deterministic beta.

## Promotion Boundary

This corpus scaffold and lexical InfoNCE model are not paper-faithful learned
NIMBUS evidence and must not be promoted as runtime artifacts. A paper-faithful
learned NIMBUS release still needs:

- a larger labeled session leakage corpus
- stronger grouped cross-validation on a larger corpus
- broader sealed holdout evaluation
- a production secret-context candidate store for the runtime adapter
- live gateway false negative and false positive rates
- a promotion manifest that binds the critic, corpus, evals, and runtime

## Safety Rules

- Do not use production credentials or real secret values.
- Keep credential-shaped markers such as `ghp_`, `github_pat_`, `sk_live_`,
  `AKIA`, `hny_`, or `{{CREDENTIAL:` out of records and manifests.
- Do not mix training rows into runtime audit logs.
- Do not claim `paper_faithful_learned_critic=true` from this scaffold.
- Do not wire `aegis_nimbus_infonce_model_v0.json` into runtime policy unless
  `AEGIS_NIMBUS_CRITIC_KIND=learned_infonce_beta` is explicitly set and the
  resulting evidence remains labeled non-promotable.
- Keep deterministic NIMBUS as the active runtime path until a learned critic
  has its own corpus, grouped CV, sealed holdout, live FN/FP metrics, and
  promotion evidence.
