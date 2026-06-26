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
```

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
```

Use markdown for a compact human-readable summary:

```bash
uv run aegis-nimbus-eval-infonce \
  --input introspection/data/reports/aegis_nimbus_training_corpus_v0.jsonl \
  --model introspection/data/reports/aegis_nimbus_infonce_model_v0.json \
  --output introspection/data/reports/aegis_nimbus_infonce_eval_v0.md \
  --format markdown \
  --allow-training-eval
```

The v0 evaluator reports retrieval/calibration metrics plus false positives and
false negatives separately. Same-corpus evaluation is rejected unless
`--allow-training-eval` is passed, and that report is labeled with
`training_eval_reused=true`. Current curated grouped-CV scaffold evidence has
FP rate `0.0` and FN rate `0.333333`, with encoded and partial-drip holdouts
failing. That is why it remains an offline scaffold rather than a runtime or
promotion artifact.

## Promotion Boundary

This corpus scaffold and lexical InfoNCE model are not paper-faithful learned
NIMBUS evidence and must not be promoted as runtime artifacts. A paper-faithful
learned NIMBUS release still needs:

- a larger labeled session leakage corpus
- stronger grouped cross-validation on a larger corpus
- sealed holdout evaluation
- a runtime learned session critic adapter
- live runtime false negative and false positive rates
- a promotion manifest that binds the critic, corpus, evals, and runtime

## Safety Rules

- Do not use production credentials or real secret values.
- Keep credential-shaped markers such as `ghp_`, `github_pat_`, `sk_live_`,
  `AKIA`, `hny_`, or `{{CREDENTIAL:` out of records and manifests.
- Do not mix training rows into runtime audit logs.
- Do not claim `paper_faithful_learned_critic=true` from this scaffold.
- Do not wire `aegis_nimbus_infonce_model_v0.json` into runtime policy.
- Keep deterministic NIMBUS as the active runtime path until a learned critic
  has its own corpus, grouped CV, sealed holdout, live FN/FP metrics, and
  promotion evidence.
