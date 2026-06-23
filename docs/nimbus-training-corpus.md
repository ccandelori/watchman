# NIMBUS Training Corpus

This runbook describes the repo-native bootstrap corpus for a paper-faithful
NIMBUS critic. It is separate from the runtime fixture loop and the external
redteam report loop.

## Purpose

The current runtime NIMBUS path uses a deterministic canary-aware critic so the
proxy and redteam affordances can exercise cumulative leakage behavior. A
paper-faithful NIMBUS critic needs a different artifact shape: one record per
turn with the conversation state, the current model output, the true secret
context, and fixed negative secret contexts for InfoNCE training.

The synthetic generator exists to make that contract explicit before larger
human or redteam-derived corpora are collected.

## Contract

Each JSONL row is a `nimbus-training-turn/v0` record with:

- `state_messages`: the state available before scoring the current output,
  corresponding to `X<=t, Y<t`.
- `output_text`: the current output `Y_t` to score.
- `true_secret_context`: the positive context `c`.
- `negative_secret_contexts`: 16 decoy contexts `c_j`.
- `info_nce`: the positive index, negative count, and ordered candidate ids.
- `leakage_label`: `benign`, `partial`, `encoded`, or `direct`.
- `target_turn_leakage_bits` and `target_cumulative_leakage_bits`: synthetic
  targets for evaluation and smoke training.

The positive context is always index `0` in the v0 generator, followed by the
16 negatives. This gives a one-turn InfoNCE ceiling of `log2(17)` bits.

## Generate

Write generated files under `data/nimbus_training/`; that directory is ignored
and blocked by the artifact-boundary guard.

```bash
uv run --extra dev aegis-nimbus-training-corpus \
  --output data/nimbus_training/nimbus_training.generated.jsonl
```

The generated corpus is intentionally small. It contains benign, partial drip,
encoded leak, and direct leak sessions using synthetic credential contexts only.
It should be used to validate parsers, loaders, and training harness plumbing,
not to claim model performance.

## Train And Evaluate

Train the first offline InfoNCE-shaped critic artifact:

```bash
uv run --extra dev aegis-nimbus-train-infonce \
  --input data/nimbus_training/nimbus_training.generated.jsonl \
  --output data/nimbus_training/nimbus_infonce_model.generated.json
```

Evaluate that artifact against a corpus:

```bash
uv run --extra dev aegis-nimbus-eval-infonce \
  --input data/nimbus_training/nimbus_training.generated.jsonl \
  --model data/nimbus_training/nimbus_infonce_model.generated.json \
  --output data/nimbus_training/nimbus_infonce_eval.generated.json
```

The v0 trainer uses a small lexical scorer with grid-searched non-negative
feature weights. It proves the InfoNCE objective, JSON artifact shape, evaluator
metrics, and bit conversion:

```text
estimated_bits = max(0, log2(17) - nce_loss_bits)
```

This is not a paper-grade encoder. Ambiguous partial fragments can tie with
negative contexts, and the evaluator counts tied positives conservatively rather
than treating them as retrieval wins.

The model artifact stores weights, schema metadata, label distribution, corpus
digest, and aggregate metrics. It does not store raw `state_messages`,
`output_text`, `context_text`, or credential-shaped values.

## Safety Rules

- Do not commit generated `data/nimbus_training/` artifacts.
- Do not store production credentials or real secret values in training rows.
- Do not promote the v0 lexical artifact as the default runtime critic.
- Keep the current output out of `state_messages`; previous outputs may appear
  there for multi-turn state.
- Treat labels and target bits as controlled training metadata, not as runtime
  detector evidence.
