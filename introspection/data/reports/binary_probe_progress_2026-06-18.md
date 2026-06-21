# Aegis Binary Probe Progress Note

Date: 2026-06-18

Status: exploratory checkpoint, not a validated research finding.

## Purpose

This note extends the first model-introspection checkpoint. The earlier note
showed that sampled hidden-state features could outperform a simple word-level
TF-IDF baseline on a three-class prompt classification task. This checkpoint
asks a sharper question: does the activation signal remain useful when we
remove arbitrary layer selection and evaluate the binary distinctions that
matter most for Aegis?

## Question

The central question is no longer simply whether a prompt contains secret-like
material. Aegis needs to distinguish safe secret handling from unsafe secret
handling. The key binary task is therefore:

```text
secret_present_safe vs exfiltration_intent
```

That task asks whether the detector can separate prompts that say "protect or
redact this secret" from prompts that say "send, reveal, encode, or otherwise
leak this secret."

## New Artifacts

- All-layer activation artifact: `introspection/data/activations/qwen3_0_6b_all_layers.pt`
- All-layer probe report: `introspection/data/reports/probe_all_layers.json`
- All-layer human summary: `introspection/data/reports/probe_all_layers_summary.md`
- Binary task report: `introspection/data/reports/binary_tasks.json`
- Binary task human summary: `introspection/data/reports/binary_tasks_summary.md`
- Binary task evaluator: `introspection/scripts/train_binary_tasks.py`

## All-Layer Sweep

The earlier probe sampled layers `0`, `7`, `14`, `21`, and `28`. That was a
reasonable first pass, but arbitrary. The all-layer sweep removed that choice by
extracting both `final_token` and `mean_pool` features for every hidden-state
layer from `0` through `28`.

This produced 58 feature matrices:

```text
29 layers x 2 pooling methods = 58 feature views
```

Each feature matrix has shape:

```text
90 examples x 1024 activation dimensions
```

The best all-layer feature was:

```text
mean_pool_layer_18
```

| Metric | Value |
|---|---:|
| Macro F1 | 0.9776 |
| Accuracy | 0.9778 |

The all-layer sweep improved over the earlier sampled-layer best:

| Run | Best Feature | Macro F1 | Accuracy |
|---|---|---:|---:|
| Sampled layers | `mean_pool_layer_14` | 0.9554 | 0.9556 |
| All layers | `mean_pool_layer_18` | 0.9776 | 0.9778 |

The ranking suggests that mean-pooled mid-to-late-middle activations are more
useful for this task than final-token activations or the earliest layers. The
best final-token feature was `final_token_layer_17` with macro F1 `0.9436`.

## Binary Tasks

Two binary tasks were evaluated:

1. `benign_vs_secret_related`
   - `benign` vs combined `secret_present_safe` + `exfiltration_intent`
   - This asks whether the prompt involves secret-like material at all.

2. `safe_secret_vs_exfiltration`
   - `secret_present_safe` vs `exfiltration_intent`
   - This asks whether the prompt is safely handling a secret or attempting to leak it.

Each task compared three methods:

| Method | Input |
|---|---|
| `activation_probe` | `mean_pool_layer_18` activation vectors |
| `word_tfidf` | word unigram/bigram TF-IDF over prompt text |
| `char_tfidf` | character n-gram TF-IDF over prompt text |

The character baseline is important because it is better at detecting
credential-like substrings than word-level TF-IDF.

## Results

### benign_vs_secret_related

| Method | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `activation_probe` | 1.0000 | 1.0000 | `[[30, 0], [0, 60]]` |
| `char_tfidf` | 0.9869 | 0.9889 | `[[29, 1], [0, 60]]` |
| `word_tfidf` | 0.9330 | 0.9444 | `[[25, 5], [0, 60]]` |

This task is likely too easy. Character TF-IDF nearly matches the activation
probe, which suggests that much of this distinction can be learned from surface
credential-like patterns.

### safe_secret_vs_exfiltration

| Method | Macro F1 | Accuracy | Confusion Matrix |
|---|---:|---:|---|
| `activation_probe` | 0.9131 | 0.9167 | `[[29, 1], [4, 26]]` |
| `char_tfidf` | 0.7769 | 0.7833 | `[[21, 9], [4, 26]]` |
| `word_tfidf` | 0.7712 | 0.7833 | `[[20, 10], [3, 27]]` |

This is the more relevant result. The activation probe substantially
outperformed both word-level and character-level TF-IDF on the distinction
between safe secret handling and exfiltration-oriented secret handling.

## Interpretation

The binary evaluation sharpens the earlier result. Character TF-IDF is strong
for identifying whether a prompt is secret-related at all. That is expected:
credential-like strings have local textual signatures. However, character
TF-IDF did not close the gap on the intent-sensitive task.

This suggests that `mean_pool_layer_18` may encode information about how the
prompt frames the secret, not just whether a secret-like string appears. That
is the signal Aegis is looking for: a representation that helps distinguish
safe handling from exfiltration intent.

This remains exploratory. The examples are still small, hand-authored, and
randomly cross-validated. The result should be treated as a promising checkpoint
and a guide for the next experiment, not as evidence of robust real-world
performance.

## Caveats

- The dataset still contains only 90 examples.
- The examples were written by one process and may contain stylistic artifacts.
- Random cross-validation may allow similar prompt families to appear in both
  training and test folds.
- The activation probe may still be learning construction artifacts.
- The text baselines are stronger than before but still not exhaustive.
- The probe was evaluated on prompt text, not yet on structured tool-call
  contexts or multi-turn agent traces.

## Current Position

Where we have been:

- Verified Qwen hidden-state access.
- Built a small three-class dataset.
- Extracted sampled-layer activation features.
- Trained an activation probe and word TF-IDF text baseline.
- Logged the first exploratory checkpoint.

Where we are:

- Removed arbitrary layer selection with an all-layer sweep.
- Identified `mean_pool_layer_18` as the strongest current activation feature.
- Added character n-gram TF-IDF as a stronger text baseline.
- Evaluated binary tasks aligned with Aegis detector needs.
- Found that the activation probe outperforms both text baselines on
  `safe_secret_vs_exfiltration`.

Where we are going:

- Test whether the activation advantage survives harder data and stricter splits.
- Move from prompt-only examples toward tool-call and agent-context examples.
- Convert this from an exploratory probe into a detector signal that can be
  combined with deterministic scanners, honeytokens, and policy enforcement.

## Next Steps

1. Add grouped splits by prompt family or attack pattern.
2. Add harder contrastive examples where surface cues are less reliable.
3. Add structured tool-call JSON examples.
4. Add paraphrased holdout examples generated separately from the training set.
5. Evaluate whether `mean_pool_layer_18` remains strong under the new splits.
6. Track these metrics as regression checkpoints.

## Reproduction Commands

Extract all-layer features:

```bash
.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --output introspection/data/activations/qwen3_0_6b_all_layers.pt
```

Train all-layer probe:

```bash
.venv-introspection/bin/python introspection/scripts/train_probe.py \
  --artifact introspection/data/activations/qwen3_0_6b_all_layers.pt \
  --output introspection/data/reports/probe_all_layers.json
```

Summarize all-layer probe:

```bash
.venv-introspection/bin/python introspection/scripts/summarize_probe_report.py \
  --report introspection/data/reports/probe_all_layers.json \
  --output introspection/data/reports/probe_all_layers_summary.md
```

Run binary tasks:

```bash
.venv-introspection/bin/python introspection/scripts/train_binary_tasks.py
```

Run tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  .venv-introspection/bin/python -m unittest discover -s introspection/tests
```
