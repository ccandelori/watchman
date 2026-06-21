# Aegis Grouped Binary Probe Progress Note

Date: 2026-06-18

Status: stricter exploratory checkpoint, not a validated research finding.

## Purpose

This note records the first grouped cross-validation pass for the Aegis
introspection experiment. The previous binary checkpoint used random stratified
cross-validation. That was useful for a fast signal, but it allowed related
prompt patterns to appear in both train and test folds.

This checkpoint asks whether the strongest current activation feature still
works when entire prompt families are held out together.

## Dataset Family Annotation

The prompt dataset still contains 90 examples across the same three labels:

| Label | Count | Family Count | Examples per Family |
|---|---:|---:|---:|
| `benign` | 30 | 10 | 3 |
| `secret_present_safe` | 30 | 10 | 3 |
| `exfiltration_intent` | 30 | 10 | 3 |

Each prompt now includes a required `family` field. Families describe the prompt
pattern, such as `safe_redaction_output`, `exfil_encoding`, or
`benign_security_concepts`. The grouped evaluator uses these families as
cross-validation groups, so examples from the same family do not appear in both
training and test folds.

## Artifacts

- Prompt dataset: `introspection/data/prompts.jsonl`
- All-layer activation artifact: `introspection/data/activations/qwen3_0_6b_all_layers.pt`
- Random binary task report: `introspection/data/reports/binary_tasks.json`
- Random binary task summary: `introspection/data/reports/binary_tasks_summary.md`
- Grouped binary task report: `introspection/data/reports/binary_tasks_grouped.json`
- Grouped binary task summary: `introspection/data/reports/binary_tasks_grouped_summary.md`
- Grouped binary evaluator: `introspection/scripts/train_grouped_binary_tasks.py`

## Method

The grouped run kept the same detector inputs as the previous binary checkpoint:

| Method | Input |
|---|---|
| `activation_probe` | `mean_pool_layer_18` activation vectors |
| `word_tfidf` | word unigram/bigram TF-IDF over prompt text |
| `char_tfidf` | character n-gram TF-IDF over prompt text |

The change is the split strategy:

| Report | Split Strategy |
|---|---|
| `binary_tasks.json` | `stratified_kfold` |
| `binary_tasks_grouped.json` | `stratified_group_kfold` |

## Results

### benign_vs_secret_related

This task remained easy under grouped splits.

| Method | Random Macro F1 | Grouped Macro F1 | Random Accuracy | Grouped Accuracy |
|---|---:|---:|---:|---:|
| `activation_probe` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `char_tfidf` | 0.9869 | 0.9345 | 0.9889 | 0.9444 |
| `word_tfidf` | 0.9330 | 0.9342 | 0.9444 | 0.9444 |

The activation probe still perfectly separates benign prompts from
secret-related prompts. The character baseline lost some ground, which suggests
that its near-perfect random-CV result benefited from seeing related surface
forms across folds.

### safe_secret_vs_exfiltration

This task is the more important Aegis distinction.

| Method | Random Macro F1 | Grouped Macro F1 | Random Accuracy | Grouped Accuracy |
|---|---:|---:|---:|---:|
| `activation_probe` | 0.9131 | 0.8620 | 0.9167 | 0.8667 |
| `char_tfidf` | 0.7769 | 0.7649 | 0.7833 | 0.7833 |
| `word_tfidf` | 0.7712 | 0.7054 | 0.7833 | 0.7167 |

The activation probe remains the best method, but it drops under grouped
evaluation. Its grouped confusion matrix is:

```text
[26, 4]
[4, 26]
```

That means the activation probe made eight total errors across 60 examples in
the safe-secret-vs-exfiltration task: four exfiltration examples classified as
safe handling and four safe-handling examples classified as exfiltration.

## Interpretation

The grouped result is more credible than the earlier random split because it
reduces prompt-family leakage. The activation probe still outperforms both text
baselines on the intent-sensitive task, but the performance gap is smaller than
the random split suggested.

That is the right kind of pressure test. It tells us the signal is not purely a
random-CV artifact, while also showing that the current dataset is not hard or
diverse enough to support broad claims.

The most useful finding is not the exact score. The useful finding is that
`mean_pool_layer_18` remains the strongest current detector signal after a
stricter split, and that the next iteration should stress it with harder
families rather than more near-duplicate examples.

## Caveats

- The dataset still has only 90 examples.
- Families were assigned manually and are an experimental grouping heuristic.
- Grouped cross-validation is stricter than random splitting, but it is not the
  same as a separately authored holdout set.
- The examples are still prompt-only and do not include structured tool-call
  traces, retrieved context, or multi-turn agent behavior.
- The activation probe is a lightweight logistic regression probe, not a
  production detector.

## Current Position

Where we have been:

- Built the hidden-state extraction pipeline.
- Found that sampled activation probes beat a word TF-IDF baseline.
- Ran an all-layer sweep and selected `mean_pool_layer_18`.
- Evaluated binary Aegis-relevant tasks under random stratified CV.

Where we are:

- Added prompt-family labels to the dataset.
- Added a family-aware activation artifact schema.
- Added grouped binary evaluation with `StratifiedGroupKFold`.
- Confirmed that the activation probe still leads on
  `safe_secret_vs_exfiltration`, with grouped macro F1 `0.8620`.

Where we are going:

- Add harder contrastive examples that reduce obvious surface cues.
- Add structured tool-call and agent-context examples.
- Build a separately authored holdout set.
- Track random, grouped, and holdout metrics side by side as regression
  checkpoints.

## Reproduction Commands

Regenerate all-layer features:

```bash
.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --pooling final_token,mean_pool \
  --output introspection/data/activations/qwen3_0_6b_all_layers.pt
```

Run random binary tasks:

```bash
.venv-introspection/bin/python introspection/scripts/train_binary_tasks.py
```

Run grouped binary tasks:

```bash
.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py
```

Run tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  .venv-introspection/bin/python -m unittest discover -s introspection/tests
```
