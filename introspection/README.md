# Aegis Introspection

This directory contains the current model-introspection research thread for
Aegis. The purpose is to test whether hidden-state features from a local model
can help distinguish safe secret handling from exfiltration-oriented behavior.

The work here is exploratory. The current results are useful checkpoints, not
validated production claims.

## Current State

Four datasets are currently registered in `data/lineage.json`:

| Dataset | Purpose | Rows |
|---|---|---:|
| `baseline_prompts_v1` | First hand-authored checkpoint dataset. | 90 |
| `hard_prompts_v1` | Harder contrastive successor to the baseline. | 90 |
| `hard_prompts_v2` | Targeted successor focused on V1 error clusters. | 90 |
| `hard_prompts_v3` | Held-out summary, redaction, and replacement checkpoint. | 90 |

All four datasets use the same label shape:

| Label | Count | Meaning |
|---|---:|---|
| `benign` | 30 | No secret handling or exfiltration request. |
| `secret_present_safe` | 30 | A secret-like value appears, but the prompt asks for safe handling. |
| `exfiltration_intent` | 30 | The prompt asks to reveal, transmit, encode, route, or leak secret-like material. |

Each prompt has a `family` field. Grouped evaluation uses those families to
hold related prompt patterns out together.

The fixed historical activation checkpoint is:

```text
mean_pool_layer_18
```

For the important `safe_secret_vs_exfiltration` task:

| Dataset | Evaluation | Best Method | Macro F1 | Accuracy |
|---|---|---|---:|---:|
| Baseline | Random stratified CV | `activation_probe` | 0.9131 | 0.9167 |
| Baseline | Grouped CV | `activation_probe` | 0.8620 | 0.8667 |
| Hard V1 | Random stratified CV | `activation_probe` | 0.9163 | 0.9167 |
| Hard V1 | Grouped CV | `activation_probe` | 0.8788 | 0.8833 |
| Hard V2 | Random stratified CV | `activation_probe` | 0.7470 | 0.7500 |
| Hard V2 | Grouped CV | `activation_probe` | 0.7225 | 0.7333 |
| Hard V3 | Grouped CV | `activation_probe` | 0.8324 | 0.8333 |

The grouped results are the more credible checkpoints because they reduce
prompt-family leakage between train and test folds. Hard V1 is especially
useful because both TF-IDF baselines degrade sharply on the intent-sensitive
task while the activation probe remains comparatively strong.

Hard V2 is now the strongest pressure test. It focuses on the Hard V1 error
clusters: output contracts, tool arguments, broker boundaries, policy
exceptions, and summaries. Under grouped evaluation, the fixed activation probe
drops to 0.7225 macro F1 / 0.7333 accuracy on `safe_secret_vs_exfiltration`,
while word and character TF-IDF baselines fall below chance. The full
machine-readable prediction ledger is registered in lineage. Preliminary
adjudication found the 16 activation-probe misses label-defensible, so the next
move is analysis rather than dataset repair.

A grouped Hard V2 layer sweep found the best feature was `final_token_layer_11`
at 0.9657 macro F1 / 0.9667 accuracy, while the fixed
`mean_pool_layer_18` checkpoint ranked 39th. A cross-dataset check now confirms
that `final_token_layer_11` also improves Hard V1, but loses slightly on the
original baseline:

| Dataset | Reference | Candidate | Delta Macro F1 | Winner |
|---|---:|---:|---:|---|
| Baseline | 0.8620 | 0.8445 | -0.0175 | `mean_pool_layer_18` |
| Hard V1 | 0.8788 | 0.8993 | +0.0205 | `final_token_layer_11` |
| Hard V2 | 0.7225 | 0.9657 | +0.2432 | `final_token_layer_11` |

Treat `mean_pool_layer_18` as the fixed regression checkpoint and
`final_token_layer_11` as a candidate hard-case checkpoint until the candidate
is tested against new prompt families.

Residual analysis on Hard V2 strengthened that candidate reading. On
`safe_secret_vs_exfiltration`, `final_token_layer_11` reduces activation-probe
errors from 16 to 2, fixing 14 prior misses with 0 introduced target-task
errors. Both persistent misses are in `hard_v2_safe_summary_customer_note`.
However, the candidate introduces 1 error on `benign_vs_secret_related`, so this
is still a candidate checkpoint rather than a universal replacement.

Hard V3 provides fresh held-out coverage for summary, redaction, and replacement
behavior with neutral secret identifiers. On this corrected held-out set,
`final_token_layer_11` beats the fixed reference on `safe_secret_vs_exfiltration`
with 0.8818 macro F1 / 0.8833 accuracy versus 0.8324 macro F1 / 0.8333
accuracy. The residual comparison is mixed but positive: 7 fixed errors, 3
persistent errors, and 4 introduced errors. Across all four checkpoints,
`final_token_layer_11` now wins 3 and `mean_pool_layer_18` wins 1.

The V3 layer sweep found a stronger local feature, `final_token_layer_16`, at
0.9321 macro F1 / 0.9333 accuracy. Treat that as a diagnostic result, not a new
replacement, until it is checked against the earlier datasets.

## Project Layout

```text
introspection/
├── data/
│   ├── activations/      # Serialized activation feature artifacts
│   ├── probes/           # Reserved for trained probe artifacts
│   ├── reports/          # JSON reports and narrative progress notes
│   ├── lineage.json      # Canonical experiment ledger
│   ├── prompts.jsonl      # Baseline prompt dataset
│   ├── prompts_hard.jsonl # Hard Baseline V1 dataset
│   ├── prompts_hard_v2.jsonl # Hard Baseline V2 dataset
│   └── prompts_hard_v3.jsonl # Hard Baseline V3 dataset
├── notebooks/            # Interactive exploration notebooks
├── scripts/              # CLI entry points for extraction, training, summaries, validation
├── src/aegis_introspection/
│   └── ...               # Typed implementation modules
└── tests/                # Unit tests
```

## Lineage Rules

Organization matters here. Any dataset, activation artifact, or machine-readable
report that supports a stated result should be registered in `data/lineage.json`
with its SHA256 hash.

The rule of thumb:

```text
Do not replace an experimental state that produced a reported metric.
```

Add new files and new lineage records rather than overwriting baseline
evidence. Current examples:

```text
data/prompts_hard.jsonl
data/prompts_hard_v2.jsonl
data/prompts_hard_v3.jsonl
data/activations/qwen3_0_6b_hard_all_layers.pt
data/activations/qwen3_0_6b_hard_v2_all_layers.pt
data/activations/qwen3_0_6b_hard_v3_all_layers.pt
data/reports/binary_tasks_hard_grouped.json
```

Future datasets should use new names, for example:

```text
data/prompts_tool_calls.jsonl
data/prompts_hard_v4.jsonl
data/activations/qwen3_0_6b_hard_v4_all_layers.pt
```

Validate lineage after any intentional manifest change:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/validate_lineage.py
```

## Common Commands

Run the full test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python -m unittest discover -s introspection/tests
```

Extract all-layer activation features for the baseline dataset:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --pooling final_token,mean_pool \
  --output introspection/data/activations/qwen3_0_6b_all_layers.pt
```

Train the all-layer probe sweep:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_probe.py \
  --artifact introspection/data/activations/qwen3_0_6b_all_layers.pt \
  --output introspection/data/reports/probe_all_layers.json
```

Run random binary tasks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_binary_tasks.py
```

Run grouped binary tasks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py
```

Run grouped binary tasks for Hard V1:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py \
  --artifact introspection/data/activations/qwen3_0_6b_hard_all_layers.pt \
  --output-json introspection/data/reports/binary_tasks_hard_grouped.json \
  --output-md introspection/data/reports/binary_tasks_hard_grouped_summary.md
```

Run grouped family-level error analysis for Hard V1:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/analyze_binary_errors.py
```

Run grouped binary tasks for Hard V2:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py \
  --artifact introspection/data/activations/qwen3_0_6b_hard_v2_all_layers.pt \
  --output-json introspection/data/reports/binary_tasks_hard_v2_grouped.json \
  --output-md introspection/data/reports/binary_tasks_hard_v2_grouped_summary.md
```

Build the Hard V2 adjudication worksheet:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/adjudicate_v2_errors.py
```

Run the Hard V2 grouped binary layer sweep:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/sweep_binary_layers.py
```

Run the cross-dataset candidate feature check:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_candidate_feature.py
```

Run candidate Hard V2 error analysis:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/analyze_binary_errors.py \
  --artifact introspection/data/activations/qwen3_0_6b_hard_v2_all_layers.pt \
  --activation-feature final_token_layer_11 \
  --output-json introspection/data/reports/binary_error_analysis_hard_v2_candidate_final_token_layer_11_grouped.json \
  --output-md introspection/data/reports/binary_error_analysis_hard_v2_candidate_final_token_layer_11_grouped_summary.md
```

Compare candidate and reference residual errors:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_error_residuals.py
```

Extract Hard V3 activations:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --prompts introspection/data/prompts_hard_v3.jsonl \
  --output introspection/data/activations/qwen3_0_6b_hard_v3_all_layers.pt \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --pooling final_token,mean_pool
```

Run the four-checkpoint candidate crosscheck:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_candidate_feature.py \
  --dataset-artifact baseline_prompts_v1:introspection/data/activations/qwen3_0_6b_all_layers.pt \
  --dataset-artifact hard_prompts_v1:introspection/data/activations/qwen3_0_6b_hard_all_layers.pt \
  --dataset-artifact hard_prompts_v2:introspection/data/activations/qwen3_0_6b_hard_v2_all_layers.pt \
  --dataset-artifact hard_prompts_v3:introspection/data/activations/qwen3_0_6b_hard_v3_all_layers.pt \
  --output-json introspection/data/reports/candidate_feature_crosscheck_with_hard_v3.json \
  --output-md introspection/data/reports/candidate_feature_crosscheck_with_hard_v3_summary.md
```

## Reports

Key human-readable checkpoints:

- `data/reports/activation_probe_progress_2026-06-18.md`
- `data/reports/binary_probe_progress_2026-06-18.md`
- `data/reports/grouped_binary_probe_progress_2026-06-18.md`
- `data/reports/hard_baseline_probe_progress_2026-06-18.md`
- `data/reports/baseline_vs_hard_v1_comparison_2026-06-19.md`
- `data/reports/hard_v2_probe_progress_2026-06-19.md`
- `data/reports/probe_all_layers_summary.md`
- `data/reports/binary_tasks_summary.md`
- `data/reports/binary_tasks_grouped_summary.md`
- `data/reports/binary_tasks_hard_summary.md`
- `data/reports/binary_tasks_hard_grouped_summary.md`
- `data/reports/binary_error_analysis_hard_grouped_summary.md`
- `data/reports/binary_tasks_hard_v2_summary.md`
- `data/reports/binary_tasks_hard_v2_grouped_summary.md`
- `data/reports/binary_error_analysis_hard_v2_grouped_summary.md`
- `data/reports/hard_v2_error_adjudication_summary.md`
- `data/reports/hard_v2_error_adjudication_notes_2026-06-19.md`
- `data/reports/binary_layer_sweep_hard_v2_grouped_summary.md`
- `data/reports/hard_v2_layer_sweep_progress_2026-06-19.md`
- `data/reports/candidate_feature_crosscheck_summary.md`
- `data/reports/candidate_feature_crosscheck_progress_2026-06-19.md`
- `data/reports/binary_error_analysis_hard_v2_candidate_final_token_layer_11_grouped_summary.md`
- `data/reports/hard_v2_candidate_residual_error_comparison_summary.md`
- `data/reports/hard_v2_candidate_error_adjudication_summary.md`
- `data/reports/hard_v2_candidate_residual_progress_2026-06-19.md`
- `data/reports/candidate_feature_crosscheck_hard_v3_summary.md`
- `data/reports/candidate_feature_crosscheck_with_hard_v3_summary.md`
- `data/reports/binary_error_analysis_hard_v3_reference_grouped_summary.md`
- `data/reports/binary_error_analysis_hard_v3_candidate_final_token_layer_11_grouped_summary.md`
- `data/reports/hard_v3_candidate_residual_error_comparison_summary.md`
- `data/reports/binary_layer_sweep_hard_v3_grouped_summary.md`
- `data/reports/hard_v3_candidate_error_adjudication_summary.md`
- `data/reports/hard_v3_heldout_validation_progress_2026-06-19.md`

Key machine-readable reports registered in lineage:

- `data/reports/probe_baseline.json`
- `data/reports/text_baseline.json`
- `data/reports/probe_all_layers.json`
- `data/reports/binary_tasks.json`
- `data/reports/binary_tasks_grouped.json`
- `data/reports/binary_tasks_hard.json`
- `data/reports/binary_tasks_hard_grouped.json`
- `data/reports/binary_error_analysis_hard_grouped.json`
- `data/reports/binary_tasks_hard_v2.json`
- `data/reports/binary_tasks_hard_v2_grouped.json`
- `data/reports/binary_error_analysis_hard_v2_grouped.json`
- `data/reports/hard_v2_error_adjudication.json`
- `data/reports/binary_layer_sweep_hard_v2_grouped.json`
- `data/reports/candidate_feature_crosscheck.json`
- `data/reports/binary_error_analysis_hard_v2_candidate_final_token_layer_11_grouped.json`
- `data/reports/hard_v2_candidate_residual_error_comparison.json`
- `data/reports/hard_v2_candidate_error_adjudication.json`
- `data/reports/candidate_feature_crosscheck_hard_v3.json`
- `data/reports/candidate_feature_crosscheck_with_hard_v3.json`
- `data/reports/binary_error_analysis_hard_v3_reference_grouped.json`
- `data/reports/binary_error_analysis_hard_v3_candidate_final_token_layer_11_grouped.json`
- `data/reports/hard_v3_candidate_residual_error_comparison.json`
- `data/reports/binary_layer_sweep_hard_v3_grouped.json`
- `data/reports/hard_v3_candidate_error_adjudication.json`

## Next Moves

The next experimental step is a feature-stability check, not silent promotion of
the latest sweep winner.

Recommended sequence:

1. Human-review the Hard V3 candidate errors, especially the introduced safe
   examples classified as exfiltration.
2. Cross-check `final_token_layer_16` against baseline, Hard V1, Hard V2, and
   Hard V3 as a diagnostic candidate.
3. Keep `mean_pool_layer_18` as the fixed regression checkpoint while
   final-token candidates remain under evaluation.
4. Define a feature-selection rule before promoting any post-hoc sweep winner.
5. Keep registering every dataset, artifact, and machine-readable report in
   `data/lineage.json`.

The research question remains narrow and concrete:

```text
Does the activation signal still help when surface text cues become less generous?
```
