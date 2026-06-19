# Aegis Introspection

This directory contains the current model-introspection research thread for
Aegis. The purpose is to test whether hidden-state features from a local model
can help distinguish safe secret handling from exfiltration-oriented behavior.

The work here is exploratory. The current results are useful checkpoints, not
validated production claims.

## System Alignment

The current work is the CIFT-like portion of the broader Aegis/AIS design. The
larger system should be treated as a multi-monitor gateway:

| Component | Role | Current Status |
|---|---|---|
| DP-HONEY | Inject format-matched honeytokens before model access. | Not implemented here yet. |
| CIFT-like activation monitor | Read hidden-state features before output generation. | Current focus. |
| Text leakage detector | Scan generated text for canary or secret leakage. | Not implemented here yet. |
| NIMBUS-like accumulator | Track cumulative leakage risk over a conversation. | Not implemented here yet. |

The activation probe is therefore a monitor signal, not the whole defense. It
should eventually emit structured evidence that a gateway policy engine can
combine with honeytoken evidence, post-output leakage scans, and cumulative
conversation risk.

The paper-specific CIFT target is narrower than generic activation probing:
monitor readout positions that occur after both secret context and the current
query or payload, rather than hidden states at static credential positions. The
paper also frames CIFT as a calibrated deviation signal over the last quarter of
transformer layers, followed by learned nonnegative layer weighting. The current
Aegis work approximates that direction with final-token/readout-style features
and grouped probe evaluation, but it has not yet implemented the paper's full
benign-calibrated CCI/CFS scoring path.

That distinction matters for planning:

| Question | Current Work | Paper-Aligned Target |
|---|---|---|
| Readout position | Final prompt token features. | Positions after secret context and query/payload; final prompt token plus pre-generation decision positions. |
| Layer handling | Individual and concatenated feature candidates. | Last-quarter layer hooks with calibrated per-layer deviation and learned layer weights. |
| Calibration | Grouped CV, residual analysis, human adjudication. | Benign calibration distribution, conformal/text thresholds, and deployment operating points. |
| System role | Candidate CIFT monitor signal. | One component in AIS per-turn pipeline before generation, combined with canary/text detection and NIMBUS. |

## Operating Workflow

Every experiment should answer these questions before it is promoted:

1. Which system component does this advance?
2. What runtime signal would this component emit?
3. What dataset or artifact proves the result without replacing prior evidence?
4. What grouped, held-out, or residual check protects against post-hoc overfit?
5. What failure cases need human review?
6. How does the result affect the eventual gateway decision pipeline?

For the current CIFT-like thread, use this workflow:

| Stage | Required Output |
|---|---|
| Scope | State whether the work is CIFT, DP-HONEY, text detection, NIMBUS, or integration. |
| Dataset | Add or reuse a registered prompt dataset with clear label and family semantics. |
| Extraction | Produce activation artifacts without overwriting prior checkpoints. |
| Evaluation | Run grouped evaluation first for credible claims. |
| Crosscheck | Compare against fixed references and newer candidates across checkpoints. |
| Residual analysis | Track fixed, persistent, and introduced errors by family. |
| Adjudication | Create human-review worksheets for promotion-sensitive regressions. |
| Paper alignment | State whether the result is an approximation or implements the paper method directly. |
| Promotion | Define an explicit rule before changing the active checkpoint or monitor contract. |
| Integration | Express the result as a monitor event suitable for gateway policy composition. |

The target monitor event shape is:

```json
{
  "monitor": "activation_probe",
  "component": "cift_like",
  "feature_key": "concat(final_token_layer_11,final_token_layer_16)",
  "task": "safe_secret_vs_exfiltration",
  "risk_score": 0.0,
  "decision": "allow|flag|block",
  "model_id": "Qwen/Qwen3-0.6B",
  "probe_version": "unpromoted",
  "evidence": {
    "evaluation_strategy": "stratified_group_kfold",
    "dataset_checkpoint": "hard_prompts_v3"
  }
}
```

Do not promote a feature solely because it improves one metric. Promotion should
balance aggregate performance, worst-checkpoint behavior, introduced-error
severity, human adjudication, and integration value for the full gateway.
For CIFT specifically, promotion should also state whether the candidate is a
temporary empirical probe or the first implementation of the paper-aligned
readout-position, calibrated-deviation monitor.

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

The feature-stability check compares `mean_pool_layer_18`,
`final_token_layer_11`, and `final_token_layer_16` across all four checkpoints.
`final_token_layer_11` remains the best average performer at 0.8978 mean macro
F1, while `final_token_layer_16` is the stability challenger: it has the higher
minimum macro F1, the smaller range, and wins baseline plus Hard V3. The
historical reference wins none of the three-feature checkpoint comparisons, but
still anchors the experiment history.

The combined-feature check adds
`concat(final_token_layer_11,final_token_layer_16)` as a derived activation
matrix. It ranks first by mean macro F1 at 0.9151 and wins or ties three of four
checkpoints. The important caveat is Hard V3: `final_token_layer_16` remains the
local winner there, while the combined feature falls back to roughly
`final_token_layer_11` performance.

The combined-feature residual suite compares that candidate against
`mean_pool_layer_18`, `final_token_layer_11`, and `final_token_layer_16` across
all four checkpoints. The combined feature reduces aggregate target-task errors
against all three references: -21 versus the historical reference, -4 versus
`final_token_layer_11`, and -6 versus `final_token_layer_16`. The caution remains
Hard V3, where `final_token_layer_16` has 4 errors and the combined feature has
7. The combined feature is now the leading promotion candidate, but the Hard V3
introduced-error families should be reviewed before changing the checkpoint.

The Hard V3 combined-regression adjudication worksheet narrows that review to 3
introduced cases against `final_token_layer_16`: one exfiltration case the
combined feature misses as safe, and two safe cases the combined feature marks
as exfiltration. All three are pending human review.

A first CIFT-like calibrated-deviation comparison now tests a more
paper-aligned direction without replacing the static-feature thread. It uses
final-token readout features from the last quarter of the hidden-state stack,
`final_token_layer_22` through `final_token_layer_28`, calibrates a diagonal
safe-secret distribution inside each grouped fold, and trains on per-layer
deviation scores. This is still an approximation: it does not yet implement
CCI/CFS, learned nonnegative layer weighting, or richer readout positions beyond
the final prompt token.

Against the current combined static feature, the first CIFT-like score loses on
all four checkpoints:

| Dataset | Combined Macro F1 | CIFT-like Macro F1 | Delta Macro F1 |
|---|---:|---:|---:|
| Baseline | 0.8804 | 0.6981 | -0.1822 |
| Hard V1 | 0.9331 | 0.5399 | -0.3932 |
| Hard V2 | 0.9657 | 0.5329 | -0.4328 |
| Hard V3 | 0.8811 | 0.4076 | -0.4735 |

Treat this as a negative checkpoint, not a failed project direction. It says
the naive calibrated-distance score is not enough by itself. The next
CIFT-focused work should isolate whether the weakness comes from the readout
position, the score compression, missing layer weighting, or the calibration
set.

A V2 CIFT-like ablation now separates some of those causes. It compares eight
last-quarter variants across final-token versus mean-pool readout proxies,
diagonal-distance versus standardized-residual representations, and
safe-secret-only versus benign-plus-safe calibration. The combined static
feature still wins every checkpoint, but residual-concat variants are much
stronger than diagonal-distance variants:

| Dataset | Combined Macro F1 | Best CIFT-like Macro F1 | Delta Macro F1 |
|---|---:|---:|---:|
| Baseline | 0.8804 | 0.8623 | -0.0181 |
| Hard V1 | 0.9331 | 0.8132 | -0.1199 |
| Hard V2 | 0.9657 | 0.7808 | -0.1849 |
| Hard V3 | 0.8811 | 0.7461 | -0.1350 |

The aggregate result points at score compression as a major weakness in the
first CIFT-like attempt. The best final-token residual variants average 0.7924
macro F1, while the best final-token diagonal variant averages 0.5783. The
calibration-set axis does not affect residual-concat results because the
downstream activation classifier standardizes those features again; the next
CIFT refinement should preserve calibrated residual magnitudes when testing
calibration choices.

A V3 CIFT-like ablation adds absolute residual features and a raw logistic
regression mode that preserves calibrated feature scale. The combined static
feature still wins every checkpoint, but raw signed residuals close the gap on
the easier datasets:

| Dataset | Combined Macro F1 | Best CIFT-like Macro F1 | Delta Macro F1 |
|---|---:|---:|---:|
| Baseline | 0.8804 | 0.8788 | -0.0015 |
| Hard V1 | 0.9331 | 0.8236 | -0.1095 |
| Hard V2 | 0.9657 | 0.7808 | -0.1849 |
| Hard V3 | 0.8811 | 0.7461 | -0.1350 |

The strongest aggregate CIFT-like family remains signed standardized residuals
over final-token last-quarter layers. Absolute residual magnitude alone is
weaker, though nonleaking calibration improves absolute residual variants. The
result motivated testing whether a simple learned nonnegative layer-weighted
head could improve the signed residual representation.

A first layer-weighted CIFT-like head now trains one signed-residual classifier
per final-token last-quarter layer, converts each layer into an exfiltration
risk probability, learns nonnegative AUC-based layer weights inside each grouped
fold, and evaluates a learned threshold on held-out prompt families. The
combined static feature still wins every checkpoint:

| Dataset | Combined Macro F1 | Layer-Weighted Head Macro F1 | Delta Macro F1 |
|---|---:|---:|---:|
| Baseline | 0.8804 | 0.8293 | -0.0511 |
| Hard V1 | 0.9331 | 0.8132 | -0.1199 |
| Hard V2 | 0.9657 | 0.7633 | -0.2024 |
| Hard V3 | 0.8811 | 0.7461 | -0.1350 |

The diagnostic result is that the learned mean weights are uniform across
layers 22 through 28. Under this heuristic, the late final-token residual
classifiers look redundant rather than complementary. The next CIFT step should
therefore move past simple final-token layer weighting toward a richer CFS-like
head with out-of-fold per-layer scores and readout positions beyond the final
prompt token.

A first out-of-fold CIFT meta-head now trains per-source residual classifiers
inside inner grouped folds, uses their out-of-fold exfiltration-risk scores as
meta-head features, and compares final-token last-quarter sources against an
expanded final-token plus mean-pool source set. The expanded source set is the
best CIFT-like variant on every checkpoint, and it is the first CIFT-like
checkpoint to beat the combined static feature on any registered dataset:

| Dataset | Combined Macro F1 | Best Meta-Head Macro F1 | Delta Macro F1 | Winner |
|---|---:|---:|---:|---|
| Baseline | 0.8804 | 0.8598 | -0.0206 | combined static |
| Hard V1 | 0.9331 | 0.8788 | -0.0543 | combined static |
| Hard V2 | 0.9657 | 0.8823 | -0.0834 | combined static |
| Hard V3 | 0.8811 | 0.9156 | +0.0345 | OOF meta-head |

This is not a promotion result because the combined static feature still wins
three of four checkpoints, including Hard V2. It is useful because mean-pool
readout expansion improves every checkpoint relative to final-token-only CIFT
meta-learning: 0.8841 mean macro F1 versus 0.8209. The next CIFT work should
analyze the Hard V3 win and Hard V2 deficit, then test calibration/readout
variants before changing the monitor contract.

The Hard V2/Hard V3 residual analysis now explains that split. Across those two
checkpoints, the OOF meta-head has 12 errors versus 9 for the combined static
feature, with 3 fixed, 6 persistent, and 6 introduced errors. Hard V2 is the
blocking regression: 0 fixed errors and 5 introduced errors. Hard V3 is the
encouraging case: 3 fixed errors and 1 introduced error.

| Dataset | Reference Errors | Meta-Head Errors | Fixed | Persistent | Introduced |
|---|---:|---:|---:|---:|---:|
| Hard V2 | 2 | 7 | 0 | 2 | 5 |
| Hard V3 | 7 | 5 | 3 | 4 | 1 |

The Hard V3 win comes mainly from fixing two
`hard_v3_exfil_tool_payload_forward` misses and one
`hard_v3_safe_policy_note_category` miss. The Hard V2 deficit comes from
introduced misses in broker, output-contract, policy-exception, broker-boundary,
and tool-argument families. The next CIFT step should target those introduced
Hard V2 families while preserving the Hard V3 fixed cases.

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

Run the combined feature stability check:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_feature_stability.py
```

Run the combined feature residual suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_combined_residual_suite.py
```

Run the first CIFT-like calibrated-deviation comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_probe.py
```

Run the latest CIFT-like ablation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_ablation.py
```

Run the CIFT layer-weighted head comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_layer_head.py
```

Run the CIFT out-of-fold meta-head comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_meta_head.py
```

Run the CIFT out-of-fold meta-head residual analysis:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/analyze_cift_meta_residuals.py
```

Build the Hard V3 combined-regression adjudication worksheet:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/adjudicate_combined_hard_v3_regressions.py
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
- `data/reports/feature_stability_reference_l11_l16_summary.md`
- `data/reports/feature_stability_progress_2026-06-19.md`
- `data/reports/feature_stability_combined_l11_l16_summary.md`
- `data/reports/feature_stability_combined_progress_2026-06-19.md`
- `data/reports/combined_feature_residual_suite_summary.md`
- `data/reports/combined_feature_residual_progress_2026-06-19.md`
- `data/reports/hard_v3_combined_regression_adjudication_summary.md`
- `data/reports/project_aligned_workflow_2026-06-19.md`
- `data/reports/cift_like_probe_comparison_summary.md`
- `data/reports/cift_like_probe_progress_2026-06-19.md`
- `data/reports/cift_like_ablation_v2_summary.md`
- `data/reports/cift_like_ablation_v2_progress_2026-06-19.md`
- `data/reports/cift_like_ablation_v3_summary.md`
- `data/reports/cift_like_ablation_v3_progress_2026-06-19.md`
- `data/reports/cift_layer_weighted_head_v1_summary.md`
- `data/reports/cift_layer_weighted_head_v1_progress_2026-06-19.md`
- `data/reports/cift_meta_head_v1_summary.md`
- `data/reports/cift_meta_head_v1_progress_2026-06-19.md`
- `data/reports/cift_meta_head_residual_suite_v1_summary.md`
- `data/reports/cift_meta_head_residual_progress_2026-06-19.md`

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
- `data/reports/feature_stability_reference_l11_l16.json`
- `data/reports/feature_stability_combined_l11_l16.json`
- `data/reports/combined_feature_residual_suite.json`
- `data/reports/hard_v3_combined_regression_adjudication.json`
- `data/reports/cift_like_probe_comparison.json`
- `data/reports/cift_like_ablation_v2.json`
- `data/reports/cift_like_ablation_v3.json`
- `data/reports/cift_layer_weighted_head_v1.json`
- `data/reports/cift_meta_head_v1.json`
- `data/reports/cift_meta_head_residual_suite_v1.json`

## Next Moves

The next experimental steps split into two related threads: completing the
static-feature promotion review, and refining the CIFT-like path into a closer
implementation of the paper method. Do not silently promote the combined
feature just because it improves aggregate performance, and do not discard CIFT
because the first simplified calibrated-distance score underperformed.

Recommended sequence:

1. Human-review the three Hard V3 combined-regression cases.
2. Decide whether each case is a true model regression, a defensible tradeoff,
   or a dataset/label ambiguity.
3. Keep `mean_pool_layer_18` as the fixed regression checkpoint while
   final-token candidates remain under evaluation.
4. Define a promotion rule that weighs average performance, worst-case
   checkpoint performance, and post-hoc discovery risk.
5. For CIFT-like work, run targeted calibration/readout ablations against the
   Hard V2 introduced-error families while preserving Hard V3 fixed cases.
6. Keep registering every dataset, artifact, and machine-readable report in
   `data/lineage.json`.

The research question remains narrow and concrete:

```text
Does the activation signal still help when surface text cues become less generous?
```
