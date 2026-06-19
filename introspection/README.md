# Aegis Introspection

This directory contains the current model-introspection research thread for
Aegis. The purpose is to test whether hidden-state features from a local model
can help distinguish safe secret handling from exfiltration-oriented behavior.

The work here is exploratory. The current results are useful checkpoints, not
validated production claims.

## Current State

Two datasets are currently registered in `data/lineage.json`:

| Dataset | Purpose | Rows |
|---|---|---:|
| `baseline_prompts_v1` | First hand-authored checkpoint dataset. | 90 |
| `hard_prompts_v1` | Harder contrastive successor to the baseline. | 90 |

Both datasets use the same label shape:

| Label | Count | Meaning |
|---|---:|---|
| `benign` | 30 | No secret handling or exfiltration request. |
| `secret_present_safe` | 30 | A secret-like value appears, but the prompt asks for safe handling. |
| `exfiltration_intent` | 30 | The prompt asks to reveal, transmit, encode, route, or leak secret-like material. |

Each prompt has a `family` field. Grouped evaluation uses those families to
hold related prompt patterns out together.

The strongest current activation feature is:

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

The grouped results are the more credible checkpoints because they reduce
prompt-family leakage between train and test folds. Hard V1 is especially
useful because both TF-IDF baselines degrade sharply on the intent-sensitive
task while the activation probe remains comparatively strong.

Current grouped error analysis for Hard V1 shows the activation probe's
remaining `safe_secret_vs_exfiltration` errors are concentrated in a small set
of contrast families: `hard_safe_output_contract` is the main weak family, with
single-error clusters around broker impersonation, output contract abuse,
policy override, safe summaries, and tool-argument review. The full
machine-readable prediction ledger is registered in lineage.

## Project Layout

```text
introspection/
├── data/
│   ├── activations/      # Serialized activation feature artifacts
│   ├── probes/           # Reserved for trained probe artifacts
│   ├── reports/          # JSON reports and narrative progress notes
│   ├── lineage.json      # Canonical experiment ledger
│   ├── prompts.jsonl      # Baseline prompt dataset
│   └── prompts_hard.jsonl # Hard Baseline V1 dataset
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
data/activations/qwen3_0_6b_hard_all_layers.pt
data/reports/binary_tasks_hard_grouped.json
```

Future datasets should use new names, for example:

```text
data/prompts_hard_v2.jsonl
data/prompts_tool_calls.jsonl
data/activations/qwen3_0_6b_hard_v2_all_layers.pt
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

## Reports

Key human-readable checkpoints:

- `data/reports/activation_probe_progress_2026-06-18.md`
- `data/reports/binary_probe_progress_2026-06-18.md`
- `data/reports/grouped_binary_probe_progress_2026-06-18.md`
- `data/reports/hard_baseline_probe_progress_2026-06-18.md`
- `data/reports/probe_all_layers_summary.md`
- `data/reports/binary_tasks_summary.md`
- `data/reports/binary_tasks_grouped_summary.md`
- `data/reports/binary_tasks_hard_summary.md`
- `data/reports/binary_tasks_hard_grouped_summary.md`
- `data/reports/binary_error_analysis_hard_grouped_summary.md`

Key machine-readable reports registered in lineage:

- `data/reports/probe_baseline.json`
- `data/reports/text_baseline.json`
- `data/reports/probe_all_layers.json`
- `data/reports/binary_tasks.json`
- `data/reports/binary_tasks_grouped.json`
- `data/reports/binary_tasks_hard.json`
- `data/reports/binary_tasks_hard_grouped.json`
- `data/reports/binary_error_analysis_hard_grouped.json`

## Next Moves

The next experimental pressure test is targeted dataset expansion, not a more
elaborate model.

Recommended sequence:

1. Generate a comparison summary across baseline and hard checkpoints.
2. Decide whether Hard Baseline V2 should expand weak families or split
   structured tool-call examples into a separate dataset.
3. Expand the currently weak contrast families, especially output contracts and
   tool-argument review.
4. Keep registering every dataset, artifact, and machine-readable report in
   `data/lineage.json`.

The research question remains narrow and concrete:

```text
Does the activation signal still help when surface text cues become less generous?
```
