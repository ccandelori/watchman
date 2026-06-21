# DP-HONEY-lite V2 Activation Checkpoint

Date: 2026-06-20

## Scope

This checkpoint advances the DP-HONEY-lite and CIFT-like integration path. It
uses the hard V2 proxy-shaped prompt dataset to extract activations from
readout windows and static prompt positions in the same artifact.

It does not implement full DP-HONEY. The honeytokens remain deterministic,
format-shaped test values rather than differentially private n-gram samples.

## Artifact

| Artifact | Dataset | Features | Rows |
|---|---|---:|---:|
| `qwen3_0_6b_dp_honey_lite_v2_all_pooling_v1` | `dp_honey_lite_prompts_v2` | 87 | 240 |

The artifact contains all 29 Qwen 0.6B hidden-state layers for:

- `final_token`
- `mean_pool`
- `readout_window`

This lets readout-window features and same-dataset static features be compared
without rerunning extraction.

## Results

Grouped evaluation used 5 stratified group folds over the 10 scenario families.

| Task | Best Feature | Best Macro F1 | Readout Layer 11 Macro F1 | Perfect Feature Count |
|---|---|---:|---:|---:|
| `safe_secret_vs_exfiltration` | `final_token_layer_04` | 1.0000 | 1.0000 | 54 |
| `benign_vs_secret_related` | `final_token_layer_02` | 1.0000 | 1.0000 | 69 |

The grouped binary report with `readout_window_layer_11` also compared text
baselines:

| Task | Activation Macro F1 | Word TF-IDF Macro F1 | Char TF-IDF Macro F1 |
|---|---:|---:|---:|
| `safe_secret_vs_exfiltration` | 1.0000 | 1.0000 | 0.9492 |
| `benign_vs_secret_related` | 1.0000 | 1.0000 | 0.9822 |

## Interpretation

The extraction and grouped evaluation pipeline is working on V2. The result is
not evidence that readout windows outperform static or text features. Word
TF-IDF is also perfect on both tasks, and dozens of activation features reach
perfect macro F1.

The likely explanation is that V2 is still lexically and structurally too easy:
the paired families are balanced, but the label intent wording remains
separable enough for a surface text baseline. This is useful because it catches
an evaluation weakness before we make a detector claim.

## Next Step

Build DP-HONEY-lite V3 as an adversarial lexical-control dataset:

- keep the same structured span schema;
- keep scenario-level grouped folds;
- make safe and exfil variants share more verbs, nouns, and payload shape;
- move the label distinction into policy semantics rather than obvious words;
- add a baseline-gating rule: a dataset is not promotion-grade if word TF-IDF is
  perfect or near-perfect.
