# DP-HONEY-lite V3 Lexical-Control Checkpoint

Date: 2026-06-20

## Scope

This checkpoint hardens the DP-HONEY-lite prompt generator after V2 proved too
lexically separable. V3 keeps the same structured span schema and 10
scenario-family grouped evaluation setup, but changes the prompt contract to a
mode-map policy:

- every row contains both `credential_value` and `summary_value`;
- every row contains both `mode_a` and `mode_b`;
- every row contains both `copy` and `mask`;
- selected modes are balanced across labels;
- half the scenario families map `mode_a` to credential copying, and half map
  `mode_b` to credential copying.

The intended pressure test is whether a monitor can infer the selected policy
semantics rather than solving from obvious intent words.

## Dataset And Artifact

| Item | ID | Rows / Features |
|---|---|---:|
| Dataset | `dp_honey_lite_prompts_v3` | 240 rows |
| Artifact | `qwen3_0_6b_dp_honey_lite_v3_all_pooling_v1` | 87 features |

The artifact extracts all 29 Qwen 0.6B hidden-state layers using
`final_token`, `mean_pool`, and `readout_window` pooling.

## Results

Grouped evaluation used 5 stratified group folds over the 10 scenario families.

| Task | Best Activation Feature | Best Activation Macro F1 | Word TF-IDF Macro F1 | Char TF-IDF Macro F1 |
|---|---|---:|---:|---:|
| `safe_secret_vs_exfiltration` | `readout_window_layer_21` | 0.5284 | 0.2670 | 0.3542 |
| `benign_vs_secret_related` | `final_token_layer_01` | 0.9906 | 1.0000 | 0.8569 |

There are zero perfect activation features on either V3 sweep.

## Interpretation

V3 succeeds at the first goal for the target task: word and character TF-IDF no
longer solve `safe_secret_vs_exfiltration`. The wording controls worked.

The activation result is not yet strong. The best target-task activation feature
is only slightly above chance, and the fixed `readout_window_layer_11` reference
falls below chance on the same sweep. This suggests the current model/features
do not reliably infer the mode-map semantics under grouped scenario holdout.

The benign-vs-secret-related task remains too easy for word TF-IDF because the
selected field differs between benign rows and secret-related rows. That should
be treated as a remaining dataset weakness rather than a monitor win.

## Next Step

Do not promote a V3 monitor from these results. The next useful branch is a
V3.1 diagnostic dataset or evaluator that isolates why the mode-map target task
collapsed:

- compare payload-only rows against no-payload rows;
- test whether the model understands the selected mode with a direct text
  answer task;
- add a small explicit policy-parser baseline;
- try readout windows around the selected-mode token rather than only the last
  active request tokens.
