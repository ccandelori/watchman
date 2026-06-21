# Hard V3 Held-Out Validation Progress - 2026-06-19

## Purpose

Hard V3 is a held-out checkpoint focused on the boundary left open by Hard V2:
prompts that contain a secret-like value but ask for safe summary, redaction, or
replacement, contrasted with prompts that ask to reproduce or forward the value.

The committed dataset uses neutral secret identifiers such as `hv3-401` rather
than label-coded values. This matters because label-coded secret strings can
create artificial separability.

## Dataset Shape

Hard V3 keeps the same balanced shape as the prior checkpoints:

| Label | Rows | Families | Rows per Family |
|---|---:|---:|---:|
| `benign` | 30 | 10 | 3 |
| `secret_present_safe` | 30 | 10 | 3 |
| `exfiltration_intent` | 30 | 10 | 3 |

The target families emphasize summary, replacement, redaction, customer notes,
support tickets, audit excerpts, logs, tool payloads, policy notes, and operator
messages.

## Candidate Crosscheck

On Hard V3, the candidate feature `final_token_layer_11` beats the fixed
historical feature `mean_pool_layer_18` on `safe_secret_vs_exfiltration`.

| Feature | Macro F1 | Accuracy | Errors |
|---|---:|---:|---:|
| `mean_pool_layer_18` | 0.8324 | 0.8333 | 10 |
| `final_token_layer_11` | 0.8818 | 0.8833 | 7 |

Across all four checkpoints, the candidate now wins three and the reference wins
one:

| Dataset | Reference Macro F1 | Candidate Macro F1 | Delta Macro F1 | Winner |
|---|---:|---:|---:|---|
| `baseline_prompts_v1` | 0.8620 | 0.8445 | -0.0175 | `mean_pool_layer_18` |
| `hard_prompts_v1` | 0.8788 | 0.8993 | +0.0205 | `final_token_layer_11` |
| `hard_prompts_v2` | 0.7225 | 0.9657 | +0.2432 | `final_token_layer_11` |
| `hard_prompts_v3` | 0.8324 | 0.8818 | +0.0494 | `final_token_layer_11` |

## Residual Comparison

The candidate improves V3 overall but still changes the error profile:

| Reference Errors | Candidate Errors | Fixed Errors | Persistent Errors | Introduced Errors |
|---:|---:|---:|---:|---:|
| 10 | 7 | 7 | 3 | 4 |

The candidate fixes seven reference errors, keeps three errors, and introduces
four new errors. The introduced errors are all safe examples classified as
exfiltration: incident report omission, operator placeholder, policy note
category, and remediation sentinel replacement.

## Layer Sweep

The V3 sweep ranks `final_token_layer_16` first at 0.9321 macro F1 / 0.9333
accuracy. `final_token_layer_11` ranks fifth at 0.8818 macro F1 / 0.8833
accuracy. The fixed reference `mean_pool_layer_18` ranks eighteenth at 0.8324
macro F1 / 0.8333 accuracy.

This means `final_token_layer_11` remains a credible candidate, but feature
stability is not settled. V3 supports the candidate over the historical
reference, while also showing that a nearby final-token feature can outperform
it on this checkpoint.

## Interpretation

The candidate has survived a fresh held-out checkpoint after correcting for
label leakage. That is meaningful evidence. It is still not enough to promote
the candidate as a universal detector feature because the original baseline
remains a small regression and the V3 layer sweep found a different local winner.

The next step should be a cross-check of `final_token_layer_16` against the same
four checkpoints, followed by a policy for selecting features without tuning on
the newest dataset.
