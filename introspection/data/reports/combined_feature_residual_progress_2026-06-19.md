# Combined Feature Residual Progress

## Question

Does `concat(final_token_layer_11,final_token_layer_16)` reduce residual errors
against the fixed historical reference and both single-layer candidate features?

## Method

The suite compares the combined feature against three references:

- `mean_pool_layer_18`
- `final_token_layer_11`
- `final_token_layer_16`

Each comparison uses grouped cross-validation on `safe_secret_vs_exfiltration`
across the four registered checkpoint datasets. The combined feature is derived
at evaluation time from existing activation artifacts.

## Aggregate Result

| Reference Feature | Comparisons | Reference Errors | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mean_pool_layer_18` | 4 | 41 | 20 | 30 | 11 | 9 | -21 |
| `final_token_layer_11` | 4 | 24 | 20 | 6 | 18 | 2 | -4 |
| `final_token_layer_16` | 4 | 26 | 20 | 9 | 17 | 3 | -6 |

Negative net error delta means the combined feature has fewer total errors than
the reference feature across the same comparisons.

## Dataset Caveat

The aggregate result favors the combined feature, but Hard V3 still favors
`final_token_layer_16` locally:

| Dataset | Reference | Reference Errors | Combined Errors | Fixed | Persistent | Introduced |
|---|---|---:|---:|---:|---:|---:|
| `hard_prompts_v3` | `final_token_layer_16` | 4 | 7 | 0 | 4 | 3 |

The introduced Hard V3 families are:

- `hard_v3_exfil_log_line_copy`
- `hard_v3_safe_operator_message_placeholder`
- `hard_v3_safe_policy_note_category`

## Interpretation

The combined feature is now the leading promotion candidate because it improves
aggregate residual behavior against all three references. It should still go
through human review before promotion because its Hard V3 regression against
`final_token_layer_16` is concentrated in safety-relevant prompt families.

The next step is to define a feature-selection rule that explicitly balances
aggregate error reduction, worst-checkpoint behavior, and post-hoc discovery
risk.
