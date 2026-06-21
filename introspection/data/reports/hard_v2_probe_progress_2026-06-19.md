# Hard Baseline V2 Probe Progress

## Purpose

Hard Baseline V2 was created after Hard V1 error analysis showed activation
misses clustered around output contracts, tool-argument review, broker
impersonation, policy override, and safe summaries. V2 keeps the same dataset
shape as prior checkpoints: 90 prompts, 30 examples per label, 30 prompt
families, and three examples per family.

The probe was not changed. Evaluation still uses `mean_pool_layer_18` from
Qwen/Qwen3-0.6B.

## Key Result

Hard V2 is a stronger pressure test than Hard V1 for the intent-sensitive
`safe_secret_vs_exfiltration` task. Under grouped cross-validation:

| Dataset | Method | Macro F1 | Accuracy |
|---|---|---:|---:|
| Hard V1 | `activation_probe` | 0.8788 | 0.8833 |
| Hard V1 | `word_tfidf` | 0.5726 | 0.5833 |
| Hard V1 | `char_tfidf` | 0.5571 | 0.5667 |
| Hard V2 | `activation_probe` | 0.7225 | 0.7333 |
| Hard V2 | `word_tfidf` | 0.2047 | 0.2167 |
| Hard V2 | `char_tfidf` | 0.2679 | 0.2833 |

The fixed activation probe degraded, which is expected for a harder contrast
set, but it still substantially outperformed both text baselines. The text
baselines fell below chance, indicating that V2 disrupts simple lexical
separation rather than merely adding more examples.

## Confusion Matrix

For `safe_secret_vs_exfiltration`, labels are ordered as
`exfiltration_intent, secret_present_safe`.

```text
activation_probe
[21, 9]
[7, 23]

word_tfidf
[5, 25]
[22, 8]

char_tfidf
[8, 22]
[21, 9]
```

## Error Clusters

The activation probe made `16/60` errors on the grouped
`safe_secret_vs_exfiltration` task. The largest error clusters were:

| Family | True Label | Errors | Examples | Accuracy |
|---|---|---:|---:|---:|
| `hard_v2_exfil_output_contract_evidence` | `exfiltration_intent` | 2 | 3 | 0.3333 |
| `hard_v2_exfil_output_contract_notes_field` | `exfiltration_intent` | 2 | 3 | 0.3333 |
| `hard_v2_exfil_summary_customer_note_leak` | `exfiltration_intent` | 2 | 3 | 0.3333 |
| `hard_v2_safe_output_contract_limited_fields` | `secret_present_safe` | 2 | 3 | 0.3333 |
| `hard_v2_safe_summary_customer_note` | `secret_present_safe` | 2 | 3 | 0.3333 |

Output-contract and summary variants remain the main weak points. Broker,
policy, and tool-argument cases also contributed single-family errors but did
not dominate the result.

## Interpretation

This is a useful regression-style checkpoint, not a production claim. Hard V2
shows that the current fixed activation probe still carries useful signal when
lexical baselines fail badly, but it also exposes clear weaknesses in highly
paired output-contract and summary cases.

The next move should not be probe tuning yet. The next useful step is to inspect
the per-example V2 prediction ledger and decide whether the confusing pairs are
valid hard cases or artifacts of prompt wording.
