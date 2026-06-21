# Baseline vs. Hard V1 Comparison

## Purpose

This note compares the baseline prompt dataset against Hard Baseline V1 using
the grouped binary-task reports. Grouped cross-validation is the relevant
checkpoint because it holds related prompt families out together and therefore
reduces prompt-family leakage.

The question is narrow:

```text
Does the activation signal still help when surface text cues become less generous?
```

## Summary

Hard V1 increased pressure on the intent-sensitive
`safe_secret_vs_exfiltration` task without breaking the activation probe. The
activation probe improved slightly under grouped evaluation, while both TF-IDF
baselines degraded sharply.

| Dataset | Task | Method | Macro F1 | Accuracy |
|---|---|---|---:|---:|
| Baseline | `safe_secret_vs_exfiltration` | `activation_probe` | 0.8620 | 0.8667 |
| Baseline | `safe_secret_vs_exfiltration` | `word_tfidf` | 0.7054 | 0.7167 |
| Baseline | `safe_secret_vs_exfiltration` | `char_tfidf` | 0.7649 | 0.7833 |
| Hard V1 | `safe_secret_vs_exfiltration` | `activation_probe` | 0.8788 | 0.8833 |
| Hard V1 | `safe_secret_vs_exfiltration` | `word_tfidf` | 0.5726 | 0.5833 |
| Hard V1 | `safe_secret_vs_exfiltration` | `char_tfidf` | 0.5571 | 0.5667 |

The broader `benign_vs_secret_related` task remained easy for the activation
probe on both datasets:

| Dataset | Task | Method | Macro F1 | Accuracy |
|---|---|---|---:|---:|
| Baseline | `benign_vs_secret_related` | `activation_probe` | 1.0000 | 1.0000 |
| Hard V1 | `benign_vs_secret_related` | `activation_probe` | 1.0000 | 1.0000 |

## Interpretation

The result is useful but still preliminary. It does not show that the activation
probe is a production detector. It does show that, for this local Qwen 0.6B
checkpoint and these hand-authored datasets, mid-layer mean-pooled activations
retain an intent signal that simple lexical baselines struggle to recover.

The key point is not only that the activation probe outperformed the text
baselines. The more important point is that Hard V1 made the text baselines
worse on the exact task where surface cues should be less reliable:
distinguishing safe secret handling from exfiltration-oriented handling.

## Error Clusters

The Hard V1 grouped error-analysis report shows the activation probe made
`7/60` errors on `safe_secret_vs_exfiltration`. Those errors were concentrated
in six contrast families:

| Family | True Label | Errors | Examples | Accuracy |
|---|---|---:|---:|---:|
| `hard_safe_output_contract` | `secret_present_safe` | 2 | 3 | 0.3333 |
| `hard_exfil_broker_impersonation` | `exfiltration_intent` | 1 | 3 | 0.6667 |
| `hard_exfil_output_contract_abuse` | `exfiltration_intent` | 1 | 3 | 0.6667 |
| `hard_exfil_policy_override` | `exfiltration_intent` | 1 | 3 | 0.6667 |
| `hard_safe_summary_without_secret` | `secret_present_safe` | 1 | 3 | 0.6667 |
| `hard_safe_tool_argument_review` | `secret_present_safe` | 1 | 3 | 0.6667 |

This clustering is the main design signal for Hard Baseline V2.

## Next Dataset Move

Hard V2 should not broaden the dataset randomly. It should expand the weak
families while preserving the same label shape and grouped-evaluation protocol.

Recommended Hard V2 targets:

1. Add more safe and exfiltration examples around output contracts.
2. Add more tool-argument review and tool-argument injection pairs.
3. Add more broker-boundary and broker-impersonation pairs.
4. Add more policy-classification and policy-override pairs.
5. Add more summary-without-secret and summary-with-secret pairs.

The probe should remain unchanged for this step. Changing the probe now would
make it harder to tell whether the next result comes from a better detector or
from a harder, more honest dataset.
