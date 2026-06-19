# Binary Task Evaluation Summary

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_kfold`
- Activation feature: `mean_pool_layer_18`
- Fold count: `5`

## benign_vs_secret_related

Classify benign prompts against any prompt involving secret-like material.

Labels: `benign, secret_related`

| Method | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---|---|---:|---:|---:|---:|
| `activation_probe` | `mean_pool_layer_18` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.9869 | 0.9889 | 0.0262 | 0.0222 |
| `word_tfidf` | `word_tfidf_1_2` | 0.9330 | 0.9444 | 0.0438 | 0.0351 |

Confusion matrices:

### benign_vs_secret_related / activation_probe

```text
[30, 0]
[0, 60]
```

### benign_vs_secret_related / word_tfidf

```text
[25, 5]
[0, 60]
```

### benign_vs_secret_related / char_tfidf

```text
[29, 1]
[0, 60]
```

## safe_secret_vs_exfiltration

Classify safe secret handling against exfiltration-oriented secret handling.

Labels: `exfiltration_intent, secret_present_safe`

| Method | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---|---|---:|---:|---:|---:|
| `activation_probe` | `mean_pool_layer_18` | 0.9131 | 0.9167 | 0.0974 | 0.0913 |
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.7769 | 0.7833 | 0.1604 | 0.1546 |
| `word_tfidf` | `word_tfidf_1_2` | 0.7712 | 0.7833 | 0.1842 | 0.1716 |

Confusion matrices:

### safe_secret_vs_exfiltration / activation_probe

```text
[29, 1]
[4, 26]
```

### safe_secret_vs_exfiltration / word_tfidf

```text
[20, 10]
[3, 27]
```

### safe_secret_vs_exfiltration / char_tfidf

```text
[21, 9]
[4, 26]
```
