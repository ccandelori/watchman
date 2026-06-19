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
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.9738 | 0.9778 | 0.0321 | 0.0272 |
| `word_tfidf` | `word_tfidf_1_2` | 0.9607 | 0.9667 | 0.0321 | 0.0272 |

Confusion matrices:

### benign_vs_secret_related / activation_probe

```text
[30, 0]
[0, 60]
```

### benign_vs_secret_related / word_tfidf

```text
[27, 3]
[0, 60]
```

### benign_vs_secret_related / char_tfidf

```text
[28, 2]
[0, 60]
```

## safe_secret_vs_exfiltration

Classify safe secret handling against exfiltration-oriented secret handling.

Labels: `exfiltration_intent, secret_present_safe`

| Method | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---|---|---:|---:|---:|---:|
| `activation_probe` | `mean_pool_layer_18` | 0.7470 | 0.7500 | 0.1199 | 0.1179 |
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.2733 | 0.2833 | 0.1319 | 0.1354 |
| `word_tfidf` | `word_tfidf_1_2` | 0.2732 | 0.2833 | 0.1072 | 0.1130 |

Confusion matrices:

### safe_secret_vs_exfiltration / activation_probe

```text
[22, 8]
[7, 23]
```

### safe_secret_vs_exfiltration / word_tfidf

```text
[9, 21]
[22, 8]
```

### safe_secret_vs_exfiltration / char_tfidf

```text
[8, 22]
[21, 9]
```
