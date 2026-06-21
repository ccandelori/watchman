# Binary Task Evaluation Summary

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Activation feature: `mean_pool_layer_18`
- Fold count: `5`

## benign_vs_secret_related

Classify benign prompts against any prompt involving secret-like material.

Labels: `benign, secret_related`

| Method | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---|---|---:|---:|---:|---:|
| `activation_probe` | `mean_pool_layer_18` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.9345 | 0.9444 | 0.0717 | 0.0609 |
| `word_tfidf` | `word_tfidf_1_2` | 0.9342 | 0.9444 | 0.0591 | 0.0497 |

Confusion matrices:

### benign_vs_secret_related / activation_probe

```text
[30, 0]
[0, 60]
```

### benign_vs_secret_related / word_tfidf

```text
[26, 4]
[1, 59]
```

### benign_vs_secret_related / char_tfidf

```text
[26, 4]
[1, 59]
```

## safe_secret_vs_exfiltration

Classify safe secret handling against exfiltration-oriented secret handling.

Labels: `exfiltration_intent, secret_present_safe`

| Method | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---|---|---:|---:|---:|---:|
| `activation_probe` | `mean_pool_layer_18` | 0.8620 | 0.8667 | 0.0727 | 0.0667 |
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.7649 | 0.7833 | 0.2085 | 0.1871 |
| `word_tfidf` | `word_tfidf_1_2` | 0.7054 | 0.7167 | 0.1072 | 0.1000 |

Confusion matrices:

### safe_secret_vs_exfiltration / activation_probe

```text
[26, 4]
[4, 26]
```

### safe_secret_vs_exfiltration / word_tfidf

```text
[18, 12]
[5, 25]
```

### safe_secret_vs_exfiltration / char_tfidf

```text
[21, 9]
[4, 26]
```
