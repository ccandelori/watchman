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
| `word_tfidf` | `word_tfidf_1_2` | 0.9341 | 0.9444 | 0.0439 | 0.0351 |

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
[28, 2]
[0, 60]
```

## safe_secret_vs_exfiltration

Classify safe secret handling against exfiltration-oriented secret handling.

Labels: `exfiltration_intent, secret_present_safe`

| Method | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---|---|---:|---:|---:|---:|
| `activation_probe` | `mean_pool_layer_18` | 0.9163 | 0.9167 | 0.0527 | 0.0527 |
| `word_tfidf` | `word_tfidf_1_2` | 0.5572 | 0.5667 | 0.1023 | 0.0972 |
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.5329 | 0.5500 | 0.1934 | 0.1871 |

Confusion matrices:

### safe_secret_vs_exfiltration / activation_probe

```text
[28, 2]
[3, 27]
```

### safe_secret_vs_exfiltration / word_tfidf

```text
[17, 13]
[13, 17]
```

### safe_secret_vs_exfiltration / char_tfidf

```text
[17, 13]
[14, 16]
```
