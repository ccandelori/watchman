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
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.9357 | 0.9556 | 0.1286 | 0.0889 |
| `word_tfidf` | `word_tfidf_1_2` | 0.9226 | 0.9444 | 0.1246 | 0.0861 |

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
[26, 4]
[0, 60]
```

## safe_secret_vs_exfiltration

Classify safe secret handling against exfiltration-oriented secret handling.

Labels: `exfiltration_intent, secret_present_safe`

| Method | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---|---|---:|---:|---:|---:|
| `activation_probe` | `mean_pool_layer_18` | 0.8788 | 0.8833 | 0.0907 | 0.0850 |
| `word_tfidf` | `word_tfidf_1_2` | 0.5726 | 0.5833 | 0.1407 | 0.1394 |
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.5571 | 0.5667 | 0.2080 | 0.2000 |

Confusion matrices:

### safe_secret_vs_exfiltration / activation_probe

```text
[27, 3]
[4, 26]
```

### safe_secret_vs_exfiltration / word_tfidf

```text
[16, 14]
[11, 19]
```

### safe_secret_vs_exfiltration / char_tfidf

```text
[16, 14]
[12, 18]
```
