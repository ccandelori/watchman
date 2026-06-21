# Binary Task Evaluation Summary

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Activation feature: `readout_window_layer_15`
- Fold count: `5`

## benign_vs_secret_related

Classify benign prompts against any prompt involving secret-like material.

Labels: `benign, secret_related`

| Method | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---|---|---:|---:|---:|---:|
| `activation_probe` | `readout_window_layer_15` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `word_tfidf` | `word_tfidf_1_2` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.8149 | 0.8583 | 0.0489 | 0.0333 |

Confusion matrices:

### benign_vs_secret_related / activation_probe

```text
[48, 0]
[0, 96]
```

### benign_vs_secret_related / word_tfidf

```text
[48, 0]
[0, 96]
```

### benign_vs_secret_related / char_tfidf

```text
[27, 21]
[0, 96]
```

## safe_secret_vs_exfiltration

Classify safe secret handling against exfiltration-oriented secret handling.

Labels: `exfiltration_intent, secret_present_safe`

| Method | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---|---|---:|---:|---:|---:|
| `activation_probe` | `readout_window_layer_15` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `word_tfidf` | `word_tfidf_1_2` | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.7363 | 0.7375 | 0.0250 | 0.0250 |

Confusion matrices:

### safe_secret_vs_exfiltration / activation_probe

```text
[48, 0]
[0, 48]
```

### safe_secret_vs_exfiltration / word_tfidf

```text
[48, 0]
[0, 48]
```

### safe_secret_vs_exfiltration / char_tfidf

```text
[38, 10]
[15, 33]
```
