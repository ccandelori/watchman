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
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.8037 | 0.8583 | 0.1222 | 0.0816 |

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
[25, 23]
[0, 96]
```

## safe_secret_vs_exfiltration

Classify safe secret handling against exfiltration-oriented secret handling.

Labels: `exfiltration_intent, secret_present_safe`

| Method | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---|---|---:|---:|---:|---:|
| `activation_probe` | `readout_window_layer_15` | 0.7646 | 0.7875 | 0.1607 | 0.1225 |
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.2851 | 0.3063 | 0.0784 | 0.0848 |
| `word_tfidf` | `word_tfidf_1_2` | 0.1172 | 0.1187 | 0.2062 | 0.2077 |

Confusion matrices:

### safe_secret_vs_exfiltration / activation_probe

```text
[34, 14]
[6, 42]
```

### safe_secret_vs_exfiltration / word_tfidf

```text
[8, 40]
[38, 10]
```

### safe_secret_vs_exfiltration / char_tfidf

```text
[20, 28]
[38, 10]
```
