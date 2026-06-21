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
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.8569 | 0.8875 | 0.0550 | 0.0386 |

Confusion matrices:

### benign_vs_secret_related / activation_probe

```text
[80, 0]
[0, 160]
```

### benign_vs_secret_related / word_tfidf

```text
[80, 0]
[0, 160]
```

### benign_vs_secret_related / char_tfidf

```text
[53, 27]
[0, 160]
```

## safe_secret_vs_exfiltration

Classify safe secret handling against exfiltration-oriented secret handling.

Labels: `exfiltration_intent, secret_present_safe`

| Method | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---|---|---:|---:|---:|---:|
| `activation_probe` | `readout_window_layer_15` | 0.7736 | 0.7812 | 0.0709 | 0.0656 |
| `char_tfidf` | `char_wb_tfidf_3_5` | 0.3542 | 0.3625 | 0.1758 | 0.1818 |
| `word_tfidf` | `word_tfidf_1_2` | 0.2670 | 0.3000 | 0.2023 | 0.2206 |

Confusion matrices:

### safe_secret_vs_exfiltration / activation_probe

```text
[59, 21]
[14, 66]
```

### safe_secret_vs_exfiltration / word_tfidf

```text
[33, 47]
[65, 15]
```

### safe_secret_vs_exfiltration / char_tfidf

```text
[31, 49]
[53, 27]
```
