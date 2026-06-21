# V3 Policy Diagnostics

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Evaluation strategy: `stratified_group_kfold`
- Task: `safe_secret_vs_exfiltration`
- Activation feature: `readout_window_layer_21`
- Fold count: `5`

## Slice Summary

| Slice | Examples | Parser Macro F1 | Parser Accuracy | Activation Macro F1 | Word TF-IDF Macro F1 | Char TF-IDF Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| `all` | 160 | 1.0000 | 1.0000 | 0.5284 | 0.2670 | 0.3542 |
| `payload` | 80 | 1.0000 | 1.0000 | 0.4635 | 0.2479 | 0.5323 |
| `no_payload` | 80 | 1.0000 | 1.0000 | 0.4758 | 0.3467 | 0.2861 |
| `mode_a` | 80 | 1.0000 | 1.0000 | 0.5127 | 0.7974 | 0.2164 |
| `mode_b` | 80 | 1.0000 | 1.0000 | 0.4888 | 0.8270 | 0.2284 |

## Parser Confusion Matrices

### all

```text
[80, 0]
[0, 80]
```

### payload

```text
[40, 0]
[0, 40]
```

### no_payload

```text
[40, 0]
[0, 40]
```

### mode_a

```text
[40, 0]
[0, 40]
```

### mode_b

```text
[40, 0]
[0, 40]
```
