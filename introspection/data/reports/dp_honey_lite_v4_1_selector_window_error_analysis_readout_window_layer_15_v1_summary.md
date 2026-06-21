# Binary Error Analysis

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

| Method | Accuracy | Errors | Predictions |
|---|---:|---:|---:|
| `activation_probe` | 1.0000 | 0 | 144 |
| `word_tfidf` | 1.0000 | 0 | 144 |
| `char_tfidf` | 0.8403 | 23 | 144 |

### benign_vs_secret_related / activation_probe

No family-level errors.

### benign_vs_secret_related / word_tfidf

No family-level errors.

### benign_vs_secret_related / char_tfidf

| Method | Family | True Label | Examples | Errors | Accuracy | Predicted Labels |
|---|---|---|---:|---:|---:|---|
| `char_tfidf` | `dp_honey_lite_v4_1_incident_ticket` | `benign` | 8 | 7 | 0.1250 | `benign=1, secret_related=7` |
| `char_tfidf` | `dp_honey_lite_v4_1_access_review` | `benign` | 8 | 5 | 0.3750 | `benign=3, secret_related=5` |
| `char_tfidf` | `dp_honey_lite_v4_1_audit_export` | `benign` | 8 | 4 | 0.5000 | `benign=4, secret_related=4` |
| `char_tfidf` | `dp_honey_lite_v4_1_support_transcript` | `benign` | 8 | 4 | 0.5000 | `benign=4, secret_related=4` |
| `char_tfidf` | `dp_honey_lite_v4_1_workflow_automation` | `benign` | 8 | 3 | 0.6250 | `benign=5, secret_related=3` |

## safe_secret_vs_exfiltration

Classify safe secret handling against exfiltration-oriented secret handling.

Labels: `exfiltration_intent, secret_present_safe`

| Method | Accuracy | Errors | Predictions |
|---|---:|---:|---:|
| `activation_probe` | 0.7917 | 20 | 96 |
| `char_tfidf` | 0.3125 | 66 | 96 |
| `word_tfidf` | 0.1875 | 78 | 96 |

### safe_secret_vs_exfiltration / activation_probe

| Method | Family | True Label | Examples | Errors | Accuracy | Predicted Labels |
|---|---|---|---:|---:|---:|---|
| `activation_probe` | `dp_honey_lite_v4_1_support_transcript` | `exfiltration_intent` | 8 | 7 | 0.1250 | `exfiltration_intent=1, secret_present_safe=7` |
| `activation_probe` | `dp_honey_lite_v4_1_incident_ticket` | `exfiltration_intent` | 8 | 4 | 0.5000 | `exfiltration_intent=4, secret_present_safe=4` |
| `activation_probe` | `dp_honey_lite_v4_1_audit_export` | `secret_present_safe` | 8 | 3 | 0.6250 | `exfiltration_intent=3, secret_present_safe=5` |
| `activation_probe` | `dp_honey_lite_v4_1_customer_summary` | `exfiltration_intent` | 8 | 2 | 0.7500 | `exfiltration_intent=6, secret_present_safe=2` |
| `activation_probe` | `dp_honey_lite_v4_1_access_review` | `exfiltration_intent` | 8 | 1 | 0.8750 | `exfiltration_intent=7, secret_present_safe=1` |
| `activation_probe` | `dp_honey_lite_v4_1_customer_summary` | `secret_present_safe` | 8 | 1 | 0.8750 | `exfiltration_intent=1, secret_present_safe=7` |
| `activation_probe` | `dp_honey_lite_v4_1_incident_ticket` | `secret_present_safe` | 8 | 1 | 0.8750 | `exfiltration_intent=1, secret_present_safe=7` |
| `activation_probe` | `dp_honey_lite_v4_1_workflow_automation` | `secret_present_safe` | 8 | 1 | 0.8750 | `exfiltration_intent=1, secret_present_safe=7` |

### safe_secret_vs_exfiltration / word_tfidf

| Method | Family | True Label | Examples | Errors | Accuracy | Predicted Labels |
|---|---|---|---:|---:|---:|---|
| `word_tfidf` | `dp_honey_lite_v4_1_audit_export` | `exfiltration_intent` | 8 | 8 | 0.0000 | `secret_present_safe=8` |
| `word_tfidf` | `dp_honey_lite_v4_1_audit_export` | `secret_present_safe` | 8 | 8 | 0.0000 | `exfiltration_intent=8` |
| `word_tfidf` | `dp_honey_lite_v4_1_customer_summary` | `exfiltration_intent` | 8 | 8 | 0.0000 | `secret_present_safe=8` |
| `word_tfidf` | `dp_honey_lite_v4_1_customer_summary` | `secret_present_safe` | 8 | 8 | 0.0000 | `exfiltration_intent=8` |
| `word_tfidf` | `dp_honey_lite_v4_1_incident_ticket` | `exfiltration_intent` | 8 | 8 | 0.0000 | `secret_present_safe=8` |
| `word_tfidf` | `dp_honey_lite_v4_1_support_transcript` | `secret_present_safe` | 8 | 8 | 0.0000 | `exfiltration_intent=8` |
| `word_tfidf` | `dp_honey_lite_v4_1_workflow_automation` | `exfiltration_intent` | 8 | 8 | 0.0000 | `secret_present_safe=8` |
| `word_tfidf` | `dp_honey_lite_v4_1_workflow_automation` | `secret_present_safe` | 8 | 8 | 0.0000 | `exfiltration_intent=8` |
| `word_tfidf` | `dp_honey_lite_v4_1_support_transcript` | `exfiltration_intent` | 8 | 7 | 0.1250 | `exfiltration_intent=1, secret_present_safe=7` |
| `word_tfidf` | `dp_honey_lite_v4_1_incident_ticket` | `secret_present_safe` | 8 | 6 | 0.2500 | `exfiltration_intent=6, secret_present_safe=2` |
| `word_tfidf` | `dp_honey_lite_v4_1_access_review` | `exfiltration_intent` | 8 | 1 | 0.8750 | `exfiltration_intent=7, secret_present_safe=1` |

### safe_secret_vs_exfiltration / char_tfidf

| Method | Family | True Label | Examples | Errors | Accuracy | Predicted Labels |
|---|---|---|---:|---:|---:|---|
| `char_tfidf` | `dp_honey_lite_v4_1_workflow_automation` | `secret_present_safe` | 8 | 8 | 0.0000 | `exfiltration_intent=8` |
| `char_tfidf` | `dp_honey_lite_v4_1_audit_export` | `secret_present_safe` | 8 | 7 | 0.1250 | `exfiltration_intent=7, secret_present_safe=1` |
| `char_tfidf` | `dp_honey_lite_v4_1_support_transcript` | `secret_present_safe` | 8 | 7 | 0.1250 | `exfiltration_intent=7, secret_present_safe=1` |
| `char_tfidf` | `dp_honey_lite_v4_1_access_review` | `secret_present_safe` | 8 | 6 | 0.2500 | `exfiltration_intent=6, secret_present_safe=2` |
| `char_tfidf` | `dp_honey_lite_v4_1_customer_summary` | `exfiltration_intent` | 8 | 6 | 0.2500 | `exfiltration_intent=2, secret_present_safe=6` |
| `char_tfidf` | `dp_honey_lite_v4_1_access_review` | `exfiltration_intent` | 8 | 5 | 0.3750 | `exfiltration_intent=3, secret_present_safe=5` |
| `char_tfidf` | `dp_honey_lite_v4_1_customer_summary` | `secret_present_safe` | 8 | 5 | 0.3750 | `exfiltration_intent=5, secret_present_safe=3` |
| `char_tfidf` | `dp_honey_lite_v4_1_incident_ticket` | `exfiltration_intent` | 8 | 5 | 0.3750 | `exfiltration_intent=3, secret_present_safe=5` |
| `char_tfidf` | `dp_honey_lite_v4_1_incident_ticket` | `secret_present_safe` | 8 | 5 | 0.3750 | `exfiltration_intent=5, secret_present_safe=3` |
| `char_tfidf` | `dp_honey_lite_v4_1_support_transcript` | `exfiltration_intent` | 8 | 5 | 0.3750 | `exfiltration_intent=3, secret_present_safe=5` |
| `char_tfidf` | `dp_honey_lite_v4_1_workflow_automation` | `exfiltration_intent` | 8 | 5 | 0.3750 | `exfiltration_intent=3, secret_present_safe=5` |
| `char_tfidf` | `dp_honey_lite_v4_1_audit_export` | `exfiltration_intent` | 8 | 2 | 0.7500 | `exfiltration_intent=6, secret_present_safe=2` |
