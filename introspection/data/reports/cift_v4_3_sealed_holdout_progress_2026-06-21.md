# CIFT V4.3 Sealed Holdout Progress

## Purpose

V4.3 establishes the first sealed holdout checkpoint for the CIFT-like
activation-monitor thread. It is designed to test whether the V4.1 trained
selector-window detector generalizes to fresh proxy-shaped scenarios without
using the holdout for training, model selection, calibration, threshold tuning,
detector export, or error inspection before an explicit unseal.

The seal is a process guard. The synthetic prompts are committed for
reproducibility, but they must not provide model-feedback until the team names
the frozen evaluation setup and records the unseal decision.

## Registered Datasets

| Dataset | Path | Rows | SHA-256 |
|---|---|---:|---|
| `dp_honey_lite_prompts_v4_3_sealed` | `introspection/data/prompts_dp_honey_lite_v4_3_sealed.jsonl` | 144 | `74a0599de5910d4e184b027fe67fe83a36af3a3e0f2de64c6c11c300c622207e` |
| `dp_honey_lite_v4_3_sealed_selector_windows` | `introspection/data/prompts_dp_honey_lite_v4_3_sealed_selector_windows.jsonl` | 144 | `0a36cd68c20c9254a6156d2b0469d8205e60b6553e4caae603181d3a49f6120c` |

Both datasets are balanced:

| Label | Rows |
|---|---:|
| `benign` | 48 |
| `secret_present_safe` | 48 |
| `exfiltration_intent` | 48 |

The six scenario families are fresh relative to V4.1:

```text
dp_honey_lite_v4_3_backup_restore
dp_honey_lite_v4_3_billing_reconciliation
dp_honey_lite_v4_3_data_retention
dp_honey_lite_v4_3_partner_integration
dp_honey_lite_v4_3_release_gate
dp_honey_lite_v4_3_sandbox_provisioning
```

Each family preserves the V4.1 relational-control shape: `mode_a` and `mode_b`
policies remain stable within a family, selected modes are balanced by label,
payload and no-payload rows are balanced by label, and each row carries
character spans, token spans, honeytoken metadata, and readout-token indices.

## Guard Rails

Sensitive CIFT scripts now refuse sealed paths or sealed row tags unless invoked
with `--allow-sealed-holdout`.

Guarded operations include:

| Operation | Script |
|---|---|
| Activation extraction | `introspection/scripts/extract_activations.py` |
| Full-train CIFT bundle training | `introspection/scripts/train_cift_model_bundle.py` |
| Grouped CIFT calibration | `introspection/scripts/calibrate_cift_detector.py` |
| Trained detector-result export | `introspection/scripts/export_trained_cift_detector_results.py` |
| Calibrated detector-result export | `introspection/scripts/export_calibrated_cift_detector_results.py` |
| Binary error analysis | `introspection/scripts/analyze_binary_errors.py` |

Dataset generation and selector-window derivation are intentionally allowed.
They create the sealed artifact but do not score a model or inspect failures.

## Unseal Protocol

Before first V4.3 scoring, record the unseal decision in the issue, pull
request, or project notes. The record must name:

| Field | Required Value |
|---|---|
| Model bundle | Frozen bundle identifier and file hash |
| Feature key | Frozen activation feature or feature expression |
| Threshold | Frozen operating threshold and score semantics |
| Metric suite | Exact grouped or holdout metrics to compute |
| Baselines | Text and activation references to compare |
| Outputs | Exact output report paths |

The first evaluation should then run once with `--allow-sealed-holdout`. Report
the result before changing templates, model bundles, thresholds, calibration, or
feature choices. After first scoring, mark V4.3 as unsealed or used in the next
lineage/documentation update.

## Current Status

V4.3 is generated, selector-window-derived, lineage-registered, and guarded.
No V4.3 activation artifact, calibration report, detector-result export, model
bundle, or error-analysis report exists in this milestone.
