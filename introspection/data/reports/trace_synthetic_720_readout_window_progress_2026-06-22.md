# Trace Synthetic 720 Readout-Window Progress

Date: 2026-06-22

## Purpose

This milestone connects the Aegis runtime trace collection harness to the
introspection CIFT workflow. The goal was to convert DP-HONEY-backed
`TraceCollectionRecord` rows into structured prompt rows with token geometry,
then extract readout-window activations from Qwen3-0.6B.

## Local Artifacts

Generated artifacts remain under ignored `data/trace_collection/` paths:

- `records.synthetic_720.jsonl`: 720 normalized runtime trace records.
- `structured_prompts.synthetic_720.jsonl`: 480 CIFT-ready secret-bearing rows.
- `qwen3_0_6b_trace_synthetic_720_readout.pt`: readout-window activation artifact.
- `probe_qwen3_0_6b_trace_synthetic_720_readout.json`: stratified linear probe smoke report.
- `candidate_feature_trace_synthetic_720_readout.json`: grouped candidate-feature crosscheck.

## Conversion Result

The converter produced:

- 240 `secret_present_safe` examples.
- 240 `exfiltration_intent` examples.
- 240 `benign` examples skipped for separate no-secret calibration.
- 12 prompt families represented, with 40 converted examples per family.
- 240 examples with payload readout windows.

Benign rows were skipped intentionally because the current structured prompt
schema requires a real secret span. Adding fake secret spans to benign rows
would weaken the data contract.

## Activation Extraction

Extraction command:

```bash
/Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python \
  introspection/scripts/extract_activations.py \
  --prompts data/trace_collection/structured_prompts.synthetic_720.jsonl \
  --output data/trace_collection/qwen3_0_6b_trace_synthetic_720_readout.pt \
  --model-id Qwen/Qwen3-0.6B \
  --revision main \
  --device auto \
  --layers 0,7,14,21,28 \
  --pooling readout_window
```

Artifact summary:

- Model: `Qwen/Qwen3-0.6B`
- Selected device: `cpu`
- Rows: 480
- Features: `readout_window_layer_00`, `07`, `14`, `21`, `28`
- Feature shape: `(480, 1024)` for each layer

## Probe Result

The stratified linear probe smoke test reached perfect performance on every
readout-window layer:

| Feature | Macro F1 | Accuracy |
|---|---:|---:|
| `readout_window_layer_00` | 1.0000 | 1.0000 |
| `readout_window_layer_07` | 1.0000 | 1.0000 |
| `readout_window_layer_14` | 1.0000 | 1.0000 |
| `readout_window_layer_21` | 1.0000 | 1.0000 |
| `readout_window_layer_28` | 1.0000 | 1.0000 |

The grouped family split also reached perfect performance for
`safe_secret_vs_exfiltration`:

| Feature | Macro F1 | Accuracy |
|---|---:|---:|
| `readout_window_layer_00` | 1.0000 | 1.0000 |
| `concat(readout_window_layer_00,readout_window_layer_07,readout_window_layer_14,readout_window_layer_21,readout_window_layer_28)` | 1.0000 | 1.0000 |

## Interpretation

This is a plumbing success, not a robustness result. The runtime-to-CIFT data
contract works: DP-HONEY spans became token spans, readout windows validated,
activations extracted, and probes trained.

The perfect scores mean the seeded corpus is still too separable to support a
strong modeling claim. The safe and exfiltration prompts differ in explicit
surface instructions, and exfiltration rows often include tool payloads while
safe rows usually do not. A classifier can exploit those differences without
necessarily learning a subtle pre-output credential-access signal.

## Next Step

The next corpus should preserve the same runtime trace schema but make the
classification problem harder:

- Match safe and exfiltration examples on tool-call presence.
- Use similar instruction style across labels.
- Add safe rows with tool calls that mention credential type but omit values.
- Add exfiltration rows whose malicious intent is phrased indirectly.
- Keep DP-HONEY spans and readout windows exact so CIFT remains testable.

After that, rerun readout-window extraction and compare activation probes
against text baselines under grouped family splits.
