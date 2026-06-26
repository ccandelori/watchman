# NIMBUS InfoNCE Evaluation

- Model: `nimbus-infonce-lexical-v0`
- Records: `1000`
- Split groups: `50`
- Eval corpus SHA-256: `4a826a9b54d3561fc543614b773cc6d98ea5777796521ceff0c6c976c9d8a1e4`
- Training eval reused: `false`
- Training eval allowed: `false`
- Attack top-1 accuracy: `0.964706`
- Mean NCE loss bits: `2.95765`
- Mean estimated leakage bits: `1.77612`
- Mean absolute error bits: `1.46003`
- False positive rate: `0.438926`
- False negative rate: `0.027451`
- Session false positive rate: `0`
- Session false negative rate: `0`
- Promotion status: `not_promotable_offline_scaffold`
- Paper-faithful learned critic: `false`

| Label | Count | Top-1 accuracy | Mean target bits | Mean estimated bits | MAE bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| benign | 745 | 0.390604 | 0 | 1.15306 | 1.15306 |
| delayed | 45 | 1 | 1.07 | 3.55936 | 2.49505 |
| direct | 45 | 1 | 2 | 3.89037 | 1.89037 |
| encoded | 39 | 1 | 1.2 | 4.08743 | 2.88743 |
| paraphrased | 41 | 1 | 0.8 | 3.65418 | 2.85418 |
| partial | 42 | 0.785714 | 0.45 | 2.64458 | 2.34458 |
| tool_output | 43 | 1 | 2 | 3.75691 | 1.75691 |
