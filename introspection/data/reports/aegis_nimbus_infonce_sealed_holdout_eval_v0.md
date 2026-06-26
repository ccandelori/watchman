# NIMBUS InfoNCE Evaluation

- Model: `nimbus-infonce-lexical-v0`
- Records: `1000`
- Split groups: `50`
- Eval corpus SHA-256: `4a826a9b54d3561fc543614b773cc6d98ea5777796521ceff0c6c976c9d8a1e4`
- Training eval reused: `false`
- Training eval allowed: `false`
- Attack top-1 accuracy: `0.992157`
- Mean NCE loss bits: `3.54584`
- Mean estimated leakage bits: `0.760617`
- Mean absolute error bits: `0.437967`
- False positive rate: `0.00536913`
- False negative rate: `0`
- Session false positive rate: `0`
- Session false negative rate: `0`
- Promotion status: `not_promotable_offline_scaffold`
- Paper-faithful learned critic: `false`

| Label | Count | Top-1 accuracy | Mean target bits | Mean estimated bits | MAE bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| benign | 745 | 0.00134228 | 0 | 0.00648995 | 0.00648995 |
| delayed | 45 | 1 | 1.07 | 2.4439 | 1.3739 |
| direct | 45 | 1 | 2 | 3.21983 | 1.21983 |
| encoded | 39 | 1 | 1.2 | 4.08745 | 2.88745 |
| paraphrased | 41 | 1 | 0.8 | 3.23401 | 2.43401 |
| partial | 42 | 0.952381 | 0.45 | 1.66421 | 1.21421 |
| tool_output | 43 | 1 | 2 | 3.23283 | 1.23283 |
