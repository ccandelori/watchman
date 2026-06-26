# NIMBUS InfoNCE Evaluation

- Model: `nimbus-infonce-lexical-v0`
- Records: `14`
- Split groups: `7`
- Eval corpus SHA-256: `ff2a41a688684cb2b887f589e5a9b96523e231b29f8c7edb99afaef4b1914131`
- Training eval reused: `true`
- Training eval allowed: `true`
- Attack top-1 accuracy: `0.777778`
- Mean NCE loss bits: `2.83973`
- Mean estimated leakage bits: `1.85418`
- Mean absolute error bits: `1.28275`
- False positive rate: `0`
- False negative rate: `0.222222`
- Promotion status: `not_promotable_offline_scaffold`
- Paper-faithful learned critic: `false`

| Label | Count | Top-1 accuracy | Mean target bits | Mean estimated bits | MAE bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| benign | 5 | 0 | 0 | 0 | 0 |
| delayed | 1 | 1 | 2 | 4.03135 | 2.03135 |
| direct | 1 | 1 | 2 | 4.08462 | 2.08462 |
| encoded | 1 | 1 | 1.2 | 4.08462 | 2.88462 |
| paraphrased | 1 | 1 | 0.8 | 3.2422 | 2.4422 |
| partial | 4 | 0.5 | 0.35 | 1.6211 | 1.6211 |
| tool_output | 1 | 1 | 2 | 4.03135 | 2.03135 |
