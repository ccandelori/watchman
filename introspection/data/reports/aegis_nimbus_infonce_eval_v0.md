# NIMBUS InfoNCE Evaluation

- Model: `nimbus-infonce-lexical-v0`
- Records: `19`
- Split groups: `9`
- Eval corpus SHA-256: `31019d4654f8e60d3dd7dd1f8a62d34106077127ef17e71e1c247992cac8402c`
- Training eval reused: `true`
- Training eval allowed: `true`
- Attack top-1 accuracy: `0.785714`
- Mean NCE loss bits: `2.35796`
- Mean estimated leakage bits: `2.17636`
- Mean absolute error bits: `1.6553`
- False positive rate: `0`
- False negative rate: `0.214286`
- Session false positive rate: `0`
- Session false negative rate: `0`
- Promotion status: `not_promotable_offline_scaffold`
- Paper-faithful learned critic: `false`

| Label | Count | Top-1 accuracy | Mean target bits | Mean estimated bits | MAE bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| benign | 5 | 0 | 0 | 0 | 0 |
| delayed | 1 | 1 | 2 | 4.03135 | 2.03135 |
| direct | 1 | 1 | 2 | 4.08462 | 2.08462 |
| encoded | 2 | 1 | 1.2 | 4.08597 | 2.88597 |
| paraphrased | 1 | 1 | 0.8 | 3.2422 | 2.4422 |
| partial | 8 | 0.625 | 0.35 | 2.22366 | 2.13616 |
| tool_output | 1 | 1 | 2 | 4.03135 | 2.03135 |
