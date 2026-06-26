# NIMBUS InfoNCE Evaluation

- Model: `nimbus-infonce-lexical-v0`
- Records: `1000`
- Split groups: `50`
- Eval corpus SHA-256: `303c905307debd9a75b91c36820a5bd2f05a3b56d981d2c7712517d15ec571c1`
- Training eval reused: `true`
- Training eval allowed: `true`
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
