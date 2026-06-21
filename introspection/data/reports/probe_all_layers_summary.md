# Probe Layer Sweep Summary

## Source

- Model: `Qwen/Qwen3-0.6B`
- Revision: `main`
- Extraction device: `cpu`
- Fold count: `5`
- Labels: `benign, exfiltration_intent, secret_present_safe`

## Best Feature

- Feature: `mean_pool_layer_18`
- Macro F1: `0.9776`
- Accuracy: `0.9778`
- `mean_pool[d] = average(token_1[d], token_2[d], ..., token_19[d])`

## Feature Ranking

| Rank | Feature | Pooling | Layer | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `mean_pool_layer_18` | `mean_pool` | 18 | 0.9776 | 0.9778 | 0.0274 | 0.0272 |
| 2 | `mean_pool_layer_06` | `mean_pool` | 6 | 0.9664 | 0.9667 | 0.0274 | 0.0272 |
| 3 | `mean_pool_layer_17` | `mean_pool` | 17 | 0.9664 | 0.9667 | 0.0274 | 0.0272 |
| 4 | `mean_pool_layer_14` | `mean_pool` | 14 | 0.9554 | 0.9556 | 0.0416 | 0.0416 |
| 5 | `mean_pool_layer_07` | `mean_pool` | 7 | 0.9552 | 0.9556 | 0.0224 | 0.0222 |
| 6 | `mean_pool_layer_13` | `mean_pool` | 13 | 0.9552 | 0.9556 | 0.0224 | 0.0222 |
| 7 | `mean_pool_layer_21` | `mean_pool` | 21 | 0.9552 | 0.9556 | 0.0224 | 0.0222 |
| 8 | `mean_pool_layer_05` | `mean_pool` | 5 | 0.9540 | 0.9556 | 0.0438 | 0.0416 |
| 9 | `mean_pool_layer_09` | `mean_pool` | 9 | 0.9540 | 0.9556 | 0.0438 | 0.0416 |
| 10 | `mean_pool_layer_19` | `mean_pool` | 19 | 0.9442 | 0.9444 | 0.0351 | 0.0351 |
| 11 | `mean_pool_layer_20` | `mean_pool` | 20 | 0.9442 | 0.9444 | 0.0351 | 0.0351 |
| 12 | `mean_pool_layer_24` | `mean_pool` | 24 | 0.9442 | 0.9444 | 0.0351 | 0.0351 |
| 13 | `mean_pool_layer_22` | `mean_pool` | 22 | 0.9441 | 0.9444 | 0.0353 | 0.0351 |
| 14 | `final_token_layer_17` | `final_token` | 17 | 0.9436 | 0.9444 | 0.0361 | 0.0351 |
| 15 | `mean_pool_layer_10` | `mean_pool` | 10 | 0.9436 | 0.9444 | 0.0361 | 0.0351 |
| 16 | `mean_pool_layer_01` | `mean_pool` | 1 | 0.9436 | 0.9444 | 0.0361 | 0.0351 |
| 17 | `mean_pool_layer_16` | `mean_pool` | 16 | 0.9332 | 0.9333 | 0.0415 | 0.0416 |
| 18 | `mean_pool_layer_23` | `mean_pool` | 23 | 0.9330 | 0.9333 | 0.0221 | 0.0222 |
| 19 | `mean_pool_layer_02` | `mean_pool` | 2 | 0.9325 | 0.9333 | 0.0422 | 0.0416 |
| 20 | `mean_pool_layer_03` | `mean_pool` | 3 | 0.9325 | 0.9333 | 0.0422 | 0.0416 |
| 21 | `mean_pool_layer_25` | `mean_pool` | 25 | 0.9325 | 0.9333 | 0.0422 | 0.0416 |
| 22 | `final_token_layer_16` | `final_token` | 16 | 0.9324 | 0.9333 | 0.0233 | 0.0222 |
| 23 | `mean_pool_layer_08` | `mean_pool` | 8 | 0.9324 | 0.9333 | 0.0233 | 0.0222 |
| 24 | `mean_pool_layer_15` | `mean_pool` | 15 | 0.9322 | 0.9333 | 0.0665 | 0.0648 |
| 25 | `final_token_layer_06` | `final_token` | 6 | 0.9319 | 0.9333 | 0.0429 | 0.0416 |
| 26 | `final_token_layer_19` | `final_token` | 19 | 0.9214 | 0.9222 | 0.0278 | 0.0272 |
| 27 | `final_token_layer_07` | `final_token` | 7 | 0.9208 | 0.9222 | 0.0465 | 0.0444 |
| 28 | `final_token_layer_18` | `final_token` | 18 | 0.9206 | 0.9222 | 0.0288 | 0.0272 |
| 29 | `final_token_layer_20` | `final_token` | 20 | 0.9206 | 0.9222 | 0.0288 | 0.0272 |
| 30 | `final_token_layer_21` | `final_token` | 21 | 0.9206 | 0.9222 | 0.0288 | 0.0272 |
| 31 | `final_token_layer_22` | `final_token` | 22 | 0.9206 | 0.9222 | 0.0288 | 0.0272 |
| 32 | `final_token_layer_14` | `final_token` | 14 | 0.9111 | 0.9111 | 0.0741 | 0.0754 |
| 33 | `mean_pool_layer_28` | `mean_pool` | 28 | 0.9104 | 0.9111 | 0.0452 | 0.0444 |
| 34 | `final_token_layer_15` | `final_token` | 15 | 0.9098 | 0.9111 | 0.0454 | 0.0444 |
| 35 | `final_token_layer_23` | `final_token` | 23 | 0.9098 | 0.9111 | 0.0462 | 0.0444 |
| 36 | `final_token_layer_24` | `final_token` | 24 | 0.9098 | 0.9111 | 0.0462 | 0.0444 |
| 37 | `final_token_layer_25` | `final_token` | 25 | 0.9098 | 0.9111 | 0.0462 | 0.0444 |
| 38 | `final_token_layer_08` | `final_token` | 8 | 0.9097 | 0.9111 | 0.0462 | 0.0444 |
| 39 | `mean_pool_layer_11` | `mean_pool` | 11 | 0.9097 | 0.9111 | 0.0281 | 0.0272 |
| 40 | `mean_pool_layer_12` | `mean_pool` | 12 | 0.9097 | 0.9111 | 0.0281 | 0.0272 |
| 41 | `mean_pool_layer_04` | `mean_pool` | 4 | 0.9095 | 0.9111 | 0.0283 | 0.0272 |
| 42 | `final_token_layer_09` | `final_token` | 9 | 0.9092 | 0.9111 | 0.0465 | 0.0444 |
| 43 | `mean_pool_layer_26` | `mean_pool` | 26 | 0.8993 | 0.9000 | 0.0423 | 0.0416 |
| 44 | `final_token_layer_11` | `final_token` | 11 | 0.8986 | 0.9000 | 0.0408 | 0.0416 |
| 45 | `final_token_layer_04` | `final_token` | 4 | 0.8978 | 0.9000 | 0.0552 | 0.0544 |
| 46 | `final_token_layer_13` | `final_token` | 13 | 0.8975 | 0.9000 | 0.0678 | 0.0648 |
| 47 | `final_token_layer_26` | `final_token` | 26 | 0.8871 | 0.8889 | 0.0368 | 0.0351 |
| 48 | `final_token_layer_12` | `final_token` | 12 | 0.8863 | 0.8889 | 0.0615 | 0.0609 |
| 49 | `final_token_layer_10` | `final_token` | 10 | 0.8857 | 0.8889 | 0.0368 | 0.0351 |
| 50 | `final_token_layer_05` | `final_token` | 5 | 0.8856 | 0.8889 | 0.0730 | 0.0703 |
| 51 | `mean_pool_layer_27` | `mean_pool` | 27 | 0.8762 | 0.8778 | 0.0421 | 0.0416 |
| 52 | `final_token_layer_28` | `final_token` | 28 | 0.8652 | 0.8667 | 0.0251 | 0.0272 |
| 53 | `final_token_layer_27` | `final_token` | 27 | 0.8650 | 0.8667 | 0.0455 | 0.0444 |
| 54 | `mean_pool_layer_00` | `mean_pool` | 0 | 0.8389 | 0.8444 | 0.1108 | 0.1077 |
| 55 | `final_token_layer_02` | `final_token` | 2 | 0.8335 | 0.8333 | 0.0611 | 0.0609 |
| 56 | `final_token_layer_03` | `final_token` | 3 | 0.8212 | 0.8222 | 0.0245 | 0.0222 |
| 57 | `final_token_layer_01` | `final_token` | 1 | 0.7747 | 0.7778 | 0.0939 | 0.0930 |
| 58 | `final_token_layer_00` | `final_token` | 0 | 0.1653 | 0.3222 | 0.0118 | 0.0222 |

## Pooling Summary

| Pooling | Best Layer | Best Macro F1 | Best Accuracy |
|---|---:|---:|---:|
| `final_token` | 17 | 0.9436 | 0.9444 |
| `mean_pool` | 18 | 0.9776 | 0.9778 |

## Takeaway
The middle-to-late layers, around 18 in this case, are the stronger region for feature extraction.