# NIMBUS Promotion Decision

Verdict: reject learned NIMBUS runtime promotion for this release and keep deterministic canary NIMBUS as the default runtime critic.

Authoritative artifact: `introspection/data/reports/aegis_nimbus_promotion_evidence_v2.json`

SHA-256: `ae389ec1a37533b70a94d1e694c657c1fd9c168cbd13e48f73df1386f6312f3c`

## Decision

The learned InfoNCE path is useful live beta evidence, but it is not a promoted runtime artifact:

- `promote_learned_runtime=false`
- `promote_hybrid_runtime=false`
- `keep_deterministic_default=true`
- `reject_learned_runtime=true`

The learned beta passes grouped, sealed, runtime beta, benign false-block, and supported live-gateway metric gates. It fails promotion because the package still lacks a common live head-to-head corpus for deterministic, learned, and hybrid policies; latency evidence; complete audit/readiness/console binding; learned failure-mode evidence; hybrid-policy evaluation; and a promoted runtime manifest.

## Evidence

Learned model: `introspection/data/reports/aegis_nimbus_infonce_model_v0.json`

- Model id: `nimbus-infonce-lexical-v0`
- `paper_faithful_learned_critic=false`
- Feature weights: `[1.0, 4.0, 0.0]`
- `state_token_overlap` policy: diagnostic only
- Negative contexts: 16

Grouped CV and sealed holdout:

- Records: 1000
- Split groups: 50
- Turn FP: 4
- Turn FN: 0
- Turn FPR: `0.005369127516778523`
- Turn FNR: `0.0`
- Session FP: 0
- Session FN: 0

Runtime beta:

- Attack sessions detected: 42/42
- Benign-only sessions false-blocked: 0/8
- False-block rate: `0.0`
- Mean first block turn index: `3.9285714285714284`

Live gateway smoke: `introspection/data/reports/aegis_default_mock_provider_smoke_learned_nimbus_beta_v3.json`

SHA-256: `504e598182f2c9e93492e916f614befc3cac259fb5c1a5dc055b6dac8fc3e939`

- `/ready`: `nimbus_status=learned_runtime_beta`
- Critic: `learned_infonce_beta`
- Model artifact SHA-256: `8c5bd62b4f54d9a0758c90cca93521b9498ce4252c6f143a4d2cb2a6cd8725e8`
- Samples: 6
- TP: 4
- TN: 2
- FP: 0
- FN: 0
- Covered live scenarios: benign allow, adversarial-benign allow, exact leak block, encoded leak block, partial leak block, tool-argument leak pre-dispatch block

## Commands

Run the live learned gateway smoke:

```bash
AEGIS_NIMBUS_CRITIC_KIND=learned_infonce_beta \
AEGIS_NIMBUS_INFONCE_MODEL_PATH=introspection/data/reports/aegis_nimbus_infonce_model_v0.json \
AEGIS_NIMBUS_CRITIC_VERSION=nimbus-infonce-lexical-v0 \
AEGIS_AUDIT_JSONL_PATH=introspection/data/reports/aegis_default_mock_provider_smoke_learned_nimbus_beta_audit_v3.jsonl \
uv run aegis-proxy --host 127.0.0.1 --port 8788

uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8788 \
  --timeout 10 \
  --nimbus-profile strict-partial-block \
  --output introspection/data/reports/aegis_default_mock_provider_smoke_learned_nimbus_beta_v3.json
```

Build the promotion decision artifact:

```bash
uv run aegis-nimbus-promotion-evidence \
  --deterministic-eval introspection/data/reports/aegis_nimbus_deterministic_beta_eval_v1.json \
  --calibration-manifest introspection/data/reports/aegis_nimbus_training_corpus_manifest_v0.json \
  --sealed-manifest introspection/data/reports/aegis_nimbus_sealed_holdout_corpus_manifest_v0.json \
  --infonce-model introspection/data/reports/aegis_nimbus_infonce_model_v0.json \
  --grouped-cv introspection/data/reports/aegis_nimbus_infonce_grouped_cv_v0.json \
  --sealed-holdout introspection/data/reports/aegis_nimbus_infonce_sealed_holdout_eval_v0.json \
  --gateway-smoke introspection/data/reports/aegis_default_mock_provider_smoke_learned_nimbus_beta_v3.json \
  --runtime-beta-eval introspection/data/reports/aegis_nimbus_runtime_beta_eval_v0.json \
  --output introspection/data/reports/aegis_nimbus_promotion_evidence_v2.json
```
