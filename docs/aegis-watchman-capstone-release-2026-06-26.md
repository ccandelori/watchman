# Aegis/Watchman Capstone Release Package - 2026-06-26

## Verdict

This is a strong local capstone release candidate, not a production/commercial
release. The release package is defensible for a local evaluator who can run the
gateway, inspect the console, and verify saved evidence artifacts.

Safe claim:

> Aegis/Watchman operationalizes the paper's vision as a local sentinel:
> certified CIFT for Qwen3-4B/MPS, paper-faithful+ candidate DP-HONEY,
> explicit-slot credential substitution, provider egress prevention,
> deterministic NIMBUS plus learned NIMBUS live beta, redacted audit/explain
> evidence, and a local console.

Do not claim:

- universal model support
- production/commercial readiness
- paper-faithful+ learned NIMBUS
- external real-provider production evidence
- autonomous semantic credential-need inference beyond deterministic v1

## Demo Path

Prerequisites:

- `uv sync --extra dev`
- MPS-capable Python environment at `.venv-mps313`
- Qwen/Qwen3-4B available to the local Transformers cache
- `AEGIS_CIFT_EXTRACTOR_API_KEY` set to a non-empty local secret

Terminal 1: start the certified Qwen3-4B/MPS CIFT sidecar.

```bash
export AEGIS_CIFT_EXTRACTOR_API_KEY="set-a-deployment-secret"

PYTHONPATH=src:introspection/src \
.venv-mps313/bin/python introspection/scripts/check_cift_device_preflight.py \
  --device mps

PYTHONPATH=src:introspection/src \
.venv-mps313/bin/python introspection/scripts/run_cift_extractor_sidecar.py \
  --model-id Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --device mps \
  --dtype device \
  --feature-key selected_choice_window_layer_21 \
  --feature-key final_token_layer_12 \
  --selected-choice-readout-token-count 4 \
  --host 127.0.0.1 \
  --port 9000 \
  --api-key-env-var AEGIS_CIFT_EXTRACTOR_API_KEY
```

Terminal 2: start the strict certified gateway.

```bash
export AEGIS_CIFT_EXTRACTOR_API_KEY="set-a-deployment-secret"
source introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_strict_deployment_env.sh
source introspection/data/certifications/qwen3_4b_watchman_v13_freeform_final_token_l12/reports/qwen3_4b_watchman_v13_freeform_final_token_l12_strict_deployment_env.sh
AEGIS_AUDIT_JSONL_PATH=/tmp/aegis-cift-demo-audit.jsonl \
uv run aegis-proxy --host 127.0.0.1 --port 8000
```

Terminal 3: open the local console for the strict gateway.

```bash
uv run aegis-console \
  --gateway-url http://127.0.0.1:8000 \
  --host 127.0.0.1 \
  --port 8780 \
  --smoke-report introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_gateway_smoke_integrated_refresh_v1.json \
  --sample-audit-jsonl introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_gateway_smoke_integrated_refresh_audit_v1.jsonl
```

Open `http://127.0.0.1:8780`. Expected state: gateway ready, CIFT certified for
Qwen/Qwen3-4B on MPS, DP-HONEY ready, NIMBUS deterministic beta by default,
recent allow/block decisions, detector activity, and stage timelines.

Terminal 4: run the strict CIFT and integrated sentinel smokes.

```bash
uv run aegis-proxy-cift-smoke \
  --url http://127.0.0.1:8000 \
  --sidecar-url http://127.0.0.1:9000 \
  --gateway-model qwen3:4b \
  --report-id qwen3_4b_launcher_cift_freeform_smoke_v1 \
  --timeout 120 \
  --detector-name cift_runtime \
  --sidecar-feature-key final_token_layer_12 \
  --expected-gateway-feature-source self_hosted_activation_extractor \
  --expected-extractor-id trusted-activation-sidecar \
  --expected-sidecar-model-id Qwen/Qwen3-4B \
  --expected-sidecar-revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --expected-sidecar-device mps \
  --expected-sidecar-hidden-size 2560 \
  --expected-sidecar-layer-count 36 \
  --expected-sidecar-tokenizer-fingerprint-sha256 41e00eccf531cffc2e562d38bdd879d41e5044ea279af5b73c6a32aabcc8fe04 \
  --expected-sidecar-special-tokens-map-sha256 edcb2fc2acbbe77f858a9c4fe51295ffdb84711efba5703ec5906b3d67282569 \
  --expected-sidecar-chat-template-sha256 a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8 \
  --selected-choice-readout-token-count 4 \
  --sidecar-api-key-env-var AEGIS_CIFT_EXTRACTOR_API_KEY \
  --output introspection/data/reports/qwen3_4b_launcher_cift_freeform_smoke_v1.json

uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8000 \
  --timeout 120 \
  --require-cift-pre-generation-block \
  --output introspection/data/reports/aegis_self_hosted_cift_smoke_v1.json
```

Learned NIMBUS beta is intentionally a separate opt-in runtime path. Use a
separate gateway instance so the beta status is obvious.

```bash
AEGIS_NIMBUS_CRITIC_KIND=learned_infonce_beta \
AEGIS_NIMBUS_INFONCE_MODEL_PATH=introspection/data/reports/aegis_nimbus_infonce_model_v0.json \
AEGIS_NIMBUS_CRITIC_VERSION=nimbus-infonce-lexical-v0 \
AEGIS_AUDIT_JSONL_PATH=/tmp/aegis-learned-nimbus-beta-audit.jsonl \
uv run aegis-proxy --host 127.0.0.1 --port 8788

uv run aegis-proxy-smoke \
  --url http://127.0.0.1:8788 \
  --timeout 10 \
  --nimbus-profile strict-partial-block \
  --output introspection/data/reports/aegis_default_mock_provider_smoke_learned_nimbus_beta_v3.json

uv run aegis-console \
  --gateway-url http://127.0.0.1:8788 \
  --host 127.0.0.1 \
  --port 8781 \
  --smoke-report introspection/data/reports/aegis_default_mock_provider_smoke_learned_nimbus_beta_v3.json \
  --sample-audit-jsonl introspection/data/reports/aegis_default_mock_provider_smoke_learned_nimbus_beta_audit_v3.jsonl
```

Expected learned NIMBUS beta state: `/ready` and the console show
`learned_runtime_beta / learned_infonce_beta /
learned_runtime_beta_not_promotable`. Benign traffic allows; encoded,
metadata-slot, tool-argument, and partial leak probes produce learned NIMBUS
block/escalation evidence. `/audit/explain` reconstructs the NIMBUS stage.

If the 4B sidecar is too heavy to run in the evaluation environment, use the
saved CIFT evidence below and state that live Qwen3-4B/MPS rehearsal was not
rerun in that environment. Do not replace it with CPU evidence.

MPS troubleshooting: if `check_cift_device_preflight.py --device mps` reports
`mps_built=True` but `torch.backends.mps.is_available() is false` inside a
sandboxed harness, rerun the same command from a normal local terminal or with
hardware access enabled. The release claim requires `selected_device=mps` and a
smoke tensor on `mps:0`; CPU fallback is not acceptable CIFT evidence for the
certified Qwen3-4B/MPS profile.

## Component Readiness

| Component | Release posture | Evidence-backed claim | Boundaries |
| --- | --- | --- | --- |
| CIFT | Certified reference | Qwen/Qwen3-4B on MPS is model-specifically certified with immutable revision, tokenizer/template hashes, selected-choice geometry, strict release gate, and live gateway smoke evidence. Readiness/capabilities/console report support states from `unsupported` through `runtime-enforceable`. | Other models are `unsupported` until they pass their own calibration, sealed holdout, live runtime, gateway smoke, and release gate. `self_hosted_introspection` alone is not a model-support claim. |
| DP-HONEY | Paper-faithful+ candidate | Segment-aware generator, scanner, provider-like redacted morphology corpus, statistical distinguisher suite, runtime substitution, and audit metadata are bound by evidence. | Not a proof of indistinguishability from real production secrets and not a complete external secret-manager integration. |
| Deterministic NIMBUS | Default runtime beta | Canary-aware session critic tracks exact, encoded, partial, and tool-argument leakage and remains the default runtime critic. | Deterministic beta is not the paper's learned session critic. |
| Learned NIMBUS beta | Live beta, not promotable | `learned_infonce_beta` loads the lexical InfoNCE artifact, emits model hash, selected context hash, negative-context count, estimated bits, confidence, readiness, audit, smoke, and console evidence. | `paper_faithful_learned_critic=false`; `promotion_status=learned_runtime_beta_not_promotable`; no promoted learned runtime manifest. |
| Gateway/sentinel loop | Local release candidate | Gateway wires DP-HONEY, CIFT, provider egress guard, provider, canary detectors, NIMBUS, policy, and audit with smoke evidence. | Needs external release-environment rehearsal before public signoff. |
| Provider egress guard | Release candidate for known sensitive payloads | Raw credential-shaped tool payloads and planted canary tool arguments are blocked before provider dispatch in saved smoke evidence. | Sink/payload coverage should expand over time. |
| Audit/explain | Evidence-grade local | Redacted JSONL audit plus `/audit/explain` reconstruct normalize, DP-HONEY, CIFT, provider egress, provider, canary, NIMBUS, policy, and audit stages. | Local JSONL only: no production retention, authz, or indexed store. |
| Console | Local operator console | Shows readiness, protected/degraded state, CIFT model binding, DP-HONEY, NIMBUS critic/promotion status, recent decisions, detector activity, and timelines. | Local/debug oriented, not hardened multi-user operations UI. |
| Real-provider mode | Hardened and loopback-proven | OpenAI-compatible adapter, HTTPS validation, mock-control rejection, loopback smoke, and evidence verifier exist. | No external credentialed provider evidence in the current release package. |

## Evidence Bundle

| Evidence | Path | SHA-256 | Key result |
| --- | --- | --- | --- |
| CIFT certification verification | `introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_certification_verification_current_turn_v1.json` | `ce3935dd6812ae9eac8857a31dbdbb97c9f30cffe9d7af53fcea0eadee35f757` | `status=certified`, model-specific Qwen3-4B reference only, release gate eligible |
| CIFT strict gateway smoke | `introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_gateway_smoke_integrated_refresh_v1.json` | `b886e9e66b4e66b214c69363d61a1517197d2367b3c12ea287283be0df1c2c7f` | Benign allow, exfiltration block before provider/model completion, FN=0, FP=0 |
| CIFT strict gateway audit | `introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_gateway_smoke_integrated_refresh_audit_v1.jsonl` | `6297d50ecfd991e943319ecc00c19f02063a23567c26f5a66303ca745f0c54ff` | Redacted audit JSONL for strict smoke |
| CIFT freeform certification report | `docs/aegis-cift-freeform-runtime-certification-2026-06-27.md` | local doc | Selected-choice and freeform routes are separate certified Qwen3-4B/MPS routes; ordinary local-agent chat uses `final_token_layer_12` |
| CIFT freeform certification verification | `introspection/data/certifications/qwen3_4b_watchman_v13_freeform_final_token_l12/reports/qwen3_4b_watchman_v13_freeform_final_token_l12_certification_verification_v1.json` | `f105fb9288568b9fb6cfb165fb8c8be73826281a8d795e3e763ff2683235ab17` | `status=certified`, `support_state=runtime-enforceable`, runtime SHA `90ba1dbfaebbe48be27baa72b17e06e07793485150517165228572e85bdb8f86` |
| CIFT freeform gateway smoke | `introspection/data/reports/qwen3_4b_launcher_cift_freeform_smoke_v1.json` | `1db81add8a682053e7b126a3eb455f77b66f5c55c358b07c60452f7a7a3b7f51` | `Say OK.` reaches provider, safe credential is honeytoken-substituted, exfiltration blocks before provider, selected-choice recheck passes, FN=0, FP=0 |
| DP-HONEY paper evidence | `introspection/data/reports/dp_honey_paper_evidence_v5.json` | `ae834ba0e1a57dd666629149aa38dd13f183564e49ac8338d369263820fb8ed8` | Checklist 9 met, 0 partial, 0 missing under provider-like morphology path |
| DP-HONEY runtime smoke | `introspection/data/reports/aegis_default_mock_provider_smoke_dp_honey_segment_v2.json` | `0282c1c7edbebd22ebb111a55fcf13ab67b631e953c14715a5861d0389067859` | Honeytoken substitution, raw egress block, canary tool pre-dispatch block |
| Learned NIMBUS live beta smoke | `introspection/data/reports/aegis_default_mock_provider_smoke_learned_nimbus_beta_v3.json` | `504e598182f2c9e93492e916f614befc3cac259fb5c1a5dc055b6dac8fc3e939` | TP=4, TN=2, FP=0, FN=0 in live beta smoke; includes adversarial-benign no-block; readiness says non-promotable |
| Learned NIMBUS promotion binder | `introspection/data/reports/aegis_nimbus_promotion_evidence_v2.json` | `ae389ec1a37533b70a94d1e694c657c1fd9c168cbd13e48f73df1386f6312f3c` | `reject_learned_runtime=true`; `keep_deterministic_default=true`; missing head-to-head, latency, hybrid, and promotion-manifest gates |
| Console evidence | `introspection/data/reports/aegis_console_integrated_refresh_v1.json` | `6caa3d97f6f365ab16414398f2d7b450575b02f618dfe4519c6876d32fd89f3e` | Console API summarizes gateway, CIFT, DP-HONEY, NIMBUS, smoke, and timelines |
| Loopback provider verification | `introspection/data/reports/aegis_loopback_real_provider_smoke_evidence_verification_v2.json` | `964ea69a8a11a96655ce88d1c3a207f0c9ea1ebad162c5cbc5c2c3bf7cffaccd` | Local OpenAI-compatible adapter path verified without external network |

## Verification Commands

Focused release checks:

```bash
env UV_CACHE_DIR=/private/tmp/aegis-uv-cache \
uv run ruff check \
  README.md \
  docs/aegis-watchman-capstone-release-2026-06-26.md \
  docs/aegis-watchman-release-readiness-2026-06-25.md \
  docs/nimbus-training-corpus.md \
  src/aegis/proxy/smoke.py \
  src/aegis/proxy/mock_app.py \
  src/aegis/proxy/nimbus_profile.py \
  src/aegis/detectors/nimbus.py \
  src/aegis/console/service.py \
  tests/aegis/test_proxy_smoke.py \
  tests/aegis/test_nimbus_runtime.py \
  tests/aegis/test_console.py

env UV_CACHE_DIR=/private/tmp/aegis-uv-cache \
uv run pytest --no-cov -q \
  tests/aegis/test_console.py \
  tests/aegis/test_nimbus_runtime.py \
  tests/aegis/test_nimbus_runtime_beta_eval.py \
  tests/aegis/test_nimbus_profile.py \
  tests/aegis/test_nimbus_promotion_evidence.py \
  tests/aegis/test_proxy_smoke.py \
  tests/aegis/test_dp_honey_paper_evidence.py \
  tests/aegis/test_proxy_cift_smoke.py \
  tests/aegis/test_proxy_cift_capability.py
```

Evidence hygiene checks for the release-critical artifacts:

```bash
rg -n "safe-canary|ghp_real|sk_live_|github_pat_" \
  introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_gateway_smoke_integrated_refresh_v1.json \
  introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_gateway_smoke_integrated_refresh_audit_v1.jsonl \
  introspection/data/reports/aegis_default_mock_provider_smoke_learned_nimbus_beta_v3.json \
  introspection/data/reports/aegis_default_mock_provider_smoke_learned_nimbus_beta_audit_v3.jsonl \
  introspection/data/reports/aegis_default_mock_provider_smoke_dp_honey_segment_v2.json \
  introspection/data/reports/aegis_default_mock_provider_smoke_dp_honey_segment_audit_v2.jsonl
```

Expected result: no matches. Documentation may contain illustrative forbidden
marker strings in commands; release evidence artifacts must not persist raw
credential values.

## Artifact Policy

The release bundle uses committed, hashed evidence reports under
`introspection/data/reports/`. Local generated corpora under
`introspection/data/runtime_turns_watchman_semantic_*.jsonl`,
`introspection/data/structured_prompts_watchman_semantic_*.jsonl`, and
`introspection/data/trace_records_watchman_semantic_*.jsonl` are intentionally
ignored unless a future evidence binder explicitly requires them by hash. The
local `Research/2304.14997v4.pdf` download is also ignored because it is not
part of the selected release evidence bundle.

## Remaining Gaps

- Learned NIMBUS promotion: needs promoted learned runtime artifact, common live
  head-to-head corpus, and evidence that the learned signal beats or complements
  deterministic NIMBUS under sealed/live gates.
- External real-provider evidence: loopback OpenAI-compatible evidence exists,
  but no credentialed external provider smoke is included.
- Credential-need inference: deterministic explicit-slot/tool/schema/env/config
  coverage exists, but broad semantic credential-need inference is not claimed.
- Observability productionization: console/audit are local and redacted, but not
  backed by production storage, retention, authz, or multi-operator controls.
