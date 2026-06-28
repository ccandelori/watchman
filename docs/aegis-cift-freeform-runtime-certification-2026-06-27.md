# Aegis CIFT Freeform Runtime Certification - 2026-06-27

## Verdict

CIFT now has two separate Qwen/Qwen3-4B on MPS runtime routes:

| Route | Status | Runtime feature | Use |
| --- | --- | --- | --- |
| Selected-choice | Certified reference route | `selected_choice_window_layer_21` | Paper-shaped semantic-indirection prompts with selected-choice metadata. |
| Freeform | Certified reference route | `final_token_layer_12` | Ordinary Hermes/Ollama/OpenAI-compatible local-agent chat when selected-choice metadata is absent. |

This is not a full Aegis/Watchman production-readiness claim. The narrower supported claim is:

> Aegis CIFT supports model-specific certified runtime enforcement for Qwen/Qwen3-4B on MPS, with a selected-choice route for paper-shaped prompts and a freeform route for ordinary local-agent traffic. Other models are unsupported until they pass their own calibration, sealed holdout, live runtime, gateway smoke, and hardened release gate.

## Model Binding

| Field | Value |
| --- | --- |
| Model id | `Qwen/Qwen3-4B` |
| Revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Device | `mps` |
| Dtype | `device` |
| Hidden size | `2560` |
| Layer count | `36` |
| Tokenizer hash | `41e00eccf531cffc2e562d38bdd879d41e5044ea279af5b73c6a32aabcc8fe04` |
| Special tokens hash | `edcb2fc2acbbe77f858a9c4fe51295ffdb84711efba5703ec5906b3d67282569` |
| Chat template hash | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |

## Corpus

The promoted freeform candidate uses the v14 freeform corpus. It is shaped for normal chat/runtime traffic and does not rely on selected-choice geometry.

| Split | Rows | Label balance | Path | SHA-256 |
| --- | ---: | --- | --- | --- |
| Normalized trace records | 5,550 | 1,850 benign / 1,850 exfiltration_intent / 1,850 secret_present_safe | `introspection/data/trace_records_watchman_v14_freeform_5550_normalized.jsonl` | `0ced732cfebc527bca33af04d61f6f41b0c54eb05c74707d174b1b198107ca1c` |
| Calibration trace records | 4,428 | 1,476 each label | `introspection/data/trace_records_watchman_v14_freeform_4428_calibration.jsonl` | `fcf425a9d5d79bd2bbe22e0f6a529827f3a9d7f7c55f6aacc7e11527aa0e0527` |
| Sealed trace records | 1,122 | 374 each label | `introspection/data/trace_records_watchman_v14_freeform_1122_sealed_holdout.jsonl` | `ace5e31e87a409afaafa12a38f842e423cd740475500fa50df7123272f37ee94` |
| Runtime turns | 1,122 | 374 each label | `introspection/data/runtime_turns_watchman_v14_freeform_1122_sealed_holdout_qwen3_4b.jsonl` | `f09af980920f4a22ddd75fe4fa3ff9d08b7f8bbae7af2319e507a0abcc419cec` |
| Calibration prompts | 4,428 | 1,476 each label | `introspection/data/structured_prompts_watchman_v14_freeform_4428_calibration_qwen3_4b.jsonl` | `1fc6c482c7bfa725164f5e40bddf57eb6d647ad718b955f419ffc25e29ee1444` |
| Sealed prompts | 1,122 | 374 each label | `introspection/data/structured_prompts_watchman_v14_freeform_1122_sealed_holdout_qwen3_4b.jsonl` | `91f43c3a4c752213dea3469c2a8ad6a0057cafa6451138e3b881f2bb60232a5c` |

The v14 supplement adds hard freeform shapes, including DP-HONEY gateway cases, direct-paste smoke shape, protected-context cases, near-neighbor phrasing, tool payloads, and redacted raw-secret print-exactly requests.

## Probe Results

Paper-faithful+ was applied: the paper MLP was trained and evaluated, but it was not promoted because the corrected live/sealed evidence favored the linear challenger.

| Evaluation | Paper MLP `K -> 128 -> 64 -> 1` | Linear/logistic challenger | Decision |
| --- | ---: | ---: | --- |
| Grouped CV macro F1 | `0.9899215270114706` | `0.9898373983739838` | Paper MLP slightly higher in grouped CV. |
| Grouped CV FNR | `0.013550135501354976` | `0.013550135501354976` | Tie. |
| Grouped CV FPR | `0.006662149954832882` | `0.006775067750677506` | Paper MLP slightly lower. |
| Sealed/live macro F1 | `0.4` | `1.0` | Linear wins operational evidence. |
| Sealed/live FN/FNR | `374 / 1.0` | `0 / 0.0` | Linear wins decisively. |
| Sealed/live FP/FPR | `0 / 0.0` | `0 / 0.0` | Tie. |

The promoted freeform runtime is linear/logistic. The paper MLP remains useful as a baseline, but the exported live runtime MLP candidate allowed every exfiltration-intent turn in the corrected sealed/live run.

## Runtime Evidence

| Evidence | Result | Path |
| --- | --- | --- |
| Promoted freeform runtime | Runtime SHA `9a9bff28f4dc43a58b9ed21b7de053e3a70fd412e753916fe1cda64d439ea9ed`; feature `final_token_layer_12`; model bundle `cift_qwen3_4b_watchman_v14_freeform_final_token_l12_linear_v1` | `introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/models/cift_qwen3_4b_watchman_v14_freeform_final_token_l12_promoted_runtime_v1.json` |
| Hardened release gate | `production_release_eligible=true`; no failed requirements | `introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_release_gate_v1.json` |
| Existing certification verification | `status=certified`; `support_state=runtime-enforceable`; runtime/model/evidence hashes matched | `introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_certification_verification_v1.json` |
| Evidence-bound manifest | `support_state=certified`; 19 required artifacts | `introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_certification_workflow_v1.json` |
| Workflow run | Execute-mode verification passed; no replayed sealed/live workload | `introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_certification_workflow_run_v1.json` |
| Linear live runtime prevention | 1,122 turns; FN=0/FNR=0.0; FP=0/FPR=0.0; 374 blocks and 748 allows | `introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_linear_runtime_prevention_v1.json` |
| Paper MLP live runtime prevention | 1,122 turns; FN=374/FNR=1.0; FP=0/FPR=0.0; not promoted | `introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_paper_mlp_runtime_prevention_v1.json` |
| Official strict gateway smoke input | Strict `/ready` captured; benign allowed with provider completion; exfiltration blocked before provider; safe credential honeytoken-substituted; selected-choice route rechecked; FN=0/FNR=0.0; FP=0/FPR=0.0 | `introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_gateway_smoke_v1.json` |
| Post-release strict gateway smoke | Fresh gateway started from the final strict env; `/ready` advertises release gate SHA `9bca27cd68010c39aa5711b1e63d3e411d5a46ac91e678f359017da30bf22412` and runtime SHA `9a9bff28f4dc43a58b9ed21b7de053e3a70fd412e753916fe1cda64d439ea9ed`; FN=0/FNR=0.0; FP=0/FPR=0.0 | `introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_post_release_gateway_smoke_v1.json` |
| Selected-choice release gate | `production_release_eligible=true`; selected-choice readout count is `4` | `introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_release_gate_v1.json` |

The official gateway smoke is the release-gate input. The post-release smoke is intentionally separate: strict readiness reports runtime and release-gate hashes, while the release gate also hashes the smoke report. Keeping the post-release smoke outside the promotion hash avoids a circular self-hash while still proving that the final strict env starts and blocks correctly.

## Operator Commands

Verify the existing freeform certification without replaying offline evidence:

```bash
UV_CACHE_DIR=/private/tmp/aegis-uv-cache \
uv run certify-cift-local-model verify-existing \
  --repository-root . \
  --runtime-model introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/models/cift_qwen3_4b_watchman_v14_freeform_final_token_l12_promoted_runtime_v1.json \
  --expected-runtime-sha256 9a9bff28f4dc43a58b9ed21b7de053e3a70fd412e753916fe1cda64d439ea9ed \
  --certification-manifest introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_certification_workflow_v1.json \
  --certification-report introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_certification_workflow_run_v1.json \
  --certification-artifact-root . \
  --release-gate-report introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_release_gate_v1.json \
  --verification-report introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_certification_verification_v1.json \
  --certification-manifest-sha256 9bd48584b8a120e08847c87349d0540c47282187e7cad5cc9e0568437467f1fb \
  --certification-report-sha256 5af0c3a611617464b2b38d216a98079e20d2d967edd53492358b1bb708c699c1 \
  --release-gate-report-sha256 9bca27cd68010c39aa5711b1e63d3e411d5a46ac91e678f359017da30bf22412 \
  --model-id Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --required-device mps \
  --expected-hidden-size 2560 \
  --expected-layer-count 36 \
  --expected-tokenizer-sha256 41e00eccf531cffc2e562d38bdd879d41e5044ea279af5b73c6a32aabcc8fe04 \
  --expected-special-tokens-sha256 edcb2fc2acbbe77f858a9c4fe51295ffdb84711efba5703ec5906b3d67282569 \
  --expected-chat-template-sha256 a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8 \
  --expected-feature-key final_token_layer_12 \
  --expected-pooling-method final_token \
  --expected-dtype-name device \
  --expected-detector-name cift_runtime \
  --expected-extractor-id trusted-activation-sidecar \
  --expected-feature-source self_hosted_activation_extractor \
  --expected-prompt-renderer aegis_trace_bridge_v1 \
  --expected-selected-choice-geometry semantic_indirection_v1 \
  --expected-selected-choice-readout-token-count 4
```

Verify the materialized evidence-bound certification:

```bash
UV_CACHE_DIR=/private/tmp/aegis-uv-cache \
uv run python introspection/scripts/run_cift_certification_workflow.py \
  --repository-root . \
  --workflow-manifest introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_certification_workflow_v1.json \
  --output introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_certification_workflow_run_v1.json \
  --execute \
  --command-timeout-seconds 30.0
```

Start the sidecar with both certified feature keys:

```bash
export AEGIS_CIFT_EXTRACTOR_API_KEY="set-a-deployment-secret"

UV_CACHE_DIR=/private/tmp/aegis-uv-cache \
uv run python introspection/scripts/run_cift_extractor_sidecar.py \
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

Start the gateway with both strict route bindings:

```bash
export AEGIS_CIFT_EXTRACTOR_API_KEY="set-a-deployment-secret"
export AEGIS_PROVIDER=openai_compatible
export AEGIS_OPENAI_BASE_URL=http://127.0.0.1:11434/v1
export AEGIS_OPENAI_API_KEY=ollama
export AEGIS_OPENAI_MODEL=qwen3:4b
export AEGIS_AUDIT_JSONL_PATH=/tmp/aegis-cift-freeform-audit.jsonl

set -a
source introspection/data/reports/qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_strict_deployment_env.sh
source introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_strict_deployment_env.sh
set +a

UV_CACHE_DIR=/private/tmp/aegis-uv-cache \
uv run aegis-proxy --host 127.0.0.1 --port 8000
```

Run the live gateway smoke:

```bash
AEGIS_CIFT_EXTRACTOR_API_KEY="$AEGIS_CIFT_EXTRACTOR_API_KEY" \
UV_CACHE_DIR=/private/tmp/aegis-uv-cache \
uv run aegis-proxy-cift-smoke \
  --url http://127.0.0.1:8000 \
  --sidecar-url http://127.0.0.1:9000 \
  --gateway-model qwen3:4b \
  --report-id qwen3_4b_watchman_v14_freeform_final_token_l12_post_release_gateway_smoke_v1 \
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
  --output introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/qwen3_4b_watchman_v14_freeform_final_token_l12_post_release_gateway_smoke_v1.json
```

The generic certification entry point for a new local hidden-state-capable model is:

```bash
UV_CACHE_DIR=/private/tmp/aegis-uv-cache \
uv run python introspection/scripts/certify_cift_local_model.py \
  --repository-root . \
  --certification-id <model-specific-certification-id> \
  --model-id <huggingface-or-local-model-id> \
  --revision <immutable-revision-or-content-hash> \
  --corpus <calibration-jsonl> \
  --runtime-turns <sealed-runtime-turns-jsonl> \
  --selected-choice-runtime-model <certified-selected-choice-runtime-json> \
  --output-dir <model-specific-output-dir> \
  --workflow-manifest <model-specific-workflow-manifest-json> \
  --run-report <model-specific-workflow-run-json> \
  --training-dataset-id <dataset-id> \
  --task non_exfiltration_vs_exfiltration \
  --positive-label exfiltration_intent \
  --behavior-id freeform_exfiltration_intent \
  --behavior-description "<model-specific behavior claim>" \
  --layers <candidate-layer-list> \
  --pooling <candidate-pooling-list> \
  --candidate-feature <selected-feature-key> \
  --device <mps-or-cuda-or-cpu-policy> \
  --prompt-renderer aegis_trace_bridge_v1 \
  --selected-choice-geometry semantic_indirection_v1 \
  --selected-choice-readout-token-count 4 \
  --dtype device \
  --metric-threshold 0.95 \
  --ablation-delta-threshold 0.0 \
  --created-at <iso-8601-timestamp> \
  --execute \
  --command-timeout-seconds 30.0
```

## Readiness Boundaries

| Component | Status | Boundary |
| --- | --- | --- |
| CIFT selected-choice route | Certified reference for Qwen3-4B/MPS | Requires selected-choice metadata and readout count `4`. Metadata failures fail closed. |
| CIFT freeform route | Certified reference for Qwen3-4B/MPS | Covers ordinary local-agent chat only for the exact bound model/tokenizer/template/runtime artifact. |
| Generic CIFT workflow | Runnable certification path | Other models remain unsupported until their own workflow, sealed/live metrics, gateway smoke, and release gate pass. |
| DP-HONEY | Demo/local runtime integration visible in smoke | Real external secret-manager integration and autonomous credential brokering remain production work. |
| NIMBUS | Deterministic/beta evidence path | Paper-faithful learned/session leakage critic is not promoted. |
| Gateway | Local operational proof for Qwen3-4B CIFT | Broader provider matrix and production deployment hardening remain separate work. |
| Observability | Evidence-grade local JSON reports and audit paths | No hardened multi-user operator store yet. |

## Remaining Gaps

- Certify a second model only as a portability check after Qwen3-4B remains stable; Qwen3-0.6B is not supported from Qwen3-4B evidence.
- Keep selected-choice and freeform routes separate; do not treat missing selected-choice metadata as selected-choice success.
- Resolve the paper MLP live-runtime failure before any future MLP promotion.
- Continue hardening DP-HONEY, NIMBUS, gateway provider coverage, and durable observability before making a full Aegis/Watchman production claim.
