# Aegis Introspection

This directory contains the current model-introspection research thread for
Aegis. The purpose is to test whether hidden-state features from a local model
can help distinguish safe secret handling from exfiltration-oriented behavior.

The work here is exploratory. The current results are useful checkpoints, not
validated production claims.

## System Alignment

The current work is the CIFT-like portion of the broader Aegis/AIS design. The
larger system should be treated as a multi-monitor gateway:

| Component | Role | Current Status |
|---|---|---|
| DP-HONEY | Inject format-matched honeytokens before model access. | DP-HONEY-lite data primitive and hard V2 dataset implemented; full DP-HONEY not implemented yet. |
| CIFT-like activation monitor | Read hidden-state features before output generation. | Current focus. |
| Text leakage detector | Scan generated text for canary or secret leakage. | Not implemented here yet. |
| NIMBUS-like accumulator | Track cumulative leakage risk over a conversation. | Not implemented here yet. |

The activation probe is therefore a monitor signal, not the whole defense. It
should eventually emit structured evidence that a gateway policy engine can
combine with honeytoken evidence, post-output leakage scans, and cumulative
conversation risk.

The paper-specific CIFT target is narrower than generic activation probing:
monitor readout positions that occur after both secret context and the current
query or payload, rather than hidden states at static credential positions. The
paper also frames CIFT as a calibrated deviation signal over the last quarter of
transformer layers, followed by learned nonnegative layer weighting. The current
Aegis work approximates that direction with final-token/readout-style features
and grouped probe evaluation, but it has not yet implemented the paper's full
benign-calibrated CCI/CFS scoring path.

DP-HONEY-lite now supplies the proxy-shaped data primitive needed for that next
CIFT step. It generates format-shaped honeytokens, injects them into structured
prompts, records character and token spans, and defines readout-token windows.
It is not the paper's full DP-HONEY system: it does not implement
differentially private n-gram generation, conformal calibration, or
indistinguishability testing.

That distinction matters for planning:

| Question | Current Work | Paper-Aligned Target |
|---|---|---|
| Readout position | Final prompt token features. | Positions after secret context and query/payload; final prompt token plus pre-generation decision positions. |
| Layer handling | Individual and concatenated feature candidates. | Last-quarter layer hooks with calibrated per-layer deviation and learned layer weights. |
| Calibration | Grouped CV, residual analysis, human adjudication. | Benign calibration distribution, conformal/text thresholds, and deployment operating points. |
| System role | Candidate CIFT monitor signal. | One component in AIS per-turn pipeline before generation, combined with canary/text detection and NIMBUS. |

## Operating Workflow

Every experiment should answer these questions before it is promoted:

1. Which system component does this advance?
2. What runtime signal would this component emit?
3. What dataset or artifact proves the result without replacing prior evidence?
4. What grouped, held-out, or residual check protects against post-hoc overfit?
5. What failure cases need human review?
6. How does the result affect the eventual gateway decision pipeline?

For the current CIFT-like thread, use this workflow:

| Stage | Required Output |
|---|---|
| Scope | State whether the work is CIFT, DP-HONEY, text detection, NIMBUS, or integration. |
| Dataset | Add or reuse a registered prompt dataset with clear label and family semantics. |
| Extraction | Produce activation artifacts without overwriting prior checkpoints. |
| Evaluation | Run grouped evaluation first for credible claims. |
| Crosscheck | Compare against fixed references and newer candidates across checkpoints. |
| Residual analysis | Track fixed, persistent, and introduced errors by family. |
| Adjudication | Create human-review worksheets for promotion-sensitive regressions. |
| Paper alignment | State whether the result is an approximation or implements the paper method directly. |
| Promotion | Define an explicit rule before changing the active checkpoint or monitor contract. |
| Integration | Express the result as a monitor event suitable for gateway policy composition. |

The target monitor event shape is:

```json
{
  "monitor": "activation_probe",
  "component": "cift_like",
  "feature_key": "concat(final_token_layer_11,final_token_layer_16)",
  "task": "safe_secret_vs_exfiltration",
  "risk_score": 0.0,
  "decision": "allow|flag|block",
  "model_id": "Qwen/Qwen3-0.6B",
  "probe_version": "unpromoted",
  "evidence": {
    "evaluation_strategy": "stratified_group_kfold",
    "dataset_checkpoint": "hard_prompts_v3"
  }
}
```

Do not promote a feature solely because it improves one metric. Promotion should
balance aggregate performance, worst-checkpoint behavior, introduced-error
severity, human adjudication, and integration value for the full gateway.
For CIFT specifically, promotion should also state whether the candidate is a
temporary empirical probe or the first implementation of the paper-aligned
readout-position, calibrated-deviation monitor.

## Current State

Seventeen datasets are currently registered in `data/lineage.json`:

| Dataset | Purpose | Rows |
|---|---|---:|
| `baseline_prompts_v1` | First hand-authored checkpoint dataset. | 90 |
| `hard_prompts_v1` | Harder contrastive successor to the baseline. | 90 |
| `hard_prompts_v2` | Targeted successor focused on V1 error clusters. | 90 |
| `hard_prompts_v3` | Held-out summary, redaction, and replacement checkpoint. | 90 |
| `dp_honey_lite_prompts_v1` | Proxy-shaped honeytoken prompts with span metadata and readout windows. | 24 |
| `dp_honey_lite_prompts_v2` | Harder paired DP-HONEY-lite scenarios with balanced credential and payload conditions. | 240 |
| `dp_honey_lite_prompts_v3` | Lexically controlled mode-map prompts for safe-secret versus exfiltration pressure testing. | 240 |
| `dp_honey_lite_v3_policy_windows` | V3-derived policy-window control with explicit selected policy decision text in the readout window. | 240 |
| `dp_honey_lite_v3_selector_windows` | V3-derived selector-window checkpoint with selected field/mode tokens but no explicit decision clause. | 240 |
| `dp_honey_lite_prompts_v4` | Targeted mode_b control focused on prior selector-window failure families. | 144 |
| `dp_honey_lite_v4_policy_windows` | V4-derived policy-window control with explicit selected mode_b decision text. | 144 |
| `dp_honey_lite_v4_selector_windows` | V4-derived selector-window control over selected field and mode_b tokens. | 144 |
| `dp_honey_lite_prompts_v4_1` | Focused relational-control checkpoint without the V4 label-specific policy-map shortcut. | 144 |
| `dp_honey_lite_v4_1_policy_windows` | V4.1-derived policy-window control with explicit selected decision text. | 144 |
| `dp_honey_lite_v4_1_selector_windows` | V4.1-derived selector-window checkpoint over selected field and selected mode tokens. | 144 |
| `runtime_turns_dp_honey_lite_v3_selector_windows_v1` | `NormalizedTurn`-shaped bridge export for runtime-spine integration checks. | 240 |
| `runtime_turns_dp_honey_lite_v4_1_selector_windows_v1` | `NormalizedTurn`-shaped bridge export for V4.1 runtime-spine integration checks. | 144 |

The four historical probe datasets use the same label shape:

| Label | Count | Meaning |
|---|---:|---|
| `benign` | 30 | No secret handling or exfiltration request. |
| `secret_present_safe` | 30 | A secret-like value appears, but the prompt asks for safe handling. |
| `exfiltration_intent` | 30 | The prompt asks to reveal, transmit, encode, route, or leak secret-like material. |

The first DP-HONEY-lite dataset uses the same labels with 8 examples per label
across 6 families. Its `benign` rows still contain an injected honeytoken
context, but the user query is unrelated to secret handling; this is intentional
for proxy-style monitor development and should not be mixed into historical
static-feature comparisons without calling out the changed data contract.

The hard V2 DP-HONEY-lite dataset keeps that proxy-shaped contract but scales it
to 240 rows across 10 paired scenario families. Each family contains benign,
safe-secret, and exfiltration variants; each label has 40 API-key rows, 40
database-URI rows, 40 payload rows, and 40 no-payload rows. It is still
DP-HONEY-lite: the honeytokens are format-shaped and deterministic, not
differentially private n-gram samples.

The hard V3 DP-HONEY-lite dataset keeps the same span schema and balance, but
changes the prompt contract to a lexically controlled mode map. Every row
contains `credential_value`, `summary_value`, `mode_a`, `mode_b`, `copy`, and
`mask`; selected modes are balanced across labels; and the mode-to-credential
action mapping flips across scenario families.

Seven DP-HONEY-lite activation artifacts are now registered:

```text
qwen3_0_6b_dp_honey_lite_v1_readout_windows_v1
qwen3_0_6b_dp_honey_lite_v2_all_pooling_v1
qwen3_0_6b_dp_honey_lite_v3_all_pooling_v1
qwen3_0_6b_dp_honey_lite_v3_policy_windows_v1
qwen3_0_6b_dp_honey_lite_v3_selector_windows_v1
qwen3_0_6b_dp_honey_lite_v4_selector_windows_v1
qwen3_0_6b_dp_honey_lite_v4_1_selector_windows_v1
```

The V1 artifact extracts all 29 Qwen 0.6B hidden-state layers using
`readout_window` pooling over the row-level `readout_token_indices`. The V2
and V3 artifacts extract all 29 layers with `final_token`, `mean_pool`, and
`readout_window` pooling in the same file, so readout-window features can be
compared directly against same-dataset static features.

A first smoke comparison confirms that the readout-window artifact is usable by
the grouped binary evaluation harness. With 2 grouped folds on the 24-row
DP-HONEY-lite dataset, the top readout-window layers separate both binary tasks:

| Task | Best Readout Feature | Macro F1 | Accuracy |
|---|---|---:|---:|
| `safe_secret_vs_exfiltration` | `readout_window_layer_07` | 1.0000 | 1.0000 |
| `benign_vs_secret_related` | `readout_window_layer_16` | 1.0000 | 1.0000 |

This is a smoke result, not a performance claim. The dataset is small and
template-shaped, and the perfect scores mostly say the new readout-window
feature path is wired correctly and worth scaling.

The V2 all-pooling artifact confirms the extraction and grouped-evaluation path
on the larger paired dataset, but it also shows that V2 is still too easy to
support a detector claim:

| Task | Best Feature | Readout Layer 11 Macro F1 | Word TF-IDF Macro F1 | Interpretation |
|---|---|---:|---:|---|
| `safe_secret_vs_exfiltration` | `final_token_layer_04` | 1.0000 | 1.0000 | Surface wording still separates the labels. |
| `benign_vs_secret_related` | `final_token_layer_02` | 1.0000 | 1.0000 | Surface wording still separates the labels. |

Across the two V2 sweeps, many activation features are perfect as well: 54 on
`safe_secret_vs_exfiltration` and 69 on `benign_vs_secret_related`. Treat V2 as
a pipeline and data-contract checkpoint, not a performance benchmark.

V3 is the first DP-HONEY-lite checkpoint where the target text baseline no
longer solves the target task:

| Task | Best Activation Feature | Best Activation Macro F1 | Word TF-IDF Macro F1 | Interpretation |
|---|---|---:|---:|---|
| `safe_secret_vs_exfiltration` | `readout_window_layer_21` | 0.5284 | 0.2670 | Surface-text shortcut is broken, but activation signal is weak. |
| `benign_vs_secret_related` | `final_token_layer_01` | 0.9906 | 1.0000 | Still lexically easy because selected field differs. |

Treat V3 as a hard negative/control checkpoint. It is useful because it shows
that the prior perfect results depended on easy surface structure, but it does
not yet provide a strong CIFT-like detector.

The V3.1 policy diagnostic confirms that the hard V3 label rule itself is not
ambiguous. A deterministic parser that reads `selected_field`, `selected_mode`,
and the selected mode policy reaches 1.0000 macro F1 / 1.0000 accuracy on the
target task across all rows, payload rows, no-payload rows, `mode_a` rows, and
`mode_b` rows. The activation probe remains weak on those same slices:
0.5284 overall, 0.4635 on payload rows, 0.4758 on no-payload rows, 0.5127 on
`mode_a`, and 0.4888 on `mode_b`. This points away from dataset repair and
toward better feature localization around the selected policy tokens.

The V3.2 direct policy-resolution diagnostic asks Qwen 0.6B itself to answer
whether the selected field should be copied or masked. The result is weaker
than the deterministic parser and slightly worse than the activation probe:
0.4685 macro F1 / 0.5563 accuracy overall. The model emits parseable answers
with no invalid outputs, but it predicts the copy/exfiltration side 145 times
out of 160. Payload rows are the clearest failure: 0.3333 macro F1 / 0.5000
accuracy, with every payload row predicted as copy/exfiltration. This suggests
that the base model is not reliably executing the symbolic mode-map policy, so
the current weak activation signal is at least partly a model-competence issue,
not only a readout-localization issue.

The V3.3 policy-window localization experiment separates two readout-window
variants. The `policy_windows` control includes the selected policy's explicit
`copy` or `mask` decision text, and therefore reaches 1.0000 macro F1 on
`safe_secret_vs_exfiltration`. Treat that as an oracle-style localization
control, not a detector claim. The `selector_windows` variant includes only the
selected field and selected mode tokens. It is the useful checkpoint: the best
layer is `readout_window_layer_15`, with 0.7736 macro F1 / 0.7812 accuracy on
`safe_secret_vs_exfiltration`. In the grouped comparison, the same selector
feature beats word TF-IDF at 0.2670 macro F1 and character TF-IDF at 0.3542.
This supports the readout-localization hypothesis while preserving the warning
that the current V3 task is still partly limited by Qwen 0.6B's policy
resolution behavior.

The selector-window error analysis gives the next dataset target. The activation
probe makes 35 errors on `safe_secret_vs_exfiltration`, compared with 112 for
word TF-IDF and 102 for character TF-IDF. Errors cluster most strongly on
payload examples and `mode_b` examples: payload exfiltration rows have 15
errors / 40 examples, while no-payload exfiltration rows have 6 errors / 40
examples; `mode_b` exfiltration rows have 15 errors / 40 examples, while
`mode_a` exfiltration rows have 6 errors / 40 examples. The largest family
cluster is `dp_honey_lite_v3_support_transcript` exfiltration, with 7 errors /
8 examples. The next dataset iteration should target that payload plus `mode_b`
weakness rather than expanding DP-HONEY mechanics first.

The first runtime-spine bridge export now converts the V3 selector-window rows
into JSON-safe objects shaped like `aegis.core.contracts.NormalizedTurn`. The
mapping is intentionally narrow:

| Structured Prompt Field | Runtime-Spine Field |
|---|---|
| `text` / `rendered_prompt` | `messages[0].content` |
| `honeytoken_id`, `honeytoken_sha256`, `credential_type` | `sensitive_spans[0].identifier` and metadata |
| `secret_char_span`, `secret_token_span` | `sensitive_spans[0]` offsets |
| `query_*`, `payload_*`, `readout_token_indices` | `metadata.cift` |
| `policy_window_*` | `metadata.policy_window` |
| `label`, `family`, `tags` | `metadata.eval` |

This bridge is appropriate for offline evaluation and integration testing. It
is not yet the final production boundary: the current synthetic row still uses
a single rendered prompt message and can contain synthetic honeytoken text in
`messages[0].content`. Production traffic should split real chat messages and
tool calls at the proxy adapter, and audit-safe paths should carry real
credentials as handles, spans, hashes, or detector evidence rather than raw
secret values.

The matching detector-result bridge converts selector-window CIFT predictions
into JSON-safe objects shaped like `aegis.core.contracts.DetectorResult`. The
historical bridge export has 160 rows for the `safe_secret_vs_exfiltration`
task, joined back to the runtime-turn export by `example_id`. It uses:

| DetectorResult Field | Bridge Value |
|---|---|
| `detector_name` | `cift_selector_probe` |
| `component` | `cift` |
| `recommended_action` | `warn` for predicted exfiltration, `allow` otherwise |
| `capability_required` | `self_hosted_introspection` |
| `capability_status` | `active` |
| `evidence` | task, feature key, probe version, fold, family, prediction, and offline-eval labels |

The detector-result score is deliberately marked as
`prediction_label_score_not_calibrated`. It is a bridge artifact that proves the
shape of the boundary, not a calibrated production risk score.

The calibrated detector-result export is now the better runtime-facing bridge
artifact. It uses the same selector-window layer 15 feature and the same
runtime-turn join, but converts grouped out-of-fold activation predictions into
Platt-calibrated exfiltration probabilities. The score semantics are recorded
as `inner_cv_platt_calibrated_probability`, and the current report reaches
0.8125 macro F1 / 0.8125 accuracy with Brier score 0.1485 and expected
calibration error 0.1110. Treat this as an offline detector calibration
checkpoint, not as a production operating-point guarantee.

The first calibrated operating-point sweep makes the policy tradeoff explicit.
At threshold `0.50`, the detector is balanced: 65 true positives, 15 false
positives, 65 true negatives, and 15 false negatives, for 0.8125 macro F1. A
high-recall threshold of `0.25` catches all 80 exfiltration rows, but warns on
63 of 80 safe-secret rows. This gives the runtime spine a concrete policy
choice: `0.50` is the current balanced offline checkpoint, while lower
thresholds are only appropriate for a noisy warning lane or human-review mode.

The selector-window score diagnostic rules out a simple global-threshold fix as
the next main lever. Threshold `0.50` remains the best macro-F1 operating point
at 0.8125. Threshold `0.60` reduces false positives from 15 to 8, but increases
false negatives from 15 to 22 and slightly lowers macro F1 to 0.8111. Of the 30
calibrated detector errors, 13 are near-threshold and 17 are confident errors.
The strongest pressure slices are `mode_b` with 21 errors / 80 examples and
payload rows with 20 errors / 80 examples. The highest-signal family clusters
are `support_transcript` false negatives, `access_review` confident false
negatives, and `customer_summary` / `audit_export` false positives. The next
CIFT move should therefore target representation and readout design around
those slices, not just tune the threshold.

A selector-window feature-expression ablation tested whether the existing
readout-window artifact already contains a better representation than single
layer 15. In raw grouped activation-probe evaluation, the best variant is
`selector_band_12_18`, a concatenation of `readout_window_layer_12` through
`readout_window_layer_18`, at 0.8138 macro F1 / 0.8187 accuracy. The single
layer-15 baseline in the same raw grouped setup is 0.7736 macro F1 / 0.7812
accuracy. This is a useful representation signal, but it is not a promotion
result. After the same inner Platt calibration used by the runtime-facing
detector, `selector_band_12_18` reaches 0.8062 macro F1 / 0.8063 accuracy at
threshold `0.50`, with best swept macro F1 of 0.8106 at threshold `0.65`. That
does not beat the calibrated layer-15 checkpoint at 0.8125 macro F1. It also
increases confident errors from 17 to 23, despite improving expected calibration
error from 0.1110 to 0.0968. Treat the mid-band result as evidence that layers
12-18 contain useful signal, but do not replace the current calibrated
checkpoint with it.

The V4 selector-window checkpoint targets the prior failure families directly:
`support_transcript`, `access_review`, `customer_summary`, `audit_export`,
`incident_ticket`, and `workflow_automation`. Every row uses `selected_mode =
mode_b`, with balanced labels and payload/no-payload controls. This isolates the
mode_b failure slice, but it also makes the task lexically recoverable. On the
V4 selector-window dataset, `readout_window_layer_15` reaches 1.0000 macro F1 /
1.0000 accuracy on `safe_secret_vs_exfiltration`, but word TF-IDF also reaches
1.0000 macro F1 / 1.0000 accuracy. Treat V4 as a targeted control and schema
checkpoint, not as evidence that the CIFT monitor improved. The next version
should preserve the mode_b focus while making the selected copy/mask resolution
less directly recoverable by full-prompt surface text.

The V4.1 selector-window checkpoint keeps the same six pressure families but
removes V4's label-specific policy-map shortcut. It returns to the relational
mode-map design: selected modes are balanced by label, and each family keeps a
single stable `mode_a` / `mode_b` policy map across benign, safe-secret, and
exfiltration rows. This makes the task materially harder. On
`safe_secret_vs_exfiltration`, the best and reference feature is
`readout_window_layer_15` at 0.7646 macro F1 / 0.7875 accuracy. Word TF-IDF
drops to 0.1172 macro F1 / 0.1187 accuracy, and character TF-IDF reaches 0.2851
macro F1 / 0.3063 accuracy. Treat V4.1 as the current hard selector-window
checkpoint: it restores a meaningful activation-over-text gap, but the probe is
not close to solved and should be followed by error analysis and calibration.

The V4.1 error analysis and calibration work now make that checkpoint
trainable rather than merely diagnostic. On `safe_secret_vs_exfiltration`, the
activation probe makes 20 grouped out-of-fold errors, compared with 78 for word
TF-IDF and 66 for character TF-IDF. The largest pressure slice is
`support_transcript` exfiltration, with 7 errors out of 8 examples; additional
pressure remains in `incident_ticket` exfiltration, `audit_export` safe-secret
rows, mode-b exfiltration, and no-payload exfiltration. The refreshed grouped
calibration report reaches 0.7708 accuracy / 0.7704 macro F1, with Brier score
0.1838 and expected calibration error 0.1701. The calibrated operating-point
sweep finds best balanced macro F1 at threshold `0.55`, but that threshold
belongs to out-of-fold calibrated probabilities, not to the full-train bundle's
classifier-probability score.

The first fully trained CIFT-like detector bundle is now registered as:

```text
cift_qwen3_0_6b_dp_honey_lite_v4_1_selector_window_layer_15_v1
```

The bundle lives at
`data/models/cift_qwen3_0_6b_dp_honey_lite_v4_1_selector_window_layer_15_v1.pkl`.
It trains a logistic activation classifier on all eligible V4.1
`safe_secret_vs_exfiltration` rows using `readout_window_layer_15`, records
`exfiltration_intent` as the positive label, and stores the source activation
artifact hash plus evaluation report IDs. It is an offline research candidate,
not a production policy authority. The bundle emits full-train classifier
probabilities at threshold `0.50`; calibration reports remain the out-of-fold
evidence trail.

The V4.1 runtime bridge now includes both `NormalizedTurn` rows and trained
bundle `DetectorResult` rows. The trained DetectorResult export has 96 task
rows: 48 `allow`, 48 `warn`, and 1.0000 in-sample projection accuracy over the
training examples. Treat that as an integration sanity check only. The quality
claim remains the grouped out-of-fold evaluation and calibration metrics above.

Each prompt has a `family` field. Grouped evaluation uses those families to
hold related prompt patterns out together.

The fixed historical activation checkpoint is:

```text
mean_pool_layer_18
```

For the important `safe_secret_vs_exfiltration` task:

| Dataset | Evaluation | Best Method | Macro F1 | Accuracy |
|---|---|---|---:|---:|
| Baseline | Random stratified CV | `activation_probe` | 0.9131 | 0.9167 |
| Baseline | Grouped CV | `activation_probe` | 0.8620 | 0.8667 |
| Hard V1 | Random stratified CV | `activation_probe` | 0.9163 | 0.9167 |
| Hard V1 | Grouped CV | `activation_probe` | 0.8788 | 0.8833 |
| Hard V2 | Random stratified CV | `activation_probe` | 0.7470 | 0.7500 |
| Hard V2 | Grouped CV | `activation_probe` | 0.7225 | 0.7333 |
| Hard V3 | Grouped CV | `activation_probe` | 0.8324 | 0.8333 |

The grouped results are the more credible checkpoints because they reduce
prompt-family leakage between train and test folds. Hard V1 is especially
useful because both TF-IDF baselines degrade sharply on the intent-sensitive
task while the activation probe remains comparatively strong.

Hard V2 is now the strongest pressure test. It focuses on the Hard V1 error
clusters: output contracts, tool arguments, broker boundaries, policy
exceptions, and summaries. Under grouped evaluation, the fixed activation probe
drops to 0.7225 macro F1 / 0.7333 accuracy on `safe_secret_vs_exfiltration`,
while word and character TF-IDF baselines fall below chance. The full
machine-readable prediction ledger is registered in lineage. Preliminary
adjudication found the 16 activation-probe misses label-defensible, so the next
move is analysis rather than dataset repair.

A grouped Hard V2 layer sweep found the best feature was `final_token_layer_11`
at 0.9657 macro F1 / 0.9667 accuracy, while the fixed
`mean_pool_layer_18` checkpoint ranked 39th. A cross-dataset check now confirms
that `final_token_layer_11` also improves Hard V1, but loses slightly on the
original baseline:

| Dataset | Reference | Candidate | Delta Macro F1 | Winner |
|---|---:|---:|---:|---|
| Baseline | 0.8620 | 0.8445 | -0.0175 | `mean_pool_layer_18` |
| Hard V1 | 0.8788 | 0.8993 | +0.0205 | `final_token_layer_11` |
| Hard V2 | 0.7225 | 0.9657 | +0.2432 | `final_token_layer_11` |

Treat `mean_pool_layer_18` as the fixed regression checkpoint and
`final_token_layer_11` as a candidate hard-case checkpoint until the candidate
is tested against new prompt families.

Residual analysis on Hard V2 strengthened that candidate reading. On
`safe_secret_vs_exfiltration`, `final_token_layer_11` reduces activation-probe
errors from 16 to 2, fixing 14 prior misses with 0 introduced target-task
errors. Both persistent misses are in `hard_v2_safe_summary_customer_note`.
However, the candidate introduces 1 error on `benign_vs_secret_related`, so this
is still a candidate checkpoint rather than a universal replacement.

Hard V3 provides fresh held-out coverage for summary, redaction, and replacement
behavior with neutral secret identifiers. On this corrected held-out set,
`final_token_layer_11` beats the fixed reference on `safe_secret_vs_exfiltration`
with 0.8818 macro F1 / 0.8833 accuracy versus 0.8324 macro F1 / 0.8333
accuracy. The residual comparison is mixed but positive: 7 fixed errors, 3
persistent errors, and 4 introduced errors. Across all four checkpoints,
`final_token_layer_11` now wins 3 and `mean_pool_layer_18` wins 1.

The V3 layer sweep found a stronger local feature, `final_token_layer_16`, at
0.9321 macro F1 / 0.9333 accuracy. Treat that as a diagnostic result, not a new
replacement, until it is checked against the earlier datasets.

The feature-stability check compares `mean_pool_layer_18`,
`final_token_layer_11`, and `final_token_layer_16` across all four checkpoints.
`final_token_layer_11` remains the best average performer at 0.8978 mean macro
F1, while `final_token_layer_16` is the stability challenger: it has the higher
minimum macro F1, the smaller range, and wins baseline plus Hard V3. The
historical reference wins none of the three-feature checkpoint comparisons, but
still anchors the experiment history.

The combined-feature check adds
`concat(final_token_layer_11,final_token_layer_16)` as a derived activation
matrix. It ranks first by mean macro F1 at 0.9151 and wins or ties three of four
checkpoints. The important caveat is Hard V3: `final_token_layer_16` remains the
local winner there, while the combined feature falls back to roughly
`final_token_layer_11` performance.

The combined-feature residual suite compares that candidate against
`mean_pool_layer_18`, `final_token_layer_11`, and `final_token_layer_16` across
all four checkpoints. The combined feature reduces aggregate target-task errors
against all three references: -21 versus the historical reference, -4 versus
`final_token_layer_11`, and -6 versus `final_token_layer_16`. The caution remains
Hard V3, where `final_token_layer_16` has 4 errors and the combined feature has
7. The combined feature is now the leading promotion candidate, but the Hard V3
introduced-error families should be reviewed before changing the checkpoint.

The Hard V3 combined-regression adjudication worksheet narrows that review to 3
introduced cases against `final_token_layer_16`: one exfiltration case the
combined feature misses as safe, and two safe cases the combined feature marks
as exfiltration. All three are pending human review.

A first CIFT-like calibrated-deviation comparison now tests a more
paper-aligned direction without replacing the static-feature thread. It uses
final-token readout features from the last quarter of the hidden-state stack,
`final_token_layer_22` through `final_token_layer_28`, calibrates a diagonal
safe-secret distribution inside each grouped fold, and trains on per-layer
deviation scores. This is still an approximation: it does not yet implement
CCI/CFS, learned nonnegative layer weighting, or richer readout positions beyond
the final prompt token.

Against the current combined static feature, the first CIFT-like score loses on
all four checkpoints:

| Dataset | Combined Macro F1 | CIFT-like Macro F1 | Delta Macro F1 |
|---|---:|---:|---:|
| Baseline | 0.8804 | 0.6981 | -0.1822 |
| Hard V1 | 0.9331 | 0.5399 | -0.3932 |
| Hard V2 | 0.9657 | 0.5329 | -0.4328 |
| Hard V3 | 0.8811 | 0.4076 | -0.4735 |

Treat this as a negative checkpoint, not a failed project direction. It says
the naive calibrated-distance score is not enough by itself. The next
CIFT-focused work should isolate whether the weakness comes from the readout
position, the score compression, missing layer weighting, or the calibration
set.

A V2 CIFT-like ablation now separates some of those causes. It compares eight
last-quarter variants across final-token versus mean-pool readout proxies,
diagonal-distance versus standardized-residual representations, and
safe-secret-only versus benign-plus-safe calibration. The combined static
feature still wins every checkpoint, but residual-concat variants are much
stronger than diagonal-distance variants:

| Dataset | Combined Macro F1 | Best CIFT-like Macro F1 | Delta Macro F1 |
|---|---:|---:|---:|
| Baseline | 0.8804 | 0.8623 | -0.0181 |
| Hard V1 | 0.9331 | 0.8132 | -0.1199 |
| Hard V2 | 0.9657 | 0.7808 | -0.1849 |
| Hard V3 | 0.8811 | 0.7461 | -0.1350 |

The aggregate result points at score compression as a major weakness in the
first CIFT-like attempt. The best final-token residual variants average 0.7924
macro F1, while the best final-token diagonal variant averages 0.5783. The
calibration-set axis does not affect residual-concat results because the
downstream activation classifier standardizes those features again; the next
CIFT refinement should preserve calibrated residual magnitudes when testing
calibration choices.

A V3 CIFT-like ablation adds absolute residual features and a raw logistic
regression mode that preserves calibrated feature scale. The combined static
feature still wins every checkpoint, but raw signed residuals close the gap on
the easier datasets:

| Dataset | Combined Macro F1 | Best CIFT-like Macro F1 | Delta Macro F1 |
|---|---:|---:|---:|
| Baseline | 0.8804 | 0.8788 | -0.0015 |
| Hard V1 | 0.9331 | 0.8236 | -0.1095 |
| Hard V2 | 0.9657 | 0.7808 | -0.1849 |
| Hard V3 | 0.8811 | 0.7461 | -0.1350 |

The strongest aggregate CIFT-like family remains signed standardized residuals
over final-token last-quarter layers. Absolute residual magnitude alone is
weaker, though nonleaking calibration improves absolute residual variants. The
result motivated testing whether a simple learned nonnegative layer-weighted
head could improve the signed residual representation.

A first layer-weighted CIFT-like head now trains one signed-residual classifier
per final-token last-quarter layer, converts each layer into an exfiltration
risk probability, learns nonnegative AUC-based layer weights inside each grouped
fold, and evaluates a learned threshold on held-out prompt families. The
combined static feature still wins every checkpoint:

| Dataset | Combined Macro F1 | Layer-Weighted Head Macro F1 | Delta Macro F1 |
|---|---:|---:|---:|
| Baseline | 0.8804 | 0.8293 | -0.0511 |
| Hard V1 | 0.9331 | 0.8132 | -0.1199 |
| Hard V2 | 0.9657 | 0.7633 | -0.2024 |
| Hard V3 | 0.8811 | 0.7461 | -0.1350 |

The diagnostic result is that the learned mean weights are uniform across
layers 22 through 28. Under this heuristic, the late final-token residual
classifiers look redundant rather than complementary. The next CIFT step should
therefore move past simple final-token layer weighting toward a richer CFS-like
head with out-of-fold per-layer scores and readout positions beyond the final
prompt token.

A first out-of-fold CIFT meta-head now trains per-source residual classifiers
inside inner grouped folds, uses their out-of-fold exfiltration-risk scores as
meta-head features, and compares final-token last-quarter sources against an
expanded final-token plus mean-pool source set. The expanded source set is the
best CIFT-like variant on every checkpoint, and it is the first CIFT-like
checkpoint to beat the combined static feature on any registered dataset:

| Dataset | Combined Macro F1 | Best Meta-Head Macro F1 | Delta Macro F1 | Winner |
|---|---:|---:|---:|---|
| Baseline | 0.8804 | 0.8598 | -0.0206 | combined static |
| Hard V1 | 0.9331 | 0.8788 | -0.0543 | combined static |
| Hard V2 | 0.9657 | 0.8823 | -0.0834 | combined static |
| Hard V3 | 0.8811 | 0.9156 | +0.0345 | OOF meta-head |

This is not a promotion result because the combined static feature still wins
three of four checkpoints, including Hard V2. It is useful because mean-pool
readout expansion improves every checkpoint relative to final-token-only CIFT
meta-learning: 0.8841 mean macro F1 versus 0.8209. The next CIFT work should
analyze the Hard V3 win and Hard V2 deficit, then test calibration/readout
variants before changing the monitor contract.

The Hard V2/Hard V3 residual analysis now explains that split. Across those two
checkpoints, the OOF meta-head has 12 errors versus 9 for the combined static
feature, with 3 fixed, 6 persistent, and 6 introduced errors. Hard V2 is the
blocking regression: 0 fixed errors and 5 introduced errors. Hard V3 is the
encouraging case: 3 fixed errors and 1 introduced error.

| Dataset | Reference Errors | Meta-Head Errors | Fixed | Persistent | Introduced |
|---|---:|---:|---:|---:|---:|
| Hard V2 | 2 | 7 | 0 | 2 | 5 |
| Hard V3 | 7 | 5 | 3 | 4 | 1 |

The Hard V3 win comes mainly from fixing two
`hard_v3_exfil_tool_payload_forward` misses and one
`hard_v3_safe_policy_note_category` miss. The Hard V2 deficit comes from
introduced misses in broker, output-contract, policy-exception, broker-boundary,
and tool-argument families. The next CIFT step should target those introduced
Hard V2 families while preserving the Hard V3 fixed cases.

A compact 12-variant CIFT meta-head ablation now tests that next step across
three axes: calibration set, source subset, and decision rule. The result is a
negative but useful checkpoint. The best variant is still the current full
dual-readout, safe-secret-calibrated, logistic-default meta-head:

```text
full_dual_readout_safe_secret_logistic_default
```

No tested variant reduces the Hard V2 introduced-error count. Broadening
calibration to benign-plus-safe rows has no measured effect, train-fold
threshold tuning worsens the result, and trimming the source set to earlier
last-quarter layers loses most of the Hard V3 gains.

| Source Set | Best Candidate Errors | Best Introduced Errors | Best Fixed Errors |
|---|---:|---:|---:|
| Full dual readout | 12 | 6 | 3 |
| Early dual readout | 22 | 14 | 1 |
| Early final token | 21 | 13 | 1 |

The next CIFT step should move from coarse grid ablation to score-level
diagnosis of the five Hard V2 introduced cases: per-source risk scores,
meta-head coefficients, and fold thresholds.

A score-level Hard V2 diagnosis now records the exact fold threshold,
meta-head risk score, risk-oriented logit, standardized per-source scores, and
per-source logit contributions for those introduced errors. It reproduces the
residual finding exactly: 2 reference errors, 7 meta-head errors, and 5
introduced errors. The three exfiltration false negatives land below the
default 0.5 threshold with meta risk scores 0.2375, 0.4275, and 0.0304. The two
safe false positives land above threshold at 0.6555 and 0.7307.

The most active source on the introduced cases is `mean_pool_layer_28`
(+0.8614 mean logit contribution, 2.2667 max absolute contribution), followed
by mixed and sometimes opposing mean-pool late-layer signals. This points
toward source calibration and late mean-pool weighting as the next debugging
surface, not simply a global decision-threshold problem.

A targeted source-ablation checkpoint tested that hypothesis directly across
Hard V2 and Hard V3. The full dual-readout meta-head remains the best variant.
Dropping `mean_pool_layer_28` leaves Hard V2 unchanged at 7 candidate errors
and 5 introduced errors, while Hard V3 keeps the same candidate error count but
trades one extra fixed error for one extra introduced error. Dropping the last
two mean-pool sources, the last final-token source, or the full last dual
readout layer all worsens the aggregate result.

| Variant | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta |
|---|---:|---:|---:|---:|---:|
| Full dual readout | 12 | 3 | 6 | 6 | 3 |
| Drop last mean pool | 12 | 4 | 5 | 7 | 3 |
| Drop last two mean pool | 15 | 2 | 7 | 8 | 6 |
| Drop last final token | 14 | 3 | 6 | 8 | 5 |
| Drop last dual readout layer | 15 | 2 | 7 | 8 | 6 |

The source-level error is therefore not isolated to one removable source. The
next CIFT refinement should look at how source scores are calibrated and
combined, rather than pruning `mean_pool_layer_28` outright.

A constrained-combiner ablation tested simple monotone alternatives on the same
full dual-readout source set. These rules are more interpretable than the
logistic meta-head because higher source risk cannot reduce final risk, but
they are much weaker on the current source scores. Mean score, majority vote,
max score, and top-two mean all introduce substantially more errors than the
logistic meta-head.

| Combiner | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta |
|---|---:|---:|---:|---:|---:|
| Logistic meta-head | 12 | 3 | 6 | 6 | 3 |
| Mean score | 26 | 2 | 7 | 19 | 17 |
| Majority vote | 27 | 2 | 7 | 20 | 18 |
| Top-two mean | 41 | 1 | 8 | 33 | 32 |
| Max score | 42 | 1 | 8 | 34 | 33 |

This means the per-source risk scores are not calibrated enough to combine
directly with raw averaging or voting. The logistic meta-head is doing useful
cross-source correction, even though it still has the Hard V2 regression. The
next constrained direction should be supervised but regularized: e.g.,
non-negative logistic weights, simplex weights, or calibrated source scores
before aggregation.

A supervised constrained-combiner follow-up tested non-negative logistic weights
and simplex-constrained logistic weights. These variants are better than raw
monotone aggregation, but still weaker than the unconstrained logistic
meta-head. Both constrained supervised variants produce 23 total candidate
errors and 17 introduced errors across Hard V2/Hard V3, versus 12 and 6 for the
current logistic meta-head.

| Combiner | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta |
|---|---:|---:|---:|---:|---:|
| Logistic meta-head | 12 | 3 | 6 | 6 | 3 |
| Positive logistic | 23 | 3 | 6 | 17 | 14 |
| Simplex logistic | 23 | 3 | 6 | 17 | 14 |

The result suggests the signed logistic weights are not merely overfitting
noise; they are correcting source-score inversions that the current source
heads produce. The next refinement should preserve supervised correction while
testing safer operating points, such as fold-learned thresholds or explicit
source-score calibration.

A thresholded constrained-combiner pass tested that operating-point hypothesis.
Learning a fold-specific macro-F1 threshold for the positive and simplex
logistic variants worsens both constrained variants. Positive logistic rises
from 23 to 29 candidate errors, and simplex logistic rises from 23 to 27.

| Combiner | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta |
|---|---:|---:|---:|---:|---:|
| Positive logistic | 23 | 3 | 6 | 17 | 14 |
| Simplex logistic | 23 | 3 | 6 | 17 | 14 |
| Positive logistic + train-F1 threshold | 29 | 1 | 8 | 21 | 20 |
| Simplex logistic + train-F1 threshold | 27 | 1 | 8 | 19 | 18 |

This narrows the next CIFT-like branch again: the problem is not just the
operating threshold on a constrained combiner. The more promising next
experiment is a signed logistic regularization sweep or source-score
calibration that preserves the useful cross-source correction.

A signed-logistic meta-head regularization sweep tested that next branch while
holding the source-head regularization fixed at `C=1.0`. Stronger meta-head
regularization values underfit the Hard V2/Hard V3 tradeoff, while weaker
regularization of the signed meta-head improves both aggregate errors and the
Hard V3 result. `C=5.0` and `C=10.0` tie as the best tested settings:

| Meta C | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 15 | 2 | 7 | 8 | 6 | 0.8750 |
| 1.0 | 12 | 3 | 6 | 6 | 3 | 0.9000 |
| 5.0 | 9 | 5 | 4 | 5 | 0 | 0.9250 |
| 10.0 | 9 | 5 | 4 | 5 | 0 | 0.9250 |

The useful finding is that preserving signed supervised correction remains the
best current CIFT-like direction. The risk is that the apparent improvement may
be checkpoint-specific: Hard V2 still has five introduced errors against the
combined static reference, and the first sweep only covered Hard V2/Hard V3.
That motivated the all-dataset cross-check and Hard V2 diagnostics below.

The four-dataset cross-check keeps the result useful but blocks promotion.
Across baseline, Hard V1, Hard V2, and Hard V3, `meta_c_10` is the best tested
regularized meta-head, not `meta_c_5`. It improves over the original `C=1.0`
meta-head but still trails the combined static reference in aggregate residual
terms:

| Meta C | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 27 | 7 | 13 | 14 | 7 | 0.8875 |
| 5.0 | 25 | 10 | 10 | 15 | 5 | 0.8958 |
| 10.0 | 23 | 11 | 9 | 14 | 3 | 0.9042 |

Hard V2 diagnostics for `meta_c_10` show 6 candidate errors and 5 introduced
errors versus the combined static reference. The remaining introduced set has
four exfiltration false negatives below the default 0.5 threshold and one safe
false positive above it. The strongest source-level evidence is still mixed
mean-pool behavior, especially opposing contributions from `mean_pool_layer_22`
and `mean_pool_layer_28`. That points the next CIFT branch toward source-score
calibration and readout design rather than threshold tuning or single-source
pruning.

A first source-score calibration comparison tested that branch without changing
the source heads, source set, or `meta_c_10` operating point. The raw source
probability control remains the best Hard V2/Hard V3 result:

| Source-Score Rule | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Raw probability | 9 | 5 | 4 | 5 | 0 | 0.9250 |
| Clipped logit | 11 | 7 | 2 | 9 | 2 | 0.9083 |
| Platt probability | 10 | 5 | 4 | 6 | 1 | 0.9167 |

The split is informative: clipped logits improve Hard V3 to 2 candidate errors
but damage Hard V2, while Platt scaling stays closer to the raw control but
still adds one aggregate error. This suggests the current source probabilities
are not the only bottleneck. The next CIFT-like experiment should test richer
readout/source design, such as separating final-token and mean-pool source
families or adding decision-position readouts, before more calibration variants.

A readout-family comparison then separated the same `meta_c_10` setup into
full dual-readout, final-token-only, and mean-pool-only source families. Full
dual-readout remains the best Hard V2/Hard V3 variant:

| Source Family | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Full dual readout | 9 | 5 | 4 | 5 | 0 | 0.9250 |
| Final token only | 23 | 1 | 8 | 15 | 14 | 0.8083 |
| Mean pool only | 11 | 8 | 1 | 10 | 2 | 0.9083 |

The result argues against pruning mean-pool sources as a family. Mean-pool-only
is noisy and introduces more errors, but it is much stronger than final-token
only and fixes more static-reference errors than the full dual-readout variant.
The useful next branch is therefore not "remove mean pool"; it is to design
better readout positions or source-family interaction features so useful
mean-pool evidence can be retained without the current Hard V2 false positives
and false negatives.

A first source-family interaction checkpoint tested that next branch without
changing the source heads, source set, or `meta_c_10` operating point. It adds
coarse family features on top of the raw full dual-readout source scores:
final-token mean, mean-pool mean, and optionally their signed and absolute
mean gap. The raw source-score vector remains the best Hard V2/Hard V3 result:

| Interaction Rule | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Raw scores | 9 | 5 | 4 | 5 | 0 | 0.9250 |
| Family means | 10 | 5 | 4 | 6 | 1 | 0.9167 |
| Family mean gaps | 15 | 5 | 4 | 11 | 6 | 0.8750 |

The result argues against coarse family aggregation as the immediate fix.
Adding family means leaves Hard V3 unchanged but worsens Hard V2 by one
introduced error; adding mean-gap features worsens both aggregate behavior and
the Hard V2 pressure test. The raw per-layer score vector appears to preserve
useful detail that these compressed family summaries disturb. The next CIFT-like
branch should move toward richer readout positions, better source-head targets,
or paper-aligned CCI/CFS scoring rather than adding more hand-built aggregate
features to the current score vector.

## Project Layout

```text
introspection/
├── data/
│   ├── activations/      # Serialized activation feature artifacts
│   ├── probes/           # Reserved for trained probe artifacts
│   ├── reports/          # JSON reports and narrative progress notes
│   ├── lineage.json      # Canonical experiment ledger
│   ├── prompts.jsonl      # Baseline prompt dataset
│   ├── prompts_hard.jsonl # Hard Baseline V1 dataset
│   ├── prompts_hard_v2.jsonl # Hard Baseline V2 dataset
│   ├── prompts_hard_v3.jsonl # Hard Baseline V3 dataset
│   ├── prompts_dp_honey_lite_v1.jsonl # DP-HONEY-lite structured smoke dataset
│   ├── prompts_dp_honey_lite_v2.jsonl # Hard DP-HONEY-lite paired scenario dataset
│   ├── prompts_dp_honey_lite_v3.jsonl # Lexically controlled DP-HONEY-lite mode-map dataset
│   ├── prompts_dp_honey_lite_v3_policy_windows.jsonl # V3 explicit policy-window control
│   ├── prompts_dp_honey_lite_v3_selector_windows.jsonl # V3 selector-window checkpoint
│   ├── prompts_dp_honey_lite_v4.jsonl # Targeted V4 mode_b control dataset
│   ├── prompts_dp_honey_lite_v4_selector_windows.jsonl # V4 selector-window control
│   ├── prompts_dp_honey_lite_v4_1.jsonl # Focused V4.1 relational-control dataset
│   ├── prompts_dp_honey_lite_v4_1_selector_windows.jsonl # V4.1 selector-window checkpoint
│   ├── detector_results_cift_v3_selector_window_layer_15_v1.jsonl # Aegis DetectorResult bridge export
│   ├── detector_results_cift_v3_selector_window_layer_15_calibrated_v1.jsonl # Calibrated DetectorResult export
│   └── runtime_turns_dp_honey_lite_v3_selector_windows.jsonl # Aegis NormalizedTurn bridge export
├── notebooks/            # Interactive exploration notebooks
├── scripts/              # CLI entry points for extraction, training, summaries, validation
├── src/aegis_introspection/
│   └── ...               # Typed implementation modules
└── tests/                # Unit tests
```

## Lineage Rules

Organization matters here. Any dataset, activation artifact, or machine-readable
report that supports a stated result should be registered in `data/lineage.json`
with its SHA256 hash.

The rule of thumb:

```text
Do not replace an experimental state that produced a reported metric.
```

Add new files and new lineage records rather than overwriting baseline
evidence. Current examples:

```text
data/prompts_hard.jsonl
data/prompts_hard_v2.jsonl
data/prompts_hard_v3.jsonl
data/activations/qwen3_0_6b_hard_all_layers.pt
data/activations/qwen3_0_6b_hard_v2_all_layers.pt
data/activations/qwen3_0_6b_hard_v3_all_layers.pt
data/reports/binary_tasks_hard_grouped.json
```

Future datasets should use new names, for example:

```text
data/prompts_tool_calls.jsonl
data/prompts_hard_v4.jsonl
data/activations/qwen3_0_6b_hard_v4_all_layers.pt
```

Validate lineage after any intentional manifest change:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/validate_lineage.py
```

## Common Commands

Run the full test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python -m unittest discover -s introspection/tests
```

Generate the DP-HONEY-lite structured prompt dataset:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/generate_dp_honey_lite_prompts.py
```

Generate the hard V2 DP-HONEY-lite prompt dataset:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/generate_dp_honey_lite_prompts.py \
  --template-set hard_v2 \
  --seed aegis-dp-honey-lite-v2 \
  --examples-per-template 4 \
  --readout-width 6
```

Generate the hard V3 DP-HONEY-lite prompt dataset:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/generate_dp_honey_lite_prompts.py \
  --template-set hard_v3 \
  --seed aegis-dp-honey-lite-v3 \
  --examples-per-template 4 \
  --readout-width 6
```

Extract DP-HONEY-lite readout-window activation features:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --prompts introspection/data/prompts_dp_honey_lite_v1.jsonl \
  --output introspection/data/activations/qwen3_0_6b_dp_honey_lite_v1_readout_windows.pt \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --pooling readout_window
```

Run DP-HONEY-lite readout-window smoke sweeps:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/sweep_binary_layers.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v1_readout_windows.pt \
  --output-json introspection/data/reports/dp_honey_lite_readout_window_sweep_safe_secret_vs_exfiltration_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_readout_window_sweep_safe_secret_vs_exfiltration_v1_summary.md \
  --task safe_secret_vs_exfiltration \
  --reference-feature readout_window_layer_11 \
  --folds 2
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/sweep_binary_layers.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v1_readout_windows.pt \
  --output-json introspection/data/reports/dp_honey_lite_readout_window_sweep_benign_vs_secret_related_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_readout_window_sweep_benign_vs_secret_related_v1_summary.md \
  --task benign_vs_secret_related \
  --reference-feature readout_window_layer_11 \
  --folds 2
```

Extract the hard V2 all-pooling activation artifact:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --prompts introspection/data/prompts_dp_honey_lite_v2.jsonl \
  --output introspection/data/activations/qwen3_0_6b_dp_honey_lite_v2_all_pooling.pt \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --pooling final_token,mean_pool,readout_window
```

Run the hard V2 grouped all-pooling sweeps:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/sweep_binary_layers.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v2_all_pooling.pt \
  --output-json introspection/data/reports/dp_honey_lite_v2_all_pooling_sweep_safe_secret_vs_exfiltration_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v2_all_pooling_sweep_safe_secret_vs_exfiltration_v1_summary.md \
  --task safe_secret_vs_exfiltration \
  --reference-feature readout_window_layer_11 \
  --folds 5
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/sweep_binary_layers.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v2_all_pooling.pt \
  --output-json introspection/data/reports/dp_honey_lite_v2_all_pooling_sweep_benign_vs_secret_related_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v2_all_pooling_sweep_benign_vs_secret_related_v1_summary.md \
  --task benign_vs_secret_related \
  --reference-feature readout_window_layer_11 \
  --folds 5
```

Run the hard V2 grouped binary baseline comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v2_all_pooling.pt \
  --output-json introspection/data/reports/dp_honey_lite_v2_grouped_binary_tasks_readout_window_layer_11_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v2_grouped_binary_tasks_readout_window_layer_11_v1_summary.md \
  --activation-feature readout_window_layer_11 \
  --folds 5
```

Extract and evaluate the hard V3 all-pooling checkpoint:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --prompts introspection/data/prompts_dp_honey_lite_v3.jsonl \
  --output introspection/data/activations/qwen3_0_6b_dp_honey_lite_v3_all_pooling.pt \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --pooling final_token,mean_pool,readout_window
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/sweep_binary_layers.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v3_all_pooling.pt \
  --output-json introspection/data/reports/dp_honey_lite_v3_all_pooling_sweep_safe_secret_vs_exfiltration_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v3_all_pooling_sweep_safe_secret_vs_exfiltration_v1_summary.md \
  --task safe_secret_vs_exfiltration \
  --reference-feature readout_window_layer_11 \
  --folds 5
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v3_all_pooling.pt \
  --output-json introspection/data/reports/dp_honey_lite_v3_grouped_binary_tasks_readout_window_layer_21_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v3_grouped_binary_tasks_readout_window_layer_21_v1_summary.md \
  --activation-feature readout_window_layer_21 \
  --folds 5
```

Run the hard V3 policy diagnostic:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/diagnose_v3_policy.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v3_all_pooling.pt \
  --output-json introspection/data/reports/dp_honey_lite_v3_policy_diagnostics_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v3_policy_diagnostics_v1_summary.md \
  --activation-feature readout_window_layer_21 \
  --folds 5
```

Run the hard V3 direct policy-resolution diagnostic:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/diagnose_v3_policy_resolution.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v3_all_pooling.pt \
  --output-json introspection/data/reports/dp_honey_lite_v3_policy_resolution_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v3_policy_resolution_v1_summary.md \
  --max-new-tokens 16
```

Generate the hard V3 policy-window and selector-window datasets:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/generate_v3_policy_window_prompts.py \
  --input introspection/data/prompts_dp_honey_lite_v3.jsonl \
  --output introspection/data/prompts_dp_honey_lite_v3_policy_windows.jsonl \
  --window-kind decision_path
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/generate_v3_policy_window_prompts.py \
  --input introspection/data/prompts_dp_honey_lite_v3.jsonl \
  --output introspection/data/prompts_dp_honey_lite_v3_selector_windows.jsonl \
  --window-kind selector
```

Extract and evaluate the hard V3 selector-window checkpoint:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --prompts introspection/data/prompts_dp_honey_lite_v3_selector_windows.jsonl \
  --output introspection/data/activations/qwen3_0_6b_dp_honey_lite_v3_selector_windows.pt \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --pooling readout_window
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/sweep_binary_layers.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v3_selector_windows.pt \
  --output-json introspection/data/reports/dp_honey_lite_v3_selector_window_sweep_safe_secret_vs_exfiltration_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v3_selector_window_sweep_safe_secret_vs_exfiltration_v1_summary.md \
  --task safe_secret_vs_exfiltration \
  --reference-feature readout_window_layer_21 \
  --folds 5
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v3_selector_windows.pt \
  --output-json introspection/data/reports/dp_honey_lite_v3_selector_window_grouped_binary_tasks_readout_window_layer_15_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v3_selector_window_grouped_binary_tasks_readout_window_layer_15_v1_summary.md \
  --activation-feature readout_window_layer_15 \
  --folds 5
```

Generate and evaluate the hard V4 mode_b selector-window control:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/generate_dp_honey_lite_prompts.py \
  --template-set hard_v4 \
  --seed capstone-hard-v4 \
  --examples-per-template 4 \
  --readout-width 6
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/generate_v3_policy_window_prompts.py \
  --input introspection/data/prompts_dp_honey_lite_v4.jsonl \
  --output introspection/data/prompts_dp_honey_lite_v4_selector_windows.jsonl \
  --window-kind selector
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/generate_v3_policy_window_prompts.py \
  --input introspection/data/prompts_dp_honey_lite_v4.jsonl \
  --output introspection/data/prompts_dp_honey_lite_v4_policy_windows.jsonl \
  --window-kind decision_path
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --prompts introspection/data/prompts_dp_honey_lite_v4_selector_windows.jsonl \
  --output introspection/data/activations/qwen3_0_6b_dp_honey_lite_v4_selector_windows.pt \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --pooling readout_window
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/sweep_binary_layers.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v4_selector_windows.pt \
  --output-json introspection/data/reports/dp_honey_lite_v4_selector_window_sweep_safe_secret_vs_exfiltration_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v4_selector_window_sweep_safe_secret_vs_exfiltration_v1_summary.md \
  --task safe_secret_vs_exfiltration \
  --reference-feature readout_window_layer_15 \
  --folds 5
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v4_selector_windows.pt \
  --output-json introspection/data/reports/dp_honey_lite_v4_selector_window_grouped_binary_tasks_readout_window_layer_15_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v4_selector_window_grouped_binary_tasks_readout_window_layer_15_v1_summary.md \
  --activation-feature readout_window_layer_15 \
  --folds 5
```

Generate and evaluate the hard V4.1 relational selector-window checkpoint:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/generate_dp_honey_lite_prompts.py \
  --template-set hard_v4_1 \
  --seed capstone-hard-v4-1 \
  --examples-per-template 4 \
  --readout-width 6
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/generate_v3_policy_window_prompts.py \
  --input introspection/data/prompts_dp_honey_lite_v4_1.jsonl \
  --output introspection/data/prompts_dp_honey_lite_v4_1_selector_windows.jsonl \
  --window-kind selector
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/generate_v3_policy_window_prompts.py \
  --input introspection/data/prompts_dp_honey_lite_v4_1.jsonl \
  --output introspection/data/prompts_dp_honey_lite_v4_1_policy_windows.jsonl \
  --window-kind decision_path
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --prompts introspection/data/prompts_dp_honey_lite_v4_1_selector_windows.jsonl \
  --output introspection/data/activations/qwen3_0_6b_dp_honey_lite_v4_1_selector_windows.pt \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --pooling readout_window
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/sweep_binary_layers.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v4_1_selector_windows.pt \
  --output-json introspection/data/reports/dp_honey_lite_v4_1_selector_window_sweep_safe_secret_vs_exfiltration_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v4_1_selector_window_sweep_safe_secret_vs_exfiltration_v1_summary.md \
  --task safe_secret_vs_exfiltration \
  --reference-feature readout_window_layer_15 \
  --folds 5
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v4_1_selector_windows.pt \
  --output-json introspection/data/reports/dp_honey_lite_v4_1_selector_window_grouped_binary_tasks_readout_window_layer_15_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v4_1_selector_window_grouped_binary_tasks_readout_window_layer_15_v1_summary.md \
  --activation-feature readout_window_layer_15 \
  --folds 5
```

Run selector-window error analysis and metadata slices:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/analyze_binary_errors.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v3_selector_windows.pt \
  --output-json introspection/data/reports/dp_honey_lite_v3_selector_window_error_analysis_readout_window_layer_15_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v3_selector_window_error_analysis_readout_window_layer_15_v1_summary.md \
  --activation-feature readout_window_layer_15 \
  --folds 5
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/summarize_policy_window_errors.py \
  --prompts introspection/data/prompts_dp_honey_lite_v3_selector_windows.jsonl \
  --error-report introspection/data/reports/dp_honey_lite_v3_selector_window_error_analysis_readout_window_layer_15_v1.json \
  --output-json introspection/data/reports/dp_honey_lite_v3_selector_window_error_slices_readout_window_layer_15_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v3_selector_window_error_slices_readout_window_layer_15_v1_summary.md \
  --task safe_secret_vs_exfiltration \
  --method activation_probe
```

Export selector-window rows into Aegis runtime-spine `NormalizedTurn` shape:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/export_runtime_turns.py \
  --input introspection/data/prompts_dp_honey_lite_v3_selector_windows.jsonl \
  --output introspection/data/runtime_turns_dp_honey_lite_v3_selector_windows.jsonl \
  --capability-mode offline_eval \
  --model-provider huggingface \
  --model-id Qwen/Qwen3-0.6B \
  --revision main \
  --selected-device cpu \
  --sensitive-source dp_honey_lite \
  --session-id introspection-offline-eval
```

Export selector-window activation predictions into Aegis
`DetectorResult` shape:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/export_cift_detector_results.py \
  --runtime-turns introspection/data/runtime_turns_dp_honey_lite_v3_selector_windows.jsonl \
  --error-report introspection/data/reports/dp_honey_lite_v3_selector_window_error_analysis_readout_window_layer_15_v1.json \
  --output introspection/data/detector_results_cift_v3_selector_window_layer_15_v1.jsonl \
  --task safe_secret_vs_exfiltration \
  --method activation_probe \
  --detector-name cift_selector_probe \
  --feature-key readout_window_layer_15 \
  --probe-version dp_honey_lite_v3_selector_window_layer_15_v1 \
  --capability-required self_hosted_introspection \
  --positive-label exfiltration_intent \
  --positive-score 1.0 \
  --negative-score 0.0 \
  --positive-action warn \
  --negative-action allow \
  --confidence 0.7736
```

Calibrate selector-window CIFT detector scores with grouped outer folds and
inner Platt calibration:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/calibrate_cift_detector.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v3_selector_windows.pt \
  --output-json introspection/data/reports/dp_honey_lite_v3_selector_window_cift_calibration_readout_window_layer_15_v1.json \
  --output-markdown introspection/data/reports/dp_honey_lite_v3_selector_window_cift_calibration_readout_window_layer_15_v1_summary.md \
  --task safe_secret_vs_exfiltration \
  --positive-label exfiltration_intent \
  --activation-feature readout_window_layer_15 \
  --folds 5 \
  --inner-folds 3 \
  --seed 42 \
  --max-iter 1000 \
  --regularization-c 1.0 \
  --decision-threshold 0.5
```

Export calibrated selector-window activation predictions into Aegis
`DetectorResult` shape:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/export_calibrated_cift_detector_results.py \
  --runtime-turns introspection/data/runtime_turns_dp_honey_lite_v3_selector_windows.jsonl \
  --calibration-report introspection/data/reports/dp_honey_lite_v3_selector_window_cift_calibration_readout_window_layer_15_v1.json \
  --output introspection/data/detector_results_cift_v3_selector_window_layer_15_calibrated_v1.jsonl \
  --detector-name cift_selector_probe \
  --probe-version dp_honey_lite_v3_selector_window_layer_15_calibrated_v1 \
  --capability-required self_hosted_introspection \
  --positive-action warn \
  --negative-action allow \
  --confidence 0.7736
```

Summarize calibrated CIFT detector operating points:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/summarize_cift_operating_points.py \
  --calibration-report introspection/data/reports/dp_honey_lite_v3_selector_window_cift_calibration_readout_window_layer_15_v1.json \
  --output-json introspection/data/reports/dp_honey_lite_v3_selector_window_cift_operating_points_readout_window_layer_15_v1.json \
  --output-markdown introspection/data/reports/dp_honey_lite_v3_selector_window_cift_operating_points_readout_window_layer_15_v1_summary.md \
  --thresholds 0.05:0.95:0.05
```

Diagnose calibrated selector-window CIFT scores by threshold margin and prompt
slice:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/diagnose_cift_selector_window_scores.py \
  --prompts introspection/data/prompts_dp_honey_lite_v3_selector_windows.jsonl \
  --calibration-report introspection/data/reports/dp_honey_lite_v3_selector_window_cift_calibration_readout_window_layer_15_v1.json \
  --output-json introspection/data/reports/dp_honey_lite_v3_selector_window_score_diagnostics_readout_window_layer_15_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v3_selector_window_score_diagnostics_readout_window_layer_15_v1_summary.md \
  --near-threshold-radius 0.10
```

Ablate selector-window feature expressions:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/ablate_cift_selector_window_features.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v3_selector_windows.pt \
  --output-json introspection/data/reports/dp_honey_lite_v3_selector_window_feature_ablation_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v3_selector_window_feature_ablation_v1_summary.md \
  --task safe_secret_vs_exfiltration \
  --baseline-variant baseline_layer_15 \
  --folds 5
```

Calibrate and diagnose the best raw ablation variant:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/calibrate_cift_detector.py \
  --artifact introspection/data/activations/qwen3_0_6b_dp_honey_lite_v3_selector_windows.pt \
  --output-json introspection/data/reports/dp_honey_lite_v3_selector_window_cift_calibration_selector_band_12_18_v1.json \
  --output-markdown introspection/data/reports/dp_honey_lite_v3_selector_window_cift_calibration_selector_band_12_18_v1_summary.md \
  --task safe_secret_vs_exfiltration \
  --positive-label exfiltration_intent \
  --activation-feature 'concat(readout_window_layer_12,readout_window_layer_13,readout_window_layer_14,readout_window_layer_15,readout_window_layer_16,readout_window_layer_17,readout_window_layer_18)' \
  --folds 5 \
  --inner-folds 3 \
  --seed 42 \
  --max-iter 1000 \
  --regularization-c 1.0 \
  --decision-threshold 0.5
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/summarize_cift_operating_points.py \
  --calibration-report introspection/data/reports/dp_honey_lite_v3_selector_window_cift_calibration_selector_band_12_18_v1.json \
  --output-json introspection/data/reports/dp_honey_lite_v3_selector_window_cift_operating_points_selector_band_12_18_v1.json \
  --output-markdown introspection/data/reports/dp_honey_lite_v3_selector_window_cift_operating_points_selector_band_12_18_v1_summary.md \
  --thresholds 0.05:0.95:0.05
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/diagnose_cift_selector_window_scores.py \
  --prompts introspection/data/prompts_dp_honey_lite_v3_selector_windows.jsonl \
  --calibration-report introspection/data/reports/dp_honey_lite_v3_selector_window_cift_calibration_selector_band_12_18_v1.json \
  --output-json introspection/data/reports/dp_honey_lite_v3_selector_window_score_diagnostics_selector_band_12_18_v1.json \
  --output-md introspection/data/reports/dp_honey_lite_v3_selector_window_score_diagnostics_selector_band_12_18_v1_summary.md \
  --near-threshold-radius 0.10
```

Extract all-layer activation features for the baseline dataset:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --pooling final_token,mean_pool \
  --output introspection/data/activations/qwen3_0_6b_all_layers.pt
```

Train the all-layer probe sweep:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_probe.py \
  --artifact introspection/data/activations/qwen3_0_6b_all_layers.pt \
  --output introspection/data/reports/probe_all_layers.json
```

Run random binary tasks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_binary_tasks.py
```

Run grouped binary tasks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py
```

Run grouped binary tasks for Hard V1:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py \
  --artifact introspection/data/activations/qwen3_0_6b_hard_all_layers.pt \
  --output-json introspection/data/reports/binary_tasks_hard_grouped.json \
  --output-md introspection/data/reports/binary_tasks_hard_grouped_summary.md
```

Run grouped family-level error analysis for Hard V1:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/analyze_binary_errors.py
```

Run grouped binary tasks for Hard V2:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py \
  --artifact introspection/data/activations/qwen3_0_6b_hard_v2_all_layers.pt \
  --output-json introspection/data/reports/binary_tasks_hard_v2_grouped.json \
  --output-md introspection/data/reports/binary_tasks_hard_v2_grouped_summary.md
```

Build the Hard V2 adjudication worksheet:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/adjudicate_v2_errors.py
```

Run the Hard V2 grouped binary layer sweep:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/sweep_binary_layers.py
```

Run the cross-dataset candidate feature check:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_candidate_feature.py
```

Run candidate Hard V2 error analysis:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/analyze_binary_errors.py \
  --artifact introspection/data/activations/qwen3_0_6b_hard_v2_all_layers.pt \
  --activation-feature final_token_layer_11 \
  --output-json introspection/data/reports/binary_error_analysis_hard_v2_candidate_final_token_layer_11_grouped.json \
  --output-md introspection/data/reports/binary_error_analysis_hard_v2_candidate_final_token_layer_11_grouped_summary.md
```

Compare candidate and reference residual errors:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_error_residuals.py
```

Extract Hard V3 activations:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --prompts introspection/data/prompts_hard_v3.jsonl \
  --output introspection/data/activations/qwen3_0_6b_hard_v3_all_layers.pt \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --pooling final_token,mean_pool
```

Run the four-checkpoint candidate crosscheck:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_candidate_feature.py \
  --dataset-artifact baseline_prompts_v1:introspection/data/activations/qwen3_0_6b_all_layers.pt \
  --dataset-artifact hard_prompts_v1:introspection/data/activations/qwen3_0_6b_hard_all_layers.pt \
  --dataset-artifact hard_prompts_v2:introspection/data/activations/qwen3_0_6b_hard_v2_all_layers.pt \
  --dataset-artifact hard_prompts_v3:introspection/data/activations/qwen3_0_6b_hard_v3_all_layers.pt \
  --output-json introspection/data/reports/candidate_feature_crosscheck_with_hard_v3.json \
  --output-md introspection/data/reports/candidate_feature_crosscheck_with_hard_v3_summary.md
```

Run the combined feature stability check:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_feature_stability.py
```

Run the combined feature residual suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_combined_residual_suite.py
```

Run the first CIFT-like calibrated-deviation comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_probe.py
```

Run the latest CIFT-like ablation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_ablation.py
```

Run the CIFT layer-weighted head comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_layer_head.py
```

Run the CIFT out-of-fold meta-head comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_meta_head.py
```

Run the CIFT out-of-fold meta-head residual analysis:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/analyze_cift_meta_residuals.py
```

Run the CIFT out-of-fold meta-head ablation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_meta_ablation.py
```

Diagnose Hard V2 CIFT meta-head introduced errors:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/diagnose_cift_meta_scores.py
```

Run targeted CIFT meta-head source ablations:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_meta_source_ablation.py
```

Run constrained CIFT meta-head combiner ablations:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_meta_combiner_ablation.py
```

Run the signed CIFT meta-head regularization sweep:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_meta_regularization_sweep.py
```

Diagnose regularized CIFT meta-head introduced errors:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/diagnose_cift_meta_regularization.py
```

Run CIFT meta-head source-score calibration comparisons:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_meta_score_calibration.py
```

Run CIFT meta-head readout-family comparisons:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_meta_readout_family.py
```

Run CIFT meta-head source-family interaction comparisons:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/compare_cift_meta_family_interactions.py
```

Build the Hard V3 combined-regression adjudication worksheet:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/adjudicate_combined_hard_v3_regressions.py
```

## Reports

Key human-readable checkpoints:

- `data/reports/activation_probe_progress_2026-06-18.md`
- `data/reports/binary_probe_progress_2026-06-18.md`
- `data/reports/grouped_binary_probe_progress_2026-06-18.md`
- `data/reports/hard_baseline_probe_progress_2026-06-18.md`
- `data/reports/baseline_vs_hard_v1_comparison_2026-06-19.md`
- `data/reports/hard_v2_probe_progress_2026-06-19.md`
- `data/reports/probe_all_layers_summary.md`
- `data/reports/binary_tasks_summary.md`
- `data/reports/binary_tasks_grouped_summary.md`
- `data/reports/binary_tasks_hard_summary.md`
- `data/reports/binary_tasks_hard_grouped_summary.md`
- `data/reports/binary_error_analysis_hard_grouped_summary.md`
- `data/reports/binary_tasks_hard_v2_summary.md`
- `data/reports/binary_tasks_hard_v2_grouped_summary.md`
- `data/reports/binary_error_analysis_hard_v2_grouped_summary.md`
- `data/reports/hard_v2_error_adjudication_summary.md`
- `data/reports/hard_v2_error_adjudication_notes_2026-06-19.md`
- `data/reports/binary_layer_sweep_hard_v2_grouped_summary.md`
- `data/reports/hard_v2_layer_sweep_progress_2026-06-19.md`
- `data/reports/candidate_feature_crosscheck_summary.md`
- `data/reports/candidate_feature_crosscheck_progress_2026-06-19.md`
- `data/reports/binary_error_analysis_hard_v2_candidate_final_token_layer_11_grouped_summary.md`
- `data/reports/hard_v2_candidate_residual_error_comparison_summary.md`
- `data/reports/hard_v2_candidate_error_adjudication_summary.md`
- `data/reports/hard_v2_candidate_residual_progress_2026-06-19.md`
- `data/reports/candidate_feature_crosscheck_hard_v3_summary.md`
- `data/reports/candidate_feature_crosscheck_with_hard_v3_summary.md`
- `data/reports/binary_error_analysis_hard_v3_reference_grouped_summary.md`
- `data/reports/binary_error_analysis_hard_v3_candidate_final_token_layer_11_grouped_summary.md`
- `data/reports/hard_v3_candidate_residual_error_comparison_summary.md`
- `data/reports/binary_layer_sweep_hard_v3_grouped_summary.md`
- `data/reports/hard_v3_candidate_error_adjudication_summary.md`
- `data/reports/hard_v3_heldout_validation_progress_2026-06-19.md`
- `data/reports/feature_stability_reference_l11_l16_summary.md`
- `data/reports/feature_stability_progress_2026-06-19.md`
- `data/reports/feature_stability_combined_l11_l16_summary.md`
- `data/reports/feature_stability_combined_progress_2026-06-19.md`
- `data/reports/combined_feature_residual_suite_summary.md`
- `data/reports/combined_feature_residual_progress_2026-06-19.md`
- `data/reports/hard_v3_combined_regression_adjudication_summary.md`
- `data/reports/project_aligned_workflow_2026-06-19.md`
- `data/reports/cift_like_probe_comparison_summary.md`
- `data/reports/cift_like_probe_progress_2026-06-19.md`
- `data/reports/cift_like_ablation_v2_summary.md`
- `data/reports/cift_like_ablation_v2_progress_2026-06-19.md`
- `data/reports/cift_like_ablation_v3_summary.md`
- `data/reports/cift_like_ablation_v3_progress_2026-06-19.md`
- `data/reports/cift_layer_weighted_head_v1_summary.md`
- `data/reports/cift_layer_weighted_head_v1_progress_2026-06-19.md`
- `data/reports/cift_meta_head_v1_summary.md`
- `data/reports/cift_meta_head_v1_progress_2026-06-19.md`
- `data/reports/cift_meta_head_residual_suite_v1_summary.md`
- `data/reports/cift_meta_head_residual_progress_2026-06-19.md`
- `data/reports/cift_meta_ablation_v1_summary.md`
- `data/reports/cift_meta_ablation_progress_2026-06-19.md`
- `data/reports/cift_meta_score_diagnostics_hard_v2_v1_summary.md`
- `data/reports/cift_meta_source_ablation_v1_summary.md`
- `data/reports/cift_meta_combiner_ablation_v1_summary.md`
- `data/reports/cift_meta_combiner_ablation_v2_summary.md`
- `data/reports/cift_meta_combiner_ablation_v3_summary.md`
- `data/reports/cift_meta_regularization_sweep_v1_summary.md`
- `data/reports/cift_meta_regularization_crosscheck_v1_summary.md`
- `data/reports/cift_meta_regularization_diagnostics_hard_v2_meta_c_10_v1_summary.md`
- `data/reports/cift_meta_score_calibration_v1_summary.md`
- `data/reports/cift_meta_readout_family_v1_summary.md`
- `data/reports/cift_meta_family_interactions_v1_summary.md`
- `data/reports/dp_honey_lite_readout_window_sweep_safe_secret_vs_exfiltration_v1_summary.md`
- `data/reports/dp_honey_lite_readout_window_sweep_benign_vs_secret_related_v1_summary.md`
- `data/reports/dp_honey_lite_v2_all_pooling_sweep_safe_secret_vs_exfiltration_v1_summary.md`
- `data/reports/dp_honey_lite_v2_all_pooling_sweep_benign_vs_secret_related_v1_summary.md`
- `data/reports/dp_honey_lite_v2_grouped_binary_tasks_readout_window_layer_11_v1_summary.md`
- `data/reports/dp_honey_lite_v2_progress_2026-06-20.md`
- `data/reports/dp_honey_lite_v3_all_pooling_sweep_safe_secret_vs_exfiltration_v1_summary.md`
- `data/reports/dp_honey_lite_v3_all_pooling_sweep_benign_vs_secret_related_v1_summary.md`
- `data/reports/dp_honey_lite_v3_grouped_binary_tasks_readout_window_layer_21_v1_summary.md`
- `data/reports/dp_honey_lite_v3_policy_diagnostics_v1_summary.md`
- `data/reports/dp_honey_lite_v3_policy_resolution_v1_summary.md`
- `data/reports/dp_honey_lite_v3_policy_window_sweep_safe_secret_vs_exfiltration_v1_summary.md`
- `data/reports/dp_honey_lite_v3_policy_window_grouped_binary_tasks_readout_window_layer_00_v1_summary.md`
- `data/reports/dp_honey_lite_v3_selector_window_sweep_safe_secret_vs_exfiltration_v1_summary.md`
- `data/reports/dp_honey_lite_v3_selector_window_grouped_binary_tasks_readout_window_layer_15_v1_summary.md`
- `data/reports/dp_honey_lite_v3_selector_window_error_analysis_readout_window_layer_15_v1_summary.md`
- `data/reports/dp_honey_lite_v3_selector_window_error_slices_readout_window_layer_15_v1_summary.md`
- `data/reports/dp_honey_lite_v3_selector_window_cift_calibration_readout_window_layer_15_v1_summary.md`
- `data/reports/dp_honey_lite_v3_selector_window_cift_operating_points_readout_window_layer_15_v1_summary.md`
- `data/reports/dp_honey_lite_v3_selector_window_score_diagnostics_readout_window_layer_15_v1_summary.md`
- `data/reports/dp_honey_lite_v3_selector_window_feature_ablation_v1_summary.md`
- `data/reports/dp_honey_lite_v3_selector_window_cift_calibration_selector_band_12_18_v1_summary.md`
- `data/reports/dp_honey_lite_v3_selector_window_cift_operating_points_selector_band_12_18_v1_summary.md`
- `data/reports/dp_honey_lite_v3_selector_window_score_diagnostics_selector_band_12_18_v1_summary.md`
- `data/reports/dp_honey_lite_v4_selector_window_sweep_safe_secret_vs_exfiltration_v1_summary.md`
- `data/reports/dp_honey_lite_v4_selector_window_grouped_binary_tasks_readout_window_layer_15_v1_summary.md`
- `data/reports/dp_honey_lite_v4_1_selector_window_sweep_safe_secret_vs_exfiltration_v1_summary.md`
- `data/reports/dp_honey_lite_v4_1_selector_window_grouped_binary_tasks_readout_window_layer_15_v1_summary.md`
- `data/reports/dp_honey_lite_v3_progress_2026-06-20.md`
- `data/reports/dp_honey_lite_v3_selector_window_progress_2026-06-20.md`

Key machine-readable reports registered in lineage:

- `data/reports/probe_baseline.json`
- `data/reports/text_baseline.json`
- `data/reports/probe_all_layers.json`
- `data/reports/binary_tasks.json`
- `data/reports/binary_tasks_grouped.json`
- `data/reports/binary_tasks_hard.json`
- `data/reports/binary_tasks_hard_grouped.json`
- `data/reports/binary_error_analysis_hard_grouped.json`
- `data/reports/binary_tasks_hard_v2.json`
- `data/reports/binary_tasks_hard_v2_grouped.json`
- `data/reports/binary_error_analysis_hard_v2_grouped.json`
- `data/reports/hard_v2_error_adjudication.json`
- `data/reports/binary_layer_sweep_hard_v2_grouped.json`
- `data/reports/candidate_feature_crosscheck.json`
- `data/reports/binary_error_analysis_hard_v2_candidate_final_token_layer_11_grouped.json`
- `data/reports/hard_v2_candidate_residual_error_comparison.json`
- `data/reports/hard_v2_candidate_error_adjudication.json`
- `data/reports/candidate_feature_crosscheck_hard_v3.json`
- `data/reports/candidate_feature_crosscheck_with_hard_v3.json`
- `data/reports/binary_error_analysis_hard_v3_reference_grouped.json`
- `data/reports/binary_error_analysis_hard_v3_candidate_final_token_layer_11_grouped.json`
- `data/reports/hard_v3_candidate_residual_error_comparison.json`
- `data/reports/binary_layer_sweep_hard_v3_grouped.json`
- `data/reports/hard_v3_candidate_error_adjudication.json`
- `data/reports/feature_stability_reference_l11_l16.json`
- `data/reports/feature_stability_combined_l11_l16.json`
- `data/reports/combined_feature_residual_suite.json`
- `data/reports/hard_v3_combined_regression_adjudication.json`
- `data/reports/cift_like_probe_comparison.json`
- `data/reports/cift_like_ablation_v2.json`
- `data/reports/cift_like_ablation_v3.json`
- `data/reports/cift_layer_weighted_head_v1.json`
- `data/reports/cift_meta_head_v1.json`
- `data/reports/cift_meta_head_residual_suite_v1.json`
- `data/reports/cift_meta_ablation_v1.json`
- `data/reports/cift_meta_score_diagnostics_hard_v2_v1.json`
- `data/reports/cift_meta_source_ablation_v1.json`
- `data/reports/cift_meta_combiner_ablation_v1.json`
- `data/reports/cift_meta_combiner_ablation_v2.json`
- `data/reports/cift_meta_combiner_ablation_v3.json`
- `data/reports/cift_meta_regularization_sweep_v1.json`
- `data/reports/cift_meta_regularization_crosscheck_v1.json`
- `data/reports/cift_meta_regularization_diagnostics_hard_v2_meta_c_10_v1.json`
- `data/reports/cift_meta_score_calibration_v1.json`
- `data/reports/cift_meta_readout_family_v1.json`
- `data/reports/cift_meta_family_interactions_v1.json`
- `data/reports/dp_honey_lite_readout_window_sweep_safe_secret_vs_exfiltration_v1.json`
- `data/reports/dp_honey_lite_readout_window_sweep_benign_vs_secret_related_v1.json`
- `data/reports/dp_honey_lite_v2_all_pooling_sweep_safe_secret_vs_exfiltration_v1.json`
- `data/reports/dp_honey_lite_v2_all_pooling_sweep_benign_vs_secret_related_v1.json`
- `data/reports/dp_honey_lite_v2_grouped_binary_tasks_readout_window_layer_11_v1.json`
- `data/reports/dp_honey_lite_v3_all_pooling_sweep_safe_secret_vs_exfiltration_v1.json`
- `data/reports/dp_honey_lite_v3_all_pooling_sweep_benign_vs_secret_related_v1.json`
- `data/reports/dp_honey_lite_v3_grouped_binary_tasks_readout_window_layer_21_v1.json`
- `data/reports/dp_honey_lite_v3_policy_diagnostics_v1.json`
- `data/reports/dp_honey_lite_v3_policy_resolution_v1.json`
- `data/reports/dp_honey_lite_v3_policy_window_sweep_safe_secret_vs_exfiltration_v1.json`
- `data/reports/dp_honey_lite_v3_policy_window_grouped_binary_tasks_readout_window_layer_00_v1.json`
- `data/reports/dp_honey_lite_v3_selector_window_sweep_safe_secret_vs_exfiltration_v1.json`
- `data/reports/dp_honey_lite_v3_selector_window_grouped_binary_tasks_readout_window_layer_15_v1.json`
- `data/reports/dp_honey_lite_v3_selector_window_error_analysis_readout_window_layer_15_v1.json`
- `data/reports/dp_honey_lite_v3_selector_window_error_slices_readout_window_layer_15_v1.json`
- `data/reports/dp_honey_lite_v3_selector_window_cift_calibration_readout_window_layer_15_v1.json`
- `data/reports/dp_honey_lite_v3_selector_window_cift_operating_points_readout_window_layer_15_v1.json`
- `data/reports/dp_honey_lite_v3_selector_window_score_diagnostics_readout_window_layer_15_v1.json`
- `data/reports/dp_honey_lite_v3_selector_window_feature_ablation_v1.json`
- `data/reports/dp_honey_lite_v3_selector_window_cift_calibration_selector_band_12_18_v1.json`
- `data/reports/dp_honey_lite_v3_selector_window_cift_operating_points_selector_band_12_18_v1.json`
- `data/reports/dp_honey_lite_v3_selector_window_score_diagnostics_selector_band_12_18_v1.json`
- `data/reports/dp_honey_lite_v4_selector_window_sweep_safe_secret_vs_exfiltration_v1.json`
- `data/reports/dp_honey_lite_v4_selector_window_grouped_binary_tasks_readout_window_layer_15_v1.json`
- `data/reports/dp_honey_lite_v4_1_selector_window_sweep_safe_secret_vs_exfiltration_v1.json`
- `data/reports/dp_honey_lite_v4_1_selector_window_grouped_binary_tasks_readout_window_layer_15_v1.json`
- `data/detector_results_cift_v3_selector_window_layer_15_v1.jsonl`
- `data/detector_results_cift_v3_selector_window_layer_15_calibrated_v1.jsonl`

## Next Moves

The next experimental steps split into two related threads: completing the
static-feature promotion review, and refining the CIFT-like path into a closer
implementation of the paper method. Do not silently promote the combined
feature just because it improves aggregate performance, and do not discard CIFT
because the first simplified calibrated-distance score underperformed.

Recommended sequence:

1. Human-review the three Hard V3 combined-regression cases.
2. Decide whether each case is a true model regression, a defensible tradeoff,
   or a dataset/label ambiguity.
3. Keep `mean_pool_layer_18` as the fixed regression checkpoint while
   final-token candidates remain under evaluation.
4. Define a promotion rule that weighs average performance, worst-case
   checkpoint performance, and post-hoc discovery risk.
5. For CIFT-like work, treat `meta_c_10` raw full dual-readout as the historical
   regularized diagnostic target, and treat V3 selector-window layer 15 as the
   current DP-HONEY-lite localization checkpoint. The V1, V2, V3, policy-window,
   and selector-window artifacts now exist. V3 breaks the target-task text
   shortcut, V3.1 shows the selected-mode policy rule is parser-recoverable,
   V3.2 shows Qwen 0.6B is copy-biased when asked to resolve the rule directly,
   and V3.3 shows that selector-localized readout windows recover a stronger
   activation signal without using the explicit copy/mask decision text. The
   feature-expression ablation shows layers 12-18 contain useful additional raw
   signal, but the calibrated `selector_band_12_18` variant does not beat the
   calibrated layer-15 checkpoint and increases confident errors. Keep layer 15
   as the current calibrated checkpoint. Use the ablation result as a clue for
   future representation work. V4 then isolated the same failure families and
   forced `selected_mode = mode_b`, but it made the task text-baseline
   recoverable: both activation and word TF-IDF reached 1.0000 macro F1. Treat
   V4 as a targeted control, not a detector benchmark. V4.1 keeps the focused
   failure families but restores balanced selected modes and stable per-family
   policy maps, removing the full-prompt lexical shortcut. On V4.1, layer 15
   reaches 0.7646 macro F1 while word TF-IDF drops to 0.1172. The next CIFT move
   is V4.1 error analysis, calibration, and operating-point review before any
   runtime-facing bridge export is refreshed.
6. Keep registering every dataset, artifact, and machine-readable report in
   `data/lineage.json`.

The research question remains narrow and concrete:

```text
Does the activation signal still help when surface text cues become less generous?
```
