# DP-HONEY generator

Synthetic, **shape-only** honeytoken generator for the Aegis capstone's
credential-leak detection layer. It produces format-compatible decoys — strings
that *look like* AWS keys, JWTs, GitHub tokens, and so on — using a declarative
grammar plus a differential-privacy-noised character bigram model.

> ⚠️ **Safety boundary.** Every value this package emits is **synthetic** and
> **shape-only**. Outputs are **never** provider-valid, signed, decryptable,
> authenticated, or usable credentials. Do **not** treat any output as a real
> secret, and **never** train this package on real credentials. The example
> tokens in this README are illustrative and non-functional.

---

## What this is (and is not)

This package is the DP-HONEY generator plus its registry-derived scanner and
auto-decoy helper. It builds decoys, model artifacts, scanner findings, and
metrics around the same declarative format registry. It is intentionally small
and auditable.

### Out of scope

These belong to other teammates / future work and are **not** implemented here:

- Conformal calibration
- Eq.5 catch-probability **accounting**
- Runtime **gateway** integration
- **Tool-call** scanning
- Provider-valid credential generation
- Real secret corpus ingestion
- Enterprise-grade secret scanning beyond the registry-derived detector

---

## Install

Requires **Python ≥ 3.11** and **NumPy**.

```bash
# from the repository root
pip install -e .            # runtime (numpy)
pip install -e ".[dev]"     # + pytest for the test suite
```

You can also run everything without installing, straight from the repo root,
because the package is import- and `-m`-runnable in place.

---

## Quickstart (CLI)

Run as a module: `python -m detect.dp_honey <command>` (or `dp-honey <command>`
after `pip install`). All token-producing commands print a synthetic/non-functional
banner to **stderr**, so piping stdout stays clean.

```bash
# 1. Discover the registered shape-only formats
python -m detect.dp_honey list-formats

# 2. Preview the synthetic training corpus for a format
python -m detect.dp_honey preview-corpus --format github-ghp --count 3 --seed 0

# 3. Generate decoys directly (trains a model on the fly, then samples)
python -m detect.dp_honey generate --format aws-access-key-id --count 2 --seed 1
#   AKIA7X3K9Q0EXAMPLE00     (illustrative; output is synthetic & random)
#   AKIA2M8VJ5R1EXAMPLE00

# 4. Train a reusable model artifact (refuses overwrite without --force)
python -m detect.dp_honey train --format github-ghp --out models/ghp.json \
    --epsilon 1.0 --clip 1.0 --corpus-size 2000 --seed 7

# 5. Generate from a saved artifact (no retraining)
python -m detect.dp_honey generate --model models/ghp.json --count 5 --seed 1

# 6. Inspect an artifact's metadata + snapshot/hash status (lenient)
python -m detect.dp_honey inspect-model --model models/ghp.json

# 7. Strictly validate an artifact (exit 0 = valid, nonzero = rejected)
python -m detect.dp_honey validate --model models/ghp.json

# 8. Compute realism / sanity metrics for a batch (JSON)
python -m detect.dp_honey report --format github-ghp --count 100 --seed 1

# 9. Scan text for registered, scannable secret shapes (no matched values echoed)
python -m detect.dp_honey scan --file suspect.txt

# 10. Scan and replace each detected span with a matching synthetic decoy
python -m detect.dp_honey auto-decoy --file suspect.txt --seed 1

# 11. Emit held-out scanner FP/FN evidence with conformal confidence calibration
python -m detect.dp_honey eval-scanner --positive-per-format 50 --seed 11 \
    --target-alpha 0.01 \
    --negative-count 1000 \
    --calibration-count 1000 \
    --output introspection/data/reports/dp_honey_scanner_eval_v1.json

# 12. Emit all-format generation-realism evidence without raw token values
python -m detect.dp_honey eval-realism --count-per-format 25 --seed 11 \
    --output introspection/data/reports/dp_honey_generation_realism_eval_v2.json

# 13. Emit the statistical-distinguisher suite evidence named by the paper
python -m detect.dp_honey eval-statistical-distinguishers \
    --train-count-per-format 25 \
    --test-count-per-format 25 \
    --alpha 0.1 \
    --seed 11 \
    --output introspection/data/reports/dp_honey_statistical_distinguisher_eval_v2.json
```

`generate` is capped at 10000 (it streams one token at a time); `report` is
capped at 5000 (metrics require materializing the whole batch). Oversized counts
and `eval-realism` batches exit nonzero *before* any generation work begins.

## Scanner & auto-decoy

`scan` derives detection patterns from the live registry, confirms candidates
with each format's `validate()` method, and prints JSON findings shaped as
`{format,start,end,confidence}`. If no registered format matches a token-like
high-entropy string, it falls back to an `unknown-token` finding with `low`
confidence and a same-shape decoy. By default, `scan` does **not** echo matched
input values. `--show-matches` is the explicit opt-in for debugging and should be
handled carefully.

`auto-decoy` runs the same scan and generates one matching synthetic decoy per
finding, then emits `swapped_text` with detected spans replaced. It is a helper,
not a sanitizer: only detected scannable spans are replaced, and unrelated input
text is preserved.

`eval-scanner` generates deterministic registry-shaped positives and benign
negatives, reports true/false positives and false negatives separately, and
never includes raw token values in the report. It also fits a split-conformal
confidence threshold over benign calibration scores. This calibrates the scanner
confidence gate; it is not a proof that generated honeytokens are
indistinguishable from real credentials.

`eval-realism` builds a separate all-format generation evidence artifact. It
compares generated batches with same-format synthetic reference batches using
aggregate validity, duplicate-rate, character-entropy, and model-likelihood
metrics, without serializing token values. It is a bounded sanity gate, not the
paper's full statistical-distinguisher suite.

`eval-statistical-distinguishers` runs the paper-named realism tests: character
entropy, bigram likelihood, numeric-substring features, and a discriminator MLP.
It reports only aggregate metrics and pass/fail statuses. The current seeded v2
artifact passes all four bounded same-format synthetic holdout tests. That is
useful local-registry evidence, but `paper_faithful_statistical_distinguisher`
remains false until the reference source is provider-like or
real-credential-distribution evidence.

Prefix-less generic formats such as `aws-secret-access-key`, `oauth-bearer`, and
`database-password` are excluded from registry classification to avoid noisy
false positives; sufficiently token-like values may still be replaced by the
unknown fallback. Checksum-confirmed formats report `high` confidence,
registry-structural matches report `medium`, and unknown fallback matches report
`low`.

## Web UI

A local browser UI runs every command without the terminal:

```bash
pip install -e ".[dp-honey-ui]"
python -m detect.dp_honey.webui      # serves http://127.0.0.1:8000 (localhost only)
```

It calls the library directly (no subprocess), keeps the synthetic/shape-only
banner on every screen, and offers a model library (saved models in `models/`
plus the committed golden fixture). The API is a thin FastAPI layer; see `/docs`
for the auto-generated endpoint reference.

---

## Quickstart (Python API)

```python
from detect.dp_honey import (
    generate_honeytokens, build_model, train_model,
    save_model, load_model, compute_report, list_formats,
)

# Simplest: train on the fly and sample (deterministic in the seeds).
tokens = generate_honeytokens("github-ghp", count=5, sample_seed=1, train_seed=7)

# Build a reusable model, save it, reload it, and sample without retraining.
model = build_model("aws-access-key-id", epsilon=1.0, clip=1.0, corpus_size=2000, train_seed=7)
save_model(model, "models/aws.json", force=True)
reloaded = load_model("models/aws.json")
same = generate_honeytokens(model=reloaded, count=5, sample_seed=1)

# Sanity metrics for a batch.
report = compute_report(reloaded.sample(100, seed=1), reloaded)
print(report["validity_rate"], report["char_entropy_bits"])
```

`generate_honeytokens` is the high-level entry point; `train_model` (explicit
corpus), `build_model`, `save_model`, and `load_model` are the lower-level helpers.

---

## Model artifact schema (v1)

Artifacts are transparent, diffable JSON — never pickles — and **fail closed** on
load. The canonical form is reproducible from its training inputs (no timestamp).

| Group | Keys | Purpose |
| ----- | ---- | ------- |
| identity | `schema_version`, `generator` | schema + producing package/version |
| format | `format.slug`, `format.registry_version`, `format.spec_hash`, `format.spec_snapshot` | format identity + drift detection |
| privacy | `privacy.epsilon`, `privacy.clip`, `privacy.corpus_size`, `privacy.train_seed`, `privacy.trained_at` (always `null` in v1), `privacy.mechanism` | DP/training settings |
| alphabet | `alphabet.start_token`, `alphabet.symbols` | variable-character alphabet |
| transitions | `transitions` | normalized `P(next \| prev)` rows |
| safety | `safety.synthetic_only`, `safety.provider_valid`, `safety.note` | provenance + disclaimer |

Loading rejects (with a typed error, before any generation): bad JSON, missing
fields, unknown `schema_version`, **format snapshot/hash drift**, unknown format
slug, invalid alphabet membership, out-of-range privacy metadata, and non-finite /
negative / non-normalized transition probabilities.

> `inspect-model` is **lenient** and always exits 0 — it reports
> `snapshot_status: OK | DRIFT | UNKNOWN_FORMAT` for debugging. Use `validate`
> (which exits nonzero on any problem) as the pass/fail gate in scripts.

---

## Format compatibility matrix

Every format below is **shape-only**: structurally compatible with the named
secret family but never a real, valid, or usable credential.

| Slug | Name | Shape sketch | Provider-valid? |
| ---- | ---- | ------------ | --------------- |
| `aws-access-key-id` | AWS Access Key ID | `AKIA` + 16×[A–Z0–9] | No — shape only |
| `aws-secret-access-key` | AWS Secret Access Key | 40×[A–Za–z0–9/+] | No — shape only |
| `oauth-bearer` | OAuth Bearer Token | 40×[A–Za–z0–9-_] | No — shape only |
| `generic-sk` | Generic sk- API Key | `sk-` + 48×[A–Za–z0–9] | No — shape only |
| `database-password` | Database Password | 20×mixed (≥1 lower/upper/digit) | No — shape only |
| `jwt` | JWT-shaped Token | base64url(20).(40).(43), unsigned | No — shape only |
| `ssh-private-key` | SSH Private Key (marker) | PEM armor + 64×base64url body | No — shape only |
| `stripe-sk-live` | Stripe sk_live_ Secret Key | `sk_live_` + 24×[A–Za–z0–9] | No — shape only |
| `github-ghp` | GitHub Personal Access Token | `ghp_` + 30 base62 + CRC | No — shape only (checksum-valid) |
| `slack-bot-token` | Slack Bot Token | `xoxb-` + 12d-12d-24 alnum | No — shape only |
| `slack-user-token` | Slack User Token | `xoxp-` + 12d-12d-24 alnum | No — shape only |
| `slack-webhook-url` | Slack Webhook URL | hooks.slack.com/services/T../B../.. | No — shape only |
| `google-api-key` | Google API Key | `AIza` + 35 base64url | No — shape only |
| `openai-project-key` | OpenAI Project Key | `sk-proj-` + 48 base64url | No — shape only |
| `anthropic-api-key` | Anthropic API Key | `sk-ant-api03-` + 93 base64url + AA | No — shape only |
| `sendgrid-key` | SendGrid API Key | `SG.` + 22.43 base64url | No — shape only |
| `twilio-account-sid` | Twilio Account SID | `AC` + 32 hex | No — shape only |
| `twilio-api-key-sid` | Twilio API Key SID | `SK` + 32 hex | No — shape only |
| `github-oauth` | GitHub OAuth Token | `gho_` + 30 base62 + CRC | No — shape only (checksum-valid) |
| `github-user-to-server` | GitHub User-to-Server Token | `ghu_` + 30 base62 + CRC | No — shape only (checksum-valid) |
| `github-server-to-server` | GitHub Server-to-Server Token (legacy) | `ghs_` + 30 base62 + CRC | No — shape only (checksum-valid legacy shape) |
| `github-refresh` | GitHub Refresh Token | `ghr_` + 30 base62 + CRC | No — shape only (checksum-valid) |
| `github-fine-grained` | GitHub Fine-grained PAT | `github_pat_` + 22 alnum + `_` + 59 alnum | No — shape only |

`jwt` produces an **unsigned**, non-verifiable shape; `ssh-private-key` produces a
PEM-armored **marker** with a short synthetic body, not a real or decryptable key.
Classic GitHub-family tokens carry a valid body-only base62-CRC32 checksum, but
remain non-functional because no provider record exists; fine-grained PATs are
structural-only until a public checksum vector is pinned. The `ghs_` row covers
the legacy 40-character installation-token shape, not the 2026 stateless
`ghs_APPID_JWT` rollout.

---

## Determinism & privacy notes

- **Determinism:** the same `(format, epsilon, clip, corpus_size, train_seed)`
  yields the same model, and the same `(model, count, sample_seed)` yields the
  same tokens. Determinism holds for the same code + dependency versions; it is
  not promised across NumPy major versions.
- **Privacy:** training clips each example's per-cell bigram contribution to
  `clip`, then adds `Laplace(clip / epsilon)` noise, projects to ≥0, and
  normalizes. This is a research-grade DP **sketch** over bigram counts
  (`epsilon` is the per-cell budget), not a formally audited end-to-end guarantee.
- **Realism metrics** (`report`) are sanity checks for a synthetic generator —
  **not** proof of indistinguishability from real credentials.

---

## Demo path

A committed, synthetic golden artifact lives at
`tests/dp_honey/fixtures/dp_honey/golden_model.json` (an `aws-access-key-id` model). It
needs no retraining:

```bash
python -m detect.dp_honey inspect-model --model tests/dp_honey/fixtures/dp_honey/golden_model.json
python -m detect.dp_honey validate    --model tests/dp_honey/fixtures/dp_honey/golden_model.json
python -m detect.dp_honey generate    --model tests/dp_honey/fixtures/dp_honey/golden_model.json --count 5 --seed 1
python -m detect.dp_honey report      --model tests/dp_honey/fixtures/dp_honey/golden_model.json --count 100 --seed 1
```

Locally trained artifacts belong under `models/` (gitignored). The fixture under
`tests/dp_honey/fixtures/dp_honey/` is the only model committed to the repo.

---

## Testing

```bash
python -m pytest tests/dp_honey
```

The Aegis quality gate also runs the DP-HONEY tests and lints
`src/detect/dp_honey` plus `tests/dp_honey`.
