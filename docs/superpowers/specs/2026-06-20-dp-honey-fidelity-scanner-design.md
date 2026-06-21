---
title: "DP-HONEY: format fidelity + scanner/auto-decoy design"
type: design
status: draft
date: 2026-06-20
builds_on: detect.dp_honey (generator package + CLI + web UI, on main)
---

# DP-HONEY: format fidelity + scanner / auto-decoy

## Summary

Extend the DP-HONEY generator in two phases, delivered as one combined spec:

- **Phase A — Coverage + fidelity.** Add more secret families and a structural
  **checksum-valid** capability so decoys closely resemble real tokens (and pass
  structural/checksum scanners) while staying strictly **non-functional**.
- **Phase B — Scanner + auto-decoy.** Scan arbitrary text/files, detect and
  classify secret-shaped substrings by matching the registry, and **auto-generate
  a matching honeytoken per finding** (with an optional "swap" that replaces each
  detected secret with its decoy) — so a user never hand-picks a format.

Phase B classifies against the registry built in Phase A, so A ships first.

## Goals

- Broader, more realistic coverage: new SaaS/LLM families + the GitHub family.
- A **checksum-valid** fidelity capability for families with public checksum
  algorithms (GitHub/GitLab), implemented via a declarative grammar segment.
- A scanner that turns free text into `(family, position, confidence)` findings
  and a matching decoy per finding, without manual format selection.
- Preserve every existing safety invariant (synthetic-only, shape-only,
  non-functional, never trained on real secrets) and add a new one for scanner
  input handling.

## Non-goals (this spec)

- No real cryptographic material (no genuine keypairs, no validly-signed JWTs).
  "Checksum-valid" never means "provider-valid".
- No runtime gateway / live prompt interception (the scanner is a library + CLI +
  UI utility; wiring it into an inference path is a later, separate effort).
- No persistence of scanned input or any detected real-secret values.
- Phase B is not a full enterprise secret scanner (entropy heuristics, git
  history walking, etc.); it is registry-driven detection for auto-decoy.

## Safety invariants

Existing (unchanged): R2 (synthetic data only, never train on real secrets), R3
(outputs non-functional / shape-only / never provider-valid).

New, introduced by the scanner:

- **SAFE-1.** The scanner *receives* real secrets as input (that is its job). It
  must **never log, persist, or echo** matched secret values. Findings carry only
  `(format, start, end, confidence)`; auto-decoy output carries only decoys and
  the swapped text (which contains decoys, not originals). Showing matched values
  is an explicit opt-in that defaults off.
- **SAFE-2.** "Checksum-valid" ≠ "provider-valid". A checksum-correct GitHub decoy
  has no record on GitHub's backend and authenticates nothing.

---

## Phase A — Coverage + fidelity

### A1. New declarative `Checksum` segment

A third segment type beside `Literal` and `Variable`:

- `Checksum(name, length, algorithm)` — its value is **computed**, not sampled.
- `algorithm` is a **string id** (e.g. `"github-crc32-base62"`) resolved through a
  registry in a new `checksums.py`: `CHECKSUM_ALGORITHMS = {id: fn}` where
  `fn(prefix_and_body: str) -> str` returns the `length`-char checksum.
- `to_dict()` serializes `{kind, name, length, algorithm}` — the **id**, never the
  callable. Drift-hash and fail-closed load keep working unchanged.

Grammar integration:
- `variable_segments()` still returns only `Variable`, so the **DP bigram only
  trains on / samples the random body** — checksum chars are excluded.
- `assemble()` computes each `Checksum` segment from the already-assembled prefix
  + body and appends it.
- `extract_variables()` / `validate()` consume the checksum span and **recompute +
  verify** it (a wrong checksum fails validation).

### A2. `checksums.py`

- `github-crc32-base62`: GitHub's base62-encoded CRC32 over the random body,
  left-padded to the fixed checksum length.
- **Implementation risk (must be addressed in the plan):** the exact algorithm
  (which bytes are summed, the base62 alphabet/ordering, padding) must be
  confirmed against an authoritative reference and pinned with **test vectors**.
  Generated GitHub tokens must pass an *independent* checksum validator
  (e.g. the algorithm used by detect-secrets / gitleaks), or we do not claim
  "checksum-valid".

### A3. New families

**SaaS / LLM (structural fidelity, existing grammar — no public checksums):**

| slug | shape (approx; exact lengths pinned in plan) |
| --- | --- |
| `slack-bot-token` | `xoxb-` + digits(12) + `-` + digits(12) + `-` + alnum(24) |
| `slack-user-token` | `xoxp-` + digits(12) + `-` + digits(12) + `-` + alnum(24) |
| `slack-webhook-url` | `https://hooks.slack.com/services/T` + A-Z0-9(10) + `/B` + A-Z0-9(10) + `/` + alnum(24) |
| `google-api-key` | `AIza` + base64url(35) |
| `openai-project-key` | `sk-proj-` + base64url(48) (research: real keys embed a `T3BlbkFJ` infix) |
| `anthropic-api-key` | `sk-ant-api03-` + base64url(93) + `AA` |
| `sendgrid-key` | `SG.` + base64url(22) + `.` + base64url(43) |
| `twilio-account-sid` | `AC` + hex(32) |
| `twilio-api-key-sid` | `SK` + hex(32) |

(Adds a `HEX` charset constant to `grammar.py`.)

**GitHub family (checksum-valid, uses the `Checksum` segment):**

| slug | shape |
| --- | --- |
| `github-ghp` (**upgrade** existing shape-only) | `ghp_` + base62(30) + crc(6) |
| `github-oauth` (`gho_`) | `gho_` + base62(30) + crc(6) |
| `github-user-to-server` (`ghu_`) | `ghu_` + base62(30) + crc(6) |
| `github-server-to-server` (`ghs_`) | `ghs_` + base62(30) + crc(6) |
| `github-refresh` (`ghr_`) | `ghr_` + base62(30) + crc(6) |
| `github-fine-grained` (`github_pat_`) | `github_pat_` + base62(22) + `_` + base62(59) + crc (research exact layout) |

### A4. Registry, surfaces, housekeeping

- `REGISTRY_VERSION` → `"2"` (the format set and `github-ghp` spec changed).
- `list-formats`, the CLI, and the web UI pick up new families automatically
  (registry-driven). README compatibility matrix + its doc-consistency test gain
  the new rows.
- Upgrading `github-ghp` changes its `spec_hash`, so any previously-saved `ghp_`
  artifact **fails closed on load (drift)** — intended. The committed golden
  fixture is `aws-access-key-id` (spec unchanged) and still loads; its stored
  `registry_version` is informational (not equality-checked), so no regeneration
  is required, though the plan may regenerate it for tidiness.

---

## Phase B — Scanner + auto-decoy

### B1. Scannability

Not every format is safe to scan for: prefix-less, high-entropy formats
(`aws-secret-access-key`, `database-password`, `oauth-bearer`) would match almost
any random string and flood false positives. Add `FormatSpec.scannable: bool`
(default derived: `True` when the spec has a distinctive literal anchor, a
checksum, or is structurally unmistakable like `jwt`; `False` for the generic
prefix-less families). The scanner only considers `scannable` formats.

### B2. `scanner.py`

- `detection_pattern(spec) -> re.Pattern` — built from the spec's segments
  (`re.escape` literals; `[charset]{length}` for variable/checksum segments),
  bounded by non-secret-char lookarounds to avoid partial hits. Cached per format.
- `scan(text) -> list[Finding]` — for each scannable format, regex-find
  candidates; confirm each with `spec.validate()` (which includes checksum
  verification for checksummed families). `Finding = {format, start, end,
  confidence}` where confidence is `"high"` for checksum-confirmed matches and
  `"medium"` for structural-validate matches. Overlapping matches are de-duped,
  preferring higher confidence then longer span. **No secret value is stored in a
  Finding** (SAFE-1).
- `auto_decoy(text, *, seed=0) -> {findings, decoys, swapped_text}` — for each
  finding, generate one matching decoy for its family (deterministic in `seed`),
  and produce `swapped_text` by replacing each matched span with its decoy
  (right-to-left to preserve indices). `decoys` aligns 1:1 with `findings`.

### B3. Surfaces

- **CLI:** `scan` (read `--file` or stdin → findings JSON, **no secret values**)
  and `auto-decoy` (`--file`/stdin `[--seed S]` → decoys + swapped text). A
  `--show-matches` flag (default off) is the only way to echo matched values, with
  a warning.
- **Web UI:** a "Scan & auto-decoy" panel — a textarea posts to `POST /api/scan`
  and `POST /api/auto-decoy`; render findings (family, position, confidence) +
  the matching decoys + the swapped text. Reuses the existing service/app/JS
  structure (thin routes over a `scanner` service).

### B4. Data flow

```
text ──► scan() ──► findings[{format,start,end,confidence}]
                      │
                      ▼ (per finding)
                 generate matching decoy ──► decoys[]
                      │
                      ▼
                 swap spans ──► swapped_text (decoys only, no originals)
```

---

## Error handling

- Unknown/invalid input to generation paths still raises typed `DPHoneyError`
  (→ HTTP 400 in the UI), as today.
- A malformed artifact still fails closed (Phase A's `Checksum` segment adds
  algorithm-id validation: an unknown algorithm id in an artifact →
  `ModelSchemaError`).
- The scanner never raises on "no matches" — it returns an empty findings list.

## Testing

**Phase A:** each new family validates its own synthetic examples; the GitHub
family is verified **checksum-valid against an independent reference + test
vectors**; the `github-ghp` upgrade's drift is asserted; `REGISTRY_VERSION == "2"`;
README-matrix/registry consistency; the `Checksum` segment round-trips through an
artifact (snapshot stores the algorithm id; unknown id rejected).

**Phase B:** `scan` detects each scannable family inside seeded sample text
(built from generator output); classification is correct; checksum matches report
`high` confidence; prefix-less formats are **not** scanned (no false positives on
random strings); `auto_decoy` returns a valid matching decoy per finding and a
correctly swapped text; **no Finding or scan output contains a matched secret
value** (SAFE-1); CLI `scan`/`auto-decoy` and the UI endpoints work.

## Requirements

**Phase A (fidelity/coverage):**
- FX1. Add the SaaS/LLM families and the GitHub family listed in A3.
- FX2. Add a declarative `Checksum` segment: computed at assemble, verified at
  validate, excluded from bigram training, serialized by algorithm id, drift-safe.
- FX3. GitHub family is checksum-valid, verified against an independent reference.
- FX4. All families stay non-functional / non-crypto; corpus synthetic-from-specs;
  per-format safety notes.
- FX5. Bump `REGISTRY_VERSION`; update README matrix + doc test; CLI/UI auto-pick-up.

**Phase B (scanner/auto-decoy):**
- SC1. `scan(text)` detects + classifies scannable families via registry-derived
  patterns confirmed by `validate()`.
- SC2. `auto_decoy(text)` returns a matching decoy per finding + optional swapped
  text; deterministic in seed.
- SC3. Scanner never logs/persists/echoes matched secret values (SAFE-1);
  `--show-matches` is opt-in and default off.
- SC4. Prefix-less/generic formats are excluded from scanning by default.
- SC5. CLI `scan` + `auto-decoy`; web UI scan panel.
- SC6. Tests for detection, classification, confidence, auto-decoy, swap, and the
  no-secret-value safety property.

## Open / deferred

- Exact GitHub (and any future GitLab) checksum algorithm + test vectors — a
  research step at the start of Phase A implementation.
- Exact real lengths/infixes for `openai-project-key`, `anthropic-api-key`,
  `github_pat_` — confirm during implementation; the table values are starting
  points.
- Runtime gateway / live prompt injection wiring — explicitly out of scope here.
