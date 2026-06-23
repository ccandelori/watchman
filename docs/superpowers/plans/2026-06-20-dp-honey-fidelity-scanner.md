# DP-HONEY Fidelity + Scanner/Auto-decoy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Broaden DP-HONEY's format coverage with structurally-accurate SaaS/LLM families and a checksum-valid GitHub family, then add a registry-driven scanner that detects secrets in text and auto-generates a matching (non-functional) decoy for each.

**Architecture:** Phase A extends the declarative grammar with a computed `Checksum` segment and adds families to the registry. Phase B adds a `scanner` module that derives a detection regex from each `FormatSpec`, confirms hits with `validate()`, and reuses the existing generator to mint a matching decoy per finding — surfaced via CLI and the web UI. The scanner never logs or echoes matched secret values.

**Tech Stack:** Python 3.11+, numpy, FastAPI/uvicorn (UI), pytest, ruff. Stdlib `re`, `zlib` (CRC32), `binascii`.

**Spec:** `docs/superpowers/specs/2026-06-20-dp-honey-fidelity-scanner-design.md` (requirements FX1–FX5, SC1–SC6; safety SAFE-1, SAFE-2).

**Branch:** `feat/dp-honey-fidelity-scanner` (already created off `main`).

---

## File structure

```text
detect/dp_honey/
  grammar.py     # MODIFY: add HEX charset + Checksum segment type
  checksums.py   # CREATE: checksum algorithm registry (github-crc32-base62)
  formats.py     # MODIFY: new families, GitHub family, scannable flag, REGISTRY_VERSION="2"
  model_io.py    # MODIFY: reject unknown checksum algorithm id on load
  scanner.py     # CREATE (Phase B): detection_pattern, scan, auto_decoy
  __main__.py    # MODIFY (Phase B): `scan` and `auto-decoy` CLI commands
  README.md      # MODIFY: compatibility matrix rows + scanner section
  webui/
    service.py   # MODIFY (Phase B): run_scan / run_auto_decoy wrappers
    app.py       # MODIFY (Phase B): /api/scan, /api/auto-decoy routes
    static/app.js# MODIFY (Phase B): "Scan & auto-decoy" nav + panel
tests/
  test_dp_honey_checksums.py   # CREATE: checksum vectors + round-trip
  test_dp_honey_grammar_checksum.py # CREATE: Checksum segment behavior
  test_dp_honey_formats.py     # MODIFY: new families validate; generics scannable=False
  test_dp_honey_model_io.py    # MODIFY: unknown checksum algorithm rejected
  test_dp_honey_docs.py        # (unchanged logic; passes after README matrix update)
  test_dp_honey_scanner.py     # CREATE (Phase B): scan/auto_decoy + SAFE-1
  test_dp_honey_cli.py         # MODIFY (Phase B): scan/auto-decoy commands
  test_dp_honey_webui.py       # MODIFY (Phase B): scan service functions
  test_dp_honey_webui_http.py  # MODIFY (Phase B): /api/scan smoke test
```

---

# PHASE A — Coverage + fidelity

## Task A0: Research & pin the GitHub checksum algorithm

**Files:**
- Create: `detect/dp_honey/checksums.py`
- Test: `tests/test_dp_honey_checksums.py`

GitHub tokens (`ghp_`, `gho_`, …) append a base62-encoded CRC32 checksum of the
random body. The exact scheme (which bytes are summed, the base62 alphabet
ordering, the checksum length/padding) MUST be confirmed against an independent
reference — do not trust memory.

- [ ] **Step 1: Research the algorithm**

Use WebSearch/WebFetch to confirm GitHub's token checksum scheme from at least one
authoritative source (GitHub's "About authentication / token formats" docs, or the
implementation in `detect-secrets` / `gitleaks` / `trufflehog`). Capture:
- the base62 alphabet and ordering (typically `0-9A-Za-z`),
- which characters are CRC'd (the 30-char random body after the `_`),
- the checksum length (6) and left-padding rule.
Record 1–2 concrete `(random_body -> checksum)` test vectors from a reference impl.

- [ ] **Step 2: Write the failing test (with the verified vectors)**

Create `tests/test_dp_honey_checksums.py` (fill the vector(s) from Step 1):

```python
"""Tests for checksum algorithms used by structurally-faithful formats."""

from __future__ import annotations

import pytest

from detect.dp_honey.checksums import CHECKSUM_ALGORITHMS, compute_checksum


def test_github_crc32_base62_known_vectors():
    fn = CHECKSUM_ALGORITHMS["github-crc32-base62"]
    # Vector(s) confirmed in Step 1 against an independent reference:
    body = "0123456789ABCDEFGHIJKLMNOPQRST"  # 30-char example body
    expected = "<PASTE_VERIFIED_CHECKSUM>"     # replace with the reference value
    assert fn(body, length=6) == expected
    assert len(fn(body, length=6)) == 6


def test_compute_checksum_unknown_algorithm_raises():
    with pytest.raises(KeyError):
        compute_checksum("not-an-algorithm", "abc", length=6)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_checksums.py -q`
Expected: FAIL (module missing).

- [ ] **Step 4: Implement `checksums.py`**

```python
"""Checksum algorithms for structurally-faithful (checksum-valid) formats.

These make a decoy pass a provider's *structural* checksum check. A checksum-valid
decoy is still NON-FUNCTIONAL: the provider has no record of it, so it
authenticates nothing. No real cryptography is used here.
"""

from __future__ import annotations

import zlib
from typing import Callable

_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _to_base62(number: int) -> str:
    if number == 0:
        return _BASE62[0]
    digits = []
    while number:
        number, rem = divmod(number, 62)
        digits.append(_BASE62[rem])
    return "".join(reversed(digits))


def github_crc32_base62(body: str, *, length: int = 6) -> str:
    """Base62-encoded CRC32 of *body*, left-padded with '0' to *length* chars.

    Confirmed in Task A0 Step 1 against an independent reference implementation.
    """
    checksum = zlib.crc32(body.encode("ascii")) & 0xFFFFFFFF
    encoded = _to_base62(checksum)
    return encoded.rjust(length, "0")[:length]


CHECKSUM_ALGORITHMS: dict[str, Callable[..., str]] = {
    "github-crc32-base62": github_crc32_base62,
}


def compute_checksum(algorithm: str, body: str, *, length: int) -> str:
    """Resolve *algorithm* from the registry and compute the checksum for *body*."""
    return CHECKSUM_ALGORITHMS[algorithm](body, length=length)
```

- [ ] **Step 5: Reconcile test and implementation**

If the verified reference value from Step 1 disagrees with this implementation
(e.g. GitHub CRCs the prefix too, or uses a different base62 ordering), adjust
`github_crc32_base62` until the test vector passes. The reference vector is the
source of truth — the claim "checksum-valid" depends on it.

Run: `python -m pytest tests/test_dp_honey_checksums.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add detect/dp_honey/checksums.py tests/test_dp_honey_checksums.py
git commit -m "feat(dp-honey): GitHub base62-CRC32 checksum (verified vectors)"
```

---

## Task A1: `Checksum` grammar segment + HEX charset

**Files:**
- Modify: `detect/dp_honey/grammar.py`
- Test: `tests/test_dp_honey_grammar_checksum.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dp_honey_grammar_checksum.py`:

```python
"""Tests for the computed Checksum grammar segment."""

from __future__ import annotations

import numpy as np
import pytest

from detect.dp_honey.checksums import github_crc32_base62
from detect.dp_honey.grammar import ALNUM, Checksum, FormatSpec, Literal, Variable


def _spec() -> FormatSpec:
    return FormatSpec(
        slug="t-cksum",
        name="t",
        description="checksum test format",
        category="test",
        segments=(Literal("ghp_"), Variable("body", ALNUM, 30), Checksum("crc", 6, "github-crc32-base62")),
        safety_note="synthetic test format; not a real credential",
    )


def test_checksum_segment_excluded_from_variable_segments():
    spec = _spec()
    assert [s.name for s in spec.variable_segments()] == ["body"]  # only Variable


def test_assemble_appends_correct_checksum_and_validates():
    spec = _spec()
    token = spec.random_example(np.random.default_rng(0))
    assert spec.validate(token)
    body = token[len("ghp_"): len("ghp_") + 30]
    assert token.endswith(github_crc32_base62(body, length=6))


def test_wrong_checksum_fails_validation():
    spec = _spec()
    token = spec.random_example(np.random.default_rng(1))
    broken = token[:-1] + ("0" if token[-1] != "0" else "1")
    assert not spec.validate(broken)


def test_checksum_segment_serializes_algorithm_id_only():
    spec = _spec()
    snap = spec.to_snapshot()
    seg = snap["segments"][-1]
    assert seg == {"kind": "checksum", "name": "crc", "length": 6, "algorithm": "github-crc32-base62"}
    import json
    json.loads(json.dumps(snap))  # pure data, no callables


def test_unknown_algorithm_raises_on_assemble():
    spec = FormatSpec(
        slug="t-bad", name="t", description="t", category="test",
        segments=(Literal("x_"), Variable("b", ALNUM, 4), Checksum("c", 6, "nope")),
        safety_note="t",
    )
    with pytest.raises(KeyError):
        spec.random_example(np.random.default_rng(0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_grammar_checksum.py -q`
Expected: FAIL (`ImportError: cannot import name 'Checksum'`).

- [ ] **Step 3: Implement in `grammar.py`**

Add the `HEX` charset near the other charset constants:

```python
HEX = "0123456789abcdef"
```

Add the `Checksum` dataclass after the `Variable` dataclass:

```python
@dataclass(frozen=True)
class Checksum:
    """A computed segment: its value is a checksum of the preceding token text.

    ``algorithm`` is an id resolved through :mod:`detect.dp_honey.checksums`. The
    value is never sampled by the bigram model; it is derived at assembly time and
    verified at validation time. Only the id is serialized (no callables).
    """

    name: str
    length: int
    algorithm: str

    def to_dict(self) -> dict:
        return {"kind": "checksum", "name": self.name, "length": self.length, "algorithm": self.algorithm}
```

Update the `Segment` union:

```python
Segment = Union[Literal, Variable, Checksum]
```

In `FormatSpec.assemble()`, handle `Checksum` segments by computing over the
text assembled so far. Replace the body of `assemble` with:

```python
    def assemble(self, variables: list[str]) -> str:
        from .checksums import compute_checksum

        out: list[str] = []
        vi = 0
        for seg in self.segments:
            if isinstance(seg, Literal):
                out.append(seg.text)
            elif isinstance(seg, Variable):
                out.append(variables[vi])
                vi += 1
            else:  # Checksum
                out.append(compute_checksum(seg.algorithm, "".join(out), length=seg.length))
        return "".join(out)
```

In `FormatSpec.extract_variables()`, consume `Checksum` spans and verify them.
Replace the loop body so the segment walk handles three kinds:

```python
    def extract_variables(self, token: str) -> Optional[list[str]]:
        from .checksums import compute_checksum

        pos = 0
        variables: list[str] = []
        for seg in self.segments:
            if isinstance(seg, Literal):
                if not token.startswith(seg.text, pos):
                    return None
                pos += len(seg.text)
            elif isinstance(seg, Variable):
                chunk = token[pos : pos + seg.length]
                if len(chunk) != seg.length or any(c not in seg.alphabet for c in chunk):
                    return None
                variables.append(chunk)
                pos += seg.length
            else:  # Checksum: recompute over the text consumed so far and compare
                chunk = token[pos : pos + seg.length]
                if len(chunk) != seg.length:
                    return None
                expected = compute_checksum(seg.algorithm, token[:pos], length=seg.length)
                if chunk != expected:
                    return None
                pos += seg.length
        if pos != len(token):
            return None
        return variables
```

`variable_segments()` already returns only `Variable` instances, so the bigram
model automatically ignores `Checksum` segments — no change needed there.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_grammar_checksum.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `python -m pytest -q` then `python -m ruff check detect tests conftest.py`
Expected: all pass; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add detect/dp_honey/grammar.py tests/test_dp_honey_grammar_checksum.py
git commit -m "feat(dp-honey): computed Checksum grammar segment and HEX charset"
```

---

## Task A2: `scannable` field on `FormatSpec`

**Files:**
- Modify: `detect/dp_honey/grammar.py`
- Test: `tests/test_dp_honey_grammar_checksum.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dp_honey_grammar_checksum.py`:

```python
def test_scannable_defaults_true_and_is_overridable():
    base = FormatSpec(slug="s", name="s", description="s", category="t",
                      segments=(Literal("x_"), Variable("b", ALNUM, 4)), safety_note="t")
    assert base.scannable is True
    off = FormatSpec(slug="s2", name="s", description="s", category="t",
                     segments=(Variable("b", ALNUM, 4),), safety_note="t", scannable=False)
    assert off.scannable is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_grammar_checksum.py -k scannable -q`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'scannable'`).

- [ ] **Step 3: Implement in `grammar.py`**

Add a field to the `FormatSpec` dataclass (after `provider_valid`):

```python
    provider_valid: bool = False
    scannable: bool = True
    extra_predicate: Optional[Callable[[str], bool]] = None
```

`scannable` is intentionally NOT part of `to_snapshot()` (it is a scanning hint,
not part of the artifact's format identity) — leave `to_snapshot()` unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_grammar_checksum.py -k scannable -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/grammar.py tests/test_dp_honey_grammar_checksum.py
git commit -m "feat(dp-honey): add scannable hint to FormatSpec"
```

---

## Task A3: SaaS/LLM families

**Files:**
- Modify: `detect/dp_honey/formats.py`
- Test: `tests/test_dp_honey_formats.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_dp_honey_formats.py`, **edit the existing top-of-file
`REQUIRED_SLUGS` set literal** (do NOT append a reassignment at the bottom — the
`@pytest.mark.parametrize("slug", sorted(REQUIRED_SLUGS))` decorator on
`test_synthetic_examples_validate_against_their_own_spec` captures the set at
definition time, so only editing the literal extends coverage). Add these entries
to that set:

```python
    # add to the existing REQUIRED_SLUGS set literal:
    "slack-bot-token", "slack-user-token", "slack-webhook-url",
    "google-api-key", "openai-project-key", "anthropic-api-key",
    "sendgrid-key", "twilio-account-sid", "twilio-api-key-sid",
```

The parametrized validate test now exercises these slugs too. Then append a
focused structure test (imports are function-local to avoid module-level
`E402`; `get_format` is also already imported at the top):

```python
def test_saas_prefixes_present():
    import numpy as np
    from detect.dp_honey import get_format
    cases = {
        "slack-bot-token": "xoxb-",
        "google-api-key": "AIza",
        "openai-project-key": "sk-proj-",
        "anthropic-api-key": "sk-ant-api03-",
        "sendgrid-key": "SG.",
        "slack-webhook-url": "https://hooks.slack.com/services/",
    }
    for slug, prefix in cases.items():
        tok = get_format(slug).random_example(np.random.default_rng(0))
        assert tok.startswith(prefix), (slug, tok)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_formats.py -q`
Expected: FAIL (`UnknownFormatError` / missing slugs).

- [ ] **Step 3: Implement in `formats.py`**

Add `HEX` to the grammar import, then add these specs to the `_SPECS` tuple
(structural fidelity, existing grammar; all have prefixes so `scannable` stays
default `True`):

```python
    FormatSpec(
        slug="slack-bot-token", name="Slack Bot Token", category="token",
        description="Slack bot token (xoxb-).",
        segments=(Literal("xoxb-"), Variable("a", DIGITS, 12), Literal("-"),
                  Variable("b", DIGITS, 12), Literal("-"), Variable("c", ALNUM, 24)),
        safety_note="'xoxb-' Slack-bot-token shape. " + SHAPE_ONLY,
    ),
    FormatSpec(
        slug="slack-user-token", name="Slack User Token", category="token",
        description="Slack user token (xoxp-).",
        segments=(Literal("xoxp-"), Variable("a", DIGITS, 12), Literal("-"),
                  Variable("b", DIGITS, 12), Literal("-"), Variable("c", ALNUM, 24)),
        safety_note="'xoxp-' Slack-user-token shape. " + SHAPE_ONLY,
    ),
    FormatSpec(
        slug="slack-webhook-url", name="Slack Webhook URL", category="webhook",
        description="Slack incoming webhook URL.",
        segments=(Literal("https://hooks.slack.com/services/T"), Variable("t", UPPER_DIGITS, 10),
                  Literal("/B"), Variable("b", UPPER_DIGITS, 10), Literal("/"), Variable("s", ALNUM, 24)),
        safety_note="Slack-webhook-URL shape. " + SHAPE_ONLY,
    ),
    FormatSpec(
        slug="google-api-key", name="Google API Key", category="api-key",
        description="Google API key (AIza...).",
        segments=(Literal("AIza"), Variable("body", BASE64URL, 35)),
        safety_note="'AIza' Google-API-key shape. " + SHAPE_ONLY,
    ),
    FormatSpec(
        slug="openai-project-key", name="OpenAI Project Key", category="api-key",
        description="OpenAI project API key (sk-proj-).",
        segments=(Literal("sk-proj-"), Variable("body", BASE64URL, 48)),
        safety_note="'sk-proj-' OpenAI-key shape. " + SHAPE_ONLY,
    ),
    FormatSpec(
        slug="anthropic-api-key", name="Anthropic API Key", category="api-key",
        description="Anthropic API key (sk-ant-api03-).",
        segments=(Literal("sk-ant-api03-"), Variable("body", BASE64URL, 93), Literal("AA")),
        safety_note="'sk-ant-api03-' Anthropic-key shape. " + SHAPE_ONLY,
    ),
    FormatSpec(
        slug="sendgrid-key", name="SendGrid API Key", category="api-key",
        description="SendGrid API key (SG.).",
        segments=(Literal("SG."), Variable("a", BASE64URL, 22), Literal("."), Variable("b", BASE64URL, 43)),
        safety_note="'SG.' SendGrid-key shape. " + SHAPE_ONLY,
    ),
    FormatSpec(
        slug="twilio-account-sid", name="Twilio Account SID", category="cloud-key",
        description="Twilio Account SID (AC + 32 hex).",
        segments=(Literal("AC"), Variable("body", HEX, 32)),
        safety_note="'AC' Twilio-Account-SID shape. " + SHAPE_ONLY,
    ),
    FormatSpec(
        slug="twilio-api-key-sid", name="Twilio API Key SID", category="cloud-key",
        description="Twilio API Key SID (SK + 32 hex).",
        segments=(Literal("SK"), Variable("body", HEX, 32)),
        safety_note="'SK' Twilio-API-key-SID shape. " + SHAPE_ONLY,
    ),
```

Update the grammar import line in `formats.py` to include `HEX`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_formats.py -q`
Expected: PASS (new slugs validate; prefixes correct).

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/formats.py tests/test_dp_honey_formats.py
git commit -m "feat(dp-honey): add Slack/Google/OpenAI/Anthropic/SendGrid/Twilio families"
```

---

## Task A4: GitHub family (checksum-valid) + generics non-scannable + version bump

**Files:**
- Modify: `detect/dp_honey/formats.py`
- Test: `tests/test_dp_honey_formats.py`

- [ ] **Step 1: Write the failing test**

First, **add the GitHub slugs to the top-of-file `REQUIRED_SLUGS` set literal**
(same reason as A3 — extends the parametrized validate test):

```python
    # add to the existing REQUIRED_SLUGS set literal:
    "github-oauth", "github-user-to-server",
    "github-server-to-server", "github-refresh", "github-fine-grained",
```

(`github-ghp` is already in the set.) Then add **two new imports to the existing
top import block** (do not import mid-file — `np`, `get_format`,
`list_format_slugs` are already imported at the top):

```python
from detect.dp_honey.checksums import github_crc32_base62
from detect.dp_honey.formats import REGISTRY_VERSION
```

Then append these tests (they use the already-imported `np` and `get_format`):

```python
GITHUB_SLUGS = {
    "github-ghp", "github-oauth", "github-user-to-server",
    "github-server-to-server", "github-refresh", "github-fine-grained",
}


def test_registry_version_bumped():
    assert REGISTRY_VERSION == "2"


@pytest.mark.parametrize("slug", sorted(GITHUB_SLUGS - {"github-fine-grained"}))
def test_classic_github_tokens_are_checksum_valid(slug):
    spec = get_format(slug)
    tok = spec.random_example(np.random.default_rng(3))
    assert spec.validate(tok)
    # last 6 chars are the base62 CRC32 of the 30-char body after the prefix
    prefix_len = tok.index("_") + 1
    body = tok[prefix_len:prefix_len + 30]
    assert tok[prefix_len + 30:] == github_crc32_base62(body, length=6)


def test_generics_are_not_scannable():
    for slug in ("aws-secret-access-key", "database-password", "oauth-bearer"):
        assert get_format(slug).scannable is False


def test_prefixed_formats_are_scannable():
    for slug in ("github-ghp", "google-api-key", "stripe-sk-live"):
        assert get_format(slug).scannable is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_formats.py -k "github or scannable or registry_version" -q`
Expected: FAIL.

- [ ] **Step 3: Implement in `formats.py`**

Bump the version constant:

```python
REGISTRY_VERSION = "2"
```

Replace the existing `github-ghp` spec with the checksum-valid version and add the
rest of the GitHub family (import `Checksum` and `ALNUM` as `BASE62` is `ALNUM`):

```python
    FormatSpec(
        slug="github-ghp", name="GitHub Personal Access Token (ghp_)", category="vcs-token",
        description="GitHub classic PAT (checksum-valid).",
        segments=(Literal("ghp_"), Variable("body", ALNUM, 30), Checksum("crc", 6, "github-crc32-base62")),
        safety_note="'ghp_' GitHub-token shape with valid checksum. " + SHAPE_ONLY,
    ),
    FormatSpec(
        slug="github-oauth", name="GitHub OAuth Token (gho_)", category="vcs-token",
        description="GitHub OAuth access token (checksum-valid).",
        segments=(Literal("gho_"), Variable("body", ALNUM, 30), Checksum("crc", 6, "github-crc32-base62")),
        safety_note="'gho_' GitHub-OAuth-token shape with valid checksum. " + SHAPE_ONLY,
    ),
    FormatSpec(
        slug="github-user-to-server", name="GitHub User-to-Server Token (ghu_)", category="vcs-token",
        description="GitHub app user-to-server token (checksum-valid).",
        segments=(Literal("ghu_"), Variable("body", ALNUM, 30), Checksum("crc", 6, "github-crc32-base62")),
        safety_note="'ghu_' GitHub-token shape with valid checksum. " + SHAPE_ONLY,
    ),
    FormatSpec(
        slug="github-server-to-server", name="GitHub Server-to-Server Token (ghs_)", category="vcs-token",
        description="GitHub app server-to-server token (checksum-valid).",
        segments=(Literal("ghs_"), Variable("body", ALNUM, 30), Checksum("crc", 6, "github-crc32-base62")),
        safety_note="'ghs_' GitHub-token shape with valid checksum. " + SHAPE_ONLY,
    ),
    FormatSpec(
        slug="github-refresh", name="GitHub Refresh Token (ghr_)", category="vcs-token",
        description="GitHub refresh token (checksum-valid).",
        segments=(Literal("ghr_"), Variable("body", ALNUM, 30), Checksum("crc", 6, "github-crc32-base62")),
        safety_note="'ghr_' GitHub-token shape with valid checksum. " + SHAPE_ONLY,
    ),
    FormatSpec(
        slug="github-fine-grained", name="GitHub Fine-grained PAT (github_pat_)", category="vcs-token",
        description="GitHub fine-grained PAT (checksum-valid).",
        segments=(Literal("github_pat_"), Variable("a", ALNUM, 22), Literal("_"),
                  Variable("b", ALNUM, 59), Checksum("crc", 6, "github-crc32-base62")),
        safety_note="'github_pat_' fine-grained-PAT shape with valid checksum. " + SHAPE_ONLY,
    ),
```

> Note on `github-fine-grained`: confirm in Task A0's research whether the
> fine-grained checksum covers the whole body (`a` + `_` + `b`) or only `b`. The
> `Checksum` segment computes over ALL preceding text by default (so over
> `github_pat_<a>_<b>`); if the reference says otherwise, split the body so the
> checksummed portion precedes the `Checksum` segment exactly. Verify with a
> reference vector as in A0.

Set `scannable=False` on the three generic, prefix-less formats already in the
registry — edit their `FormatSpec(...)` calls to add `scannable=False`:
`aws-secret-access-key`, `database-password`, `oauth-bearer`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_formats.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite + ruff**

Run: `python -m pytest -q` and `python -m ruff check detect tests conftest.py`
Expected: pass / clean. (The existing `github-ghp` CLI/generation tests still pass
because `validate()` now includes the checksum and generation produces valid
tokens.)

- [ ] **Step 6: Commit**

```bash
git add detect/dp_honey/formats.py tests/test_dp_honey_formats.py
git commit -m "feat(dp-honey): checksum-valid GitHub family; mark generics non-scannable; REGISTRY_VERSION=2"
```

---

## Task A5: Reject unknown checksum algorithm id on artifact load

**Files:**
- Modify: `detect/dp_honey/model_io.py`
- Test: `tests/test_dp_honey_model_io.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dp_honey_model_io.py`:

```python
def test_unknown_checksum_algorithm_in_snapshot_raises():
    from detect.dp_honey import build_model, model_to_dict, load_model
    from detect.dp_honey.errors import ModelSchemaError

    data = model_to_dict(build_model("github-ghp", corpus_size=20, train_seed=1))
    # Tamper the checksum segment's algorithm id to an unregistered one.
    for seg in data["format"]["spec_snapshot"]["segments"]:
        if seg.get("kind") == "checksum":
            seg["algorithm"] = "totally-unknown"
    with pytest.raises(ModelSchemaError):
        load_model(data)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_model_io.py -k checksum -q`
Expected: FAIL — without the check it raises `FormatSpecMismatchError` (drift) or
nothing useful; we want a clear `ModelSchemaError` for an unknown algorithm.

- [ ] **Step 3: Implement in `model_io.py`**

In `load_model`, after `stored_snapshot = _field(fmt, "spec_snapshot")` and BEFORE
the drift comparison, validate any checksum-segment algorithm ids:

```python
    from .checksums import CHECKSUM_ALGORITHMS

    if isinstance(stored_snapshot, dict):
        for segment in stored_snapshot.get("segments", []):
            if isinstance(segment, dict) and segment.get("kind") == "checksum":
                algo = segment.get("algorithm")
                _require(algo in CHECKSUM_ALGORITHMS, f"unknown checksum algorithm: {algo!r}")
```

(`_require` already raises `ModelSchemaError`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_model_io.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/model_io.py tests/test_dp_honey_model_io.py
git commit -m "feat(dp-honey): reject unknown checksum algorithm id on artifact load"
```

---

## Task A6: README compatibility matrix + doc test

**Files:**
- Modify: `detect/dp_honey/README.md`
- Test: `tests/test_dp_honey_docs.py` (logic unchanged; must pass)

- [ ] **Step 1: Run the doc test to see it fail**

Run: `python -m pytest tests/test_dp_honey_docs.py -q`
Expected: FAIL — `test_every_format_appears_in_matrix` fails because the new
slugs are missing from the README matrix.

- [ ] **Step 2: Add matrix rows in `README.md`**

In the `## Format compatibility matrix` table, add one row per new slug, each
backtick-wrapped in the first cell and ending with the shape-only wording, e.g.:

```markdown
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
| `github-user-to-server` | GitHub User-to-Server | `ghu_` + 30 base62 + CRC | No — shape only (checksum-valid) |
| `github-server-to-server` | GitHub Server-to-Server | `ghs_` + 30 base62 + CRC | No — shape only (checksum-valid) |
| `github-refresh` | GitHub Refresh Token | `ghr_` + 30 base62 + CRC | No — shape only (checksum-valid) |
| `github-fine-grained` | GitHub Fine-grained PAT | `github_pat_` + base62 + CRC | No — shape only (checksum-valid) |
```

Also update the existing `github-ghp` row to note it is now checksum-valid, and
add a one-line sentence under the matrix that GitHub-family tokens carry a valid
base62-CRC32 checksum but remain non-functional (no provider record).

- [ ] **Step 3: Run the doc test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_docs.py -q`
Expected: PASS (matrix slugs == registry slugs; shape-only wording present).

- [ ] **Step 4: Full suite + ruff**

Run: `python -m pytest -q` and `python -m ruff check detect tests conftest.py`
Expected: pass / clean.

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/README.md
git commit -m "docs(dp-honey): matrix rows for new families and checksum-valid GitHub tokens"
```

---

# PHASE B — Scanner + auto-decoy

## Task B1: `scanner.py` — detection + `scan()`

**Files:**
- Create: `detect/dp_honey/scanner.py`
- Test: `tests/test_dp_honey_scanner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dp_honey_scanner.py`:

```python
"""Tests for the registry-driven scanner (SC1, SC3, SC4)."""

from __future__ import annotations

import numpy as np

from detect.dp_honey import get_format
from detect.dp_honey import scanner


def _example(slug, seed=0):
    return get_format(slug).random_example(np.random.default_rng(seed))


def test_scan_detects_prefixed_and_checksummed_families():
    ghp = _example("github-ghp", 1)
    slack = _example("slack-bot-token", 2)
    text = f"here is a token {ghp} and a slack one {slack} end"
    found = {f["format"]: f for f in scanner.scan(text)}
    assert "github-ghp" in found and "slack-bot-token" in found
    assert found["github-ghp"]["confidence"] == "high"   # checksum-confirmed
    assert found["slack-bot-token"]["confidence"] == "medium"


def test_findings_never_contain_the_secret_value():
    ghp = _example("github-ghp", 5)
    finding = scanner.scan(f"x {ghp} y")[0]
    assert set(finding) == {"format", "start", "end", "confidence"}  # SAFE-1: no value
    # the recorded span maps back to the token, but the value itself isn't stored
    assert f"x {ghp} y"[finding["start"]:finding["end"]] == ghp


def test_generic_prefixless_formats_do_not_false_positive():
    # a random 20-char mixed string must NOT be reported as database-password etc.
    text = "Th1sIsJustSomeR4ndomX configuration value"
    assert all(f["format"] not in {"database-password", "aws-secret-access-key", "oauth-bearer"}
               for f in scanner.scan(text))


def test_scan_of_plain_text_is_empty():
    assert scanner.scan("nothing secret here, just words") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_scanner.py -q`
Expected: FAIL (no `scanner` module).

- [ ] **Step 3: Implement `scanner.py`**

```python
"""Registry-driven secret scanner + auto-decoy.

Derives a detection regex from each scannable FormatSpec, finds candidate
substrings, and confirms them with the spec's own validate() (which includes
checksum verification). Per SAFE-1 the scanner NEVER stores, logs, or returns the
matched secret value — only its position, detected family, and confidence.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .formats import get_format, list_formats
from .grammar import Checksum, FormatSpec, Literal, Variable

_BOUNDARY_BEFORE = r"(?<![A-Za-z0-9_./+-])"
_BOUNDARY_AFTER = r"(?![A-Za-z0-9_./+-])"


def _segment_pattern(spec: FormatSpec) -> str:
    parts = []
    for seg in spec.segments:
        if isinstance(seg, Literal):
            parts.append(re.escape(seg.text))
        elif isinstance(seg, Variable):
            parts.append(f"[{re.escape(seg.alphabet)}]{{{seg.length}}}")
        elif isinstance(seg, Checksum):
            parts.append(f"[0-9A-Za-z]{{{seg.length}}}")
    return "".join(parts)


@lru_cache(maxsize=None)
def detection_pattern(slug: str) -> "re.Pattern[str]":
    spec = get_format(slug)
    return re.compile(_BOUNDARY_BEFORE + _segment_pattern(spec) + _BOUNDARY_AFTER)


def _scannable_specs() -> list[FormatSpec]:
    return [s for s in list_formats() if s.scannable]


def _has_checksum(spec: FormatSpec) -> bool:
    return any(isinstance(seg, Checksum) for seg in spec.segments)


def scan(text: str) -> list[dict]:
    """Return findings ``{format, start, end, confidence}`` for *text*.

    SAFE-1: the matched secret value is never included.
    """
    raw: list[dict] = []
    for spec in _scannable_specs():
        checksummed = _has_checksum(spec)
        for m in detection_pattern(spec.slug).finditer(text):
            candidate = m.group(0)
            if not spec.validate(candidate):
                continue
            raw.append({
                "format": spec.slug,
                "start": m.start(),
                "end": m.end(),
                "confidence": "high" if checksummed else "medium",
            })
    return _dedupe(raw)


_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}


def _dedupe(findings: list[dict]) -> list[dict]:
    """Drop findings whose span overlaps a kept one, preferring higher confidence
    then longer span. Returns findings sorted by start."""
    ordered = sorted(
        findings,
        key=lambda f: (-_CONFIDENCE_RANK[f["confidence"]], -(f["end"] - f["start"]), f["start"]),
    )
    kept: list[dict] = []
    for f in ordered:
        if any(f["start"] < k["end"] and k["start"] < f["end"] for k in kept):
            continue
        kept.append(f)
    return sorted(kept, key=lambda f: f["start"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_scanner.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/scanner.py tests/test_dp_honey_scanner.py
git commit -m "feat(dp-honey): registry-driven scanner with confidence + SAFE-1"
```

---

## Task B2: `auto_decoy()` + swap

**Files:**
- Modify: `detect/dp_honey/scanner.py`
- Test: `tests/test_dp_honey_scanner.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dp_honey_scanner.py`:

```python
def test_auto_decoy_generates_matching_valid_decoys_and_swaps():
    ghp = _example("github-ghp", 7)
    text = f"export TOKEN={ghp}"
    result = scanner.auto_decoy(text, seed=1)
    assert len(result["findings"]) == 1 == len(result["decoys"])
    decoy = result["decoys"][0]
    spec = get_format("github-ghp")
    assert spec.validate(decoy)          # decoy is a valid same-format token
    assert decoy != ghp                  # but not the original
    assert decoy in result["swapped_text"]
    assert ghp not in result["swapped_text"]  # original removed


def test_auto_decoy_is_deterministic():
    ghp = _example("github-ghp", 8)
    a = scanner.auto_decoy(f"a {ghp} b", seed=2)
    b = scanner.auto_decoy(f"a {ghp} b", seed=2)
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_scanner.py -k auto_decoy -q`
Expected: FAIL (`AttributeError: ... 'auto_decoy'`).

- [ ] **Step 3: Implement in `scanner.py`**

```python
from .bigram import generate_honeytokens


def auto_decoy(text: str, *, seed: int = 0) -> dict:
    """Scan *text* and produce one matching decoy per finding plus a swapped text.

    Deterministic in *seed*. Output contains only decoys and decoy-swapped text
    (never the original secret values beyond their positions in *text*).
    """
    findings = scan(text)
    decoys: list[str] = []
    for index, finding in enumerate(findings):
        # Distinct, deterministic per-finding seed so repeated families differ.
        decoy = generate_honeytokens(finding["format"], count=1, sample_seed=seed + index)[0]
        decoys.append(decoy)

    swapped = text
    # Replace right-to-left so earlier spans keep their indices.
    for finding, decoy in sorted(zip(findings, decoys), key=lambda fd: fd[0]["start"], reverse=True):
        swapped = swapped[: finding["start"]] + decoy + swapped[finding["end"] :]

    return {"findings": findings, "decoys": decoys, "swapped_text": swapped}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_scanner.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/scanner.py tests/test_dp_honey_scanner.py
git commit -m "feat(dp-honey): auto_decoy generates matching decoys and swapped text"
```

---

## Task B3: CLI `scan` and `auto-decoy`

**Files:**
- Modify: `detect/dp_honey/__main__.py`
- Test: `tests/test_dp_honey_cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dp_honey_cli.py`:

```python
def test_cli_scan_reports_findings_without_secret_values(tmp_path, capsys):
    spec = get_format("github-ghp")
    import numpy as np
    tok = spec.random_example(np.random.default_rng(1))
    f = tmp_path / "in.txt"
    f.write_text(f"TOKEN={tok}", encoding="utf-8")
    assert main(["scan", "--file", str(f)]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["findings"][0]["format"] == "github-ghp"
    assert tok not in out  # SAFE-1: secret value not echoed by default


def test_cli_auto_decoy_emits_swapped_text(tmp_path, capsys):
    spec = get_format("github-ghp")
    import numpy as np
    tok = spec.random_example(np.random.default_rng(2))
    f = tmp_path / "in.txt"
    f.write_text(f"TOKEN={tok}", encoding="utf-8")
    assert main(["auto-decoy", "--file", str(f), "--seed", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["decoys"]) == 1
    assert tok not in payload["swapped_text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_cli.py -k "scan or auto_decoy" -q`
Expected: FAIL (argparse: invalid choice 'scan').

- [ ] **Step 3: Implement in `__main__.py`**

Add `from . import scanner` to the imports. Register two subparsers in
`build_parser()` (alongside the existing ones):

```python
    p_scan = sub.add_parser("scan", help="detect secret-shaped substrings in text")
    p_scan.add_argument("--file", help="path to scan (default: stdin)")
    p_scan.add_argument("--show-matches", action="store_true",
                        help="also include matched values (OFF by default; handle with care)")
    p_scan.set_defaults(func=cmd_scan)

    p_auto = sub.add_parser("auto-decoy", help="scan text and emit a matching decoy per finding")
    p_auto.add_argument("--file", help="path to scan (default: stdin)")
    p_auto.add_argument("--seed", type=int, default=0)
    p_auto.set_defaults(func=cmd_auto_decoy)
```

Add the handlers and a small input reader:

```python
def _read_input(path: Optional[str]) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return sys.stdin.read()


def cmd_scan(args: argparse.Namespace) -> int:
    text = _read_input(args.file)
    findings = scanner.scan(text)
    output: dict = {"findings": findings}
    if args.show_matches:
        output["matches"] = [text[f["start"]:f["end"]] for f in findings]
    print(json.dumps(output, indent=2))
    return 0


def cmd_auto_decoy(args: argparse.Namespace) -> int:
    _emit_safety_banner()
    result = scanner.auto_decoy(_read_input(args.file), seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0
```

(`Path` is already imported in `__main__.py`? If not, add `from pathlib import Path`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp_honey_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite + ruff**

Run: `python -m pytest -q` and `python -m ruff check detect tests conftest.py`
Expected: pass / clean.

- [ ] **Step 6: Commit**

```bash
git add detect/dp_honey/__main__.py tests/test_dp_honey_cli.py
git commit -m "feat(dp-honey): scan and auto-decoy CLI commands (no secret echo by default)"
```

---

## Task B4: Web UI service + routes

**Files:**
- Modify: `detect/dp_honey/webui/service.py`, `detect/dp_honey/webui/app.py`
- Test: `tests/test_dp_honey_webui.py`, `tests/test_dp_honey_webui_http.py`

- [ ] **Step 1: Write the failing test (service)**

Append to `tests/test_dp_honey_webui.py`:

```python
def test_service_run_scan_and_auto_decoy():
    tok = get_format("github-ghp").random_example(__import__("numpy").random.default_rng(1))
    text = f"TOKEN={tok}"
    findings = service.run_scan(text)["findings"]
    assert findings and findings[0]["format"] == "github-ghp"
    result = service.run_auto_decoy(text, seed=1)
    assert tok not in result["swapped_text"]
    assert len(result["decoys"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp_honey_webui.py -k "scan or auto_decoy" -q`
Expected: FAIL (`AttributeError: ... 'run_scan'`).

- [ ] **Step 3: Implement in `webui/service.py`**

Add `from .. import scanner` (or `from ..scanner import scan, auto_decoy`) and:

```python
def run_scan(text: str) -> dict:
    return {"findings": scanner.scan(text)}


def run_auto_decoy(text: str, *, seed: int = 0) -> dict:
    return scanner.auto_decoy(text, seed=seed)
```

In `webui/app.py`, add two routes inside `create_app()`:

```python
    @app.post("/api/scan")
    def api_scan(body: dict) -> dict:
        return service.run_scan(body.get("text", ""))

    @app.post("/api/auto-decoy")
    def api_auto_decoy(body: dict) -> dict:
        return service.run_auto_decoy(body.get("text", ""), seed=int(body.get("seed", 0)))
```

- [ ] **Step 4: Write the failing HTTP test**

Append to `tests/test_dp_honey_webui_http.py`:

```python
def test_http_scan_and_auto_decoy():
    tok = __import__("detect.dp_honey", fromlist=["get_format"]).get_format("github-ghp").random_example(
        __import__("numpy").random.default_rng(3))
    c = _client()
    s = c.post("/api/scan", json={"text": f"k={tok}"})
    assert s.status_code == 200 and s.json()["findings"][0]["format"] == "github-ghp"
    a = c.post("/api/auto-decoy", json={"text": f"k={tok}", "seed": 1})
    assert a.status_code == 200 and tok not in a.json()["swapped_text"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dp_honey_webui.py tests/test_dp_honey_webui_http.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add detect/dp_honey/webui/service.py detect/dp_honey/webui/app.py tests/test_dp_honey_webui.py tests/test_dp_honey_webui_http.py
git commit -m "feat(webui): scan and auto-decoy service functions and routes"
```

---

## Task B5: Web UI "Scan & auto-decoy" panel

**Files:**
- Modify: `detect/dp_honey/webui/static/app.js`

Presentation only; verified manually in Task B6.

- [ ] **Step 1: Add the command to the nav and a form/handler**

In `app.js`, add an entry to the `COMMANDS` array:

```javascript
  { id: "scan", label: "Scan & auto-decoy" },
```

In `buildForm(cmd)`, add a branch that renders a textarea + a seed input:

```javascript
  } else if (cmd === "scan") {
    const ta = el("textarea", { name: "text", rows: "8", style: "width:100%; font-family:ui-monospace,monospace;" });
    const wrap = el("label", {}, "paste text to scan");
    wrap.append(ta);
    form.append(wrap);
    form.append(field("seed", numInput("seed", 1)));
  }
```

In `runCommand(cmd, form)`, add a branch:

```javascript
    } else if (cmd === "scan") {
      const result = await api("/api/auto-decoy", body);
      const lines = result.findings.map(
        (f) => `${f.format}  [${f.start}-${f.end}]  ${f.confidence}`
      );
      const out = el("div", {});
      out.append(el("div", {}, `${result.findings.length} finding(s) — synthetic decoys below`));
      out.append(el("pre", {}, lines.join("\n") || "no secrets detected"));
      out.append(el("div", {}, "swapped text (secrets replaced with decoys):"));
      out.append(el("pre", {}, result.swapped_text));
      setOutput(out);
    }
```

(`readForm` already coerces `seed` to a number; `text` passes through as a string.)

- [ ] **Step 2: Commit**

```bash
git add detect/dp_honey/webui/static/app.js
git commit -m "feat(webui): scan & auto-decoy panel in the frontend"
```

---

## Task B6: Integration verify + README scanner section

**Files:**
- Modify: `detect/dp_honey/README.md`

- [ ] **Step 1: Full suite + ruff**

Run: `python -m pytest -q` and `python -m ruff check detect tests conftest.py`
Expected: all pass; ruff clean.

- [ ] **Step 2: Manual end-to-end check (CLI)**

```bash
python -m detect.dp_honey generate --format github-ghp --count 1 --seed 1 > tmp_tok.txt
python -m detect.dp_honey scan --file tmp_tok.txt          # finding, NO token echoed
python -m detect.dp_honey auto-decoy --file tmp_tok.txt --seed 1   # decoy + swapped text
```
Confirm `scan` output does not contain the original token; `auto-decoy` swapped
text contains a different valid `ghp_…`. Remove `tmp_tok.txt`.

- [ ] **Step 3: Manual end-to-end check (UI)**

`python -m detect.dp_honey.webui`, open `http://127.0.0.1:8000`, pick "Scan &
auto-decoy", paste a generated `ghp_…` token, Run; confirm a finding (confidence
`high`) and a swapped text with a decoy. Ctrl+C to stop.

- [ ] **Step 4: Add a "Scanner & auto-decoy" section to `README.md`**

Document the `scan` / `auto-decoy` CLI commands, the SAFE-1 no-echo behavior, the
`--show-matches` opt-in, and that prefix-less/generic formats are excluded from
scanning to avoid false positives.

- [ ] **Step 5: Commit**

```bash
git add detect/dp_honey/README.md
git commit -m "docs(dp-honey): document the scanner and auto-decoy"
```

---

## Self-review notes

- **Spec coverage:** FX1→A3/A4; FX2→A1 (+A5 load guard); FX3→A0/A4; FX4→A3/A4 safety notes + no-crypto checksums; FX5→A4/A6; SC1→B1; SC2→B2; SC3→B1 (Finding has no value) + B3 (`--show-matches` off by default); SC4→A4 (`scannable=False`) + B1 (`_scannable_specs`); SC5→B3/B4/B5; SC6→B1/B2/B4 tests.
- **Checksum correctness** is gated on Task A0's reference verification — every later GitHub claim depends on it. If A0 can't confirm an exact algorithm, downgrade the GitHub family to structural (no checksum) and update the matrix wording rather than claim checksum-validity falsely.
- **Determinism:** `auto_decoy` uses `seed + index` per finding; reuses the library's seeded generation. No new global RNG.
- **SAFE-1** is asserted directly (`test_findings_never_contain_the_secret_value`, CLI/UI no-echo tests).
- **Back-compat:** upgrading `github-ghp` intentionally drifts old `ghp_` artifacts (fail-closed). The golden fixture is `aws-access-key-id` (unchanged) and still loads.
```
