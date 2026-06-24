# NIMBUS Redteam Evaluation

This runbook explains how Aegis consumes external redteam results to inspect
NIMBUS behavior. The redteam runner remains a separate black-box HTTP client.
Aegis owns the proxy, detector metadata, policy metadata, and NIMBUS report
interpretation.

## Contract

The external runner writes JSONL records with one redteam result per line. Aegis
does not import that package. It only reads the fields needed for NIMBUS
evaluation:

- `scenario_name`
- `turn_results[].turn_index`
- `turn_results[].policy_decision.final_action`
- `turn_results[].detector_results[]`
- the `nimbus` detector result's `evidence`

The NIMBUS evidence must include:

- `turn_estimated_leakage_bits`
- `cumulative_estimated_leakage_bits`
- `budget_fraction`
- `warn_threshold`, `sanitize_threshold`, and `block_threshold` when the
  detector action is not present
- optional `critic_evidence.critic_kind`
- optional `critic_evidence.partial_match_count`

Reports are metric-first. They do not render prompts, assistant output, raw
honeytokens, or credential-shaped values.

Minimal hand-authored fixture shape:

```json
{
  "scenario_name": "multi_turn_drip",
  "turn_results": [
    {
      "turn_index": 0,
      "detector_results": [
        {
          "detector_name": "nimbus",
          "component": "nimbus",
          "recommended_action": "warn",
          "evidence": {
            "turn_estimated_leakage_bits": 0.38,
            "cumulative_estimated_leakage_bits": 0.38,
            "budget_fraction": 0.38,
            "critic_evidence": {
              "critic_kind": "canary",
              "partial_match_count": 1
            }
          }
        }
      ],
      "policy_decision": {
        "final_action": "warn",
        "triggered_detectors": ["nimbus"]
      }
    }
  ]
}
```

## Local Run

Start the Aegis development proxy:

```bash
uv run --extra dev aegis-proxy --host 127.0.0.1 --port 8765
```

From the separate redteam repository, run its scenarios against the HTTP target
and write JSONL results:

```bash
uv run --extra dev aegis-redteam run scenarios \
  --target http://127.0.0.1:8765 \
  --output results/aegis-local.jsonl
```

Then return to this repository and summarize the result file:

```bash
uv run --extra dev aegis-nimbus-report \
  --input ../watchman-redteam/results/aegis-local.jsonl
```

Use JSON output when another tool needs the summary:

```bash
uv run --extra dev aegis-nimbus-report \
  --input ../watchman-redteam/results/aegis-local.jsonl \
  --format json
```

## Internal Fixture Loop

When the external redteam runner is not available, Aegis can generate a small
in-process fixture JSONL through the same mock proxy and report parser:

```bash
uv run --extra dev aegis-nimbus-fixtures \
  --output /tmp/aegis-nimbus-fixtures.jsonl

uv run --extra dev aegis-nimbus-report \
  --input /tmp/aegis-nimbus-fixtures.jsonl
```

This fixture loop is for fast regression checks of current runtime behavior. It
does not replace black-box redteam validation because it does not exercise HTTP
transport, external scenario loading, or redteam expectation scoring.

## Reading the Report

The report shows one row per scenario:

- policy action progression, such as `warn -> sanitize -> block`
- NIMBUS action progression
- max cumulative leakage bits
- final budget fraction
- whether public canary detectors triggered

Public canary detectors and NIMBUS answer different questions. `text_canary`
and `encoded_canary` are immediate post-output detectors with their own match
thresholds. NIMBUS uses the session critic's leakage estimate and can accumulate
partial leakage across turns even when a public canary detector does not fire on
an individual turn.

For the current deterministic development critic, a healthy multi-turn partial
leak smoke should show cumulative NIMBUS risk strengthening over the session.
The exact actions depend on the runtime thresholds, but a representative
`multi_turn_drip` scenario should move toward a final block once the leakage
budget is exhausted.

## Safety Rules

- Do not commit generated redteam result files.
- If parser fixtures are needed, hand-author minimal synthetic records with only
  the parser contract fields. Do not include `request`, `assistant_content`,
  `raw_response`, or `failures` payloads.
- Do not store production credentials or real secret values in report inputs.
- Treat the current report as evaluation of `CanaryNimbusCritic`, not the
  future paper-faithful learned NIMBUS critic.
- Use `POST /test/reset` between stateful redteam runs when scenarios share a
  session id.
