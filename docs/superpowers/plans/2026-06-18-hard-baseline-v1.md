# Hard Baseline V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and evaluate `prompts_hard.jsonl` as a harder, lineage-tracked successor to the baseline prompt dataset.

**Architecture:** Add one new dataset file with the same schema as the baseline, then run the existing extraction and binary-evaluation scripts with explicit paths. Add a focused dataset-shape test, register new machine-readable outputs in `lineage.json`, update the living README, and write a narrative checkpoint report.

**Tech Stack:** Python 3.12, PyTorch/Transformers through the existing extraction script, scikit-learn through the existing binary evaluators, JSONL, unittest, SHA256 lineage validation.

---

## File Structure

- Create `introspection/data/prompts_hard.jsonl`
  - Hard Baseline V1 dataset, 90 examples, same schema as `prompts.jsonl`.
- Create `introspection/tests/test_hard_dataset.py`
  - Validates the hard dataset exists, parses, and has exact label/family counts.
- Create `introspection/data/activations/qwen3_0_6b_hard_all_layers.pt`
  - Generated activation artifact for hard-v1.
- Create `introspection/data/reports/binary_tasks_hard.json`
  - Generated random stratified binary evaluation report.
- Create `introspection/data/reports/binary_tasks_hard_summary.md`
  - Generated random stratified binary evaluation summary.
- Create `introspection/data/reports/binary_tasks_hard_grouped.json`
  - Generated grouped binary evaluation report.
- Create `introspection/data/reports/binary_tasks_hard_grouped_summary.md`
  - Generated grouped binary evaluation summary.
- Create `introspection/data/reports/hard_baseline_probe_progress_2026-06-18.md`
  - Narrative checkpoint comparing baseline grouped results to hard-v1 grouped results.
- Modify `introspection/data/lineage.json`
  - Register `hard_prompts_v1`, `qwen3_0_6b_hard_all_layers_v1`, `hard_binary_random_v1`, and `hard_binary_grouped_v1`.
- Modify `introspection/README.md`
  - Add hard-v1 current state, outputs, and next-step update.

## Task 1: Dataset Shape Test

**Files:**
- Create: `introspection/tests/test_hard_dataset.py`

- [ ] **Step 1: Write the failing test**

Create `introspection/tests/test_hard_dataset.py` with:

```python
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from aegis_introspection.prompts import load_prompt_examples


HARD_PROMPTS_PATH = Path(__file__).resolve().parents[1] / "data" / "prompts_hard.jsonl"
EXPECTED_LABELS = ("benign", "secret_present_safe", "exfiltration_intent")


class HardDatasetTest(unittest.TestCase):
    def test_hard_dataset_has_expected_shape(self) -> None:
        examples = load_prompt_examples(HARD_PROMPTS_PATH)

        label_counts = Counter(example.label for example in examples)
        family_counts_by_label: dict[str, Counter[str]] = defaultdict(Counter)
        for example in examples:
            family_counts_by_label[example.label].update((example.family,))

        self.assertEqual(90, len(examples))
        self.assertEqual({label: 30 for label in EXPECTED_LABELS}, dict(label_counts))
        for label in EXPECTED_LABELS:
            family_counts = family_counts_by_label[label]
            self.assertEqual(10, len(family_counts))
            self.assertEqual([3] * 10, sorted(family_counts.values()))

    def test_hard_dataset_uses_hard_ids_and_tags(self) -> None:
        examples = load_prompt_examples(HARD_PROMPTS_PATH)

        for example in examples:
            self.assertTrue(example.id.startswith("hard_"))
            self.assertIn("hard", example.tags)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python -m unittest introspection.tests.test_hard_dataset
```

Expected: failure because `introspection/data/prompts_hard.jsonl` does not exist.

## Task 2: Hard Dataset

**Files:**
- Create: `introspection/data/prompts_hard.jsonl`
- Test: `introspection/tests/test_hard_dataset.py`

- [ ] **Step 1: Create hard-v1 JSONL dataset**

Create `introspection/data/prompts_hard.jsonl` with 90 JSONL records:

- IDs `hard_benign_001` through `hard_benign_030`.
- IDs `hard_safe_001` through `hard_safe_030`.
- IDs `hard_exfil_001` through `hard_exfil_030`.
- Labels exactly `benign`, `secret_present_safe`, and `exfiltration_intent`.
- Families exactly the 30 families listed in the approved design.
- Three examples per family.
- Every row includes the tag `hard`.

- [ ] **Step 2: Run hard dataset test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python -m unittest introspection.tests.test_hard_dataset
```

Expected: `Ran 2 tests` and `OK`.

## Task 3: Generate Hard Activation Artifact

**Files:**
- Create: `introspection/data/activations/qwen3_0_6b_hard_all_layers.pt`

- [ ] **Step 1: Extract all-layer activations**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/extract_activations.py \
  --prompts introspection/data/prompts_hard.jsonl \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --pooling final_token,mean_pool \
  --output introspection/data/activations/qwen3_0_6b_hard_all_layers.pt
```

Expected: extraction prints 90 prompt rows and writes the hard all-layer artifact.

- [ ] **Step 2: Inspect artifact schema**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python -c 'from pathlib import Path; from aegis_introspection.artifacts import load_activation_artifact; artifact=load_activation_artifact(Path("introspection/data/activations/qwen3_0_6b_hard_all_layers.pt")); print(len(artifact["example_ids"])); print(len(artifact["families"])); print(len(artifact["features"]))'
```

Expected:

```text
90
90
58
```

## Task 4: Generate Hard Binary Reports

**Files:**
- Create: `introspection/data/reports/binary_tasks_hard.json`
- Create: `introspection/data/reports/binary_tasks_hard_summary.md`
- Create: `introspection/data/reports/binary_tasks_hard_grouped.json`
- Create: `introspection/data/reports/binary_tasks_hard_grouped_summary.md`

- [ ] **Step 1: Run random binary tasks**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_binary_tasks.py \
  --artifact introspection/data/activations/qwen3_0_6b_hard_all_layers.pt \
  --output-json introspection/data/reports/binary_tasks_hard.json \
  --output-md introspection/data/reports/binary_tasks_hard_summary.md
```

Expected: prints best method and metrics for both binary tasks.

- [ ] **Step 2: Run grouped binary tasks**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/train_grouped_binary_tasks.py \
  --artifact introspection/data/activations/qwen3_0_6b_hard_all_layers.pt \
  --output-json introspection/data/reports/binary_tasks_hard_grouped.json \
  --output-md introspection/data/reports/binary_tasks_hard_grouped_summary.md
```

Expected: prints best method and metrics for both grouped binary tasks.

## Task 5: Register Lineage

**Files:**
- Modify: `introspection/data/lineage.json`

- [ ] **Step 1: Compute hashes for hard-v1 records**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python -c 'from pathlib import Path; import hashlib; paths=[Path("introspection/data/prompts_hard.jsonl"),Path("introspection/data/activations/qwen3_0_6b_hard_all_layers.pt"),Path("introspection/data/reports/binary_tasks_hard.json"),Path("introspection/data/reports/binary_tasks_hard_grouped.json")]; [print(hashlib.sha256(path.read_bytes()).hexdigest(), path) for path in paths]'
```

Expected: four SHA256 hashes.

- [ ] **Step 2: Add hard-v1 records to lineage**

Modify `introspection/data/lineage.json`:

- Add dataset record `hard_prompts_v1`.
- Add artifact record `qwen3_0_6b_hard_all_layers_v1`.
- Add report record `hard_binary_random_v1`.
- Add report record `hard_binary_grouped_v1`.

- [ ] **Step 3: Validate lineage**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/validate_lineage.py
```

Expected: lineage validates with 2 datasets, 3 artifacts, and 7 reports.

## Task 6: Documentation

**Files:**
- Modify: `introspection/README.md`
- Create: `introspection/data/reports/hard_baseline_probe_progress_2026-06-18.md`

- [ ] **Step 1: Write checkpoint report**

Create `introspection/data/reports/hard_baseline_probe_progress_2026-06-18.md` with:

- dataset purpose and shape
- random hard-v1 metrics
- grouped hard-v1 metrics
- comparison against baseline grouped metrics
- caveats
- next steps toward Hard Baseline V2

- [ ] **Step 2: Update README**

Modify `introspection/README.md`:

- Add `hard_prompts_v1` under current state.
- Add hard-v1 random and grouped metrics.
- Add hard-v1 outputs under reports.
- Keep the next move focused on inspecting hard-v1 failures before V2.

## Task 7: Final Verification

**Files:**
- Verify all changed and generated files.

- [ ] **Step 1: Run lineage validation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python introspection/scripts/validate_lineage.py
```

Expected: lineage validates.

- [ ] **Step 2: Run full test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/sheep/Desktop/Gauntlet/Capstone/introspection/src \
  /Users/sheep/Desktop/Gauntlet/Capstone/.venv-introspection/bin/python -m unittest discover -s introspection/tests
```

Expected: all tests pass.

- [ ] **Step 3: Check cache noise**

Run:

```bash
find introspection -name __pycache__ -type d
```

Expected: no output.

If Python creates cache directories, remove only generated cache directories after confirming the paths.

## Self-Review

- Spec coverage: The plan creates the hard dataset, evaluates it, registers lineage, updates README, and writes the narrative checkpoint.
- Placeholder scan: The plan contains no deferred implementation markers. Dataset examples are authored in Task 2 and verified by exact shape tests.
- Type consistency: The plan uses the existing prompt schema, artifact schema, lineage record names, and report script parameters consistently.
