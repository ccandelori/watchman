from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aegis_introspection.sealed_holdout_policy import (
    SealedHoldoutPolicyError,
    assert_unsealed_jsonl_tags,
    assert_unsealed_path,
    assert_unsealed_tag_rows,
    path_is_sealed_holdout,
    tag_rows_are_sealed_holdout,
)


class SealedHoldoutPolicyTest(unittest.TestCase):
    def test_path_is_sealed_holdout_detects_sealed_filename(self) -> None:
        self.assertTrue(path_is_sealed_holdout(Path("prompts_dp_honey_lite_v4_3_sealed.jsonl")))
        self.assertFalse(path_is_sealed_holdout(Path("prompts_dp_honey_lite_v4_1.jsonl")))
        self.assertFalse(path_is_sealed_holdout(Path("unsealed_report.jsonl")))

    def test_tag_rows_are_sealed_holdout_detects_row_tag(self) -> None:
        self.assertTrue(tag_rows_are_sealed_holdout((("dp_honey_lite", "sealed_holdout"),)))
        self.assertFalse(tag_rows_are_sealed_holdout((("dp_honey_lite", "hard_v4_1"),)))

    def test_assert_unsealed_path_rejects_without_override(self) -> None:
        with self.assertRaisesRegex(SealedHoldoutPolicyError, "path 'sealed_prompts.jsonl' is marked sealed"):
            assert_unsealed_path(
                path=Path("sealed_prompts.jsonl"),
                allow_sealed_holdout=False,
                context="unit test",
            )

    def test_assert_unsealed_tag_rows_rejects_without_override(self) -> None:
        with self.assertRaisesRegex(SealedHoldoutPolicyError, "row tags include 'sealed_holdout'"):
            assert_unsealed_tag_rows(
                tag_rows=(("dp_honey_lite", "sealed_holdout"),),
                allow_sealed_holdout=False,
                context="unit test",
            )

    def test_assert_unsealed_jsonl_tags_rejects_runtime_turn_eval_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime_turns.jsonl"
            path.write_text(
                json.dumps({"metadata": {"example_id": "row-1", "eval": {"tags": ["sealed_holdout"]}}}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SealedHoldoutPolicyError, "row tags include 'sealed_holdout'"):
                assert_unsealed_jsonl_tags(
                    path=path,
                    allow_sealed_holdout=False,
                    context="unit test",
                )

    def test_assert_unsealed_jsonl_tags_wraps_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime_turns.jsonl"
            path.write_text("{not-json}\n", encoding="utf-8")

            with self.assertRaisesRegex(SealedHoldoutPolicyError, "invalid JSON"):
                assert_unsealed_jsonl_tags(
                    path=path,
                    allow_sealed_holdout=False,
                    context="unit test",
                )


if __name__ == "__main__":
    unittest.main()
