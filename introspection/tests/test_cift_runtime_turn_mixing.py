from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aegis_introspection.cift_runtime_turn_mixing import (
    CiftRuntimeTurnMixConfig,
    CiftRuntimeTurnMixingError,
    build_mixed_cift_window_runtime_turns,
    load_runtime_turn_jsonl,
    write_runtime_turn_jsonl,
)


class CiftRuntimeTurnMixingTest(unittest.TestCase):
    def test_mixer_preserves_selected_choice_rows_and_strips_fallback_rows_by_label(self) -> None:
        turns = (
            _runtime_turn("safe-0", "secret_present_safe"),
            _runtime_turn("safe-1", "secret_present_safe"),
            _runtime_turn("exfil-0", "exfiltration_intent"),
            _runtime_turn("exfil-1", "exfiltration_intent"),
        )

        result = build_mixed_cift_window_runtime_turns(
            turns=turns,
            config=CiftRuntimeTurnMixConfig(fallback_modulus=2, fallback_remainder=1),
        )

        self.assertEqual({"selected_choice": 2, "payload_query_fallback": 2}, result.window_family_counts)
        first_safe_cift = result.turns[0]["metadata"]["cift"]
        second_safe_cift = result.turns[1]["metadata"]["cift"]
        first_exfil_cift = result.turns[2]["metadata"]["cift"]
        second_exfil_cift = result.turns[3]["metadata"]["cift"]

        self.assertIsInstance(first_safe_cift, dict)
        self.assertIsInstance(second_safe_cift, dict)
        self.assertIsInstance(first_exfil_cift, dict)
        self.assertIsInstance(second_exfil_cift, dict)
        assert isinstance(first_safe_cift, dict)
        assert isinstance(second_safe_cift, dict)
        assert isinstance(first_exfil_cift, dict)
        assert isinstance(second_exfil_cift, dict)

        self.assertEqual("selected_choice", result.turns[0]["metadata"]["eval"]["expected_cift_window_family"])
        self.assertEqual(
            "payload_query_fallback",
            result.turns[1]["metadata"]["eval"]["expected_cift_window_family"],
        )
        self.assertEqual("selected_choice", result.turns[2]["metadata"]["eval"]["expected_cift_window_family"])
        self.assertEqual(
            "payload_query_fallback",
            result.turns[3]["metadata"]["eval"]["expected_cift_window_family"],
        )
        self.assertIn("selected_choice_readout_token_indices", first_safe_cift)
        self.assertNotIn("selected_choice_readout_token_indices", second_safe_cift)
        self.assertIn("selected_choice_readout_token_indices", first_exfil_cift)
        self.assertNotIn("selected_choice_readout_token_indices", second_exfil_cift)

        self.assertNotIn("expected_cift_window_family", turns[0]["metadata"]["eval"])

    def test_mixer_rejects_invalid_config(self) -> None:
        with self.assertRaisesRegex(CiftRuntimeTurnMixingError, "fallback_modulus"):
            CiftRuntimeTurnMixConfig(fallback_modulus=1, fallback_remainder=0)
        with self.assertRaisesRegex(CiftRuntimeTurnMixingError, "fallback_remainder"):
            CiftRuntimeTurnMixConfig(fallback_modulus=2, fallback_remainder=2)

    def test_mixer_rejects_primary_row_without_selected_choice_geometry(self) -> None:
        turn = _runtime_turn("safe-0", "secret_present_safe")
        cift_metadata = turn["metadata"]["cift"]
        assert isinstance(cift_metadata, dict)
        cift_metadata.pop("selected_choice_readout_token_indices")

        with self.assertRaisesRegex(CiftRuntimeTurnMixingError, "selected_choice_readout_token_indices"):
            build_mixed_cift_window_runtime_turns(
                turns=(turn,),
                config=CiftRuntimeTurnMixConfig(fallback_modulus=2, fallback_remainder=1),
            )

    def test_runtime_turn_jsonl_round_trip(self) -> None:
        result = build_mixed_cift_window_runtime_turns(
            turns=(_runtime_turn("safe-0", "secret_present_safe"), _runtime_turn("safe-1", "secret_present_safe")),
            config=CiftRuntimeTurnMixConfig(fallback_modulus=2, fallback_remainder=1),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed_runtime_turns.jsonl"
            write_runtime_turn_jsonl(path=path, turns=result.turns)
            loaded = load_runtime_turn_jsonl(path=path)

        self.assertEqual(result.turns, loaded)
        encoded = "\n".join(json.dumps(turn) for turn in loaded)
        self.assertIn("expected_cift_window_family", encoded)


def _runtime_turn(example_id: str, label: str) -> dict[str, object]:
    return {
        "trace_id": f"trace-{example_id}",
        "session_id": "session-test",
        "turn_index": 1,
        "capability_mode": "offline_eval",
        "model": {
            "provider": "huggingface",
            "model_id": "Qwen/Qwen3-test",
            "revision": "main",
            "selected_device": "cpu",
        },
        "messages": [{"role": "user", "content": f"message for {example_id}"}],
        "tool_calls": [],
        "sensitive_spans": [],
        "metadata": {
            "example_id": example_id,
            "eval": {
                "label": label,
                "family": "family-a",
                "tags": ["test"],
            },
            "cift": {
                "secret_token_span": [1, 2],
                "query_token_span": [2, 4],
                "readout_token_indices": [6, 7],
                "selected_choice_char_span": [10, 20],
                "selected_choice_token_span": [4, 8],
                "selected_choice_readout_token_indices": [5, 6, 7],
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
