from __future__ import annotations

# ruff: noqa: E402
import unittest

import pytest

torch = pytest.importorskip("torch")
from aegis_introspection.activations import HiddenStateForwardPass
from aegis_introspection.cift_live_extractor import (
    CiftLiveExtractorError,
    LiveCiftFeatureExtractor,
    LiveCiftFeatureSetExtractor,
    parse_live_cift_feature_key,
)

from aegis.core.contracts import CapabilityMode, Message, ModelInfo, NormalizedTurn


class CiftLiveExtractorTest(unittest.TestCase):
    def test_parse_live_concat_feature_key(self) -> None:
        specs = parse_live_cift_feature_key("concat(readout_window_layer_01,readout_window_layer_02)")

        self.assertEqual(("readout_window_layer_01", "readout_window_layer_02"), tuple(spec.key for spec in specs))
        self.assertEqual((1, 2), tuple(spec.layer_index for spec in specs))

    def test_live_extractor_concatenates_readout_window_features(self) -> None:
        runner = RecordingRunner(
            forward_pass=_forward_pass(
                prompt="rendered prompt",
                hidden_states=(
                    _hidden_state(((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))),
                    _hidden_state(((1.0, 10.0), (10.0, 20.0), (30.0, 40.0))),
                    _hidden_state(((2.0, 20.0), (100.0, 200.0), (300.0, 400.0))),
                ),
            )
        )
        extractor = LiveCiftFeatureExtractor(
            runner=runner,
            feature_key="concat(readout_window_layer_01,readout_window_layer_02)",
        )

        vector = extractor.extract_feature_vector(
            turn=_turn(messages=(Message(role="user", content="rendered prompt"),), metadata=_cift_metadata((1, 2))),
            feature_key="concat(readout_window_layer_01,readout_window_layer_02)",
        )

        self.assertEqual(("rendered prompt",), tuple(runner.prompts))
        self.assertEqual((20.0, 30.0, 200.0, 300.0), vector)

    def test_live_extractor_returns_none_when_readout_indices_are_missing(self) -> None:
        extractor = LiveCiftFeatureExtractor(
            runner=RecordingRunner(forward_pass=_forward_pass(prompt="rendered prompt", hidden_states=())),
            feature_key="readout_window_layer_01",
        )

        vector = extractor.extract_feature_vector(
            turn=_turn(messages=(Message(role="user", content="rendered prompt"),), metadata={}),
            feature_key="readout_window_layer_01",
        )

        self.assertIsNone(vector)

    def test_live_extractor_concatenates_query_tail_window_features(self) -> None:
        runner = RecordingRunner(
            forward_pass=_forward_pass(
                prompt="rendered prompt",
                hidden_states=(
                    _hidden_state(((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))),
                    _hidden_state(((1.0, 10.0), (10.0, 20.0), (30.0, 40.0))),
                ),
            )
        )
        extractor = LiveCiftFeatureExtractor(
            runner=runner,
            feature_key="query_tail_window_layer_01",
        )

        vector = extractor.extract_feature_vector(
            turn=_turn(
                messages=(Message(role="user", content="rendered prompt"),),
                metadata={"cift": {"query_tail_readout_token_indices": [0, 2]}},
            ),
            feature_key="query_tail_window_layer_01",
        )

        self.assertEqual((15.5, 25.0), vector)
        self.assertEqual(("rendered prompt",), tuple(runner.prompts))

    def test_live_extractor_rejects_multi_message_prompt_shape(self) -> None:
        extractor = LiveCiftFeatureExtractor(
            runner=RecordingRunner(forward_pass=_forward_pass(prompt="rendered prompt", hidden_states=())),
            feature_key="readout_window_layer_01",
        )

        with self.assertRaisesRegex(CiftLiveExtractorError, "exactly one"):
            extractor.extract_feature_vector(
                turn=_turn(
                    messages=(
                        Message(role="system", content="system"),
                        Message(role="user", content="user"),
                    ),
                    metadata=_cift_metadata((1,)),
                ),
                feature_key="readout_window_layer_01",
            )

    def test_feature_set_extractor_reuses_one_forward_pass_for_multiple_feature_keys(self) -> None:
        runner = RecordingRunner(
            forward_pass=_forward_pass(
                prompt="rendered prompt",
                hidden_states=(
                    _hidden_state(((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))),
                    _hidden_state(((1.0, 10.0), (10.0, 20.0), (30.0, 40.0))),
                ),
            )
        )
        extractor = LiveCiftFeatureSetExtractor(
            runner=runner,
            feature_keys=("selected_choice_window_layer_01", "readout_window_layer_01"),
        )
        turn = _turn(
            messages=(Message(role="user", content="rendered prompt"),),
            metadata={"cift": {"readout_token_indices": [1, 2], "selected_choice_readout_token_indices": [2]}},
        )

        selected_vector = extractor.extract_feature_vector(turn=turn, feature_key="selected_choice_window_layer_01")
        fallback_vector = extractor.extract_feature_vector(turn=turn, feature_key="readout_window_layer_01")

        self.assertEqual((30.0, 40.0), selected_vector)
        self.assertEqual((20.0, 30.0), fallback_vector)
        self.assertEqual(("rendered prompt",), tuple(runner.prompts))

    def test_feature_set_extractor_skips_forward_pass_when_required_window_is_missing(self) -> None:
        runner = RecordingRunner(forward_pass=_forward_pass(prompt="rendered prompt", hidden_states=()))
        extractor = LiveCiftFeatureSetExtractor(
            runner=runner,
            feature_keys=("selected_choice_window_layer_01", "readout_window_layer_01"),
        )

        vector = extractor.extract_feature_vector(
            turn=_turn(messages=(Message(role="user", content="rendered prompt"),), metadata=_cift_metadata((1,))),
            feature_key="selected_choice_window_layer_01",
        )

        self.assertIsNone(vector)
        self.assertEqual((), tuple(runner.prompts))


class RecordingRunner:
    def __init__(self, forward_pass: HiddenStateForwardPass) -> None:
        self.prompts: list[str] = []
        self._forward_pass = forward_pass

    def run(self, prompt: str) -> HiddenStateForwardPass:
        self.prompts.append(prompt)
        return self._forward_pass


def _turn(messages: tuple[Message, ...], metadata: dict[str, object]) -> NormalizedTurn:
    return NormalizedTurn(
        trace_id="trace-live-cift",
        session_id="session-live-cift",
        turn_index=1,
        capability_mode=CapabilityMode.OFFLINE_EVAL,
        model=ModelInfo(provider="huggingface", model_id="Qwen/Qwen3-test", revision="main", selected_device="cpu"),
        messages=messages,
        tool_calls=(),
        sensitive_spans=(),
        metadata=metadata,
    )


def _cift_metadata(readout_token_indices: tuple[int, ...]) -> dict[str, object]:
    return {"cift": {"readout_token_indices": list(readout_token_indices)}}


def _forward_pass(prompt: str, hidden_states: tuple[torch.Tensor, ...]) -> HiddenStateForwardPass:
    return HiddenStateForwardPass(
        prompt=prompt,
        input_ids=torch.tensor(((1, 2, 3),), dtype=torch.int64),
        attention_mask=torch.tensor(((1, 1, 1),), dtype=torch.int64),
        hidden_states=hidden_states,
    )


def _hidden_state(values: tuple[tuple[float, float], ...]) -> torch.Tensor:
    return torch.tensor((values,), dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
