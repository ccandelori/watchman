from __future__ import annotations

import hashlib
import json
import unittest
from types import SimpleNamespace
from typing import ClassVar

from aegis_introspection.cift_model_metadata import (
    CiftModelMetadataConfig,
    CiftModelMetadataError,
    cift_model_metadata_report_from_loaded_objects,
    cift_model_metadata_report_to_json,
)


class CiftModelMetadataTest(unittest.TestCase):
    def test_metadata_report_fingerprints_tokenizer_and_chat_template(self) -> None:
        config = CiftModelMetadataConfig(
            model_id="Qwen/Qwen3-test",
            revision="main",
            requested_device="cpu",
            dtype_name="device",
            selected_readout_candidates=("selected_choice_window_layer_1",),
            local_files_only=True,
            trust_remote_code=False,
        )
        model_config = SimpleNamespace(
            model_type="qwen3",
            hidden_size=4096,
            num_hidden_layers=36,
            output_hidden_states=False,
            _commit_hash="0123456789abcdef0123456789abcdef01234567",
        )
        tokenizer = _FakeTokenizer()

        report = cift_model_metadata_report_from_loaded_objects(
            config=config,
            model_config=model_config,
            tokenizer=tokenizer,
        )
        decoded = cift_model_metadata_report_to_json(report)

        self.assertEqual("aegis_introspection.cift_model_metadata/v1", decoded["schema_version"])
        self.assertEqual("calibration-ready", decoded["support_state"])
        self.assertEqual("Qwen/Qwen3-test", decoded["model_id"])
        self.assertEqual("main", decoded["revision"])
        self.assertEqual("0123456789abcdef0123456789abcdef01234567", decoded["resolved_revision"])
        self.assertEqual("qwen3", decoded["model_type"])
        self.assertEqual(4096, decoded["hidden_size"])
        self.assertEqual(36, decoded["layer_count"])
        self.assertEqual("cpu", decoded["requested_device"])
        self.assertEqual("cpu", decoded["selected_device"])
        self.assertEqual("device", decoded["dtype_name"])
        self.assertEqual("torch.float32", decoded["resolved_torch_dtype"])
        self.assertEqual("configurable_output_hidden_states", decoded["hidden_state_support"])
        self.assertTrue(decoded["hidden_state_capable"])
        self.assertEqual(["selected_choice_window_layer_1"], decoded["selected_readout_candidates"])
        self.assertIsNone(decoded["failure_reason"])
        self.assertEqual("_FakeTokenizer", decoded["tokenizer_class"])
        self.assertEqual(32000, decoded["tokenizer_vocab_size"])
        self.assertEqual(hashlib.sha256(b'{"model":"tokenizer"}').hexdigest(), decoded["tokenizer_fingerprint_sha256"])
        self.assertEqual(hashlib.sha256(b"{{ messages }}").hexdigest(), decoded["chat_template_sha256"])
        self.assertTrue(decoded["chat_template_present"])

    def test_metadata_report_rejects_missing_hidden_size(self) -> None:
        config = CiftModelMetadataConfig(
            model_id="Qwen/Qwen3-test",
            revision="main",
            requested_device="cpu",
            dtype_name="device",
            selected_readout_candidates=("selected_choice_window_layer_1",),
            local_files_only=True,
            trust_remote_code=False,
        )
        model_config = SimpleNamespace(model_type="qwen3", num_hidden_layers=36, output_hidden_states=False)

        with self.assertRaisesRegex(CiftModelMetadataError, "hidden_size"):
            cift_model_metadata_report_from_loaded_objects(
                config=config,
                model_config=model_config,
                tokenizer=_FakeTokenizer(),
            )

    def test_metadata_report_fingerprints_vocab_when_backend_is_absent(self) -> None:
        config = CiftModelMetadataConfig(
            model_id="Qwen/Qwen3-test",
            revision="main",
            requested_device="cpu",
            dtype_name="device",
            selected_readout_candidates=("selected_choice_window_layer_1",),
            local_files_only=True,
            trust_remote_code=False,
        )
        model_config = SimpleNamespace(
            model_type="qwen3",
            hidden_size=1024,
            num_hidden_layers=24,
            output_hidden_states=False,
        )
        tokenizer = _FakeSlowTokenizer()

        report = cift_model_metadata_report_from_loaded_objects(
            config=config,
            model_config=model_config,
            tokenizer=tokenizer,
        )

        expected_vocab_json = json.dumps({"a": 1, "b": 2}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        self.assertEqual(
            hashlib.sha256(expected_vocab_json.encode("utf-8")).hexdigest(),
            report.tokenizer_fingerprint_sha256,
        )
        self.assertFalse(report.chat_template_present)


class _FakeBackendTokenizer:
    def to_str(self) -> str:
        return '{"model":"tokenizer"}'


class _FakeTokenizer:
    vocab_size = 32000
    backend_tokenizer = _FakeBackendTokenizer()
    special_tokens_map: ClassVar[dict[str, str]] = {"eos_token": "<|endoftext|>"}
    chat_template = "{{ messages }}"


class _FakeSlowTokenizer:
    vocab_size = 2
    special_tokens_map: ClassVar[dict[str, str]] = {}
    chat_template = None

    def get_vocab(self) -> dict[str, int]:
        return {"b": 2, "a": 1}


if __name__ == "__main__":
    unittest.main()
