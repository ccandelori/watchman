import unittest
from unittest.mock import Mock, patch

import torch
from aegis_introspection.model_loader import (
    DeviceUnavailableError,
    ModelLoadConfig,
    UnsupportedDeviceError,
    UnsupportedModelDTypeError,
    load_causal_lm,
    parse_model_dtype,
    resolve_model_load_dtype,
    select_device,
)


class ModelLoaderTest(unittest.TestCase):
    def test_select_device_returns_cpu_selection(self) -> None:
        selection = select_device("cpu")

        self.assertEqual("cpu", selection.name)
        self.assertEqual(torch.device("cpu"), selection.torch_device)
        self.assertEqual(torch.float32, selection.torch_dtype)

    def test_select_device_auto_prefers_cuda_over_mps(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("aegis_introspection.model_loader._mps_is_available", return_value=True),
        ):
            selection = select_device("auto")

        self.assertEqual("cuda", selection.name)
        self.assertEqual(torch.device("cuda"), selection.torch_device)
        self.assertEqual(torch.float16, selection.torch_dtype)

    def test_select_device_auto_uses_mps_when_cuda_is_unavailable(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("aegis_introspection.model_loader._mps_is_available", return_value=True),
        ):
            selection = select_device("auto")

        self.assertEqual("mps", selection.name)
        self.assertEqual(torch.device("mps"), selection.torch_device)
        self.assertEqual(torch.float16, selection.torch_dtype)

    def test_select_device_auto_uses_cpu_when_accelerators_are_unavailable(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("aegis_introspection.model_loader._mps_is_available", return_value=False),
        ):
            selection = select_device("auto")

        self.assertEqual("cpu", selection.name)
        self.assertEqual(torch.device("cpu"), selection.torch_device)
        self.assertEqual(torch.float32, selection.torch_dtype)

    def test_select_device_accepts_gpu_as_cuda_alias(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            selection = select_device("gpu")

        self.assertEqual("cuda", selection.name)
        self.assertEqual(torch.device("cuda"), selection.torch_device)
        self.assertEqual(torch.float16, selection.torch_dtype)

    def test_select_device_rejects_unavailable_cuda(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            self.assertRaises(DeviceUnavailableError),
        ):
            select_device("cuda")

    def test_select_device_rejects_unknown_device(self) -> None:
        with self.assertRaises(UnsupportedDeviceError):
            select_device("tpu")

    def test_parse_model_dtype_accepts_auto(self) -> None:
        self.assertEqual("auto", parse_model_dtype("auto"))

    def test_parse_model_dtype_rejects_unknown_dtype(self) -> None:
        with self.assertRaises(UnsupportedModelDTypeError):
            parse_model_dtype("float8")

    def test_resolve_model_load_dtype_uses_device_default(self) -> None:
        device = select_device("cpu")

        self.assertEqual(torch.float32, resolve_model_load_dtype("device", device))

    def test_resolve_model_load_dtype_preserves_auto(self) -> None:
        device = select_device("cpu")

        self.assertEqual("auto", resolve_model_load_dtype("auto", device))

    def test_load_causal_lm_passes_auto_dtype_and_trust_remote_code(self) -> None:
        config = ModelLoadConfig(
            model_id="local-model",
            revision="main",
            requested_device="cpu",
            local_files_only=True,
            dtype_name="auto",
            trust_remote_code=True,
        )
        tokenizer = Mock()
        model = Mock()

        with (
            patch(
                "aegis_introspection.model_loader.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as tokenizer_loader,
            patch(
                "aegis_introspection.model_loader.AutoModelForCausalLM.from_pretrained",
                return_value=model,
            ) as model_loader,
        ):
            loaded_model = load_causal_lm(config)

        tokenizer_loader.assert_called_once_with(
            "local-model",
            revision="main",
            local_files_only=True,
            trust_remote_code=True,
        )
        model_loader.assert_called_once_with(
            "local-model",
            revision="main",
            local_files_only=True,
            trust_remote_code=True,
            dtype="auto",
        )
        model.to.assert_called_once_with(torch.device("cpu"))
        model.eval.assert_called_once_with()
        self.assertEqual("local-model", loaded_model.model_id)


if __name__ == "__main__":
    unittest.main()
