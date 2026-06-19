import unittest
from unittest.mock import patch

import torch

from aegis_introspection.model_loader import DeviceUnavailableError, UnsupportedDeviceError, select_device


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


if __name__ == "__main__":
    unittest.main()
