"""Unit tests for SAM3 platform selection."""

import unittest

import torch

from sam3.runtime import _convert_floating_weights, runtime_config


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.linear2 = torch.nn.Linear(4, 2)
        self.convertible = torch.nn.Linear(2, 2)


class RuntimeConfigTests(unittest.TestCase):
    def test_gb10_defaults_to_bf16(self):
        config = runtime_config({"SAM3_PLATFORM": "gb10"})
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.precision, "bf16-weight")
        self.assertEqual(config.resolution, 1008)

    def test_gb10_fp32_fallback(self):
        config = runtime_config(
            {"SAM3_PLATFORM": "gb10", "SAM3_GB10_PRECISION": "fp32"}
        )
        self.assertEqual(config.dtype, torch.float32)
        self.assertEqual(config.precision, "fp32")

    def test_unknown_gb10_precision_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "SAM3_GB10_PRECISION"):
            runtime_config(
                {"SAM3_PLATFORM": "gb10", "SAM3_GB10_PRECISION": "fp16"}
            )

    def test_b580_alias_selects_fp16_xpu_profile(self):
        config = runtime_config({"SAM3_PLATFORM": "b580"})
        self.assertEqual(config.platform, "intel_arc")
        self.assertEqual(config.device, "xpu")
        self.assertEqual(config.precision, "fp16-weight")
        self.assertEqual(config.resolution, 1008)

    def test_resolution_must_match_checkpoint_geometry(self):
        with self.assertRaisesRegex(RuntimeError, "must be 1008"):
            runtime_config(
                {"SAM3_PLATFORM": "intel_arc", "SAM3_RESOLUTION": "560"}
            )

    def test_unknown_platform_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "SAM3_PLATFORM"):
            runtime_config({"SAM3_PLATFORM": "unknown"})

    def test_conversion_retains_decoder_ffn_in_fp32(self):
        model = TransformerDecoderLayer()
        _convert_floating_weights(model, torch.bfloat16)
        self.assertEqual(model.linear1.weight.dtype, torch.float32)
        self.assertEqual(model.linear2.weight.dtype, torch.float32)
        self.assertEqual(model.convertible.weight.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
