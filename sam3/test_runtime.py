"""Unit tests for SAM3 platform selection."""

import unittest

from sam3.runtime import runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_gb10_defaults_preserve_existing_runtime(self):
        config = runtime_config({"SAM3_PLATFORM": "gb10"})
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.precision, "fp32")
        self.assertEqual(config.resolution, 1008)

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


if __name__ == "__main__":
    unittest.main()
