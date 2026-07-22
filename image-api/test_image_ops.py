"""Invariant tests for CPU-only image operations and RPC image artifacts."""

import base64
import io
import unittest

import numpy as np
from PIL import Image

from image_ops import optional_rpc_image, outpaint_canvas, transform_image


class ImageOperationTests(unittest.TestCase):
    def test_outpaint_mask_preserves_every_original_pixel(self):
        source = Image.new("RGB", (4, 3), (1, 2, 3))
        canvas, mask = outpaint_canvas(source, 10, 9, "center")
        pixels = np.asarray(mask)
        left, top = 3, 3
        self.assertTrue(np.all(pixels[top : top + 3, left : left + 4] == 0))
        self.assertEqual(
            canvas.crop((left, top, left + 4, top + 3)).tobytes(),
            source.tobytes(),
        )

    def test_transform_rejects_partial_crop(self):
        with self.assertRaises(Exception):
            transform_image(Image.new("RGB", (10, 10)), (0, 0, None, 5), 0, True)


def _attachment(name: str, color: str) -> dict:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
    return {
        "name": name,
        "media_type": "image/png",
        "data_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


class RpcImageArtifactTests(unittest.TestCase):
    def test_returns_named_diagnostic_artifact(self):
        result = {"attachments": [_attachment("pre_composite_image", "red")]}

        image = optional_rpc_image(result, "pre_composite_image")

        self.assertIsNotNone(image)
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_returns_none_when_artifact_is_absent(self):
        self.assertIsNone(
            optional_rpc_image({"attachments": []}, "pre_composite_image")
        )

    def test_returns_conditioning_image_artifact(self):
        result = {"attachments": [_attachment("conditioning_image", "magenta")]}

        image = optional_rpc_image(result, "conditioning_image")

        self.assertIsNotNone(image)
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 255))


if __name__ == "__main__":
    unittest.main()
