"""Invariant tests for CPU-only image operations."""

import unittest

import numpy as np
from PIL import Image

from image_ops import outpaint_canvas, transform_image


class ImageOperationTests(unittest.TestCase):
    def test_outpaint_mask_preserves_every_original_pixel(self):
        source = Image.new("RGB", (4, 3), (1, 2, 3))
        canvas, mask = outpaint_canvas(source, 10, 9, "center")
        pixels = np.asarray(mask)
        left, top = 3, 3
        self.assertTrue(np.all(pixels[top:top + 3, left:left + 4] == 0))
        self.assertEqual(canvas.crop((left, top, left + 4, top + 3)).tobytes(), source.tobytes())

    def test_transform_rejects_partial_crop(self):
        with self.assertRaises(Exception):
            transform_image(Image.new("RGB", (10, 10)), (0, 0, None, 5), 0, True)


if __name__ == "__main__":
    unittest.main()
