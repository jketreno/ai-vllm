"""Unit tests for lossless SAM3 API mask serialization."""

import base64
import importlib
import io
import sys
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image


managers = types.ModuleType("managers")
annotation_manager = types.ModuleType("managers.annotation_manager")
annotation_manager.SAM3Annotator = mock.Mock
managers.annotation_manager = annotation_manager
sys.modules.setdefault("managers", managers)
sys.modules.setdefault("managers.annotation_manager", annotation_manager)

api = importlib.import_module("sam3.api")


class MaskSerializationTests(unittest.TestCase):
    def test_mask_png_round_trips_every_pixel(self):
        mask = np.array(
            [[False, True, False], [True, True, False]],
            dtype=bool,
        )

        data_uri = api._mask_to_data_uri(mask)

        prefix, payload = data_uri.split(",", 1)
        decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(payload))))
        self.assertEqual(prefix, "data:image/png;base64")
        self.assertEqual(decoded.dtype, np.uint8)
        np.testing.assert_array_equal(decoded, mask.astype(np.uint8) * 255)


if __name__ == "__main__":
    unittest.main()
