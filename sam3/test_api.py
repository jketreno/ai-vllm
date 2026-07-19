"""Unit tests for SAM3 worker mask attachment serialization."""

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
    def test_mask_attachment_round_trips_every_pixel(self):
        mask = np.array([[False, True, False], [True, True, False]], dtype=bool)
        attachment = api._mask_attachment(mask, "mask:0")
        decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(attachment["data_base64"]))))
        self.assertEqual(attachment["name"], "mask:0")
        self.assertEqual(attachment["media_type"], "image/png")
        np.testing.assert_array_equal(decoded, mask.astype(np.uint8) * 255)


if __name__ == "__main__":
    unittest.main()
