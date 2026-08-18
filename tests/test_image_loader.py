import base64
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from utils.image_loader import (
    InvalidOCRImage,
    decode_base64_image,
    load_image_path,
    valid_image_batch,
)


def encoded_image(image_format="PNG", size=(20, 10)):
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, format=image_format)
    return base64.b64encode(output.getvalue()).decode("ascii"), output.getvalue()


class ImageLoaderSecurityTest(unittest.TestCase):
    def test_accepts_supported_strict_base64_image(self):
        encoded, _ = encoded_image()

        image = decode_base64_image(encoded)

        self.assertEqual(image.size, (20, 10))
        self.assertEqual(image.mode, "RGB")

    def test_rejects_malformed_base64_and_unsupported_format(self):
        with self.assertRaises(InvalidOCRImage):
            decode_base64_image("not base64!@")

        encoded, _ = encoded_image("GIF")
        with self.assertRaises(InvalidOCRImage):
            decode_base64_image(encoded)

    def test_rejects_byte_and_pixel_limits_before_inference(self):
        encoded, data = encoded_image(size=(20, 20))
        with patch("utils.image_loader.MAXIMUM_IMAGE_BYTES", len(data) - 1):
            with self.assertRaises(InvalidOCRImage):
                decode_base64_image(encoded)
        with patch("utils.image_loader.MAXIMUM_IMAGE_PIXELS", 399):
            with self.assertRaises(InvalidOCRImage):
                decode_base64_image(encoded)

    def test_path_mode_stays_inside_configured_roots(self):
        with tempfile.TemporaryDirectory() as allowed_directory, tempfile.TemporaryDirectory() as other_directory:
            allowed = Path(allowed_directory) / "recipe.png"
            blocked = Path(other_directory) / "private.png"
            _, data = encoded_image()
            allowed.write_bytes(data)
            blocked.write_bytes(data)
            with patch.dict(os.environ, {"OCR_ALLOWED_IMAGE_ROOTS": allowed_directory}):
                self.assertEqual(load_image_path(str(allowed)).size, (20, 10))
                with self.assertRaises(InvalidOCRImage):
                    load_image_path(str(blocked))

    def test_batch_count_is_bounded(self):
        self.assertFalse(valid_image_batch([]))
        self.assertTrue(valid_image_batch([{}] * 12))
        self.assertFalse(valid_image_batch([{}] * 13))
        self.assertFalse(valid_image_batch("not-a-list"))


if __name__ == "__main__":
    unittest.main()
