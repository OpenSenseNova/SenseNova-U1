import base64
import importlib.util
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_UTILS_PATH = REPO_ROOT / "apps" / "comfyui" / "image_utils.py"


def _load_image_utils_module():
    spec = importlib.util.spec_from_file_location("sensenova_comfyui_image_utils", IMAGE_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


IMAGE_UTILS = _load_image_utils_module()


class ComfyUIImageUtilsTest(unittest.TestCase):
    def test_jpeg_data_url_resizes_to_pixel_limit_without_upscaling(self) -> None:
        large_image = Image.new("RGB", (3000, 2000), color=(128, 64, 32))
        small_image = Image.new("RGB", (300, 200), color=(32, 64, 128))

        large_result = IMAGE_UTILS.pil_to_jpeg_data_url(large_image, quality=95, max_pixels=4_000_000)
        small_result = IMAGE_UTILS.pil_to_jpeg_data_url(small_image, quality=95, max_pixels=4_000_000)

        self.assertTrue(large_result.startswith("data:image/jpeg;base64,"))
        with Image.open(BytesIO(base64.b64decode(large_result.split(",", 1)[1]))) as decoded_large:
            self.assertEqual(decoded_large.format, "JPEG")
            self.assertLessEqual(decoded_large.width * decoded_large.height, 4_000_000)
            self.assertEqual(decoded_large.size, (2449, 1632))

        with Image.open(BytesIO(base64.b64decode(small_result.split(",", 1)[1]))) as decoded_small:
            self.assertEqual(decoded_small.size, small_image.size)


if __name__ == "__main__":
    unittest.main()
