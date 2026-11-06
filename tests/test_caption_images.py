import base64
import importlib.util
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo

if importlib.util.find_spec("openai") is None:
    raise unittest.SkipTest("Caption tests require the optional openai package")

CAPTION_PATH = Path(__file__).resolve().parents[1] / "src/sensenova_u1_5/caption/caption.py"
spec = importlib.util.spec_from_file_location("caption", CAPTION_PATH)
caption = importlib.util.module_from_spec(spec)
spec.loader.exec_module(caption)


def decode_image(data_uri):
    _, encoded = data_uri.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded, validate=True)))


class CaptionImageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_large_image_preserves_original_resolution(self):
        path = self.directory / "large.png"
        Image.new("RGB", (4096, 3072), "red").save(path)

        data_uri, width, height = caption.image_to_data_uri(path)

        with decode_image(data_uri) as uploaded:
            self.assertEqual(uploaded.size, (4096, 3072))
        self.assertEqual(base64.b64decode(data_uri.split(",", 1)[1]), path.read_bytes())
        self.assertEqual((width, height), (4096, 3072))
        self.assertEqual(caption.image_settings(width, height), ("4:3", "4096"))

    def test_compressed_image_is_base64_encoded_once(self):
        path = self.directory / "noise.png"
        self.noise_image().save(path)
        with (
            patch.object(caption, "MAX_DATA_URI_LEN", 30_000),
            patch.object(caption.base64, "b64encode", wraps=base64.b64encode) as encode,
        ):
            data_uri, _, _ = caption.image_to_data_uri(path)
        self.assertEqual(encode.call_count, 1)
        self.assertLessEqual(len(data_uri), 30_000)
        with decode_image(data_uri) as uploaded:
            self.assertEqual(uploaded.format, "JPEG")
            self.assertEqual(uploaded.size, (128, 128))

    def test_narrow_images_keep_valid_dimensions(self):
        for size in ((20, 1), (1, 20)):
            with self.subTest(size=size):
                path = self.directory / "narrow.bmp"
                Image.new("RGB", size).save(path)
                data_uri, width, height = caption.image_to_data_uri(path)
                self.assertEqual((width, height), size)
                with decode_image(data_uri) as uploaded:
                    self.assertEqual(uploaded.size, size)

    def noise_image(self, size=(128, 128)):
        return Image.frombytes("RGB", size, random.Random(0).randbytes(size[0] * size[1] * 3))

    def test_small_supported_images_reuse_original_bytes(self):
        for fmt in ("JPEG", "PNG", "WEBP"):
            with self.subTest(format=fmt):
                path = self.directory / f"small.{fmt.lower()}"
                self.noise_image().save(path, fmt)
                original = path.read_bytes()

                data_uri, width, height = caption.image_to_data_uri(path)

                prefix, encoded = data_uri.split(",", 1)
                self.assertEqual(prefix, f"data:{caption.SUPPORTED_FORMATS[fmt]};base64")
                self.assertEqual(base64.b64decode(encoded), original)
                self.assertEqual((width, height), (128, 128))
                self.assertEqual(path.read_bytes(), original)

    def test_uri_limit_includes_prefix_and_base64_padding(self):
        path = self.directory / "metadata.png"
        metadata = PngInfo()
        metadata.add_text("Comment", "x" * 1024)
        Image.new("RGBA", (32, 24), (10, 20, 30, 40)).save(path, pnginfo=metadata)
        original = path.read_bytes()
        original_uri = "data:image/png;base64," + base64.b64encode(original).decode("ascii")

        with patch.object(caption, "MAX_DATA_URI_LEN", len(original_uri)):
            data_uri, _, _ = caption.image_to_data_uri(path)
        self.assertEqual(data_uri, original_uri)

        # One character below the full URI size must reject the byte-copy path.
        # Converting to JPEG drops metadata and fits without resizing.
        with patch.object(caption, "MAX_DATA_URI_LEN", len(original_uri) - 1):
            data_uri, width, height = caption.image_to_data_uri(path)
        self.assertLess(len(data_uri), len(original_uri))
        self.assertEqual((width, height), (32, 24))
        with decode_image(data_uri) as uploaded:
            self.assertEqual(uploaded.format, "JPEG")
            self.assertEqual(uploaded.size, (32, 24))
        self.assertEqual(path.read_bytes(), original)

    def test_oversized_png_falls_back_to_jpeg(self):
        path = self.directory / "noise.png"
        self.noise_image().save(path)
        limit = 25_000
        self.assertGreater(len(path.read_bytes()), limit)

        with patch.object(caption, "MAX_DATA_URI_LEN", limit):
            data_uri, width, height = caption.image_to_data_uri(path)

        self.assertLessEqual(len(data_uri), limit)
        self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))
        self.assertEqual((width, height), (128, 128))
        with decode_image(data_uri) as uploaded:
            self.assertEqual(uploaded.format, "JPEG")
            self.assertEqual(uploaded.size, (128, 128))

    def test_jpeg_fallback_composites_transparency_on_white(self):
        path = self.directory / "transparent.png"
        image = self.noise_image().convert("RGBA")
        image.putalpha(0)
        image.save(path)

        with patch.object(caption, "MAX_DATA_URI_LEN", 2000):
            data_uri, _, _ = caption.image_to_data_uri(path)

        self.assertLessEqual(len(data_uri), 2000)
        with decode_image(data_uri) as uploaded:
            self.assertEqual(uploaded.format, "JPEG")
            self.assertEqual(uploaded.getpixel((64, 64)), (255, 255, 255))

    def test_jpeg_resizes_only_when_quality_reduction_is_insufficient(self):
        path = self.directory / "noise.jpg"
        self.noise_image((256, 256)).save(path, quality=95)

        with patch.object(caption, "MAX_DATA_URI_LEN", 2000):
            data_uri, width, height = caption.image_to_data_uri(path)

        self.assertLessEqual(len(data_uri), 2000)
        self.assertEqual((width, height), (256, 256))
        with decode_image(data_uri) as uploaded:
            self.assertLess(uploaded.width * uploaded.height, width * height)
            self.assertGreater(uploaded.width * uploaded.height, 1)

    def test_jpeg_quality_can_fall_without_further_resizing(self):
        path = self.directory / "noise.jpg"
        self.noise_image().save(path, quality=95)
        self.assertGreater(len(path.read_bytes()), 12_000)

        with patch.object(caption, "MAX_DATA_URI_LEN", 12_000):
            data_uri, width, height = caption.image_to_data_uri(path)

        self.assertLessEqual(len(data_uri), 12_000)
        with decode_image(data_uri) as uploaded:
            self.assertEqual(uploaded.size, (128, 128))
            self.assertEqual(uploaded.size, (width, height))

    def test_unsupported_format_converts_to_jpeg_without_resizing(self):
        path = self.directory / "input.bmp"
        Image.new("RGB", (32, 24), "red").save(path)

        data_uri, width, height = caption.image_to_data_uri(path)

        self.assertEqual((width, height), (32, 24))
        self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))
        with decode_image(data_uri) as uploaded:
            self.assertEqual(uploaded.size, (32, 24))

    def test_exif_orientation_is_preserved_when_converting(self):
        path = self.directory / "rotated.jpg"
        image = Image.new("RGB", (48, 32), "red")
        image.paste("blue", (24, 0, 48, 32))
        exif = Image.Exif()
        exif[274] = 6  # Rotate 90 degrees clockwise for display.
        exif[270] = "x" * 5000
        image.save(path, exif=exif)

        original_uri, width, height = caption.image_to_data_uri(path)
        self.assertEqual((width, height), (32, 48))
        self.assertEqual(base64.b64decode(original_uri.split(",", 1)[1]), path.read_bytes())

        with patch.object(caption, "MAX_DATA_URI_LEN", 4000):
            data_uri, width, height = caption.image_to_data_uri(path)

        self.assertLessEqual(len(data_uri), 4000)
        self.assertEqual((width, height), (32, 48))
        with decode_image(data_uri) as uploaded:
            self.assertEqual(uploaded.size, (32, 48))
            self.assertGreater(uploaded.getpixel((16, 8))[0], 200)
            self.assertGreater(uploaded.getpixel((16, 40))[2], 200)

    def test_impossible_limit_raises_without_base64_encoding(self):
        path = self.directory / "input.jpg"
        Image.new("RGB", (32, 24), "red").save(path)

        with (
            patch.object(caption, "MAX_DATA_URI_LEN", 1),
            patch.object(caption.base64, "b64encode", wraps=base64.b64encode) as encode,
        ):
            with self.assertRaisesRegex(ValueError, "data URI limit"):
                caption.image_to_data_uri(path)
        encode.assert_not_called()

    def test_cli_uses_original_size_and_sends_compressed_image(self):
        path = self.directory / "large.png"
        metadata = PngInfo()
        metadata.add_text("Comment", "x" * 1_000_000)
        Image.new("RGB", (4096, 3072), "red").save(path, pnginfo=metadata)
        client = Mock()
        response = {"json": {}, "nlp": "A red image", "language": "en", "task": "test"}
        client.chat.completions.create.return_value.choices = [Mock(message=Mock(content=json.dumps(response)))]
        output = StringIO()

        with (
            patch.object(caption, "MAX_DATA_URI_LEN", 400_000),
            patch.object(caption, "OpenAI", return_value=client),
            patch.object(caption.sys, "argv", ["caption.py", str(path)]),
            redirect_stdout(output),
        ):
            caption.main()

        result = json.loads(output.getvalue())
        self.assertEqual((result["ratio"], result["resolution"]), ("4:3", "4096"))
        request = client.chat.completions.create.call_args.kwargs
        data_uri = request["messages"][1]["content"][0]["image_url"]["url"]
        self.assertLessEqual(len(data_uri), 400_000)
        with decode_image(data_uri) as uploaded:
            self.assertEqual(uploaded.format, "JPEG")
            self.assertEqual(uploaded.size, (4096, 3072))


if __name__ == "__main__":
    unittest.main()
