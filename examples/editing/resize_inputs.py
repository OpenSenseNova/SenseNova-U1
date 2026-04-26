import argparse
import math
from pathlib import Path

from PIL import Image

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
TARGET_PIXELS = 2048 * 2048


def resize_to_target_pixels(img: Image.Image, target_pixels: int) -> Image.Image:
    w, h = img.size
    scale = math.sqrt(target_pixels / (w * h))
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def process_folder(src: Path, dst: Path, target_pixels: int = TARGET_PIXELS) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    if not files:
        print(f"No images found under {src}")
        return

    for path in files:
        try:
            with Image.open(path) as img:
                img.load()
                orig_size = img.size
                resized = resize_to_target_pixels(img, target_pixels)
                out_path = dst / path.name
                save_kwargs = {}
                fmt = (img.format or path.suffix.lstrip(".")).upper()
                if fmt in {"JPEG", "JPG"}:
                    save_kwargs["quality"] = 95
                    if resized.mode in {"RGBA", "P"}:
                        resized = resized.convert("RGB")
                elif fmt == "WEBP":
                    save_kwargs["quality"] = 95
                resized.save(out_path, **save_kwargs)
                print(
                    f"{path.name}: {orig_size[0]}x{orig_size[1]} -> "
                    f"{resized.size[0]}x{resized.size[1]} "
                    f"({resized.size[0] * resized.size[1]} px)"
                )
        except Exception as e:
            print(f"Failed to process {path.name}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", type=str, default="examples/editing/data/images")
    parser.add_argument("--dst", type=str, default="examples/editing/data/images_2048")
    parser.add_argument("--target-pixels", type=int, default=TARGET_PIXELS)
    args = parser.parse_args()

    process_folder(Path(args.src), Path(args.dst), args.target_pixels)


if __name__ == "__main__":
    main()
