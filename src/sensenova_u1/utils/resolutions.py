"""Pixel budgets and output sizing for the image-generation examples."""

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from decimal import Decimal, InvalidOperation
from math import isqrt

from .profiler import DEFAULT_IMAGE_PATCH_SIZE

DEFAULT_ASPECT_RATIO = "9:16"
_PIXELS_PER_M = 1024**2
_MIN_BUCKET_M = 1
_DEFAULT_BUCKET_M = 4
_MAX_BUCKET_M = 16
_DEFAULT_PIXEL_BUDGET = _DEFAULT_BUCKET_M * _PIXELS_PER_M


def calculate_image_size(
    pixel_budget: int = _DEFAULT_PIXEL_BUDGET, aspect_ratio: str = DEFAULT_ASPECT_RATIO
) -> tuple[int, int]:
    """Scale to the pixel budget and round each dimension down to the image grid."""
    width_ratio, height_ratio = map(int, aspect_ratio.split(":"))
    factor = DEFAULT_IMAGE_PATCH_SIZE
    pixels_per_cell = factor**2
    width = factor * isqrt(pixel_budget * width_ratio // (pixels_per_cell * height_ratio))
    height = factor * isqrt(pixel_budget * height_ratio // (pixels_per_cell * width_ratio))
    return width, height


def _parse_pixel_budget(value: str) -> int:
    error = f"expected a pixel budget from {_MIN_BUCKET_M}m to {_MAX_BUCKET_M}m, such as 8m or 12.5m"
    if not value.lower().endswith("m"):
        raise ArgumentTypeError(error)
    try:
        amount = Decimal(value[:-1])
    except InvalidOperation:
        raise ArgumentTypeError(error) from None
    if not amount.is_finite() or not _MIN_BUCKET_M <= amount <= _MAX_BUCKET_M:
        raise ArgumentTypeError(error)
    numerator, denominator = amount.as_integer_ratio()
    return numerator * _PIXELS_PER_M // denominator


def _parse_aspect_ratio(value: str) -> str:
    error = "expected W:H with positive integers and a ratio from 1:3 to 3:1"
    width, _, height = value.partition(":")
    if not (width.isdecimal() and height.isdecimal()):
        raise ArgumentTypeError(error)
    try:
        w, h = int(width), int(height)
    except ValueError:
        raise ArgumentTypeError(error) from None
    if min(w, h) == 0 or max(w, h) > 3 * min(w, h):
        raise ArgumentTypeError(error)
    return value


def add_resolution_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--bucket",
        dest="pixel_budget",
        type=_parse_pixel_budget,
        metavar="BUDGET",
        help=(
            f"Output pixel budget from {_MIN_BUCKET_M}m to {_MAX_BUCKET_M}m. Each m represents 1024*1024 pixels. "
            f"Start with {_DEFAULT_BUCKET_M}m, the default when only --aspect_ratio is given."
        ),
    )
    parser.add_argument(
        "--aspect_ratio",
        type=_parse_aspect_ratio,
        metavar="W:H",
        help=f"Output aspect ratio from 1:3 to 3:1. Defaults to {DEFAULT_ASPECT_RATIO} when only --bucket is given.",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Exact output width in pixels. Supply with --height to override --bucket and --aspect_ratio.",
    )
    parser.add_argument(
        "--height",
        type=int,
        help=f"Exact output height in pixels. Both dimensions must be positive multiples of {DEFAULT_IMAGE_PATCH_SIZE}.",
    )


def resolve_cli_image_size(parser: ArgumentParser, args: Namespace) -> tuple[int, int] | None:
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be supplied together")
    if args.width is not None and args.height is not None:
        size = (args.width, args.height)
        if any(side <= 0 or side % DEFAULT_IMAGE_PATCH_SIZE for side in size):
            parser.error(f"--width and --height must be positive multiples of {DEFAULT_IMAGE_PATCH_SIZE}")
        return size
    if args.pixel_budget is None and args.aspect_ratio is None:
        return None
    return calculate_image_size(args.pixel_budget or _DEFAULT_PIXEL_BUDGET, args.aspect_ratio or DEFAULT_ASPECT_RATIO)


def print_output_size(width: int, height: int, *, sample_number: int = 1) -> None:
    print(f"[resolution] sample={sample_number} --width {width} --height {height}", flush=True)
