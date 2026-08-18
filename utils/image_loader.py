import base64
import binascii
import io
import os
from pathlib import Path

from PIL import Image


MAXIMUM_IMAGE_FILES = 12
MAXIMUM_IMAGE_BYTES = 15 * 1024 * 1024
MAXIMUM_IMAGE_WIDTH = 12_000
MAXIMUM_IMAGE_HEIGHT = 12_000
MAXIMUM_IMAGE_PIXELS = 50_000_000
MAXIMUM_REQUEST_BYTES = 256 * 1024 * 1024
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG"}


class InvalidOCRImage(ValueError):
    pass


def valid_image_batch(candidates) -> bool:
    return isinstance(candidates, list) and 1 <= len(candidates) <= MAXIMUM_IMAGE_FILES


def normalize_image(image: Image.Image) -> Image.Image:
    if image.mode in ("P", "PA", "LA", "L"):
        return image.convert("RGBA" if "transparency" in image.info else "RGB")
    if image.mode not in ("RGB", "RGBA"):
        return image.convert("RGB")
    return image


def decode_base64_image(encoded: str) -> Image.Image:
    if not isinstance(encoded, str) or not encoded:
        raise InvalidOCRImage("image data must be non-empty base64")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise InvalidOCRImage("image data is not valid base64") from error

    return load_supported_image(io.BytesIO(data), len(data))


def load_image_path(raw_path: str) -> Image.Image:
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise InvalidOCRImage("image path is invalid")

    path = Path(raw_path).resolve(strict=True)
    if not path.is_file() or not _path_allowed(path):
        raise InvalidOCRImage("image path is not allowed")
    size = path.stat().st_size
    with path.open("rb") as source:
        return load_supported_image(source, size)


def load_supported_image(source, size: int) -> Image.Image:
    if size <= 0 or size > MAXIMUM_IMAGE_BYTES:
        raise InvalidOCRImage("image size is outside the allowed range")
    try:
        with Image.open(source) as opened:
            if opened.format not in SUPPORTED_IMAGE_FORMATS:
                raise InvalidOCRImage("image format is not supported")
            width, height = opened.size
            if (
                width <= 0
                or height <= 0
                or width > MAXIMUM_IMAGE_WIDTH
                or height > MAXIMUM_IMAGE_HEIGHT
                or width * height > MAXIMUM_IMAGE_PIXELS
            ):
                raise InvalidOCRImage("image dimensions are outside the allowed range")
            opened.load()
            return normalize_image(opened).copy()
    except InvalidOCRImage:
        raise
    except (Image.DecompressionBombError, OSError, ValueError) as error:
        raise InvalidOCRImage("image could not be decoded") from error


def _path_allowed(path: Path) -> bool:
    configured = os.getenv("OCR_ALLOWED_IMAGE_ROOTS", "/images")
    roots = [Path(value).resolve() for value in configured.split(os.pathsep) if value.strip()]
    return any(path.is_relative_to(root) for root in roots)
