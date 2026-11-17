"""MRI validation and preprocessing (write-up 3.12.2, 3.16.2).

The same functions run during training and inside the web application, so an
uploaded image is prepared exactly the way the training images were.
"""
from __future__ import annotations

import cv2
import numpy as np

IMG_SIZE = 128          # standard working size (pixels)
GRAY_LEVELS = 32        # quantisation used for the co-occurrence matrix
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
MAX_BYTES = 8 * 1024 * 1024
MIN_SIDE = 64


class InvalidImage(ValueError):
    """Raised when an upload fails validation. The message is safe to show."""


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def decode(data: bytes) -> np.ndarray:
    """Decode raw file bytes to a BGR image, or raise InvalidImage."""
    if not data:
        raise InvalidImage("The file is empty.")
    if len(data) > MAX_BYTES:
        raise InvalidImage("The file is larger than 8 MB.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImage("The file could not be read as an image.")
    return img


def validate(img_bgr: np.ndarray) -> None:
    """Checks that the picture plausibly is a brain MRI slice.

    These are sanity checks, not a clinical quality control step:
      1. large enough to carry texture,
      2. grayscale (MRI slices carry no colour),
      3. not blank or flat,
      4. dark background around a brighter object, as a skull-on-black slice has.
    """
    h, w = img_bgr.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise InvalidImage(f"The image is too small ({w}x{h}). Minimum side is {MIN_SIDE} px.")
    b, g, r = [c.astype(np.int16) for c in cv2.split(img_bgr)]
    colour = (np.abs(r - g).mean() + np.abs(g - b).mean()) / 2
    if colour > 6.0:
        raise InvalidImage("This looks like a colour photograph. Upload a grayscale brain MRI slice.")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.std() < 8:
        raise InvalidImage("The image is blank or almost uniform.")
    small = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    m = IMG_SIZE // 10
    border = np.concatenate([small[:m].ravel(), small[-m:].ravel(),
                             small[:, :m].ravel(), small[:, -m:].ravel()])
    centre = small[IMG_SIZE // 4: -IMG_SIZE // 4, IMG_SIZE // 4: -IMG_SIZE // 4]
    if border.mean() > 90 or centre.mean() < border.mean() + 10:
        raise InvalidImage("The image does not have the dark background and bright centre of an MRI slice.")


def preprocess(img: np.ndarray) -> np.ndarray:
    """grayscale -> 128x128 -> min-max normalise to 0..255 (uint8)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32)
    lo, hi = float(gray.min()), float(gray.max())
    if hi - lo < 1e-6:
        return np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    return np.round((gray - lo) / (hi - lo) * 255.0).astype(np.uint8)


def quantise(gray: np.ndarray, levels: int = GRAY_LEVELS) -> np.ndarray:
    """Reduce 256 gray tones to `levels` bins for the co-occurrence matrix."""
    return np.minimum(gray.astype(np.uint16) * levels // 256, levels - 1).astype(np.intp)
