"""Input helpers used before object-detection inference."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

ImageSource = Union[str, Path]


def read_image(image_path: ImageSource) -> np.ndarray:
    """Read an image and raise a clear error if it cannot be decoded."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    return validate_frame(image)


def validate_frame(frame: np.ndarray) -> np.ndarray:
    """Check that a frame is a non-empty, three-channel image."""
    if not isinstance(frame, np.ndarray):
        raise TypeError("Expected a NumPy image array.")
    if frame.size == 0:
        raise ValueError("Received an empty frame.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            "Expected an HxWx3 image; received shape "
            f"{frame.shape}."
        )
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError(f"Invalid frame dimensions: {frame.shape}")

    return frame


def resize_with_letterbox(
    image: np.ndarray,
    target_size: int = 640,
    fill_value: int = 114,
) -> np.ndarray:
    """Fit an image on a square canvas without changing its aspect ratio.

    This helper is for custom preprocessing or visualization. The
    Ultralytics high-level prediction API already applies its own
    model-specific resize and letterbox operations.
    """
    image = validate_frame(image)
    if target_size <= 0:
        raise ValueError("target_size must be positive.")

    height, width = image.shape[:2]
    scale = min(target_size / width, target_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.full(
        (target_size, target_size, 3), fill_value, dtype=np.uint8
    )

    pad_x = (target_size - new_width) // 2
    pad_y = (target_size - new_height) // 2
    canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
    return canvas
