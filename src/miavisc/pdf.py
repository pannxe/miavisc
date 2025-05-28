"""Utility function to convert PIL.Image to pdf file."""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Krit Patyarath"

from typing import TYPE_CHECKING

import imageio.v3 as iio
import img2pdf  # type: ignore # noqa: PGH003

if TYPE_CHECKING:
    from PIL.Image import Image


def convert(
    unique_images: list[Image],
    extension: str,
) -> bytes | None:
    """Return a byte of resulted pdf from a list of PIL.Image."""
    pages: list[bytes] = [
        iio.imwrite("<bytes>", image=img, extension=extension)  # type: ignore # noqa: PGH003
        for img in unique_images
    ]
    return img2pdf.convert(pages)
