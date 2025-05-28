from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import PIL.Image

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray as Frame  # noqa: N812
    from PIL.Image import Image

    import miavisc as mv

import math

import imageio.v3 as iio
import PIL


def get_source_total_frame(input_path: str) -> int:
    """Return a number of frame of a video from the given path/url."""
    metadata = iio.immeta(input_path, plugin="pyav")
    return math.ceil(metadata["fps"] * metadata["duration"])


def get_enum_images_from_indices(
    input_path: str, indices: list[int], setting: mv.settings.VideoSetting
) -> list[tuple[int, Image]]:
    with iio.imopen(input_path, "r", plugin="pyav") as video:
        read_at = functools.partial(
            video.read,
            thread_type=setting.thread_type,
            constant_framerate=setting.constant_framerate,
            filter_sequence=setting.filter_sequence,
            format=setting.format
        )
        return [(i, PIL.Image.fromarray(read_at(index=i))) for i in indices]


def get_enum_frames_iter(
    input_path: str, setting: mv.settings.VideoSetting
) -> Iterable[tuple[int, Frame]]:
    """Return enumerate object of frames from a given path/url.

    Args:
        input_path: Path or url to the input source.
        filter_sequence: FFmpeg filters to be applied.

    Returns:
        enumerate object of numpy.ndarray type of every frames

    """
    video_iter = iio.imiter(
        input_path,
        plugin="pyav",
        thread_type=setting.thread_type,
        filter_sequence=setting.filter_sequence,
        format=setting.format
    )
    return enumerate(video_iter)
