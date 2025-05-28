"""Contain types for Miavisc as a module."""

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class BackgroundSubtractorSetting:
    """Setting for background subtractor algorithms."""

    algorithm: Literal["KNN", "GMG"]
    init_frames: int
    d_threshold: float | int
    max_threshold: float
    min_threshold: float


@dataclasses.dataclass(frozen=True)
class HashSetting:
    """Setting for dhash."""

    size: int
    threshold: int
    history_size: int


@dataclasses.dataclass(frozen=True)
class VideoSetting:
    """Setting for iio."""

    filter_sequence: list[tuple[str, str]]
    thread_type: Literal["FRAME", "SCLICE"] = "FRAME"
    constant_framerate: bool = True
    format: str | None = None

