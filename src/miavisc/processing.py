from __future__ import annotations

import PIL.Image
import itertools

__version__ = "2.0.0"
__author__ = "Krit Patyarath"

from typing import TYPE_CHECKING

import cv2
import cv2.bgsegm
import PIL

import miavisc as mv

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray as Frame  # noqa: N812


def get_candidate_indices(
    enumerated_frames: list[tuple[int, Frame]] | Iterable[tuple[int, Frame]],
    bgs_setting: mv.settings.BackgroundSubtractorSetting,
    hash_setting: mv.settings.HashSetting,
) -> list[int]:
    if bgs_setting.algorithm == "KNN":
        bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=bgs_setting.init_frames,
            dist2Threshold=bgs_setting.d_threshold,
            detectShadows=False,
        )
    else:
        bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG(
            initializationFrames=bgs_setting.init_frames,
            decisionThreshold=bgs_setting.d_threshold,
        )

    hash_pool = mv.hash_utils.FrameHashPool(
        hash_setting,
        keep_unique_data=True
    )
    if isinstance(enumerated_frames, list):
        first_index, first_frame = enumerated_frames[1]
    else:
        first_index, first_frame = next(iter(enumerated_frames))

    hash_pool.add_image_if_unique(PIL.Image.fromarray(first_frame), first_index)
    captured = False

    for i, frame in enumerated_frames:
        fg_mask = bg_subtractor.apply(frame)
        perc_non_zero: float = 100 * cv2.countNonZero(fg_mask) / (1.0 * fg_mask.size)

        animation_stopped = perc_non_zero < bgs_setting.max_threshold
        if animation_stopped and not captured:
            captured = True
            hash_pool.add_image_if_unique(PIL.Image.fromarray(frame), i)

        animation_began = perc_non_zero >= bgs_setting.min_threshold
        if captured and animation_began:
            captured = False

    return hash_pool.data_pool
