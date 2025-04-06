#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
from math import ceil
from itertools import islice
import imageio.v3 as iio
import cv2
from tqdm import tqdm
import img2pdf as i2p
from imagehash import dhash
from PIL import Image

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Iterable
    from numpy import ndarray as Frame
    from imagehash import ImageHash


def similar_prev_hashes(
    current_hash,
    prev_hashes,
    hash_threshold,
    hash_hist_size
) -> bool:
    def in_hist_size(i) -> bool:
        if not hash_hist_size:
            return True
        return i < hash_hist_size

    # similar hashes should be in the back, so search in reverse.
    for i, prev_hash in enumerate(reversed(prev_hashes)):
        if not in_hist_size(i):
            return False
        if prev_hash - current_hash <= hash_threshold:
            return True
    return False


def get_indexed_frames(
    input_path: str,
    check_per_sec: int,
    crop_zoom: str,
    scale: str,
    fast: bool,
) -> Iterable[tuple[int, Frame]]:
    metadata: dict[str, Any] = iio.immeta(input_path, plugin="pyav")
    fps = metadata["fps"]
    duration = metadata["duration"]
    step = int(max(fps / check_per_sec, 1)) if check_per_sec else 1
    n_frames = ceil(duration * fps / step)

    filters = [
        fil for opt, fil in (
            (scale, ("scale", f"{scale}*in_w:{scale}*in_h")),
            (fast, ("format", "gray")),
            (crop_zoom, ("crop", f"{crop_zoom}*in_w:{crop_zoom}*in_h")),
        ) if opt
    ]
    format_type = None if fast else "bgr24"
    indexed_frames = enumerate(iio.imiter(
        input_path,
        plugin="pyav",
        thread_type="FRAME",
        filter_sequence=filters,
        format=format_type
    ))

    return tqdm(
        islice(indexed_frames, 1, None, step),
        desc="Parsing Video ",
        total=n_frames-1
    )


def get_captured_indexes(
    indexed_frames: Iterable[tuple[int, Frame]],
    init_frames: int,
    d_threshold: float | None,
    max_threshold: float,
    min_threshold: float,
    knn: bool,
    fast: bool,
    hash_size: int,
    hash_threshold: int,
    hash_hist_size: int
) -> list[int]:
    prev_hashes: list[ImageHash] = []

    def is_unique_hash(frame):
        # Only checking if `--fast` is on. This checking make running this portion of code
        # a little bit slower. However, it should save A LOT of times running `extract_indexes()`
        if not fast:
            return True

        fast_hash_threshold = int(max(1, hash_threshold/2))
        fast_hash_hist_size = int(max(1, hash_hist_size/1.5))

        current_hash = dhash(Image.fromarray(frame), hash_size=hash_size)
        is_unique = not similar_prev_hashes(
            current_hash,
            prev_hashes,
            fast_hash_threshold,
            fast_hash_hist_size
        )
        if is_unique:
            prev_hashes.append(current_hash)
        return is_unique

    captured = False

    bg_subtrator = cv2.createBackgroundSubtractorKNN(
        history=init_frames,
        dist2Threshold=d_threshold if d_threshold else 100,
        detectShadows=False
    ) if knn else \
        cv2.bgsegm.createBackgroundSubtractorGMG(
        initializationFrames=init_frames,
        decisionThreshold=d_threshold if d_threshold else 0.75
    )

    # Always include 1st frame
    capture_indexes = [0]

    for i, frame in indexed_frames:
        fg_mask = bg_subtrator.apply(frame)
        percent_non_zero = 100 * \
            cv2.countNonZero(fg_mask) / (1.0 * fg_mask.size)

        if percent_non_zero < max_threshold and not captured:
            # with `--fast`, perform a rough hash
            # so we don't have to extract so many frames later.
            if not is_unique_hash(frame):
                continue
            captured = True
            capture_indexes.append(i)

        if captured and percent_non_zero >= min_threshold:
            captured = False

    print(f"\nFound potentially {len(capture_indexes)} unique slides.\n")

    return capture_indexes


def extract_indexes(
    input_path: str,
    indexes: list[int],
    fast: bool
) -> list[Frame]:
    with iio.imopen(input_path, "r", plugin="pyav") as vid:
        images = [
            vid.read(index=i, thread_type="FRAME", constant_framerate=fast)
            for i in tqdm(indexes, desc="Getting Images")
        ]
    return images


def get_unique_indexes(
    slides: list[Frame],
    hash_size: int,
    hash_threshold: int,
    hash_hist_size: int,
) -> list[int]:
    unique_indexes: list[int] = []
    prev_hashes: list[ImageHash] = []

    for i, slide in enumerate(tqdm(slides, desc="Removing dups ")):
        current_hash = dhash(Image.fromarray(slide), hash_size=hash_size)
        is_unique = not similar_prev_hashes(
            current_hash,
            prev_hashes,
            hash_threshold,
            hash_hist_size
        )
        if not is_unique:
            continue
        unique_indexes.append(i)
        prev_hashes.append(current_hash)

    print(f"\n{len(unique_indexes)} slides remain after postprocessing.\n")

    return unique_indexes


def convert_to_pdf(
    output_path: str,
    slides: list[Frame],
    unique_indexes: list[int],
    final_extension: str
) -> None:
    def frame_to_bytes(frame) -> bytes:
        kargs: dict[str, Any] = {"optimize": True}
        if final_extension.lower() == "jpg":
            kargs.update((
                ("quality", 95),
                ("progressive", True),
                ("keep_rgb", True),
                ("subsampling", 0)
            ))

        return iio.imwrite(
            "<bytes>", frame, plugin="pillow",
            extension=final_extension,
            **kargs
        )

    unique_bytes_list: list[bytes] = [
        frame_to_bytes(slides[i])
        for i in tqdm(unique_indexes, desc="Making PDF")
    ]
    with open(output_path, "wb") as f:
        f.write(i2p.convert(unique_bytes_list))


def main():
    arg_parser = ArgumentParser(
        description="Miavisc is a video to slide converter.",
    )
    arg_parser.add_argument(
        "--input",
        type=str, required=True,
        help="Path to input video file"
    )
    arg_parser.add_argument(
        "--output",
        type=str, required=True,
        help="Path to input video file"
    )
    arg_parser.add_argument(
        "--hash_size",
        type=int, default=12,
        help="Hash size. Default = 12."
    )
    arg_parser.add_argument(
        "--hash_threshold",
        type=int, default=4,
        help="Threshold for final hash (default = 4). Also used to calculate fash hash threshold."
    )
    arg_parser.add_argument(
        "--hash_hist_size",
        type=int, default=5,
        help="Process at <num> times the original resolution. (default = 5; 0 = unlimited)"
    )
    arg_parser.add_argument(
        "--knn",
        default=False, action="store_true",
        help="Use KNN instead of GMG"
    )
    arg_parser.add_argument(
        "--max_threshold",
        type=float, default=0.15,
        help="Max threshold for GMG/KNN (in %%). (default = 0.15)"
    )
    arg_parser.add_argument(
        "--min_threshold",
        type=float, default=0.01,
        help="Min threshold for GMG/KNN (in %%). (default = 0.01)"
    )
    arg_parser.add_argument(
        "--d_threshold",
        type=float, default=None,
        help="Decision threshold for GMG. (default = 0.75) / Dist_2_Threshold for KNN. (default = 100)"
    )
    arg_parser.add_argument(
        "--init_frames",
        type=int, default=15,
        help="Number of initialization frames for GMG. (default = 15)"
    )
    arg_parser.add_argument(
        "--check_per_sec",
        type=int, default=0,
        help="How many frame to process in 1 sec. (0 = no skip frame)"
    )
    arg_parser.add_argument(
        "--fast",
        action="store_true", default=False,
        help="Use various hacks to speed up the process (might affect the final result)."
    )
    arg_parser.add_argument(
        "--crop_zoom",
        type=str, default="",
        help="Only process inner <str> of the video. Recommened: '4/5'"
    )
    arg_parser.add_argument(
        "--process_scale",
        type=str, default="0.25",
        help="Process at <num> times the original resolution. (default = 0.25)"
    )
    arg_parser.add_argument(
        "--final_extension",
        type=str, default=".png",
        help="Extension of final images (default: '.png'). Use '.jpg' should give you smaller file"
    )
    args = arg_parser.parse_args()

    indexed_frames: Iterable[tuple[int, Frame]] = get_indexed_frames(
        args.input,
        args.check_per_sec,
        args.crop_zoom,
        args.process_scale,
        args.fast
    )
    slides_indexes: list[int] = get_captured_indexes(
        indexed_frames,
        args.init_frames,
        args.d_threshold,
        args.max_threshold,
        args.min_threshold,
        args.knn,
        args.fast,
        args.hash_size,
        args.hash_threshold,
        args.hash_hist_size
    )
    slides: list[Frame] = extract_indexes(
        args.input,
        slides_indexes,
        args.fast
    )
    unique_indexes: list[int] = get_unique_indexes(
        slides,
        args.hash_size,
        args.hash_threshold,
        args.hash_hist_size
    )
    convert_to_pdf(
        args.output,
        slides,
        unique_indexes,
        args.final_extension
    )

if __name__ == "__main__":
    main() 