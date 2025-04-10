from __future__ import annotations

__version__ = "1.0.1"
__author__ = "Krit Patyarath"

from argparse import ArgumentParser
from math import ceil
from itertools import chain, tee, islice
from functools import partial
from operator import itemgetter
import imageio.v3 as iio
import cv2
from tqdm import tqdm
from imagehash import dhash
from PIL import Image
from os import cpu_count, name as os_name
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Iterable
    from PIL.Image import Image as Image_T
    from numpy import ndarray as Frame
    from imagehash import ImageHash


def similar_prev_hashes(
    current_hash: ImageHash,
    prev_hashes: list[ImageHash],
    hash_threshold: int,
    hash_hist_size: int
) -> bool:
    # similar hashes should be in the back, so search in reverse.
    for i, prev_hash in enumerate(reversed(prev_hashes)):
        if hash_hist_size > 0 and i >= hash_hist_size:
            return False
        if prev_hash - current_hash <= hash_threshold:
            return True
    return False


def get_indexed_frames_iter(
    input_path: str,
    check_per_sec: int,
    crop_width: str,
    crop_heigh: str,
    scale: str,
) -> tuple[int, Iterable[enumerate]]:
    metadata: dict[str, Any] = iio.immeta(input_path, plugin="pyav")
    fps = metadata["fps"]
    step = int(max(fps / check_per_sec, 1)) if check_per_sec else 1
    duration = metadata["duration"]
    n_frames = ceil(duration * fps / step)

    # import as grayscale for faster result.
    filters = [("format", "gray")]

    if crop_width or crop_heigh:
        w_ratio = crop_width if crop_width else "1"
        h_ratio = crop_heigh if crop_heigh else "1"
        filters.append(("crop", f"{w_ratio}*in_w:{h_ratio}*in_h"))

    if scale:
        filters.append(("scale", f"{scale}*in_w:{scale}*in_h"))

    indexed_frames = enumerate(iio.imiter(
        input_path,
        plugin="pyav",
        thread_type="FRAME",
        filter_sequence=filters,
        format=None
    ))

    return n_frames, islice(indexed_frames, None, None, step)


def extract_indexes(
    input_path: str,
    indexes: list[int],
    fast: bool,
    include_index=False,
) -> list[Image_T] | list[tuple[int, Image_T]]:
    with iio.imopen(input_path, "r", plugin="pyav") as vid:
        read_at = partial(vid.read, thread_type="FRAME", constant_framerate=fast)
        if include_index:
            return [(i, Image.fromarray(read_at(index=i))) for i in indexes]
        return [Image.fromarray(read_at(index=i)) for i in tqdm(indexes, desc="Getting Images")]


def get_candidate_frames(
    indexed_frames: Iterable[tuple[int, Frame]],
    init_frames: int,
    d_threshold: float | None,
    max_threshold: float,
    min_threshold: float,
    knn: bool,
    fast: bool,
    hash_size: int,
    hash_threshold: int,
    hash_hist_size: int,
    input_path: str,
    n_frame: int,
    proc_label=0,
    enable_pb=True
) -> list[Image_T] | list[tuple[int, Image_T]]:
    prev_hashes: list[ImageHash] = []

    def is_unique_hash(frame):
        # Only checking if `--fast` is on. This checking make running this portion of code
        # a little bit slower. However, it should save A LOT of times running `extract_indexes()`
        if not fast:
            return True

        fast_hash_threshold = int(max(1, hash_threshold/2))
        fast_hash_hist_size = int(
            max(1, hash_hist_size/1.5)) if hash_hist_size else 0

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

    # Always include 1st frame.
    is_multiproc = isinstance(indexed_frames, list)
    if is_multiproc:
        capture_indexes = [indexed_frames[0][0]]
        total = len(indexed_frames)
    else:
        # only here if single thread/process
        capture_indexes = [0]
        total = n_frame
    leave = not is_multiproc
    proc_text = f"#{proc_label}" if is_multiproc else ""

    if enable_pb:
        indexed_frames = tqdm(indexed_frames, desc="Parsing Video " + proc_text, total=total, leave=leave)

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

    return extract_indexes(input_path, capture_indexes, fast, include_index=is_multiproc)


def get_candidate_frames_concurrent(
    get_candidates: partial,
    video_iter: list[Frame],
    n_worker: int,
    n_frame: int,
    c_method: str 
) -> list[Image_T]:
    if c_method == "thread":
        pool_executor = ThreadPoolExecutor
        worker_pb = True
    else:
        pool_executor = ProcessPoolExecutor
        worker_pb = False
    with pool_executor(n_worker) as exe:
        def slice_iter(i, e):
            start = int(i * n_frame/n_worker)
            end = min(int((i+1) * n_frame/n_worker), n_frame)
            return islice(e, start, end)

        vid_gen_trimmed = [
            slice_iter(i, e) for i, e in enumerate(tee(video_iter, n_worker))
        ]
        print("Done")
        results = [
            exe.submit(
                partial(get_candidates, proc_label=i+1, enable_pb=worker_pb),
                list(e)
            ) for i, e in enumerate(tqdm(vid_gen_trimmed, desc="Load Chunks"))
        ]
        unsorted_frames = chain.from_iterable(
            e.result() for e in as_completed(results)
        )

    sorted_frames = [
        e[1] for e in sorted(unsorted_frames, key=itemgetter(0))
    ]
    return sorted_frames


def get_unique_frames(
    frames_bytes: list[bytes],
    hash_size: int,
    hash_threshold: int,
    hash_hist_size: int,
) -> list[Image_T]:
    unique_bytes_list: list[Image_T] = []
    prev_hashes: list[ImageHash] = []

    for frame_bytes in tqdm(frames_bytes, desc="Removing dups "):
        current_hash = dhash(frame_bytes, hash_size=hash_size)
        is_unique = not similar_prev_hashes(
            current_hash,
            prev_hashes,
            hash_threshold,
            hash_hist_size
        )
        if not is_unique:
            continue
        unique_bytes_list.append(frame_bytes)
        prev_hashes.append(current_hash)

    return unique_bytes_list


def convert_to_pdf(
    output_path: str,
    unique_bytes_list: list[Image_T],
) -> None:
    if not unique_bytes_list:
        print("No file was created.")
        return

    unique_bytes_list[0].save(
        output_path,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=tqdm(
            unique_bytes_list[1:], desc="Making PDF")
    )


def main():
    arg_parser = ArgumentParser(
        description="Miavisc is a video to slide converter.",
    )
    arg_parser.add_argument(
        "-i", "--input",
        type=str, required=True,
        help="Path to input video file"
    )
    arg_parser.add_argument(
        "-o", "--output",
        type=str, required=True,
        help="Path to input video file"
    )
    arg_parser.add_argument(
        "-f", "--fast",
        action="store_true", default=False,
        help="Use various hacks to speed up the process (might affect the final result)."
    )
    arg_parser.add_argument(
        "-v", "--version",
        action="version", version="1.0.0"
    )
    arg_parser.add_argument(
        "-c", "--concurrent",
        default=False,
        action="store_true",
        help="Enable concurrency"
    )
    arg_parser.add_argument(
        "-k", "--knn",
        default=False, action="store_true",
        help="Use KNN instead of GMG"
    )
    arg_parser.add_argument(
        "--hash_size",
        type=int, default=12,
        help="Hash size. (default = 12)"
    )
    arg_parser.add_argument(
        "--hash_threshold",
        type=int, default=6,
        help="Threshold for final hash (default = 6). "\
             "Larger number means larger differences are required for image to be considered different "\
             "(i.e., it become LESS sensitive to small changes)."
    )
    arg_parser.add_argument(
        "--hash_hist_size",
        type=int, default=5,
        help="Process at <num> times the original resolution. (default = 5; 0 = unlimited)"
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
        "--crop_h",
        type=str, default="1",
        help="Only process inner <str> of the video (height). Recommened: '4/5'"
    )
    arg_parser.add_argument(
        "--crop_w",
        type=str, default="1",
        help="Only process inner <str> of the video (width). Recommened: '4/5'"
    )
    arg_parser.add_argument(
        "--process_scale",
        type=str, default="0.25",
        help="Process at <num> times the original resolution. (default = 0.25)"
    )
    arg_parser.add_argument(
        "--n_worker", "--c_num",
        type=int, default=cpu_count()*2,
        help="Numer of concurrent workers (default = CPU core)"
    )
    arg_parser.add_argument(
        "--concurrent_method", "--c_type",
        type=str, default="thread",
        choices=["thread", "process"],
        help="Method of concurrent (default = thread)"
    )
    args = arg_parser.parse_args()

    n_frame, video_iter = get_indexed_frames_iter(
        args.input,
        args.check_per_sec,
        args.crop_w,
        args.crop_h,
        args.process_scale,
    )

    get_candidates = partial(
        get_candidate_frames,
        init_frames=args.init_frames,
        d_threshold=args.d_threshold,
        max_threshold=args.max_threshold,
        min_threshold=args.min_threshold,
        knn=args.knn,
        fast=args.fast,
        hash_size=args.hash_size,
        hash_threshold=args.hash_threshold,
        hash_hist_size=args.hash_hist_size,
        input_path=args.input,
        n_frame=n_frame
    )
    if args.concurrent:
        print(f"Using {args.concurrent_method} method with {args.n_worker} workers.\n"
              "\tInitializing concurrency... ", end=" ")
        candidate_frames = get_candidate_frames_concurrent(
            get_candidates, video_iter, args.n_worker, n_frame, args.concurrent_method
        )
    else:
        candidate_frames = get_candidates(video_iter)

    print(f"\tFound potentially {len(candidate_frames)} unique slides.")
    unique_bytes_list: list[bytes] = get_unique_frames(
        candidate_frames,
        args.hash_size,
        args.hash_threshold,
        args.hash_hist_size
    )
    print(f"\t{len(unique_bytes_list)} slides remain after postprocessing.")
    convert_to_pdf(args.output, unique_bytes_list)

    # Windows somehow cannot display emoji.
    print("\tDone! 🔥 🚀" if os_name != "nt" else "\tDone!")


if __name__ == "__main__":
    main()
