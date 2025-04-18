from __future__ import annotations

__version__ = "1.1.0"
__author__ = "Krit Patyarath"

from argparse import ArgumentParser
from math import ceil
from itertools import chain, tee, islice
from functools import partial
from operator import itemgetter
import imageio.v3 as iio
from cv2 import createBackgroundSubtractorKNN, countNonZero
from cv2.bgsegm import createBackgroundSubtractorGMG
from tqdm import tqdm
from imagehash import dhash
from PIL.Image import fromarray as frame_to_image
from os import cpu_count, name as os_name
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from img2pdf import convert as to_pdf
import os

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Iterable
    from PIL.Image import Image
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
    w_ratio: str,
    h_ratio: str,
    x_ratio: str,
    y_ratio: str,
    scale: str,
) -> tuple[int, Iterable[enumerate]]:
    metadata: dict[str, Any] = iio.immeta(input_path, plugin="pyav")
    fps = metadata["fps"]
    step = int(max(fps / check_per_sec, 1)) if check_per_sec else 1
    duration = metadata["duration"]
    n_frames = ceil(duration * fps / step)

    filters = [
        ("crop", f"{w_ratio}*in_w:{h_ratio}*in_h:{x_ratio}*in_w:{y_ratio}*in_h"),
        ("scale", f"{scale}*in_w:{scale}*in_h"),
        ("format", "gray")
    ]
    indexed_frames = enumerate(iio.imiter(
        input_path,
        plugin="pyav",
        thread_type="FRAME",
        filter_sequence=filters,
        format=None
    ))
    return n_frames, islice(indexed_frames, None, None, step)


def get_frames_from_indexes(
    input_path: str,
    indexes: list[int],
    fast: bool,
    w_ratio: str,
    h_ratio: str,
    x_ratio: str,
    y_ratio: str,
    include_index=False,
) -> list[Image] | list[tuple[int, Image]]:
    with iio.imopen(input_path, "r", plugin="pyav") as vid:
        read_at = partial(
            vid.read, 
            thread_type="FRAME",
            filter_sequence=[("crop", f"{w_ratio}*in_w:{h_ratio}*in_h:{x_ratio}*in_w:{y_ratio}*in_h")],
            constant_framerate=fast
        )
        if include_index:
            return [(i, frame_to_image(read_at(index=i))) for i in indexes]
        return [frame_to_image(read_at(index=i)) for i in tqdm(indexes, desc="Getting Images")]


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
    extract_indexes: partial,
    n_frame: int,
    proc_label=0,
    enable_pb=True
) -> list[Image] | list[tuple[int, Image]]:
    prev_hashes: list[ImageHash] = []

    def is_unique_hash(frame):
        fast_hash_threshold = int(max(1, hash_threshold/2))
        fast_hash_hist_size = int(max(1, hash_hist_size/1.5)
            ) if hash_hist_size else 0

        current_hash = dhash(frame_to_image(frame), hash_size=hash_size)
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

    bg_subtrator = createBackgroundSubtractorKNN(
        history=init_frames,
        dist2Threshold=d_threshold if d_threshold else 100,
        detectShadows=False
    ) if knn else \
        createBackgroundSubtractorGMG(
        initializationFrames=init_frames,
        decisionThreshold=d_threshold if d_threshold else 0.75
    )

    # Always include 1st frame.
    is_multiproc = isinstance(indexed_frames, list)
    if is_multiproc:
        captured_indexes = [indexed_frames[0][0]]
        total = len(indexed_frames)
    else:
        # only here if single thread/process
        captured_indexes = [0]
        total = n_frame
    leave = not is_multiproc
    proc_text = f"#{proc_label}" if is_multiproc else ""

    if enable_pb:
        indexed_frames = tqdm(indexed_frames, desc="Parsing Video " + proc_text, total=total, leave=leave)

    for i, frame in indexed_frames:
        fg_mask = bg_subtrator.apply(frame)
        percent_non_zero = 100 * \
            countNonZero(fg_mask) / (1.0 * fg_mask.size)

        animation_stopped = percent_non_zero < max_threshold
        if animation_stopped and not captured:
            # with `--fast`, perform a rough hash
            # so we don't have to extract so many frames later.
            # This checking make running this portion of code a little bit slower.
            # However, it should save A LOT of times running `get_frames_from_indexes()`
            if fast and not is_unique_hash(frame):
                continue
            captured = True
            captured_indexes.append(i)

        animation_began = percent_non_zero >= min_threshold
        if captured and animation_began:
            captured = False

    return extract_indexes(indexes=captured_indexes, include_index=is_multiproc)


def get_candidate_frames_concurrent(
    get_candidates: partial,
    video_iter: list[Frame],
    n_worker: int,
    n_frame: int,
    c_method: str 
) -> list[Image]:
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
) -> list[Image]:
    unique_frames: list[Image] = []
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
        unique_frames.append(frame_bytes)
        prev_hashes.append(current_hash)

    return unique_frames


def convert_to_pdf(
    output_path: str,
    unique_frames: list[Image],
    extension: str
) -> None:
    if not unique_frames:
        print("No file was created.")
        return

    with open(output_path, "wb") as f:
        f.write(to_pdf([
            iio.imwrite("<bytes>", frame, extension=extension) \
                for frame in tqdm(unique_frames, desc="Making PDF")
        ]))


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
        "-F", "--force",
        default=False, action="store_true",
        help="Force replace if output file already exists."
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
        help="Number of frame to look back when deduplicating images. (default = 5; 0 = unlimited)"
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
        "--crop_h", "-H",
        type=str, default="0:1:0",
        help="Top_Border:Content:Bottom_Border. Calculated in ratio so numbers do not have to be exactly match the source video."
    )
    arg_parser.add_argument(
        "--crop_w", "-W",
        type=str, default="0:1:0",
        help="Left_Border:Content:Right_Border. Calculated in ratio so numbers do not have to be exactly match the source video."
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
    arg_parser.add_argument(
        "--img_type", "-t",
        type=str, default=".png",
        choices=[".png" ".jpeg"],
        help="Encoding for final images. PNG provides better results, JPEG provides smaller file size. (default = .png)"
    )
    args = arg_parser.parse_args()

    if not os.access(args.input, os.R_OK):
        print(f"Error! Cannot access {args.input}")
        return
    
    output_dir = os.path.dirname(args.output)
    if not os.access(output_dir, os.F_OK):
        print(f"Error! Path {output_dir} does not exist.")
        return
    if os.path.exists(args.output) and not args.force:
        print(f"Error! {args.output} already exists. To force replace, use '--force' or '-F' option")
        return
    if not os.access(output_dir, os.W_OK):
        print(f"Error! Cannot write to {output_dir}.")
        return

    l_border, content_w, r_border = (
        float(e) if e else 0 for e in args.crop_w.split(":")
    )
    t_border, content_h, b_border = (
        float(e) if e else 0 for e in args.crop_h.split(":")
    )
    total_w = l_border + content_w + r_border
    total_h = t_border + content_h + b_border
    w_ratio = content_w / total_w
    h_ratio = content_h / total_h
    x_ratio = l_border / total_w
    y_ratio = t_border / total_h

    n_frame, video_iter = get_indexed_frames_iter(
        args.input,
        args.check_per_sec,
        w_ratio,
        h_ratio,
        x_ratio,
        y_ratio,
        args.process_scale,
    )

    extract_indexes = partial(
        get_frames_from_indexes,
        input_path=args.input,
        fast=args.fast,
        w_ratio=w_ratio,
        h_ratio=h_ratio,
        x_ratio=x_ratio,
        y_ratio=y_ratio,
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
        n_frame=n_frame,
        extract_indexes=extract_indexes
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
    unique_frames: list[Frame] = get_unique_frames(
        candidate_frames,
        args.hash_size,
        args.hash_threshold,
        args.hash_hist_size
    )
    print(f"\t{len(unique_frames)} slides remain after postprocessing.")
    convert_to_pdf(args.output, unique_frames, args.img_type)

    # Windows somehow cannot display emoji.
    print("\tDone! 🔥 🚀" if os_name != "nt" else "\tDone!")


if __name__ == "__main__":
    main()
