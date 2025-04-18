from __future__ import annotations

__version__ = "1.1.1"
__author__ = "Krit Patyarath"

import os
from argparse import ArgumentParser
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from functools import partial
from itertools import chain, islice, starmap, tee
from math import ceil
from operator import itemgetter
from os import cpu_count
from os import name as os_name
from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v3 as iio
from cv2 import countNonZero, createBackgroundSubtractorKNN
from cv2.bgsegm import createBackgroundSubtractorGMG
from imagehash import dhash
from img2pdf import convert as img_to_pdf  # type: ignore
from PIL.Image import fromarray as frame_to_image
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from typing import Any

    from imagehash import ImageHash
    from numpy import ndarray as Frame
    from PIL.Image import Image


def similar_prev_hashes(
    current_hash: ImageHash,
    prev_hashes: list[ImageHash],
    hash_threshold: int,
    hash_hist_size: int,
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
    crop_str: str,
    box_str: str | None,
    scale: str,
) -> tuple[int, Iterable[enumerate]]:
    metadata: dict[str, Any] = iio.immeta(input_path, plugin="pyav")
    fps = metadata["fps"]
    step = int(max(fps / check_per_sec, 1)) if check_per_sec else 1
    duration = metadata["duration"]
    n_frames = ceil(duration * fps / step)

    filters = [
        ("scale", f"{scale}*in_w:{scale}*in_h"),
        ("drawbox", box_str),
        ("crop", crop_str),
        ("format", "gray"),
    ]
    if not box_str:
        filters.pop(1)

    indexed_frames = enumerate(iio.imiter(
        input_path,
        plugin="pyav",
        thread_type="FRAME",
        filter_sequence=filters,
        format=None,
    ))
    return n_frames, islice(indexed_frames, None, None, step) # type: ignore


def get_candidate_frames(
    indexed_frames: Sequence[tuple[int, Frame]],
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
    proc_label: int =0,
    enable_pb: bool =True,
) -> list[Image] | list[tuple[int, Image]]:
    prev_hashes: list[ImageHash] = []

    def is_unique_hash(frame: Frame) -> bool:
        fast_hash_threshold = int(max(1, hash_threshold/2))
        fast_hash_hist_size = int(max(1, hash_hist_size/1.5),
            ) if hash_hist_size else 0

        current_hash = dhash(frame_to_image(frame), hash_size=hash_size)
        is_unique = not similar_prev_hashes(
            current_hash,
            prev_hashes,
            fast_hash_threshold,
            fast_hash_hist_size,
        )
        if is_unique:
            prev_hashes.append(current_hash)
        return is_unique

    captured = False

    bg_subtrator = createBackgroundSubtractorKNN(
        history=init_frames,
        dist2Threshold=d_threshold or 100,
        detectShadows=False,
    ) if knn else \
        createBackgroundSubtractorGMG(
        initializationFrames=init_frames,
        decisionThreshold=d_threshold or 0.75,
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
        indexed_frames = tqdm(
            indexed_frames,
            desc="Parsing Video " + proc_text,
            total=total,
            leave=leave,
        ) # type: ignore

    for i, frame in indexed_frames:
        fg_mask = bg_subtrator.apply(frame)
        percent_non_zero = 100 * \
            countNonZero(fg_mask) / (1.0 * fg_mask.size)

        animation_stopped = percent_non_zero < max_threshold
        if animation_stopped and not captured:
            # with `--fast`, perform a rough hash
            # so we don't have to extract so many frames later.
            # This checking make this portion of code a little bit slower.
            # However, it should save A LOT of times getting full images`
            if fast and not is_unique_hash(frame):
                continue
            captured = True
            captured_indexes.append(i)

        animation_began = percent_non_zero >= min_threshold
        if captured and animation_began:
            captured = False

    return extract_indexes(
        indexes=captured_indexes,
        include_index=is_multiproc)


def get_candidate_frames_concurrent(
    get_candidates: partial,
    video_iter: Iterable[enumerate[Frame]],
    n_worker: int,
    n_frame: int,
    c_method: str,
) -> list[Image]:
    if c_method == "thread":
        pool_executor = ThreadPoolExecutor
        worker_pb = True
    else:
        pool_executor = ProcessPoolExecutor # type: ignore
        worker_pb = False

    with pool_executor(n_worker) as exe:
        def slice_iter(i: int, e: Iterator) -> Iterator:
            start = int(i * n_frame/n_worker)
            end = min(int((i+1) * n_frame/n_worker), n_frame)
            return islice(e, start, end)

        vid_gen_trimmed = list(
            starmap(slice_iter, enumerate(tee(video_iter, n_worker))))

        print("Done")
        results = [
            exe.submit(
                partial(get_candidates, proc_label=i+1, enable_pb=worker_pb),
                list(e),
            ) for i, e in enumerate(tqdm(vid_gen_trimmed, desc="Load Chunks"))
        ]
        unsorted_frames = chain.from_iterable(
            e.result() for e in as_completed(results)
        )

    return [e[1] for e in sorted(unsorted_frames, key=itemgetter(0))]


def get_frames_from_indexes(
    input_path: str,
    indexes: list[int],
    fast: bool,
    crop_str: str,
    box_str: str | None,
    include_index: bool = False,
) -> list[Image] | list[tuple[int, Image]]:
    filters = [
        ("drawbox", box_str),
        ("crop", crop_str),
    ]
    if not box_str:
        filters.pop(0)

    with iio.imopen(input_path, "r", plugin="pyav") as vid:
        read_at = partial(
            vid.read,
            thread_type="FRAME",
            filter_sequence=filters, # type: ignore
            constant_framerate=fast,
        )
        if include_index:
            return [(i, frame_to_image(read_at(index=i))) for i in indexes]
        return [frame_to_image(read_at(index=i)) for i in tqdm(
            indexes, desc="Getting Images")]


def get_unique_images(
    images: list[Image],
    hash_size: int,
    hash_threshold: int,
    hash_hist_size: int,
) -> list[Image]:
    unique_frames: list[Image] = []
    prev_hashes: list[ImageHash] = []

    for img in tqdm(images, desc="Removing dups "):
        current_hash = dhash(img, hash_size=hash_size)
        is_unique = not similar_prev_hashes(
            current_hash,
            prev_hashes,
            hash_threshold,
            hash_hist_size,
        )
        if not is_unique:
            continue
        unique_frames.append(img)
        prev_hashes.append(current_hash)

    return unique_frames


def convert_to_pdf(
    output_path: str,
    unique_images: list[Image],
    extension: str,
) -> None:
    if not unique_images:
        print("No file was created.")
        return
    get_bytes = partial(iio.imwrite, uri="<bytes>", extension=extension)
    Path(output_path).write_bytes(
        img_to_pdf([get_bytes(image=img) for img in
            tqdm(unique_images, desc="Making PDF")]))


def main() -> None:
    arg_parser = ArgumentParser(
        description="Miavisc is a video to slide converter.",
    )
    arg_parser.add_argument(
        "-i", "--input",
        type=str, required=True,
        help="Path to input video file",
    )
    arg_parser.add_argument(
        "-o", "--output",
        type=str, required=True,
        help="Path to input video file",
    )
    arg_parser.add_argument(
        "-f", "--fast",
        action="store_true", default=False,
        help="Use various hacks to speed up the process"
             " (might affect the final result).",
    )
    arg_parser.add_argument(
        "-v", "--version",
        action="version", version="1.0.0",
    )
    arg_parser.add_argument(
        "-c", "--concurrent",
        default=False,
        action="store_true",
        help="Enable concurrency",
    )
    arg_parser.add_argument(
        "-k", "--knn",
        default=False, action="store_true",
        help="Use KNN instead of GMG",
    )
    arg_parser.add_argument(
        "-F", "--force",
        default=False, action="store_true",
        help="Force replace if output file already exists.",
    )
    arg_parser.add_argument(
        "--hash_size",
        type=int, default=12,
        help="Hash size. (default = 12)",
    )
    arg_parser.add_argument(
        "--hash_threshold",
        type=int, default=6,
        help="Threshold for final hash (default = 6). "
             "Larger number means larger differences are required for image "
             "to be considered different "
             "(i.e., it become LESS sensitive to small changes).",
    )
    arg_parser.add_argument(
        "--hash_hist_size",
        type=int, default=5,
        help="Number of frame to look back when deduplicating images."
             " (default = 5; 0 = unlimited)",
    )
    arg_parser.add_argument(
        "--max_threshold",
        type=float, default=0.15,
        help="Max threshold for GMG/KNN (in %%). (default = 0.15)",
    )
    arg_parser.add_argument(
        "--min_threshold",
        type=float, default=0.01,
        help="Min threshold for GMG/KNN (in %%). (default = 0.01)",
    )
    arg_parser.add_argument(
        "--d_threshold",
        type=float, default=None,
        help="Decision threshold for GMG. (default = 0.75) "
             "/ Dist_2_Threshold for KNN. (default = 100)",
    )
    arg_parser.add_argument(
        "--init_frames",
        type=int, default=15,
        help="Number of initialization frames for GMG. (default = 15)",
    )
    arg_parser.add_argument(
        "--check_per_sec",
        type=int, default=0,
        help="How many frame to process in 1 sec. (0 = no skip frame)",
    )
    arg_parser.add_argument(
        "--crop_h", "-H",
        type=str, default="0:1:0",
        help="Top_Border:Content_Height:Bottom_Border. "
             "Calculated in ratio so numbers do not have to "
             "exactly match source video.",
    )
    arg_parser.add_argument(
        "--crop_w", "-W",
        type=str, default="0:1:0",
        help="Left_Border:Content_Width:Right_Border. "\
             "Calculated in ratio so numbers do not have to "
             "exactly match source video.",
    )
    arg_parser.add_argument(
        "--box_h",
        type=str, default=None,
        help="Top_Margin:Box_Height:Bottom_Margin. "
             "Calculated in ratio so numbers do not have to "
             "exactly match source video. Applied before crop.",
    )
    arg_parser.add_argument(
        "--box_w",
        type=str, default=None,
        help="Left_Margin:Box_Width:Right_Margin. "
             "Calculated in ratio so numbers do not have to "
             "exactly match source video. Applied before crop.",
    )
    arg_parser.add_argument(
        "--box_color",
        type=str, default="0xFFFFFF",
        help="Color of the block, unproductive if --box_w & --box_h are unset"
        " (default = 0xFFFFFF; i.e., white)",
    )
    arg_parser.add_argument(
        "--process_scale",
        type=str, default="0.25",
        help="Process at <num>x the original resolution. (default = 0.25)",
    )
    arg_parser.add_argument(
        "--n_worker", "--c_num",
        type=int, default=(cpu_count() or 4) * 2,
        help="Numer of concurrent workers (default = CPU core)",
    )
    arg_parser.add_argument(
        "--concurrent_method", "--c_type",
        type=str, default="thread",
        choices=["thread", "process"],
        help="Method of concurrent (default = thread)",
    )
    arg_parser.add_argument(
        "--img_type", "-t",
        type=str, default=".png",
        choices=[".png.jpeg"],
        help="Encoding for final images. PNG provides better results." \
            "JPEG provides smaller file size. (default = .png)",
    )
    arg_parser.add_argument(
        "--no_check_input", "-U",
        action="store_true",
        help="Skip checking input path. Useful for in URL input.",
    )
    args = arg_parser.parse_args()

    if not args.no_check_input and not os.access(args.input, os.R_OK):
        raise FileNotFoundError(f"Error! Cannot access {args.input}")

    output_dir = os.path.dirname(args.output)
    if not os.access(output_dir, os.F_OK):
        raise FileNotFoundError(f"Error! Path {output_dir} does not exist")

    if os.path.exists(args.output) and not args.force:
        raise FileExistsError(f"{args.output} already exists."\
                            "To force replace, use '--force' or '-F' option")
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Error! Cannot write to {output_dir}")

    def get_ffmpeg_pos_str(hs: str, ws: str) -> str:
        l_border, content_w, r_border = (
            f"({e})" if e else "0" for e in ws.split(":")
        )
        t_border, content_h, b_border = (
            f"({e})" if e else "0" for e in hs.split(":")
        )
        if "0" in (content_h, content_w):
            raise ValueError("Content/box height or width cannot be zero")

        total_w = f"({l_border}+{content_w}+{r_border})"
        total_h = f"({t_border}+{content_h}+{b_border})"
        wr = f"{content_w}/{total_w}"
        hr =  f"{content_h}/{total_h}"
        xr = f"{l_border}/{total_w}"
        yr = f"{t_border}/{total_h}"

        return f"x={xr}*in_w:y={yr}*in_h:w={wr}*in_w:h={hr}*in_h"


    crop_str = get_ffmpeg_pos_str(args.crop_h, args.crop_w)
    box_str = (get_ffmpeg_pos_str(
        args.box_h, args.box_w) + f":c={args.box_color}@1.0:t=fill") \
        if args.box_h and args.box_w else None

    n_frame, video_iter = get_indexed_frames_iter(
        args.input,
        args.check_per_sec,
        crop_str,
        box_str,
        args.process_scale,
    )

    extract_indexes = partial(
        get_frames_from_indexes,
        input_path=args.input,
        fast=args.fast,
        box_str=box_str,
        crop_str=crop_str,
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
        extract_indexes=extract_indexes,
    )
    if args.concurrent:
        print(f"Using {args.concurrent_method} method"
              " with {args.n_worker} workers.\n"
              "\tInitializing concurrency... ", end=" ")
        candidate_frames = get_candidate_frames_concurrent(
            get_candidates,
            video_iter,
            args.n_worker,
            n_frame,
            args.concurrent_method,
        )
    else:
        candidate_frames = get_candidates(video_iter) # type: ignore

    print(f"\tFound potentially {len(candidate_frames)} unique slides.")
    unique_frames = get_unique_images(
        candidate_frames,
        args.hash_size,
        args.hash_threshold,
        args.hash_hist_size,
    )
    print(f"\t{len(unique_frames)} slides remain after postprocessing.")
    convert_to_pdf(args.output, unique_frames, args.img_type)

    # Windows somehow cannot display emoji.
    print("\tDone! 🔥 🚀" if os_name != "nt" else "\tDone!")


if __name__ == "__main__":
    main()
