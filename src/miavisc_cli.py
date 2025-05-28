from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import operator
import os
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import miavisc as mv

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray as Frame  # noqa: N812
    from PIL.Image import Image

__version__ = "2.0.0"


def get_ffmpeg_pos_str(hs: str, ws: str) -> str:
    l_border, content_w, r_border = (f"({e})" if e else "0" for e in ws.split(":"))
    t_border, content_h, b_border = (f"({e})" if e else "0" for e in hs.split(":"))

    if "0" in (content_h, content_w):
        error_msg = "Content/box height or width cannot be zero"
        raise ValueError(error_msg)

    total_w = f"({l_border}+{content_w}+{r_border})"
    total_h = f"({t_border}+{content_h}+{b_border})"
    wr = f"{content_w}/{total_w}"
    hr = f"{content_h}/{total_h}"
    xr = f"{l_border}/{total_w}"
    yr = f"{t_border}/{total_h}"

    return f"x=({xr})*in_w:y=({yr})*in_h:w=({wr})*in_w:h=({hr})*in_h"


def get_images_from_input(
    video_frames_list: list[tuple[int, Frame]] | Iterable[tuple[int, Frame]],
    input_path: str,
    full_video_setting: mv.settings.VideoSetting,
    bgs_setting: mv.settings.BackgroundSubtractorSetting,
    rough_hash_setting: mv.settings.HashSetting,
    worker_label: int = 0,
) -> list[tuple[int, Image]]:
    print(f"\tWorker # {worker_label}\tstarted processing frames.")
    candidate_indices = mv.processing.get_candidate_indices(
        video_frames_list, bgs_setting, rough_hash_setting
    )
    print(f"\tWorker # {worker_label}\tfinished.")
    return mv.video.get_enum_images_from_indices(
        input_path, candidate_indices, full_video_setting
    )


def concurrent_handler(
    video_frames_iter: Iterable[tuple[int, Frame]],
    n_frames: int,
    get_images_partial: functools.partial,
    n_worker: int,
    concurrent_method: str,
) -> list[tuple[int, Image]]:
    pool_executor = (
        ThreadPoolExecutor if concurrent_method == "thread" else ProcessPoolExecutor
    )
    print("Loading data for each workers...")
    with pool_executor(n_worker) as pe:
        results = [
            pe.submit(get_images_partial, video_frames_list=list(e), worker_label=i)
            for i, e in enumerate(
                itertools.batched(video_frames_iter, math.ceil(n_frames / n_worker))
            )
        ]
        unsorted_frames = itertools.chain.from_iterable(
            e.result() for e in as_completed(results)
        )

    return sorted(unsorted_frames, key=operator.itemgetter(0))


def get_argument_parser() -> ArgumentParser:
    arg_parser = ArgumentParser(description="Miavisc is a video to slide converter.")
    arg_parser.add_argument("-v", "--version", action="version", version=__version__)
    arg_parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to input video file"
    )
    arg_parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to input video file"
    )
    arg_parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        default=False,
        help=(
            "Use various hacks to speed up the process "
            "(might affect the final result)."
        ),
    )
    arg_parser.add_argument(
        "-u",
        "--no_check_input",
        "--url",
        action="store_true",
        help="Skip checking input path. Useful for in URL input.",
    )
    arg_parser.add_argument(
        "-c",
        "--concurrent",
        default=False,
        action="store_true",
        help="Enable concurrency",
    )
    arg_parser.add_argument(
        "-k",
        "--knn",
        default=False,
        action="store_true",
        help="Use KNN instead of GMG",
    )
    arg_parser.add_argument(
        "-F",
        "--force",
        default=False,
        action="store_true",
        help="Force replace if output file already exists.",
    )
    arg_parser.add_argument(
        "--hash_size", type=int, default=12, help="Hash size. (default = 12)"
    )
    arg_parser.add_argument(
        "--hash_threshold",
        type=int,
        default=6,
        help="Threshold for final hash (default = 6). "
        "Larger number means larger differences are required for image "
        "to be considered different "
        "(i.e., it become LESS sensitive to small changes).",
    )
    arg_parser.add_argument(
        "--hash_hist_size",
        type=int,
        default=5,
        help="Number of frame to look back when deduplicating images."
        " (default = 5; -1 = unlimited; 0 = no dhash check)",
    )
    arg_parser.add_argument(
        "--max_threshold",
        type=float,
        default=0.15,
        help="Max threshold for GMG/KNN (in %%). (default = 0.15)",
    )
    arg_parser.add_argument(
        "--min_threshold",
        type=float,
        default=0.01,
        help="Min threshold for GMG/KNN (in %%). (default = 0.01)",
    )
    arg_parser.add_argument(
        "--d_threshold",
        type=float,
        default=None,
        help=(
            "Decision threshold for GMG. (default = 0.75) "
            "/ Dist_2_Threshold for KNN. (default = 100)"
        ),
    )
    arg_parser.add_argument(
        "--init_frames",
        type=int,
        default=15,
        help="Number of initialization frames for GMG. (default = 15)",
    )
    arg_parser.add_argument(
        "--crop_h",
        "-H",
        type=str,
        default="0:1:0",
        help=(
            "Top_Border:Content_Height:Bottom_Border. "
            "Calculated in ratio so numbers do not have to "
            "exactly match source video."
        ),
    )
    arg_parser.add_argument(
        "--crop_w",
        "-W",
        type=str,
        default="0:1:0",
        help=(
            "Left_Border:Content_Width:Right_Border. "
            "Calculated in ratio so numbers do not have to "
            "exactly match source video."
        ),
    )
    arg_parser.add_argument(
        "--box_h",
        type=str,
        default=None,
        help=(
            "Top_Margin:Box_Height:Bottom_Margin. "
            "Calculated in ratio so numbers do not have to "
            "exactly match source video. Applied before crop."
        ),
    )
    arg_parser.add_argument(
        "--box_w",
        type=str,
        default=None,
        help=(
            "Left_Margin:Box_Width:Right_Margin. "
            "Calculated in ratio so numbers do not have to "
            "exactly match source video. Applied before crop."
        ),
    )
    arg_parser.add_argument(
        "--box_color",
        type=str,
        default="0xFFFFFF",
        help=(
            "Color of the block, unproductive if --box_w & --box_h are unset"
            " (default = 0xFFFFFF; i.e., white)"
        ),
    )
    arg_parser.add_argument(
        "--process_scale",
        type=str,
        default="0.25",
        help="Process at <num>x the original resolution. (default = 0.25)",
    )
    arg_parser.add_argument(
        "--n_worker",
        "--c_num",
        type=int,
        default=(os.cpu_count() or 4) * 2,
        help="Number of concurrent workers (default = CPU core x 2)",
    )
    arg_parser.add_argument(
        "--concurrent_method",
        "--c_type",
        type=str,
        default="thread",
        choices=["thread", "process"],
        help="Method of concurrent (default = thread)",
    )
    arg_parser.add_argument(
        "--img_type",
        "-t",
        type=str,
        default=".png",
        choices=[".png", ".jpeg"],
        help=(
            "Encoding for final images. PNG provides better results."
            "JPEG provides smaller file size. (default = .png)"
        ),
    )
    return arg_parser


def main() -> None:  # noqa: PLR0912, C901
    args = get_argument_parser().parse_args()
    if not args.no_check_input and not os.access(args.input, os.R_OK):
        error_msg = f"Error! Cannot access {args.input}."
        raise FileNotFoundError(error_msg)

    output = Path(args.output)

    if not os.access(output.parent, os.F_OK):
        error_msg = f"Error! Path {output.parent} does not exist"
        raise FileNotFoundError(error_msg)

    if output.exists() and not args.force:
        error_msg = (
            f"{args.output} already exists. "
            "To force replace, use '--force' or '-F' option"
        )
        raise FileExistsError(error_msg)

    if not os.access(output.parent, os.W_OK):
        error_msg = f"Error! Cannot write to {output.parent}"
        raise PermissionError(error_msg)

    filter_sequence: list[tuple[str, str]] = []
    if args.box_h and args.box_w:
        pos_str = get_ffmpeg_pos_str(args.box_h, args.box_w)
        filter_sequence.append(
            ("drawbox", f"{pos_str}:c={args.box_color}@1.0:t=fill")
        )
    filter_sequence.append(("crop", get_ffmpeg_pos_str(args.crop_h, args.crop_w)))

    bgs_video_setting = mv.settings.VideoSetting(
        filter_sequence=[
            ("scale", f"{args.process_scale}*in_w:{args.process_scale}*in_h"),
            *filter_sequence,
            ("format", "gray"),
        ],  # type: ignore  # noqa: PGH003
        thread_type="FRAME" if args.fast else "SCLICE",
        constant_framerate=args.fast,
        format=None,
    )

    if not args.fast or args.hash_hist_size == 0:
        rough_hash_hist_size = 0
    elif args.hash_hist_size < 0:  # Unlimited
        rough_hash_hist_size = 5
    else:
        rough_hash_hist_size = int(max(1, args.hash_hist_size / 1.5))

    get_images_partial = functools.partial(
        get_images_from_input,
        input_path=args.input,
        full_video_setting=dataclasses.replace(
            bgs_video_setting, filter_sequence=filter_sequence, format="rgb24"
        ),
        bgs_setting=mv.settings.BackgroundSubtractorSetting(
            algorithm="KNN" if args.knn else "GMG",
            init_frames=args.init_frames,
            d_threshold=args.d_threshold or (100.0 if args.knn else 0.75),
            max_threshold=args.max_threshold,
            min_threshold=args.min_threshold,
        ),
        rough_hash_setting=mv.settings.HashSetting(
            size=args.hash_size,
            threshold=int(max(1, args.hash_threshold / 2)),
            history_size=rough_hash_hist_size,
        ),
    )

    video_frames_iter = mv.video.get_enum_frames_iter(args.input, bgs_video_setting)

    if args.concurrent:
        if args.n_worker < 2:
            print("Warning: n_worker is set < 2. Using 2 workers instead.")

        n_worker = max(2, args.n_worker)

        print(f"Using {args.concurrent_method} method with {n_worker} workers.\n")
        captured_images = concurrent_handler(
            video_frames_iter,
            mv.video.get_source_total_frame(args.input),
            get_images_partial,
            args.n_worker,
            args.concurrent_method,
        )
    else:
        print("Using non-concurrency method.")
        captured_images = get_images_partial(video_frames_iter)

    print(f"Found potentially {len(captured_images)} unique slides.")

    hash_pool = mv.hash_utils.FrameHashPool(
        mv.settings.HashSetting(
            size=args.hash_size,
            threshold=args.hash_threshold,
            history_size=args.hash_hist_size,
        ),
        keep_unique_data=True,
    )
    for _, image in captured_images:
        hash_pool.add_image_if_unique(image, image)

    unique_image: list[Image] = hash_pool.data_pool

    print(f"\t{len(unique_image)} slides remain after deduplication.")
    if not unique_image:
        print("Output file not created.")
        return

    print("Creating pdf file...", end=" ")
    output.write_bytes(
        mv.pdf.convert(unique_image, args.img_type)  # type: ignore  # noqa: PGH003
    )

    # Windows somehow cannot display emoji.
    print("Done! 🔥 🚀" if os.name != "nt" else "Done!")


if __name__ == "__main__":
    main()
