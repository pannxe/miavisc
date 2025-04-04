from argparse import ArgumentParser
from math import ceil
from io import BytesIO
from collections.abc import Iterator
from itertools import islice
import imageio.v3 as iio
import cv2
from tqdm import tqdm
from img2pdf import convert
from imagehash import dhash, ImageHash
from PIL import Image


def frame_to_bytes(frame) -> BytesIO:
    bio = BytesIO()
    iio.imwrite(bio, frame, plugin="pillow", extension=".jpg")
    bio.seek(0)
    return bio


def get_indexed_frames(
    input_path: str,
    check_per_sec: int,
    crop_zoom: str,
    scale: str,
    fast: bool
) -> Iterator:
    metadata = iio.immeta(input_path, plugin="pyav")
    fps, duration = metadata["fps"], metadata["duration"]

    step = int(max(fps / check_per_sec, 1)) if check_per_sec else 1
    n_frames = ceil(duration * fps / step)

    filters = [fil for opt, fil in (
        (fast, ("scale", f"{scale}*in_w:{scale}*in_h")),
        (crop_zoom, ("crop", f"{crop_zoom}*in_w:{crop_zoom}*in_h")),
    ) if opt]

    thread_type = "FRAME" if fast else "SLICE"

    indexed_frames = enumerate(iio.imiter(
        input_path,
        plugin="pyav",
        thread_type=thread_type,
        filter_sequence=filters
    ))

    return tqdm(islice(indexed_frames, 1, None, step), desc="Parsing Video ", total=n_frames-1)


def get_captured_indexes(
    indexed_frames,
    init_frames,
    d_threshold,
    max_threshold,
    min_threshold,
    use_knn,
    fast
) -> list[int]:
    prev_hashes: [ImageHash] = []

    def is_unique_hash(slide):
        # only checking if `--fast` is on
        if not fast:
            return True

        hash_threshold = 1

        slide_bytes = frame_to_bytes(slide)
        current_hash = dhash(Image.open(slide_bytes), hash_size=8)

        is_unique = not any(
            prev_hash - current_hash <= hash_threshold
            for prev_hash in prev_hashes
        )
        if is_unique:
            prev_hashes.append(current_hash)
        return is_unique

    captured = False

    bg_subtrator = cv2.createBackgroundSubtractorKNN(
        history=init_frames,
        dist2Threshold=d_threshold if d_threshold else 100,
        detectShadows=False
    ) if use_knn else \
        cv2.bgsegm.createBackgroundSubtractorGMG(
        initializationFrames=init_frames,
        decisionThreshold=d_threshold if d_threshold else 0.75
    )

    # Always include 1st frame
    capture_indexes: list[int] = [0]

    for index, frame in indexed_frames:
        fg_mask = bg_subtrator.apply(frame)
        percent_non_zero = 100 * \
            cv2.countNonZero(fg_mask) / (1.0 * fg_mask.size)

        if percent_non_zero < max_threshold and not captured:
            # with `--fast`, perform a rolling rough hash so we don't have to extract so many frames later.
            if not is_unique_hash(frame):
                continue
            captured = True
            capture_indexes.append(index)

        if captured and percent_non_zero >= min_threshold:
            captured = False

    print(f"\nFound potentially {len(capture_indexes)} unique slides.\n")

    return capture_indexes


def extract_indexes(input_path, indexes) -> list[BytesIO]:
    with iio.imopen(input_path, "r", plugin="pyav") as vid:
        images = [vid.read(index=i)
                  for i in tqdm(indexes, desc="Getting Images")]
    return [frame_to_bytes(img) for img in images]


def get_unique_indexes(slides, hash_threshold) -> list[int]:
    unique_indexes = []
    prev_hashes = []

    for i, slide in enumerate(tqdm(slides, desc="Removing dups  ")):
        current_hash = dhash(Image.open(slide), hash_size=8)
        if any(prev_hash - current_hash <= hash_threshold for prev_hash in prev_hashes):
            continue
        unique_indexes.append(i)
        prev_hashes.append(current_hash)
        slide.seek(0)
    print(
        f"\nAfter further checking, {len(unique_indexes)} potentially unique slides remain.\n")

    return unique_indexes


def convert_to_pdf(output_path, slides, unique_indexes):
    with open(output_path, "wb") as f:
        f.write(convert([slides[i].read() for i in unique_indexes]))

    print("Finished making PDF file.")


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        description="Miavisc is a video to slide converter.",
    )
    arg_parser.add_argument(
        "--input",
        type=str, required=True,
        help="Path to input video file")
    arg_parser.add_argument(
        "--output",
        type=str, required=True,
        help="Path to input video file"
    )
    arg_parser.add_argument(
        "--hash_threshold",
        type=int, default=4,
        help="Threshold for final hash. (default = 4)"
    )
    arg_parser.add_argument(
        "--use_knn",
        default=False, action="store_true",
        help="Use KNN instead of GMG"
    )
    arg_parser.add_argument(
        "--max_threshold",
        type=float, default=0.15,
        help="Max threshold for GMG/KNN (in %). (default = 0.15)"
    )
    arg_parser.add_argument(
        "--min_threshold",
        type=float, default=0.01,
        help="Min threshold for GMG/KNN (in %). (default = 0.01)"
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

    args = arg_parser.parse_args()

    indexed_frames = get_indexed_frames(
        args.input,
        args.check_per_sec,
        args.crop_zoom,
        args.process_scale,
        args.fast
    )

    slides_indexes = get_captured_indexes(
        indexed_frames,
        args.init_frames,
        args.d_threshold,
        args.max_threshold,
        args.min_threshold,
        args.use_knn,
        args.fast
    )
    slides = extract_indexes(args.input, slides_indexes)
    unique_indexes = get_unique_indexes(slides, args.hash_threshold)
    convert_to_pdf(args.output, slides, unique_indexes)
