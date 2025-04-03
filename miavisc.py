from argparse import ArgumentParser
from itertools import islice
from math import ceil
from io import BytesIO

import imageio.v3 as iio
import cv2
from tqdm import tqdm
from img2pdf import convert
from imagehash import dhash
from PIL import Image

fast = False

def frame_to_bytes(frame):
    bio = BytesIO()
    iio.imwrite(bio, frame, plugin="pillow", extension=".jpg")
    bio.seek(0)
    return bio

def get_indexed_frames(
    input_path: str,
    check_per_sec: int,
    crop_zoom: str,
    scale: str
):
    metadata = iio.immeta(input_path, plugin="pyav")
    fps, duration = metadata["fps"], metadata["duration"]

    step = int(max(fps / check_per_sec, 1)) if check_per_sec else 1
    n_frames = ceil(duration * fps / step)

    filters = [fil for opt, fil in (
        (fast, ("scale", f"{scale}*in_w:{scale}*in_h")),
        (crop_zoom, ("crop", f"{crop_zoom}*in_w:{crop_zoom}*in_h"))
    ) if opt]

    thread_type = "FRAME" if fast else "SLICE"

    indexed_frames = enumerate(iio.imiter(
        input_path,
        plugin="pyav",
        thread_type=thread_type,
        filter_sequence=filters 
    ))
    return n_frames, islice(indexed_frames, 1, None, step)


def capture_slides(n_frames, indexed_frames):
    prev_hashes = []

    def is_unique_hash(slide):
        # only checking if `--fast` is on
        if fast:
            return True
        
        HASH_THRESHOLD = 2

        slide_bytes = frame_to_bytes(slide)
        current_hash = dhash(Image.open(slide_bytes), hash_size=16)

        is_unique = not any(prev_hash - current_hash <= HASH_THRESHOLD for prev_hash in prev_hashes)
        if is_unique:
            prev_hashes.append(current_hash)
        
        return is_unique

    history = 15
    decision_threshold = 0.75
    max_threshold = 0.15
    min_threshold = 0.01

    captured = False

    bg_subtrator = cv2.bgsegm.createBackgroundSubtractorGMG(
        initializationFrames=history,
        decisionThreshold=decision_threshold
    )

    # Always include 1st frame
    capture_indexes = [0]

    for index, frame in tqdm(indexed_frames, desc="Parsing Video ", total=n_frames):
        fg_mask = bg_subtrator.apply(frame)
        percent_non_zero = 100 * cv2.countNonZero(fg_mask) / (1.0 * fg_mask.size)

        if percent_non_zero < max_threshold and not captured:
            # with `--fast`, perform a rolling rough hash so we don't have to extract so many frames later.
            if is_unique_hash(frame):
                continue
            captured = True
            capture_indexes.append(index)

        if captured and percent_non_zero >= min_threshold:
            captured = False
    
    print(f"\nFound potentially {len(capture_indexes)} unique slides.\n")

    return capture_indexes


def extract_indexes(input_path, indexes):    
    with iio.imopen(input_path, "r", plugin="pyav") as vid:
        images = [vid.read(index=i) for i in tqdm(indexes, desc="Getting Images")]
    return [frame_to_bytes(img).read() for img in images]


def get_unique_indexes(slides):
    HASH_THRESHOLD = 4
    
    unique_indexes = []
    prev_hashes = []
    
    for i, slide in enumerate(tqdm(slides, desc="Removing dups  ")):
        current_hash = dhash(Image.open(BytesIO(slide)))
        if any(prev_hash - current_hash <= HASH_THRESHOLD for prev_hash in prev_hashes):
            continue
        unique_indexes.append(i)
        prev_hashes.append(current_hash)

    print(f"\nAfter further checking, found {len(unique_indexes)} unique slides.\n")

    return unique_indexes


def convert_to_pdf(output_path, slides, unique_indexes):
    with open(output_path, "wb") as f:                             
        f.write(convert([slides[i] for i in unique_indexes]))
    
    print("Finished making PDF file.")

if __name__ == "__main__":
    arg_parser = ArgumentParser(
        description="Miavisc is a video to slide converter.",
    )
    arg_parser.add_argument(
        "--input",
        type=str, required=True,
        help="Path to input video file")
    arg_parser.add_argument("--output",
                            type=str, required=True,
                            help="Path to input video file"
                            )
    arg_parser.add_argument(
        "--threshold",
        type=float, default=4,
        help="Threshold for treating different frames as different pages."
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

    fast = args.fast

    n_frames, indexed_frames = get_indexed_frames(
        args.input,
        args.check_per_sec,
        args.crop_zoom,
        args.process_scale
    )

    slides_indexes = capture_slides(n_frames, indexed_frames)
    slides = extract_indexes(args.input, slides_indexes)
    unique_indexes = get_unique_indexes(slides)
    convert_to_pdf(args.output, slides, unique_indexes)