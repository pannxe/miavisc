from argparse import ArgumentParser
from imagehash import dhash
from itertools import islice
from functools import partial
from io import BytesIO
from PIL import Image
from img2pdf import convert
from tqdm.auto import tqdm
from math import ceil

import imageio.v3 as iio


def convert_to_pdf(output_path, pages):
    with open(output_path, "wb") as f:                             
        # f.write(convert([page.read() for page in pages]))
        f.write(convert(pages))


def parse_video(
    input_path: str,
    output_path: str,
    check_per_sec: int,
    threshold: int,
    fast: int
):
    metadata = iio.immeta(input_path, plugin="pyav")

    step = 1 if check_per_sec == 0 else\
        max(int(metadata["fps"] / check_per_sec), 1)
    
    total_frames = ceil(metadata["duration"] * metadata["fps"] / step)

    page_indexes = []
    prev_hashes = []

    def hash_image(img):
        return dhash(Image.open(img))

    def exist_close_hash(current_hash):
        for prev_hash in prev_hashes:
            if prev_hash - current_hash <= threshold:
                return True
        return False

    _extension = ".jpg"
    write_image = partial(iio.imwrite, plugin="pillow", extension=_extension)

    _filter = [("scale", "0.25*in_w:0.25*in_h")] if fast else []
    _thread_type = "FRAME" if fast else "SLICE"
    frames = enumerate(iio.imiter(
        input_path,
        plugin="pyav",
        thread_type=_thread_type,
        filter_sequence=_filter 
    ))
    for index, frame in tqdm(islice(frames, 0, None, step), total=total_frames, desc="Parsing Video "):
        img_path = BytesIO()

        if not prev_hashes:
            write_image(img_path, frame)
            prev_hashes.append(hash_image(img_path))

            page_indexes.append(index)
            continue

        write_image(img_path, frame)

        img_path.seek(0)
        current_hash = hash_image(img_path)

        same_image = exist_close_hash(current_hash)
        if same_image:
            continue

        page_indexes.append(index)
        prev_hashes.append(current_hash)
    
    print(f"\nFound {len(page_indexes)} potentially unique slide(s).\n")

    with iio.imopen(input_path, "r", plugin="pyav") as vid:
        images = [vid.read(index=idx) for idx in tqdm(page_indexes, desc="Getting Images")]
    
    def frame_to_bytes(frame):
        bio = BytesIO()
        write_image(bio, frame)
        bio.seek(0)
        return bio.read()

    return [frame_to_bytes(img) for img in images]


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
        type=int, default=3,
        help="How many frame to process in 1 sec. (0 = no skip frame) Default = 3"
    )
    arg_parser.add_argument(
        "--fast",
        type=int, default=1,
        help="Use various hacks to speed up the process (might affect the final result)."
    )
    args = arg_parser.parse_args()

    pages = parse_video(
        args.input,
        args.output,
        args.check_per_sec,
        args.threshold,
        args.fast
    )
    convert_to_pdf(args.output, pages)
