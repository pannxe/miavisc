from argparse import ArgumentParser
from imagehash import dhash
from itertools import islice
from functools import partial
from io import BytesIO
from PIL import Image
from img2pdf import convert

import imageio.v3 as iio

def convert_to_pdf(output_path, pages):
    with open(output_path, "wb") as f:
        f.write(convert([page.read() for page in pages]))
        

def parse_video(input_path: str, output_path:str, check_per_sec: int, threshold: int):    
    metadata = iio.immeta(input_path, plugin="pyav")
    step =  1    if check_per_sec == 0 else\
            max(int(metadata["fps"] / check_per_sec), 1)

    def hash_image(img):
        return dhash(Image.open(img))

    pages = []
    prev_hashes = []

    def exist_close_hash(current_hash):
        for prev_hash in prev_hashes:
            if prev_hash - current_hash <= threshold:
                return True
        return False
    
    extension = ".jpg"
    write_image = partial(iio.imwrite, plugin="pillow", extension=extension)

    for frame in islice(iio.imiter(input_path, plugin="pyav"), None, None, step):
        img_path = BytesIO()

        if not prev_hashes:
            write_image(img_path, frame)
            prev_hashes.append(hash_image(img_path))

            img_path.seek(0)
            pages.append(img_path)
            continue

        write_image(img_path, frame)

        img_path.seek(0)
        current_hash = hash_image(img_path)
        
        same_image = exist_close_hash(current_hash)
        if same_image:
            continue
        
        img_path.seek(0)
        pages.append(img_path)
        prev_hashes.append(current_hash)
    
    return pages


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
    args = arg_parser.parse_args()

    pages = parse_video(args.input, args.output, args.check_per_sec, args.threshold)
    convert_to_pdf(args.output, pages)
