# Miavisc is a Video → Slide Converter

![alt text](img/image.png)

Born out of my frustration, this tool will convert video of a lecture to pdf file at a **blazzingly fast** 🚀 (sarcasm intended) speed. 

Key features includes:

- [x] **Tunable similarity threshold** — so slightly different frame due to mouse movement / lazer pointers are not treated as different page
- [x] **Selectable ignored area** — only process centre portion area (to ignore camera, etc.)

This project is inspired by 

To any professors out there, for the love of capybara and all is that holy in the world, **PLEASE PROVIDE PDF OF YOUR LECTURE VIDEO** 🔥🔥

## Dependencies
- Python 3.10 or newer
- `pip install opencv-contrib-python imagehash av img2pdf imageio tqdm`

## Brenchmark
Tested on Macbook Air M2, 512 GB SSD, 16 GM memory using 1280x720 @ 60fps, mp4, 1:30 hr lecture.

Using GMG algorithm:

- `--fast --check_per_sec 0` → 7 min. 57 sec.
- `--fast --check_per_sec 10` → 4 min. 4 sec.
- `--fast --check_per_sec 5` → 2 min. 4 sec.
- `--check_per_sec 5` → 18 min. 40 sec. (est.)

Using KNN algorithm:

- `--fast --knn --check_per_sec 0` → 4 min 21 secs.
- `--fast --knn --check_per_sec 10` → 2 min 26 secs.
- `--fast --knn --check_per_sec 5` → 1 min 32 secs.
- `--knn --check_per_sec 5` → 6 min. 10 sec. (est.)

As `--check_per_sec` goes up, risk of page-loss increases but false triggers also decreases. Sweet spot seem to be around 10.

Using GMG algorithm might give you somewhat better result but KKN is faster especially with large `--check_per_sec`. 


## Author
- [pannxe](https://github.com/pannxe) — Original author