# Miavisc is a Video → Slide Converter

![alt text](img/image.png)

Born out of my frustration, this tool will convert video of a lecture to pdf file at a **blazzingly fast speed** 🚀 (sarcasm intended).

Key features includes:

- [x] **Blezzingly fast 🚀** — compare to other similar programs[^3], Miavisc is **> 10x faster**[^4] while producing comparable result[^5].
- [x] **Tunable similarity threshold** — so slightly different frame due to mouse movement / lazer pointers are not treated as different page
- [x] **Selectable ignored area** — only process centre portion area (to ignore camera, etc.)

[^3]: That I have tried (e.g., those in reference section).
[^4]: Miavisc at 2:11 min. vs [binh234/video2slides](https://github.com/binh234/video2slides) at 22:08 min. Tested on Macbook Air M2, 512 GB SSD, 16 GM memory using 1280x720 @ 30fps, mp4, 1:11 hr lecture. Using GMG algorithm, no skip frames.
[^5]:
    Overall, results from both programs are very usable without any significant difference (extra or missing slides here and there).
    Both requires some further manual processing (e.g., delete residual duplications).
    Note that this evaluation is **SUBJECTIVE** to the creator of this program and thus should be taken with a grain of salt.

To any professors out there, for the love of capybara and all is that holy in the world, **PLEASE PROVIDE PDF OF YOUR LECTURE VIDEO** 🔥🔥

## Installation

```bash
git clone https://github.com/pannxe/miavisc.git
cd miavisc
pip install .
```

Or download the pre-build version and run `pip install miavisc-x.x.x.tar.gz`.

## Usage

It is recommend that you use `--fast --concurrent` (shortern to `-fc`) almost **without exception**.

```bash
# Default
miavisc -fc -i <PATH_TO_VIDEO> -o <PATH_TO_PDF>
```

If you want to speed thing up even more, add `--knn` (`-k`) should not change the final result significantly but you will gain about 2-3x speed !! 🚀

```bash
# Extra fast, you see what I did there *wink*.
miavisc -fck -i <PATH_TO_VIDEO> -o <PATH_TO_PDF>
```

## Brenchmark

Tested on Macbook Air M2, 512 GB SSD, 16 GM memory using 1280x720 @ 30fps, mp4, 1:11 hr lecture.

As `--check_per_sec` goes up, risk of page-loss increases but false triggers also decreases. Sweet spot seem to be around 10.

Using GMG algorithm might give you somewhat better result but KKN is faster especially with large `--check_per_sec`.

| Options                  | Exec time | Diff     |
| ------------------------ | --------- | -------- |
| `-f`                     | 6:15 min  | Base     |
| `-f --check_per_sec 10`  | 2:45 min  | `-56.3%` |
| `-fk`                    | 2:44 min  | `-56.3%` |
| `-fk --check_per_sec 10` | 1:29 min  | `-76.3%` |

Here's is what without `--fast` look like:

| Options                 | Exec time | Diff      |
| ----------------------- | --------- | --------- |
| `--check_per_sec 10`    | 3:23 min  | `+23.8%`  |
| `-k --check_per_sec 10` | 6:28 min  | `+335.1%` |

**Update**: Now, with `--concurrent` (or `-c`) you can speed thing up even more! Here's what  `-c` look like compare to without one

| Options | Exec time | Diff    |
| ------- | --------- | ------- |
| `-fc`   | 2:11 min  | `+286%` |
| `-fck`  | 1:25 min  | `+192%` |

## Author

- [pannxe](https://github.com/pannxe) — Original author

## References

- Kunal Dawn. (2023). **Build a Video to Slides Converter Application using the Power of Background Estimation and Frame Differencing in OpenCV**. _LearnOpenCV_. Accessed April 1st, 2025. [Link](https://learnopencv.com/video-to-slides-converter-using-background-subtraction/).
- [binh234/video2slides](https://github.com/binh234/video2slides) — This project is inspired by this program and a lot of references are taken from this work.
  Any comparison to this program is purely educational and mean no offense to its author.
