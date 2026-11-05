# Image and Video Processing

Ten reproducible Python experiments using OpenCV, NumPy, Matplotlib, and
FFmpeg. This repository contains only source code and documentation: no bundled
images, videos, reports, saved results, or web frontend.

The command-line runner generates an original test image and a silent
320 × 240 video (72 frames at 24 fps), processes them, and saves measurements.
No downloads, accounts, or API keys are needed to run the experiments.

## Experiments

1. Edge detection
2. Lossy and lossless image compression
3. Image segmentation and seeded region growing
4. Video frame extraction
5. Temporal sampling
6. Geometric transformations
7. Frame differencing
8. Spatial filtering
9. H.264 video compression and quality measurement
10. Fixed-bitrate H.264 encodings and HLS over localhost HTTP

## Setup and run

Install Python 3.10+ and FFmpeg, including `ffprobe`, on your PATH.
On macOS, `brew install ffmpeg`; on Ubuntu/Debian,
`sudo apt install ffmpeg python3-venv`.

```sh
git clone https://github.com/dheeraj-juvvadi/image-video-processing.git
cd image-video-processing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python assignment.py
python validate.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
The FFmpeg build must include the `libx264` encoder.

The runner creates `inputs/` and `outputs/`. These generated directories are
ignored by Git. Rerunning the runner overwrites its generated results.
The validation script independently checks decoded pixels, frame counts,
sampling order and rate, transformations, filtering, and HTTP delivery.

## Compression on your own video

```sh
python video_compression.py --input /path/to/video.mp4 --output /path/to/results
```

Use a short clip with even pixel dimensions. The implementation decodes frames
into memory, so long or high-resolution inputs can require substantial RAM.
The default target bitrates are 100, 300, and 800 kbps. To supply other rates
from Python:

```python
from video_compression import run
run("clip.mp4", "results", bitrates=(64, 300, 2500))
```

MSE uses all aligned decoded BGR pixels, and PSNR is derived from pooled MSE.
HLS is actually served and decoded over loopback HTTP. This is a local
fixed-bitrate demonstration; it does not simulate congestion, adaptive
streaming, or internet playback latency. Codec builds can change the results.

See [compression_notes.txt](compression_notes.txt) for measurement details
and manual HLS replay commands.

## Files

- `assignment.py`: generated inputs, experiments 1–8, and the complete runner.
- `video_compression.py`: experiments 9–10 and a standalone CLI.
- `validate.py`: independent validation of the generated results.
- `requirements.txt`: Python dependencies.
