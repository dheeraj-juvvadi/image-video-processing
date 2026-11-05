"""Tasks 9–10: H.264 compression, quality measurement, and local HTTP HLS streaming.

Requires Python packages numpy/OpenCV and FFmpeg executables on PATH.
Run: python video_compression.py --input inputs/original.avi --output outputs
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.request

import cv2
import numpy as np


def command(args):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"Command failed: {args!r}\n{result.stderr.decode(errors='replace')}")
    return result.stdout


def decode(path):
    # Use one FFmpeg decoder/converter for local and HTTP inputs. Mixing OpenCV
    # and system FFmpeg builds can produce small chroma-conversion differences.
    metadata = json.loads(command([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate", "-of", "json", str(path)]))
    stream = metadata["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    fps = numerator / denominator if denominator else 0
    if fps <= 0:
        raise RuntimeError(f"Invalid frame rate: {path}")
    raw = command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
                   "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, stream["height"], stream["width"], 3)
    if not len(frames):
        raise RuntimeError(f"No frames: {path}")
    return frames, fps


def quality(reference, reconstructed):
    if reference.shape != reconstructed.shape:
        raise ValueError(f"Unaligned videos: {reference.shape} != {reconstructed.shape}")
    squared_error = 0.0
    for a, b in zip(reference, reconstructed):
        delta = a.astype(np.float64) - b.astype(np.float64)
        squared_error += float(np.sum(delta * delta))
    mse = squared_error / reference.size
    return {"mse": mse, "psnr_db": "Infinity" if mse == 0 else 10 * math.log10(255 ** 2 / mse)}


def save_comparison(path, images, labels):
    panels = []
    for frame, label in zip(images, labels):
        panel = cv2.copyMakeBorder(frame, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(panel)
    if not cv2.imwrite(str(path), np.hstack(panels)):
        raise IOError(f"Could not write {path}")


def encode(source, target, options):
    command(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
             "-an", "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
             *options, "-movflags", "+faststart", str(target)])


class StreamHandler(SimpleHTTPRequestHandler):
    """Silence terminal logging; retain requested asset paths as execution evidence."""
    requests = None

    def log_message(self, fmt, *args):
        if self.requests is not None:
            self.requests.append({"path": self.path, "message": fmt % args})


def run(input_path, output_root, bitrates=(100, 300, 800)):
    if not all(shutil.which(name) for name in ("ffmpeg", "ffprobe")):
        raise RuntimeError("Install FFmpeg (including ffprobe) and add it to PATH.")
    source = Path(input_path).resolve()
    root = Path(output_root).resolve()
    compression = root / "09_video_compression"
    streaming = root / "10_bitrate_streaming"
    compression.mkdir(parents=True, exist_ok=True)
    streaming.mkdir(parents=True, exist_ok=True)
    original, fps = decode(source)
    n, height, width, _ = original.shape
    if width % 2 or height % 2:
        raise ValueError("H.264 yuv420p requires even width and height for this experiment.")
    duration = n / fps
    compressed = compression / "compressed_crf28.mp4"
    encode(source, compressed, ["-crf", "28"])
    reconstructed, reconstructed_fps = decode(compressed)
    if abs(fps - reconstructed_fps) > 0.01:
        raise ValueError("Compression unexpectedly changed frame rate")
    comparison = {
        "codec": "H.264/libx264, CRF 28, yuv420p",
        "original_bytes": source.stat().st_size,
        "compressed_bytes": compressed.stat().st_size,
        "compression_ratio": source.stat().st_size / compressed.stat().st_size,
        "size_reduction_percent": 100 * (1 - compressed.stat().st_size / source.stat().st_size),
        "frames_compared": n, "fps": fps, "duration_seconds": duration,
        "measurement": "MSE across all aligned decoded BGR pixels; PSNR from pooled MSE",
        **quality(original, reconstructed),
    }
    index = n // 2
    cv2.imwrite(str(compression / "reconstructed_frame.png"), reconstructed[index])
    save_comparison(compression / "comparison.png", [original[index], reconstructed[index]],
                    ["Original (lossless reference)", "H.264 CRF 28"])
    (compression / "metrics.json").write_text(json.dumps(comparison, indent=2))

    results = []
    del reconstructed
    previews = [original[index].copy()]
    labels = ["Original"]
    for bitrate in bitrates:
        folder = streaming / f"{bitrate}kbps"
        folder.mkdir(exist_ok=True)
        mp4 = folder / "video.mp4"
        encode(source, mp4, ["-b:v", f"{bitrate}k", "-minrate", f"{bitrate}k",
                            "-maxrate", f"{bitrate}k", "-bufsize", f"{2 * bitrate}k",
                            "-x264-params", "nal-hrd=cbr:force-cfr=1", "-g", str(round(fps)),
                            "-keyint_min", str(round(fps)), "-sc_threshold", "0"])
        playlist = folder / "stream.m3u8"
        command(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(mp4),
                 "-c", "copy", "-f", "hls", "-hls_time", "1", "-hls_playlist_type", "vod",
                 "-hls_segment_filename", str(folder / "segment_%03d.ts"), str(playlist)])
        frames, encoded_fps = decode(mp4)
        assert abs(encoded_fps - fps) < 0.01
        stream_bytes = sum(p.stat().st_size for p in folder.glob("*.ts")) + playlist.stat().st_size
        results.append({"target_video_bitrate_kbps": bitrate, "mp4_bytes": mp4.stat().st_size,
                        "actual_mp4_total_bitrate_kbps": mp4.stat().st_size * 8 / duration / 1000,
                        "hls_assets_bytes": stream_bytes, "frames_compared": n,
                        **quality(original, frames)})
        previews.append(frames[index].copy())
        del frames
        labels.append(f"H.264 {bitrate} kbps target")

    request_log = []
    handler = type("RecordingHandler", (StreamHandler,), {"requests": request_log})
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(handler, directory=str(streaming)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        for row in results:
            bitrate = row["target_video_bitrate_kbps"]
            url = f"{base}/{bitrate}kbps/stream.m3u8"
            started = time.perf_counter()
            raw = command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", url,
                           "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
            elapsed = time.perf_counter() - started
            streamed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
            row["http_hls_decoded_frames"] = len(streamed)
            row["http_hls_decode_seconds"] = elapsed
            row["http_hls_quality"] = quality(original, streamed)
            local, _ = decode(streaming / f"{bitrate}kbps/video.mp4")
            row["http_hls_matches_local_decode"] = bool(np.array_equal(local, streamed))
            if not row["http_hls_matches_local_decode"]:
                raise AssertionError("HTTP HLS decode differs from local encoded video")
            # Check an independent progressive MP4 transfer as a byte-integrity test.
            with urllib.request.urlopen(f"{base}/{bitrate}kbps/video.mp4", timeout=10) as response:
                received = response.read()
            expected = (streaming / f"{bitrate}kbps/video.mp4").read_bytes()
            row["http_mp4_received_bytes"] = len(received)
            row["http_mp4_sha256"] = hashlib.sha256(received).hexdigest()
            if received != expected:
                raise AssertionError("HTTP transfer changed bytes")
            cv2.imwrite(str(streaming / f"{bitrate}kbps/http_decoded_frame.png"), streamed[index])
            del streamed, local, raw
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
    save_comparison(streaming / "comparison.png", previews, labels)
    summary = {"method": "Fixed target CBR H.264 encodings; HLS VOD over actual localhost HTTP; FFmpeg decodes all streamed frames",
               "limitations": "Localhost, no congestion or packet-loss simulation; decode time is not playback delay. Total MP4 bitrate includes container overhead and may differ from the target video bitrate. No adaptive bitrate switching.",
               "duration_seconds": duration, "fps": fps, "variants": results}
    (streaming / "metrics.json").write_text(json.dumps(summary, indent=2))
    (streaming / "http_requests.json").write_text(json.dumps(request_log, indent=2))
    return {"09_video_compression": comparison, "10_bitrate_streaming": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("inputs/original.avi"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), indent=2))
