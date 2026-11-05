"""Reproducible implementations of image/video processing problems 1–8.
Run: python assignment.py (also runs problems 9–10).
"""
from pathlib import Path
import json
import math
import subprocess
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
FPS, N, W, H = 24, 72, 320, 240


def metrics(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return {'mse': mse, 'psnr_db': 'infinity' if mse == 0 else 10 * math.log10(255 ** 2 / mse)}


def save(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f'Cannot write {path}')


def panel(path, images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 3.5), squeeze=False)
    for ax, im, title in zip(axes[0], images, titles):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im,
                  cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_video(path, frames, fps=FPS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError('FFV1 VideoWriter unavailable')
    for frame in frames:
        writer.write(frame)
    writer.release()


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f'No decoded frames in {path}')
    return frames


def make_inputs():
    """Original procedural test media; no downloads or third-party licensing."""
    folder = ROOT / 'inputs'
    folder.mkdir(exist_ok=True)
    yy, xx = np.mgrid[:H, :W]
    base = np.stack([30 + xx // 3, 35 + yy // 3, 45 + xx // 5], axis=2).astype(np.uint8)
    # Fine texture and straight/slanted edges expose compression and edge differences.
    base[150:215, 190:300] = np.where(((xx[150:215, 190:300] // 3 + yy[150:215, 190:300] // 3) % 2)[..., None], 185, 65)
    frames = []
    for i in range(N):
        frame = base.copy()
        cv2.rectangle(frame, (20, 25), (110, 100), (80, 190, 225), -1)
        cv2.circle(frame, (155 + i, 90), 30, (220, 225, 230), -1)
        cv2.line(frame, (15, 220), (175, 135), (245, 245, 245), 3)
        cv2.putText(frame, f'IVP {i:02d}', (15, 130), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        frames.append(frame)
    save(folder / 'image.png', frames[0])
    write_video(folder / 'original.avi', frames)
    return frames


def region_grow(gray, seed, tolerance=12):
    """8-connected region whose intensities differ from the seed by <= tolerance."""
    mask = np.zeros(gray.shape, np.uint8)
    y, x = seed
    allowed = np.abs(gray.astype(np.int16) - int(gray[y, x])) <= tolerance
    stack = [(y, x)]
    mask[y, x] = 255
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < gray.shape[0] and 0 <= nx < gray.shape[1] and allowed[ny, nx] and not mask[ny, nx]:
                    mask[ny, nx] = 255
                    stack.append((ny, nx))
    return mask


def run_image_tasks(image):
    result = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    d = OUT / '01_edges'; d.mkdir(parents=True, exist_ok=True)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    sobel = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    canny = cv2.Canny(blur, 60, 140, L2gradient=True)
    save(d / 'sobel.png', sobel); save(d / 'canny.png', canny)
    panel(d / 'comparison.png', [gray, sobel, canny], ['Grayscale', 'Sobel magnitude (normalized)', 'Canny: 60 / 140'])
    result['01_edges'] = {'canny_edge_pixels': int(np.count_nonzero(canny)), 'sobel_max_raw': float(magnitude.max())}

    d = OUT / '02_image_compression'; d.mkdir(exist_ok=True)
    save(d / 'reference.bmp', image)
    cv2.imwrite(str(d / 'lossless.png'), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(str(d / 'lossy_q25.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 25])
    png, jpg = [cv2.imread(str(d / f)) for f in ['lossless.png', 'lossy_q25.jpg']]
    result['02_image_compression'] = {name: {**metrics(image, decoded), 'bytes': (d / name).stat().st_size} for name, decoded in [('lossless.png', png), ('lossy_q25.jpg', jpg)]}
    result['02_image_compression']['reference_bytes'] = (d / 'reference.bmp').stat().st_size
    panel(d / 'comparison.png', [image, png, jpg], ['Original', 'PNG: lossless reconstruction', 'JPEG: quality 25'])

    d = OUT / '03_segmentation'; d.mkdir(exist_ok=True)
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    region = region_grow(gray, (60, 60), 12)
    save(d / 'otsu.png', binary); save(d / 'region.png', region)
    panel(d / 'comparison.png', [gray, binary, region], ['Grayscale', f'Otsu threshold = {threshold:.0f}', 'Region growing: seed (60,60)'])
    result['03_segmentation'] = {'otsu_threshold': threshold, 'otsu_foreground_pixels': int(np.count_nonzero(binary)), 'region_pixels': int(np.count_nonzero(region)), 'seed_yx': [60, 60], 'tolerance': 12}
    return result


def run_video_tasks(frames):
    result = {}
    d = OUT / '04_frame_extraction'; d.mkdir(exist_ok=True)
    decoded = read_video(ROOT / 'inputs/original.avi')
    for i, frame in enumerate(decoded):
        save(d / f'frame_{i:04d}.png', frame)
    panel(d / 'comparison.png', [decoded[i] for i in [0, 24, 48]], ['Frame 0 / 0 s', 'Frame 24 / 1 s', 'Frame 48 / 2 s'])
    result['04_frame_extraction'] = {'frames_saved': len(decoded), 'fps': FPS}

    d = OUT / '05_temporal_sampling'; d.mkdir(exist_ok=True)
    sampled = decoded[::3]
    write_video(d / 'sampled_8fps.avi', sampled, 8)
    # Duplicate retained frames only for an aligned 24fps comparison of temporal error.
    held = [sampled[i // 3] for i in range(N)]
    write_video(d / 'sampled_hold_24fps.avi', held)
    panel(d / 'comparison.png', [decoded[2], held[2], cv2.absdiff(decoded[2], held[2])], ['Original at t=2/24 s', '8fps: previous retained frame', 'Temporal difference'])
    result['05_temporal_sampling'] = {'original_fps': FPS, 'sampled_fps': 8, 'original_frames': N, 'sampled_frames': len(sampled), 'duration_seconds_both': N / FPS, 'hold_reconstruction': metrics(np.asarray(decoded), np.asarray(held))}

    d = OUT / '06_transformations'; d.mkdir(exist_ok=True)
    translation = np.float32([[1, 0, 35], [0, 1, 20]])
    rotation = cv2.getRotationMatrix2D((W / 2, H / 2), 25, 1)
    scaling = cv2.getRotationMatrix2D((W / 2, H / 2), 0, .65)
    samples = [decoded[0]]
    for name, matrix in [('translated', translation), ('rotated', rotation), ('scaled', scaling)]:
        transformed = [cv2.warpAffine(f, matrix, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) for f in decoded]
        write_video(d / f'{name}.avi', transformed)
        save(d / f'{name}.png', transformed[0]); samples.append(transformed[0])
    panel(d / 'comparison.png', samples, ['Original', 'Translation (+35,+20)', 'Rotation 25 degrees', 'Scale 0.65 about center'])
    result['06_transformations'] = {'translation_xy': [35, 20], 'rotation_degrees': 25, 'scale': .65, 'canvas': [W, H]}

    d = OUT / '07_frame_difference'; d.mkdir(exist_ok=True)
    a, b = decoded[0], decoded[12]
    diff = cv2.absdiff(a, b)
    graydiff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(graydiff, 20, 255, cv2.THRESH_BINARY)
    save(d / 'difference.png', diff); save(d / 'motion_mask.png', mask)
    panel(d / 'comparison.png', [a, b, diff, mask], ['Frame 0', 'Frame 12', 'Absolute difference', 'Motion mask > 20'])
    result['07_frame_difference'] = {**metrics(a, b), 'changed_pixels': int(np.count_nonzero(mask)), 'frame_indices': [0, 12]}

    d = OUT / '08_spatial_filtering'; d.mkdir(exist_ok=True)
    rng = np.random.default_rng(480)
    noisy, filtered = [], []
    for frame in decoded:
        damaged = frame.copy()
        r = rng.random((H, W))
        damaged[r < .025] = 0
        damaged[r > .975] = 255
        noisy.append(damaged)
        filtered.append(cv2.medianBlur(damaged, 3))
    write_video(d / 'noisy.avi', noisy)
    write_video(d / 'median_filtered.avi', filtered)
    panel(d / 'comparison.png', [decoded[0], noisy[0], filtered[0]], ['Clean reference', 'Salt & pepper noise: 5%', 'Median filter: 3 x 3'])
    result['08_spatial_filtering'] = {'noise': 'salt-and-pepper, probability 0.05', 'rng_seed': 480, 'noisy': metrics(np.asarray(decoded), np.asarray(noisy)), 'filtered': metrics(np.asarray(decoded), np.asarray(filtered))}
    return result


def main():
    OUT.mkdir(exist_ok=True)
    log = []
    def record(message):
        print(message, flush=True)
        log.append(message)
        (OUT / 'execution.log').write_text('\n'.join(log) + '\n')
    record('Image and Video Processing | All 10 experiments')
    record(f'Python/OpenCV {cv2.__version__}; NumPy {np.__version__}')
    frames = make_inputs()
    record(f'INPUT: original procedural image + {N} frames, {W}x{H}, {FPS} fps')
    result = run_image_tasks(frames[0])
    for key in result: record(f'PASS {key}: {json.dumps(result[key])}')
    video = run_video_tasks(frames)
    result.update(video)
    for key in video: record(f'PASS {key}: {json.dumps(video[key])}')
    from video_compression import run
    compressed = run(ROOT / 'inputs/original.avi', OUT)
    result.update(compressed)
    for key in compressed: record(f'PASS {key}: metrics saved in outputs/metrics.json')
    (OUT / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    record('COMPLETE: 10 / 10 experiments; input, output and metrics files saved.')


if __name__ == '__main__':
    main()
