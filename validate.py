"""Validate saved deliverables independently of the processing pipeline."""
from pathlib import Path
import json
import cv2
import numpy as np
from assignment import read_video, metrics, region_grow

root = Path(__file__).resolve().parent
out = root / 'outputs'
m = json.loads((out / 'metrics.json').read_text())
assert len(m) == 10
image = cv2.imread(str(root / 'inputs/image.png'))
png = cv2.imread(str(out / '02_image_compression/lossless.png'))
assert np.array_equal(image, png), 'PNG must reconstruct exactly'
jpg = cv2.imread(str(out / '02_image_compression/lossy_q25.jpg'))
assert abs(metrics(image, jpg)['mse'] - m['02_image_compression']['lossy_q25.jpg']['mse']) < 1e-10
original = read_video(root / 'inputs/original.avi')
extracted = sorted((out / '04_frame_extraction').glob('frame_*.png'))
assert len(original) == len(extracted) == 72
assert all(np.array_equal(f, cv2.imread(str(p))) for f, p in zip(original, extracted))
sampled = read_video(out / '05_temporal_sampling/sampled_8fps.avi')
assert len(sampled) == 24
assert all(np.array_equal(a, b) for a, b in zip(original[::3], sampled))
cap = cv2.VideoCapture(str(out / '05_temporal_sampling/sampled_8fps.avi'))
assert cap.get(cv2.CAP_PROP_FPS) == 8
cap.release()
for name in ['translated', 'rotated', 'scaled']:
    frames = read_video(out / f'06_transformations/{name}.avi')
    assert len(frames) == 72 and all(f.shape == (240, 320, 3) for f in frames)
assert np.array_equal(cv2.imread(str(out / '07_frame_difference/difference.png')), cv2.absdiff(original[0], original[12]))
assert m['08_spatial_filtering']['filtered']['mse'] < m['08_spatial_filtering']['noisy']['mse']
# A disconnected patch with the same intensity must not join a seeded region.
a = np.array([[100, 100, 0, 100], [100, 100, 0, 100]], np.uint8)
assert np.count_nonzero(region_grow(a, (0, 0), 0)) == 4
for v in m['10_bitrate_streaming']['variants']:
    assert v['http_hls_decoded_frames'] == 72 and v['http_hls_matches_local_decode']
    assert v['http_mp4_received_bytes'] == v['mp4_bytes']
assert len(list(out.glob('*/comparison.png'))) == 10
print('PASS: ten tasks; exact PNG and extracted-frame equality; sampling order/fps;')
print('transformation frames; absolute difference; filter improvement; region connectivity;')
print('HTTP HLS decoding/integrity; ten comparison panels.')
