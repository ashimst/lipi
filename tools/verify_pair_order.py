"""
tools/verify_pair_order.py

Checks which member of each glyph pair is the original. The augmentation script
saved each original first and its augmented copy immediately after, choosing one
of seven transforms at random (blur, Gaussian noise sigma=30, contrast,
brightness, sharpness, elastic deformation, erosion/dilation).

Gaussian noise (sigma=30) leaves a large high-frequency residual. If the first
member of each pair is the original, that signature should appear in about 1/7
of second members and almost never in first members. Writes
runs/pair_order_check.json.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

pairs = json.load(open('data/pairs.json'))
pairs = pairs.get('pairs', pairs)


def noise_level(path):
    a = np.asarray(Image.open(path).convert('L'), dtype=np.float32)
    lap = ndimage.laplace(a)
    return float(np.median(np.abs(lap - np.median(lap))))  # robust high-frequency level


first, second = [], []
for cls, plist in pairs.items():
    for p in plist:
        a, b = p
        first.append(noise_level(Path('data/canonical') / cls / a if not Path(a).exists() else a))
        second.append(noise_level(Path('data/canonical') / cls / b if not Path(b).exists() else b))
first, second = np.array(first), np.array(second)
thr = np.percentile(np.concatenate([first, second]), 50) * 4
res = {
    'pairs': int(len(first)),
    'threshold': round(float(thr), 2),
    'first_member_noisy_pct': round(100 * float((first > thr).mean()), 2),
    'second_member_noisy_pct': round(100 * float((second > thr).mean()), 2),
    'median_first': round(float(np.median(first)), 2),
    'median_second': round(float(np.median(second)), 2),
}
print(res)
json.dump(res, open('runs/pair_order_check.json', 'w'), indent=1)
