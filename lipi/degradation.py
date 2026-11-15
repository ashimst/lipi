"""Synthetic damage applied to glyph images (dark ink on a light background).

Four corruptions, each with a severity in (0, 1], and their combination:
    erosion       strokes thinned morphologically, plus random pixel dropout
    ink_bleed     blend with a Gaussian-blurred copy
    noise         additive Gaussian noise (rough stone, fibre grain)
    illumination  radial brightness falloff
    combined      all four, in that order

The pipeline is not calibrated against real damaged material. It is meant to
rank kinds of damage, not to predict accuracy on inscriptions.
"""

import cv2
import numpy as np
import torch
from PIL import Image

from .data import list_files

CORRUPTIONS = ("erosion", "ink_bleed", "noise", "illumination", "combined")
SEVERITIES = (0.25, 0.5, 0.75, 1.0)


def erosion(img, severity, rng):
    size = int(1 + severity * 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    out = cv2.dilate(img, kernel, iterations=1)          # dilating the light background thins dark strokes
    out[rng.rand(*img.shape) < severity * 0.15] = 255    # flakes
    return out


def ink_bleed(img, severity, rng):
    blurred = cv2.GaussianBlur(img, (0, 0), 0.5 + 2.0 * severity)
    alpha = 0.3 + 0.5 * severity
    return cv2.addWeighted(img, 1 - alpha, blurred, alpha, 0)


def noise(img, severity, rng):
    noisy = img.astype(np.float32) + rng.normal(0, 45.0 * severity, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def illumination(img, severity, rng):
    h, w = img.shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    falloff = np.clip(1.0 - 0.6 * severity * np.sqrt(xx ** 2 + yy ** 2) / np.sqrt(2), 0.2, 1.0)
    return np.clip(img.astype(np.float32) * falloff, 0, 255).astype(np.uint8)


def corrupt(img: np.ndarray, kind: str, severity: float, rng) -> np.ndarray:
    steps = [erosion, ink_bleed, noise, illumination] if kind == "combined" else [globals()[kind]]
    for step in steps:
        img = step(img, severity, rng)
    return img


def load_degraded(class_names, size: int, kind: str, severity: float) -> torch.Tensor:
    """The images of `class_names`, corrupted at native resolution and then resized.

    Returns uint8 [N, size, size] in the same order as data.load_glyphs. The random
    stream depends only on (kind, severity), so every model sees the same damage.
    """
    rng = np.random.RandomState(1000 * CORRUPTIONS.index(kind) + int(100 * severity))
    out = []
    for path, *_ in list_files(class_names):
        with Image.open(path) as im:
            damaged = corrupt(np.array(im.convert("L")), kind, severity, rng)
        out.append(np.asarray(Image.fromarray(damaged).resize((size, size), Image.BILINEAR)))
    return torch.from_numpy(np.stack(out))
