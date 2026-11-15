"""Dataset access: where the files live, the splits, and image loading.

Layout of the data directory (override with the LIPI_DATA_DIR environment variable):

    data/canonical/<script>/<character>/<NNNN>.png   128x128 grayscale glyphs
    data/splits/*.json                               class lists per protocol
    data/dataset_summary.json                        characters of each script

Inside a class folder the files come in pairs: sorted position 2k is an original
glyph and 2k+1 is its augmented copy (see tools/augment_dataset.py). Everything
that samples episodes relies on this, so `GlyphSet` records it per image.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = ["brahmi", "lichhavi", "bhujimol", "nandinagari", "maithili", "tibetan", "kirat"]


def data_dir() -> Path:
    return Path(os.environ.get("LIPI_DATA_DIR", REPO / "data"))


def runs_dir() -> Path:
    return Path(os.environ.get("LIPI_RUNS_DIR", REPO / "runs"))


def script_classes(script: str) -> list[str]:
    """All classes of one script, as 'script/character' names."""
    summary = json.loads((data_dir() / "dataset_summary.json").read_text(encoding="utf-8"))
    return sorted(f"{script}/{c}" for c in summary["scripts"][script]["characters"])


def load_split(name: str) -> dict[str, list[str]]:
    """Return {'train': [...], 'val': [...], 'test': [...]} class lists.

    ancestry        train Brahmi + Licchavi, validate Nandinagari, test the other four
    loso_<script>   train on five scripts, validate on a sixth, <script> withheld
    single_<script> train on 80% of one script's classes, validate on the rest;
                    the test side is every other script (handled by the evaluation)
    """
    if name.startswith("single_"):
        script = name[len("single_"):]
        classes = script_classes(script)
        # the held-out fifth is fixed, so it does not change with the training seed
        val = sorted(random.Random(0).sample(classes, max(5, round(0.2 * len(classes)))))
        return {"train": [c for c in classes if c not in val], "val": val, "test": []}
    split = json.loads((data_dir() / "splits" / f"{name}.json").read_text(encoding="utf-8"))
    return {k: sorted(split[k]) for k in ("train", "val", "test")}


@dataclass
class GlyphSet:
    """A set of glyph images held in memory, with what the samplers need to know."""

    images: torch.Tensor       # uint8, [N, size, size]
    labels: list[int]          # class index per image
    pair_ids: list[int]        # images with the same (label, pair_id) are a glyph and its copy
    is_original: list[bool]
    class_names: list[str]

    def __len__(self):
        return len(self.labels)


def list_files(class_names: list[str]):
    """Yield (path, label, pair_id, is_original) for every image of the given classes."""
    root = data_dir() / "canonical"
    for label, name in enumerate(sorted(class_names)):
        files = sorted((root / name).glob("*.png"))
        if not files:
            raise FileNotFoundError(f"no images for class '{name}' under {root} "
                                    f"(is the dataset unpacked? see README)")
        for position, path in enumerate(files):
            yield path, label, position // 2, position % 2 == 0


def read_gray(path, size: int) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("L").resize((size, size), Image.BILINEAR)


def load_glyphs(class_names: list[str], size: int) -> GlyphSet:
    rows = list(list_files(class_names))
    images = torch.from_numpy(np.stack([np.asarray(read_gray(p, size)) for p, *_ in rows]))
    return GlyphSet(images, [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
                    sorted(class_names))


def load_omniglot(size: int) -> GlyphSet:
    """Omniglot evaluation set (20 alphabets, 659 characters); downloaded on first use."""
    from torchvision.datasets import Omniglot

    raw = Omniglot(root=str(data_dir() / "omniglot"), background=False, download=True)
    images, labels = [], []
    for path, label in raw._flat_character_images:
        folder = Path(raw.target_folder) / raw._characters[label]
        images.append(np.asarray(read_gray(folder / path, size)))
        labels.append(label)
    n = len(labels)
    # Omniglot has no augmented copies: every image is its own "pair"
    return GlyphSet(torch.from_numpy(np.stack(images)), labels, list(range(n)), [True] * n,
                    [str(c) for c in raw._characters])


def to_input(images: torch.Tensor) -> torch.Tensor:
    """uint8 [N, H, W] -> float [N, 3, H, W] in [-1, 1] (grayscale repeated on three channels)."""
    x = images.float().div(255.0).sub(0.5).div(0.5)
    return x.unsqueeze(1).expand(-1, 3, -1, -1)
