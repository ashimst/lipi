"""Validation on glyphs cropped from original manuscripts and inscriptions.

Crops are produced by tools/ingest_manuscript_crops.py and live in
    data/manuscripts/bare/<script>/<character>/*.png

For each script we use the Conv-6 models that never saw that script in training
(the `loso` runs) and ask the deployment question: with handwritten glyphs as
the support set, how well are original glyphs recognised? The same episodes are
also scored with handwritten queries, so the gap is measured on identical
classes.
"""

import json
import random

import numpy as np
import torch

from .data import GlyphSet, data_dir, load_glyphs, read_gray, runs_dir
from .episodes import build_classifier, seed_from, summarize
from .experiments import MAIN_SIZE, SEEDS
from .jobs import load_checkpoint
from .models import embed
from .train import pick_device


def load_crops(script: str, size: int):
    """Original crops of one script as a GlyphSet; classes need at least two crops."""
    root = data_dir() / "manuscripts" / "bare" / script
    names, images, labels = [], [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(folder.glob("*.png"))
        if len(files) < 2:
            continue
        for f in files:
            images.append(np.asarray(read_gray(f, size)))
            labels.append(len(names))
        names.append(f"{script}/{folder.name}")
    n = len(labels)
    return GlyphSet(torch.from_numpy(np.stack(images)), labels, list(range(n)), [True] * n, names)


@torch.no_grad()
def score_script(script: str, seed: int, device, n_episodes: int = 1000, max_queries: int = 10) -> dict:
    model, _ = load_checkpoint(f"loso_{script}-conv6-{MAIN_SIZE}px-s{seed}", device)
    crops = load_crops(script, MAIN_SIZE)
    hand = load_glyphs(crops.class_names, MAIN_SIZE)          # same classes, sorted the same way
    crop_feats = embed(model, crops.images, device)
    hand_feats = embed(model, hand.images, device)

    def by_class(glyphs, originals_only=False):
        groups = {}
        for i, label in enumerate(glyphs.labels):
            if not originals_only or glyphs.is_original[i]:
                groups.setdefault(label, []).append(i)
        return groups

    hand_pool, crop_pool = by_class(hand, originals_only=True), by_class(crops)
    n_way = min(5, len(crops.class_names))
    classifier = build_classifier("protonet", torch.zeros(1))
    out = {"classes": len(crops.class_names), "crops": len(crops), "n_way": n_way}
    for n_shot in (1, 5):
        rng = random.Random(seed_from(f"manuscripts/{script}/{n_shot}"))
        to_original, to_handwritten = [], []
        for _ in range(n_episodes):
            classes = rng.sample(sorted(crop_pool), n_way)
            support, hand_q, crop_q, q_lab = [], [], [], []
            for position, c in enumerate(classes):
                n_q = min(max_queries, len(crop_pool[c]))
                picked = rng.sample(hand_pool[c], n_shot + n_q)       # handwritten support and queries never overlap
                support += picked[:n_shot]
                hand_q += picked[n_shot:]
                crop_q += rng.sample(crop_pool[c], n_q)
                q_lab += [position] * n_q
            s_lab = torch.arange(n_way).repeat_interleave(n_shot)
            classifier.process_support_set(hand_feats[support], s_lab)
            truth = torch.tensor(q_lab)
            to_original.append((classifier(crop_feats[crop_q]).argmax(1) == truth).float().mean().item())
            to_handwritten.append((classifier(hand_feats[hand_q]).argmax(1) == truth).float().mean().item())
        out[f"{n_shot}shot"] = {"original_queries": summarize(to_original),
                                "handwritten_queries": summarize(to_handwritten)}
    return out


def run(n_episodes: int = 1000) -> dict:
    root = data_dir() / "manuscripts" / "bare"
    if not root.exists():
        raise SystemExit(f"no crops found under {root}; run tools/ingest_manuscript_crops.py first")
    device = pick_device()
    report = {}
    for script in sorted(p.name for p in root.iterdir() if p.is_dir()):
        seeds = [s for s in SEEDS if (runs_dir() / f"loso_{script}-conv6-{MAIN_SIZE}px-s{s}" / "checkpoint.pt").exists()]
        if not seeds:
            print(f"{script}: no finished loso checkpoints, skipped")
            continue
        report[script] = {f"s{s}": score_script(script, s, device, n_episodes) for s in seeds}
        print(script, json.dumps(report[script][f"s{seeds[0]}"]["5shot"]))
    out = runs_dir() / "manuscripts.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return report
