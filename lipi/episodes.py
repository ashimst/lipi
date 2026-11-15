"""Episode sampling and few-shot evaluation.

Evaluation works on embeddings computed once per model, so every method and
every setting is scored on exactly the same episodes at almost no cost.

Sampling modes
    disjoint   support and query images come from different glyph pairs, and one
               member of each pair is picked at random (the default everywhere)
    originals  like disjoint, but only original glyphs are used
    standard   images drawn at random; a glyph and its augmented copy can land on
               opposite sides of the episode (kept only to measure the leak)
"""

import math
import random
import zlib
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from easyfsl.methods import LaplacianShot, PrototypicalNetworks, SimpleShot

METHODS = ("protonet", "simpleshot", "laplacianshot")


def seed_from(tag: str) -> int:
    """Stable seed from a readable tag, so every model sees the same episodes."""
    return zlib.crc32(tag.encode())


class EpisodeSampler:
    def __init__(self, labels, pair_ids, is_original, seed: int):
        self.rng = random.Random(seed)
        pairs = defaultdict(lambda: defaultdict(list))
        for index, (label, pair, original) in enumerate(zip(labels, pair_ids, is_original)):
            members = pairs[label][pair]
            if original:
                members.insert(0, index)    # the original always comes first in its pair
            else:
                members.append(index)
        self.pairs = {label: list(p.values()) for label, p in pairs.items()}
        self.classes = sorted(self.pairs)

    def sample(self, n_way: int, n_shot: int, n_query, mode: str = "disjoint"):
        """Return (classes, support, query): per class, lists of image indices.

        n_query is an int, or a (low, high) range for unbalanced query sets.
        """
        classes = self.rng.sample(self.classes, n_way)
        support, query = [], []
        for label in classes:
            q = self.rng.randint(*n_query) if isinstance(n_query, tuple) else n_query
            if mode == "standard":
                pool = [i for pair in self.pairs[label] for i in pair]
                picked = self.rng.sample(pool, n_shot + q)
            else:
                chosen = self.rng.sample(self.pairs[label], n_shot + q)
                picked = [pair[0] if mode == "originals" else self.rng.choice(pair) for pair in chosen]
            support.append(picked[:n_shot])
            query.append(picked[n_shot:])
        return classes, support, query


def build_classifier(method: str, centre: torch.Tensor):
    """EasyFSL classifiers running on precomputed features.

    SimpleShot and LaplacianShot use CL2N features: centred on the mean feature
    of the training classes, then L2-normalised.
    """
    if method == "protonet":
        return PrototypicalNetworks(nn.Identity())
    if method == "simpleshot":
        return SimpleShot(nn.Identity(), feature_centering=centre, feature_normalization=2)
    if method == "laplacianshot":
        return LaplacianShot(nn.Identity(), feature_centering=centre, feature_normalization=2)
    raise ValueError(f"unknown method '{method}'")


@torch.no_grad()
def run_episodes(features, sampler: EpisodeSampler, method: str, centre, n_way: int, n_shot: int,
                 n_query=10, mode: str = "disjoint", n_episodes: int = 1000, query_features=None,
                 device="cpu"):
    """Yield (classes, true, predicted) per episode; labels are positions in `classes`.

    `query_features` lets the queries come from a different rendering of the same
    images (used for degraded queries with clean support).
    """
    features = features.to(device)
    query_features = features if query_features is None else query_features.to(device)
    classifier = build_classifier(method, centre.to(device))
    for _ in range(n_episodes):
        classes, support, query = sampler.sample(n_way, n_shot, n_query, mode)
        s_idx = [i for per_class in support for i in per_class]
        q_idx = [i for per_class in query for i in per_class]
        s_lab = torch.tensor([c for c, per_class in enumerate(support) for _ in per_class], device=device)
        q_lab = torch.tensor([c for c, per_class in enumerate(query) for _ in per_class], device=device)
        classifier.process_support_set(features[s_idx], s_lab)
        predicted = classifier(query_features[q_idx]).argmax(dim=1)
        yield classes, q_lab.cpu(), predicted.cpu()


def summarize(accuracies) -> dict:
    a = np.asarray(accuracies, dtype=np.float64) * 100
    return {"acc": round(float(a.mean()), 2),
            "ci95": round(float(1.96 * a.std(ddof=1) / math.sqrt(len(a))), 2),
            "episodes": len(a)}


def accuracy(features, glyphs, tag: str, method: str, centre, n_way: int, n_shot: int, **kwargs) -> dict:
    """Mean accuracy with a 95% interval over episodes. `tag` fixes the episode seed."""
    sampler = EpisodeSampler(glyphs.labels, glyphs.pair_ids, glyphs.is_original, seed_from(tag))
    accs = [(true == pred).float().mean().item()
            for _, true, pred in run_episodes(features, sampler, method, centre, n_way, n_shot, **kwargs)]
    return summarize(accs)


def confusions(features, glyphs, tag: str, centre, n_way: int = 5, n_shot: int = 5,
               n_episodes: int = 1000, top: int = 15, device="cpu") -> dict:
    """Which characters ProtoNet confuses, with episodes drawn from one set of classes."""
    sampler = EpisodeSampler(glyphs.labels, glyphs.pair_ids, glyphs.is_original, seed_from(tag))
    name = [c.split("/")[-1] for c in glyphs.class_names]
    pairs, wrong, seen = Counter(), Counter(), Counter()
    for classes, true, pred in run_episodes(features, sampler, "protonet", centre, n_way, n_shot,
                                            n_episodes=n_episodes, device=device):
        for t, p in zip(true.tolist(), pred.tolist()):
            seen[name[classes[t]]] += 1
            if t != p:
                wrong[name[classes[t]]] += 1
                pairs[(name[classes[t]], name[classes[p]])] += 1
    total, errors = sum(seen.values()), sum(wrong.values())
    hardest = sorted(wrong, key=lambda c: -wrong[c] / seen[c])[:10]
    return {"queries": total, "errors": errors, "error_rate": round(100 * errors / total, 2),
            "confusions": [{"true": a, "pred": b, "count": n} for (a, b), n in pairs.most_common(top)],
            "hardest": [{"class": c, "error_rate": round(100 * wrong[c] / seen[c], 2), "queries": seen[c]}
                        for c in hardest]}
