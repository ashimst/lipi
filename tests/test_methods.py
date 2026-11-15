"""Checks on encoders and few-shot methods with synthetic inputs (no dataset, no training)."""

import torch

from lipi.data import GlyphSet, to_input
from lipi.episodes import METHODS, accuracy
from lipi.models import build_encoder


def fake_glyphs(n_classes=8, per_class=50):
    labels = [c for c in range(n_classes) for _ in range(per_class)]
    pairs = [i // 2 for _ in range(n_classes) for i in range(per_class)]
    original = [i % 2 == 0 for _ in range(n_classes) for i in range(per_class)]
    images = torch.zeros(len(labels), 8, 8, dtype=torch.uint8)
    return GlyphSet(images, labels, pairs, original, [f"x/{c}" for c in range(n_classes)])


def test_well_separated_features_are_classified_perfectly():
    glyphs = fake_glyphs()
    torch.manual_seed(0)
    centres = torch.randn(8, 16) * 10
    feats = centres[torch.tensor(glyphs.labels)] + 0.01 * torch.randn(len(glyphs), 16)
    for method in METHODS:
        for mode in ("disjoint", "originals", "standard"):
            r = accuracy(feats, glyphs, "t", method, feats.mean(0), 5, 1, mode=mode, n_episodes=20)
            assert r["acc"] == 100.0, (method, mode, r)


def test_unbalanced_queries_and_fixed_episodes():
    glyphs = fake_glyphs()
    feats = torch.randn(len(glyphs), 16, generator=torch.Generator().manual_seed(1))
    a = accuracy(feats, glyphs, "same-tag", "protonet", feats.mean(0), 5, 5, n_query=(2, 18), n_episodes=30)
    b = accuracy(feats, glyphs, "same-tag", "protonet", feats.mean(0), 5, 5, n_query=(2, 18), n_episodes=30)
    assert a == b


def test_encoder_output_shapes():
    x = to_input(torch.zeros(2, 56, 56, dtype=torch.uint8))
    for name, dim in (("conv4", 576), ("conv6", 576), ("conv6_gap", 64), ("resnet12", 640)):
        model = build_encoder(name).eval()
        with torch.no_grad():
            assert model(x).shape == (2, dim), name
    with torch.no_grad():
        assert build_encoder("conv6").eval()(to_input(torch.zeros(2, 128, 128, dtype=torch.uint8))).shape == (2, 4096)
