"""Checks on the data and the samplers. Run with:  python -m pytest tests

These need the dataset in data/canonical but no GPU and no training.
"""

import json

import pytest

from lipi.data import SCRIPTS, data_dir, list_files, load_split, script_classes
from lipi.episodes import EpisodeSampler
from lipi.experiments import all_jobs
from lipi.genealogy import ALTERNATIVE_TREES, MAIN_TREE, tree_distances

pytestmark = pytest.mark.skipif(not (data_dir() / "canonical").exists(), reason="dataset not unpacked")


def test_every_class_has_25_pairs():
    for script in SCRIPTS:
        rows = list(list_files(script_classes(script)))
        per_class = {}
        for _, label, pair, original in rows:
            per_class.setdefault(label, []).append((pair, original))
        for members in per_class.values():
            assert len(members) == 50
            assert sorted(members) == sorted([(p, o) for p in range(25) for o in (False, True)])


def test_file_order_matches_released_pair_map():
    pairs = json.loads((data_dir() / "pairs.json").read_text(encoding="utf-8"))
    pairs = pairs.get("pairs", pairs)
    for name in ("brahmi/ka", "tibetan/pa", "kirat/a"):
        files = [p.name for p, *_ in list_files([name])]
        assert [files[i:i + 2] for i in range(0, 50, 2)] == pairs[name]


def test_splits_are_class_and_script_disjoint():
    ancestry = load_split("ancestry")
    assert len(ancestry["train"]) == 81 and len(ancestry["val"]) == 51 and len(ancestry["test"]) == 195
    for script in SCRIPTS:
        split = load_split(f"loso_{script}")
        scripts_of = lambda part: {c.split("/")[0] for c in split[part]}
        assert scripts_of("test") == {script}
        assert not scripts_of("train") & scripts_of("val")
        assert script not in scripts_of("train") | scripts_of("val")
        single = load_split(f"single_{script}")
        assert not set(single["train"]) & set(single["val"])
        assert load_split(f"single_{script}")["val"] == single["val"]      # fixed, not random


def sampler_for(classes):
    rows = list(list_files(classes))
    return EpisodeSampler([r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows], seed=1), rows


def test_disjoint_sampler_never_splits_a_pair():
    sampler, rows = sampler_for(script_classes("kirat"))
    for mode in ("disjoint", "originals"):
        for _ in range(200):
            _, support, query = sampler.sample(5, 5, 10, mode)
            for s, q in zip(support, query):
                pairs = [rows[i][2] for i in s + q]
                assert len(set(pairs)) == len(pairs) == 15
                if mode == "originals":
                    assert all(rows[i][3] for i in s + q)


def test_same_seed_gives_same_episodes():
    a, _ = sampler_for(script_classes("brahmi"))
    b, _ = sampler_for(script_classes("brahmi"))
    assert [a.sample(5, 1, 10) for _ in range(20)] == [b.sample(5, 1, 10) for _ in range(20)]


def test_job_ids_are_unique_and_grouped():
    jobs = all_jobs()
    assert len({j.id for j in jobs}) == len(jobs)
    assert all(j.groups for j in jobs)


def test_tree_distances():
    d = tree_distances(MAIN_TREE)
    assert (d == d.T).all() and d[SCRIPTS.index("tibetan"), SCRIPTS.index("kirat")] == 3
    for tree in ALTERNATIVE_TREES.values():
        assert tree_distances(tree).max() <= 7
