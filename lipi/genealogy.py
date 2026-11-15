"""Script genealogy: the paleographic tree, glyph-shape distances, and the tests that relate them.

Tree distance between two scripts is the number of edges on the path between
them. The main tree follows standard descriptions of each script; branches that
the literature disputes are varied in ALTERNATIVE_TREES so that no conclusion
depends on one contested placement.
"""

import itertools
from collections import deque

import numpy as np
from scipy import stats
from scipy.ndimage import distance_transform_edt

from .data import SCRIPTS, list_files, read_gray, script_classes

# Internal nodes are intermediate scripts that are not in the dataset.
MAIN_TREE = [
    ("ashokan", "brahmi"), ("ashokan", "gupta"),
    ("gupta", "lichhavi"),
    ("gupta", "siddhamatrka"), ("siddhamatrka", "bhujimol"),
    ("siddhamatrka", "gaudi"), ("gaudi", "maithili"),
    ("gupta", "nagari"), ("nagari", "nandinagari"),
    ("gupta", "old_tibetan"), ("old_tibetan", "tibetan"),
    ("old_tibetan", "lepcha"), ("lepcha", "kirat"),      # Limbu derives from Tibetan through Lepcha
]


def _swap(tree, remove, add):
    return [e for e in tree if e not in remove] + add


ALTERNATIVE_TREES = {
    "kirat_sister_of_tibetan": _swap(MAIN_TREE, [("old_tibetan", "lepcha"), ("lepcha", "kirat")],
                                     [("old_tibetan", "kirat")]),
    "bhujimol_via_licchavi": _swap(MAIN_TREE, [("gupta", "lichhavi"), ("siddhamatrka", "bhujimol")],
                                   [("gupta", "licchavi_line"), ("licchavi_line", "lichhavi"),
                                    ("licchavi_line", "kutila"), ("kutila", "bhujimol")]),
    "tibetan_via_siddhamatrka": _swap(MAIN_TREE, [("gupta", "old_tibetan")], [("siddhamatrka", "old_tibetan")]),
    "nagari_via_siddhamatrka": _swap(MAIN_TREE, [("gupta", "nagari")], [("siddhamatrka", "nagari")]),
}
ALTERNATIVE_TREES["all_three_alternatives"] = _swap(
    ALTERNATIVE_TREES["bhujimol_via_licchavi"],
    [("gupta", "old_tibetan"), ("gupta", "nagari")],
    [("siddhamatrka", "old_tibetan"), ("siddhamatrka", "nagari")])


def tree_distances(tree=MAIN_TREE) -> np.ndarray:
    """7x7 matrix of path lengths between the scripts, in SCRIPTS order."""
    neighbours = {}
    for a, b in tree:
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    out = np.zeros((len(SCRIPTS), len(SCRIPTS)))
    for i, start in enumerate(SCRIPTS):
        depth, queue = {start: 0}, deque([start])
        while queue:
            node = queue.popleft()
            for nxt in neighbours[node]:
                if nxt not in depth:
                    depth[nxt] = depth[node] + 1
                    queue.append(nxt)
        out[i] = [depth[s] for s in SCRIPTS]
    return out


def mean_glyph(class_name: str, size: int = 128) -> np.ndarray:
    """Average ink map of a class (ink = 1, paper = 0), from the original glyphs only."""
    images = [1.0 - np.asarray(read_gray(path, size), dtype=np.float32) / 255.0
              for path, _, _, original in list_files([class_name]) if original]
    return np.mean(images, axis=0)


def chamfer(a: np.ndarray, b: np.ndarray, threshold: float = 0.3) -> float:
    """Symmetric Chamfer distance between the thresholded ink regions, in pixels."""
    a, b = a > threshold, b > threshold
    if not a.any() or not b.any():
        return 1.0
    return float((distance_transform_edt(~b)[a].mean() + distance_transform_edt(~a)[b].mean()) / 2)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return float(1.0 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def structural_distances() -> dict:
    """Shape distance between scripts over the characters that all seven share.

    Returns the Chamfer and cosine matrices and their equal-weight combination
    (Chamfer scaled to [0, 1] by its maximum).
    """
    shared = set.intersection(*[{c.split("/")[1] for c in script_classes(s)} for s in SCRIPTS])
    glyph = {s: {ch: mean_glyph(f"{s}/{ch}") for ch in shared} for s in SCRIPTS}
    n = len(SCRIPTS)
    cham, cos = np.zeros((n, n)), np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        a, b = glyph[SCRIPTS[i]], glyph[SCRIPTS[j]]
        cham[i, j] = cham[j, i] = np.mean([chamfer(a[ch], b[ch]) for ch in shared])
        cos[i, j] = cos[j, i] = np.mean([cosine_distance(a[ch], b[ch]) for ch in shared])
    return {"shared_characters": sorted(shared), "chamfer": cham, "cosine": cos,
            "combined": 0.5 * cham / cham.max() + 0.5 * cos}


def association(tree_dist: np.ndarray, other: np.ndarray, symmetric: bool) -> dict:
    """Pearson, Spearman and an exact two-sided Mantel test between two 7x7 matrices.

    The Mantel test permutes the scripts (all 5,040 orderings), which respects
    the dependence between cells that share a script. For a symmetric matrix
    only the upper triangle is used; otherwise all off-diagonal cells.
    """
    n = len(tree_dist)
    mask = np.triu(np.ones((n, n), bool), 1) if symmetric else ~np.eye(n, dtype=bool)
    r, p_r = stats.pearsonr(tree_dist[mask], other[mask])
    rho, p_rho = stats.spearmanr(tree_dist[mask], other[mask])
    extreme = total = 0
    for order in itertools.permutations(range(n)):
        permuted = tree_dist[np.ix_(order, order)]
        extreme += abs(stats.pearsonr(permuted[mask], other[mask])[0]) >= abs(r) - 1e-12
        total += 1
    return {"n": int(mask.sum()), "pearson_r": round(float(r), 3), "pearson_p": round(float(p_r), 4),
            "spearman_rho": round(float(rho), 3), "spearman_p": round(float(p_rho), 4),
            "mantel_p": round(extreme / total, 4)}


def analyse(transfer: dict = None) -> dict:
    """Relate tree distance to glyph shape and, if given, to transfer matrices.

    `transfer` maps a name to a 7x7 accuracy matrix (rows: training condition,
    columns: evaluated script).
    """
    shape = structural_distances()
    trees = {"main": MAIN_TREE, **ALTERNATIVE_TREES}
    report = {"scripts": SCRIPTS, "shared_characters": shape["shared_characters"],
              "structural_distance": shape["combined"].round(4).tolist(), "trees": {}}
    for name, tree in trees.items():
        dist = tree_distances(tree)
        entry = {"tree_distance": dist.astype(int).tolist(),
                 "structure": association(dist, shape["combined"], symmetric=True)}
        for key, matrix in (transfer or {}).items():
            entry[key] = association(dist, np.asarray(matrix), symmetric=False)
        report["trees"][name] = entry
    return report
