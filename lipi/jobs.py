"""Run one job end to end: train, evaluate, write runs/<job id>/.

A finished job directory holds
    result.json     the job, the environment, the training history and every score
    train.log       what was printed while it ran
    checkpoint.pt   encoder weights of the best epoch (trained jobs only)

Evaluation suites (all on source-disjoint episodes unless said otherwise)
    core         ProtoNet, SimpleShot, LaplacianShot at 5/20-way, 1/5-shot on the split's test classes
    grid         ProtoNet at 5/10/15/20-way and 1/5/10-shot
    samplers     `core` again with standard sampling and with original glyphs only
    unbalanced   5-way with 2 to 18 queries per class
    scripts      ProtoNet 5-way on each of the seven scripts separately
    confusion    which characters are confused, per test script
    omniglot     Omniglot evaluation set
    degradation  clean support, synthetically damaged queries
"""

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from . import __version__
from .data import REPO, SCRIPTS, load_glyphs, load_omniglot, load_split, runs_dir, script_classes
from .degradation import CORRUPTIONS, SEVERITIES, load_degraded
from .episodes import METHODS, accuracy, confusions
from .experiments import Job
from .models import IMAGENET_NORMALIZED, build_encoder, count_parameters, embed
from .train import RECIPE, pick_device, set_seed, train_episodic, train_supervised

SETTINGS = ((5, 1), (5, 5), (20, 1), (20, 5))


class Logger:
    """Print and append to a file, with a timestamp."""

    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def environment(device, who: str) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    import easyfsl

    return {"who": who, "host": platform.node(), "platform": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__, "easyfsl": easyfsl.__version__,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else device.type,
            "lipi": __version__, "git_commit": commit}


def evaluate(job: Job, model, device, log, n_episodes: int = 1000) -> dict:
    """Run the job's suites. Embeddings are computed once per image set and reused."""
    imagenet = job.encoder in IMAGENET_NORMALIZED
    split = load_split(job.split)
    cache = {}

    def glyphs_and_features(key, classes):
        if key not in cache:
            glyphs = load_glyphs(classes, job.size)
            cache[key] = (glyphs, embed(model, glyphs.images, device, imagenet))
        return cache[key]

    def score(tag, glyphs, feats, method, n_way, n_shot, **kw):
        kw.setdefault("n_episodes", n_episodes)
        return accuracy(feats, glyphs, tag, method, centre, n_way, n_shot, device=device, **kw)

    # CL2N centre: mean feature of the classes the model was trained on
    _, train_feats = glyphs_and_features("train", split["train"])
    centre = train_feats.mean(dim=0)
    out = {}

    if split["test"]:
        test, feats = glyphs_and_features("test", split["test"])

    if "core" in job.suites:
        out["core"] = {f"{m}/{w}w{k}s": score(f"core/{w}w{k}s", test, feats, m, w, k)
                       for m in METHODS for w, k in SETTINGS}
        log("core: " + ", ".join(f"{k} {v['acc']}" for k, v in out["core"].items() if k.endswith("5w5s")))

    if "grid" in job.suites:
        out["grid"] = {f"{w}w{k}s": score(f"grid/{w}w{k}s", test, feats, "protonet", w, k)
                       for w in (5, 10, 15, 20) for k in (1, 5, 10)}

    if "samplers" in job.suites:
        # same episode tags as `core`, so only the sampling rule differs
        out["samplers"] = {f"{mode}/{m}/{w}w{k}s": score(f"core/{w}w{k}s", test, feats, m, w, k, mode=mode)
                           for mode in ("standard", "originals") for m in METHODS for w, k in SETTINGS}

    if "unbalanced" in job.suites:
        out["unbalanced"] = {f"{m}/5w{k}s": score(f"unbalanced/5w{k}s", test, feats, m, 5, k, n_query=(2, 18))
                             for m in METHODS for k in (1, 5)}

    if "scripts" in job.suites:
        out["scripts"] = {}
        for script in SCRIPTS:
            g, f = glyphs_and_features(script, script_classes(script))
            for k in (1, 5):
                out["scripts"][f"{script}/5w{k}s"] = score(f"scripts/{script}/5w{k}s", g, f, "protonet", 5, k)
        log("scripts 5w5s: " + ", ".join(f"{s} {out['scripts'][f'{s}/5w5s']['acc']}" for s in SCRIPTS))

    if "confusion" in job.suites:
        test_scripts = sorted({c.split("/")[0] for c in split["test"]})
        out["confusion"] = {}
        for script in test_scripts:
            g, f = glyphs_and_features(script, script_classes(script))
            out["confusion"][script] = confusions(f, g, f"confusion/{script}", centre,
                                                  n_episodes=n_episodes, device=device)

    if "omniglot" in job.suites:
        omni = load_omniglot(job.size)
        omni_feats = embed(model, omni.images, device, imagenet)
        out["omniglot"] = {
            f"{m}/{w}w{k}s": score(f"omniglot/{w}w{k}s", omni, omni_feats, m, w, k, mode="standard",
                                   n_query=10 if w == 5 else 5)   # 20 images per class limits 20-way queries
            for m in ("protonet", "laplacianshot") for w, k in SETTINGS}
        log(f"omniglot protonet 5w5s: {out['omniglot']['protonet/5w5s']['acc']}")

    if "degradation" in job.suites:
        out["degradation"] = {"clean": score("degradation", test, feats, "protonet", 5, 5)}
        for kind in CORRUPTIONS:
            for severity in SEVERITIES:
                damaged = embed(model, load_degraded(split["test"], job.size, kind, severity), device, imagenet)
                out["degradation"][f"{kind}/{severity}"] = score("degradation", test, feats, "protonet", 5, 5,
                                                                 query_features=damaged)
            log(f"degradation {kind} at 1.0: {out['degradation'][f'{kind}/1.0']['acc']}")
    return out


def run_job(job: Job, who: str = "unknown", device=None, recipe=RECIPE, n_episodes: int = 1000,
            out_root: Path = None) -> Path:
    """Run a job unless it is already finished. Returns the job directory."""
    folder = (out_root or runs_dir()) / job.id
    result_file = folder / "result.json"
    if result_file.exists():
        print(f"{job.id}: already done, skipping")
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    log = Logger(folder / "train.log")
    device = device or pick_device()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    log(f"=== {job.id} on {device} (run by {who}) ===")

    set_seed(job.seed)
    model = build_encoder(job.encoder).to(device)
    result = {"job": {**asdict(job), "id": job.id}, "parameters": count_parameters(model)}

    if job.kind == "supervised":
        glyphs = load_glyphs(script_classes(job.split), job.size)
        result["supervised"] = train_supervised(model, glyphs, job.seed, device, log=log)
    else:
        if job.kind == "fewshot":
            split = load_split(job.split)
            train, val = load_glyphs(split["train"], job.size), load_glyphs(split["val"], job.size)
            log(f"training on {len(train.class_names)} classes, validating on {len(val.class_names)}")
            state, result["training"] = train_episodic(model, train, val, job.seed, device, log, recipe)
            torch.save({"job": asdict(job), "state_dict": state}, folder / "checkpoint.pt")
        result["evaluation"] = evaluate(job, model, device, log, n_episodes)

    result["environment"] = {**environment(device, who), "started": started,
                             "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    # write to a temporary name first so a crash never leaves a half-written result
    tmp = folder / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    tmp.replace(result_file)
    log(f"done: {result_file}")
    return folder


def load_checkpoint(job_id: str, device):
    """Rebuild the encoder of a finished job from runs/<job_id>/checkpoint.pt."""
    saved = torch.load(runs_dir() / job_id / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = build_encoder(saved["job"]["encoder"])
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval(), saved["job"]
