"""Every run behind the paper, declared in one place.

A job is one model: it is trained (or loaded, for pretrained encoders), scored
on its evaluation suites, and leaves a self-contained result.json. Jobs belong
to groups so they can be handed out ("you take `loso`, I take `resolution`").

Groups
    encoders     six encoders on the ancestry split, 56 px        (main comparison)
    baselines    untrained encoders and frozen pretrained encoders (no training)
    resolution   Conv-6, flattened and globally pooled, at five input sizes
    loso         Conv-6 with one script withheld from training
    single       Conv-6 trained on a single script                 (pairwise transfer)
    within       supervised per-script classifiers                 (non-few-shot reference)
"""

from dataclasses import dataclass

from .data import SCRIPTS

SEEDS = (0, 1, 2, 3, 4)
MAIN_SIZE = 56
SIZES = (28, 56, 84, 112, 128)

# evaluation suites, see jobs.py
FULL = ("core", "grid", "samplers", "unbalanced", "scripts", "confusion", "omniglot", "degradation")


@dataclass(frozen=True)
class Job:
    kind: str            # fewshot | untrained | frozen | supervised
    split: str           # ancestry, loso_<script>, single_<script>; the script itself for supervised
    encoder: str
    size: int
    seed: int
    suites: tuple = ()
    groups: tuple = ()

    @property
    def id(self) -> str:
        if self.kind == "frozen":
            return f"frozen-{self.encoder}"
        if self.kind == "supervised":
            return f"within-{self.split}-s{self.seed}"
        prefix = "untrained" if self.kind == "untrained" else self.split
        return f"{prefix}-{self.encoder}-{self.size}px-s{self.seed}"


def all_jobs() -> list[Job]:
    jobs: dict[str, Job] = {}

    def add(job: Job):
        # the same model can serve two groups (Conv-6 at 56 px is in `encoders` and `resolution`)
        if job.id in jobs:
            old = jobs[job.id]
            job = Job(old.kind, old.split, old.encoder, old.size, old.seed,
                      tuple(dict.fromkeys(old.suites + job.suites)), tuple(dict.fromkeys(old.groups + job.groups)))
        jobs[job.id] = job

    COMMON_SUITES = ("core", "scripts", "omniglot", "degradation")
    for seed in SEEDS:
        suites = {"conv6": FULL}
        for encoder in ("conv4", "conv6", "shufflenet_v2", "mobilenet_v2", "resnet12", "resnet18_imagenet"):
            add(Job("fewshot", "ancestry", encoder, MAIN_SIZE, seed, suites.get(encoder, COMMON_SUITES), ("encoders",)))

        for encoder in ("conv6", "resnet12"):
            add(Job("untrained", "ancestry", encoder, MAIN_SIZE, seed, ("core", "omniglot"), ("baselines",)))

        for encoder in ("conv6", "conv6_gap"):
            for size in SIZES:
                add(Job("fewshot", "ancestry", encoder, size, seed, ("core",), ("resolution",)))

        for script in SCRIPTS:
            add(Job("fewshot", f"loso_{script}", "conv6", MAIN_SIZE, seed, ("scripts",), ("loso",)))
            add(Job("fewshot", f"single_{script}", "conv6", MAIN_SIZE, seed, ("scripts",), ("single",)))
            add(Job("supervised", script, "conv6", MAIN_SIZE, seed, (), ("within",)))

    # frozen encoders are deterministic, so one run each
    for encoder in ("resnet50_imagenet", "dinov2_small"):
        add(Job("frozen", "ancestry", encoder, 224, 0, ("core",), ("baselines",)))

    return list(jobs.values())


GROUPS = ("encoders", "baselines", "resolution", "loso", "single", "within")


def select(groups=None, ids=None, seeds=None, shard=None) -> list[Job]:
    """Filter the job list. `shard` is (i, n): take every n-th job starting at the i-th (1-based)."""
    jobs = all_jobs()
    if groups:
        jobs = [j for j in jobs if set(groups) & set(j.groups)]
    if ids:
        known = {j.id for j in jobs}
        missing = [i for i in ids if i not in known]
        if missing:
            raise SystemExit(f"unknown job id(s): {', '.join(missing)}  (see `python run.py list`)")
        jobs = [j for j in jobs if j.id in ids]
    if seeds is not None:
        jobs = [j for j in jobs if j.seed in seeds]
    if shard:
        i, n = shard
        jobs = jobs[i - 1::n]
    return jobs
