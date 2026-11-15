"""Command line: python run.py <command> ...   (see `python run.py --help`)"""

import argparse
import csv
import getpass
import json
import os
import statistics
import tempfile
import time
import zipfile
from dataclasses import replace
from pathlib import Path

from .data import REPO, SCRIPTS, runs_dir
from .experiments import GROUPS, all_jobs, select

MAX_CHECKPOINT_MB = 25      # larger checkpoints are left out of result bundles


def finished(job) -> bool:
    return (runs_dir() / job.id / "result.json").exists()


def chosen(args):
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    ids = args.jobs.split(",") if args.jobs else None
    return select(args.group, ids, seeds, shard)


def cmd_list(args):
    jobs = chosen(args)
    for job in jobs:
        print(f"{'done   ' if finished(job) else 'pending'}  {job.id:<44} {','.join(job.groups)}")
    print(f"\n{sum(finished(j) for j in jobs)} of {len(jobs)} done")


def cmd_status(args):
    print(f"runs directory: {runs_dir()}\n")
    for group in GROUPS:
        jobs = select([group])
        print(f"{group:<12} {sum(finished(j) for j in jobs):>3} / {len(jobs)}")
    jobs = all_jobs()
    print(f"{'total':<12} {sum(finished(j) for j in jobs):>3} / {len(jobs)}")


def cmd_train(args):
    from .jobs import run_job

    jobs = [j for j in chosen(args) if not finished(j)]
    print(f"{len(jobs)} job(s) to run as '{args.who}'")
    failed = []
    for n, job in enumerate(jobs, 1):
        print(f"\n--- {n}/{len(jobs)}: {job.id}")
        try:
            run_job(job, who=args.who)
        except KeyboardInterrupt:
            raise
        except Exception as error:       # keep going; one broken job should not waste a night
            failed.append(job.id)
            print(f"FAILED {job.id}: {error!r}")
    if failed:
        print(f"\n{len(failed)} job(s) failed: {', '.join(failed)}\nRe-run the same command to retry them.")
    print("\nWhen you are done:  python run.py pack")


def cmd_pack(args):
    """Bundle finished results into one zip to send back."""
    name = f"lipi_runs_{args.who}_{time.strftime('%Y%m%d_%H%M')}.zip"
    target = Path(args.out or REPO) / name
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
        for folder in sorted(p for p in runs_dir().iterdir() if (p / "result.json").exists()):
            count += 1
            for file in ("result.json", "train.log", "checkpoint.pt"):
                path = folder / file
                if path.exists() and (file != "checkpoint.pt" or path.stat().st_size < MAX_CHECKPOINT_MB * 2**20
                                      or args.all_checkpoints):
                    bundle.write(path, f"{folder.name}/{file}")
    print(f"packed {count} finished job(s) into {target}  ({target.stat().st_size / 2**20:.1f} MB)")


def cmd_unpack(args):
    """Merge bundles from co-authors into runs/. Existing results are never overwritten."""
    for bundle_path in args.bundles:
        added = skipped = 0
        with zipfile.ZipFile(bundle_path) as bundle:
            for member in bundle.namelist():
                target = runs_dir() / member
                if target.exists():
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(member))
                added += 1
        print(f"{bundle_path}: {added} file(s) added, {skipped} already present")


def flatten(prefix, node, out):
    """{'core': {'protonet/5w5s': {'acc': ..}}} -> {'core/protonet/5w5s': acc}"""
    for key, value in node.items():
        if isinstance(value, dict) and "acc" in value:
            out[f"{prefix}{key}"] = value["acc"]
        elif isinstance(value, dict):
            flatten(f"{prefix}{key}/", value, out)


def collect():
    """All finished results, grouped by experiment (a job without its seed)."""
    experiments = {}
    for job in all_jobs():
        path = runs_dir() / job.id / "result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = {}
        flatten("", result.get("evaluation", {}), metrics)
        if "supervised" in result:
            metrics["supervised"] = result["supervised"]["acc"]
        if "training" in result:
            metrics["best_epoch"] = result["training"]["best_epoch"]
        name = replace(job, seed=0).id.rsplit("-s0", 1)[0]
        for metric, value in metrics.items():
            experiments.setdefault(name, {}).setdefault(metric, {})[job.seed] = value
    return experiments


def cmd_summarize(args):
    """Mean and standard deviation over seeds for every metric -> runs/summary.csv"""
    experiments = collect()
    target = runs_dir() / "summary.csv"
    with open(target, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "metric", "seeds", "mean", "sd", "values"])
        for name in sorted(experiments):
            for metric, by_seed in sorted(experiments[name].items()):
                values = [by_seed[s] for s in sorted(by_seed)]
                sd = statistics.stdev(values) if len(values) > 1 else float("nan")
                writer.writerow([name, metric, len(values), f"{statistics.mean(values):.2f}", f"{sd:.2f}",
                                 " ".join(f"{v:g}" for v in values)])
    print(f"wrote {target}  ({len(experiments)} experiments)")
    for name in sorted(experiments):
        headline = experiments[name].get("core/protonet/5w5s") or experiments[name].get("supervised")
        if headline:
            values = list(headline.values())
            print(f"  {name:<34} {statistics.mean(values):6.2f}  (n={len(values)})")


def transfer_matrix(prefix: str, experiments: dict):
    """7x7 mean 5-way 5-shot accuracy: row = script named in the split, column = evaluated script."""
    rows = []
    for source in SCRIPTS:
        metrics = experiments.get(f"{prefix}_{source}-conv6-56px")
        if not metrics:
            return None
        rows.append([statistics.mean(metrics[f"scripts/{target}/5w5s"].values()) for target in SCRIPTS])
    return rows


def cmd_genealogy(args):
    from .genealogy import analyse

    experiments = collect()
    transfer = {name: m for name in ("loso", "single") if (m := transfer_matrix(name, experiments))}
    report = analyse(transfer)
    report["transfer_matrices"] = transfer
    target = runs_dir() / "genealogy.json"
    target.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for tree, entry in report["trees"].items():
        print(f"{tree:<28}" + "  ".join(f"{k}: r={v['pearson_r']:+.2f} p={v['mantel_p']:.2f}"
                                        for k, v in entry.items() if k != "tree_distance"))
    print(f"wrote {target}")


def cmd_manuscripts(args):
    from .manuscripts import run

    run()


def cmd_smoke(args):
    """A two-minute end-to-end check with a tiny budget. Results go to a temporary folder."""
    from .jobs import run_job
    from .train import Recipe

    tiny = Recipe(episodes_per_epoch=3, max_epochs=2, val_episodes=5)
    wanted = {"ancestry-conv6-56px-s0", "single_kirat-conv6-56px-s0", "within-kirat-s0", "untrained-conv6-56px-s0"}
    with tempfile.TemporaryDirectory() as tmp:
        for job in all_jobs():
            if job.id in wanted:
                if job.kind == "supervised":
                    continue            # the supervised trainer has no small budget; the tests cover its split
                folder = run_job(job, who="smoke-test", recipe=tiny, n_episodes=5, out_root=Path(tmp))
                assert (folder / "result.json").exists()
    print("\nsmoke test passed: training, evaluation and result files all work on this machine")


def main():
    parser = argparse.ArgumentParser(prog="run.py", description="Lipi experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    def job_filters(p):
        p.add_argument("--group", action="append", choices=GROUPS, help="can be given more than once")
        p.add_argument("--jobs", help="comma-separated job ids")
        p.add_argument("--seeds", help="comma-separated seeds, e.g. 0,1,2")
        p.add_argument("--shard", help="i/n: take every n-th job starting at the i-th, to split work")

    who = os.environ.get("LIPI_WHO") or getpass.getuser()
    p = sub.add_parser("list", help="show jobs and whether they are done"); job_filters(p); p.set_defaults(run=cmd_list)
    p = sub.add_parser("status", help="progress per group"); p.set_defaults(run=cmd_status)
    p = sub.add_parser("train", help="run jobs (finished ones are skipped)"); job_filters(p)
    p.add_argument("--who", default=who, help="your name, recorded in every result")
    p.set_defaults(run=cmd_train)
    p = sub.add_parser("pack", help="zip finished results to send back")
    p.add_argument("--who", default=who); p.add_argument("--out"); p.add_argument("--all-checkpoints", action="store_true")
    p.set_defaults(run=cmd_pack)
    p = sub.add_parser("unpack", help="merge result zips into runs/"); p.add_argument("bundles", nargs="+")
    p.set_defaults(run=cmd_unpack)
    p = sub.add_parser("summarize", help="mean and sd over seeds -> runs/summary.csv"); p.set_defaults(run=cmd_summarize)
    p = sub.add_parser("genealogy", help="tree distance vs shape and transfer"); p.set_defaults(run=cmd_genealogy)
    p = sub.add_parser("manuscripts", help="score crops from original manuscripts"); p.set_defaults(run=cmd_manuscripts)
    p = sub.add_parser("smoke", help="quick end-to-end check of the installation"); p.set_defaults(run=cmd_smoke)

    args = parser.parse_args()
    args.run(args)
