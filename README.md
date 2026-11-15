# Lipi

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24240673.svg)](https://doi.org/10.5281/zenodo.24240673)

Few-shot recognition of seven historical scripts of Nepal: Ashokan Brahmi, Licchavi,
Bhujimol, Nandinagari, Maithili (Tirhuta), Tibetan Uchen and Kirat Sirijunga.

This repository holds the benchmark code for the Lipi dataset: 8,175 handwritten glyphs
in 327 character classes, each stored with one augmented copy (16,350 images). The
dataset itself is described in [DATASET_CARD.md](DATASET_CARD.md).

## Setup

```bash
git clone https://github.com/ashimst/lipi.git && cd lipi
pip install torch torchvision          # pick the build for your GPU: https://pytorch.org
pip install -r requirements.txt
```

Download the images automatically from Zenodo:

```bash
python tools/download_zenodo.py --record 24240673
```

Check the installation:

```bash
python -m pytest tests        # data and method checks, under a minute, no GPU needed
python run.py smoke           # trains and evaluates tiny models end to end, about two minutes
```

## Running experiments

Everything goes through `run.py`:

```bash
python run.py status                      # progress per group
python run.py list --group loso           # the jobs of one group
python run.py train --group loso          # run them; finished jobs are skipped
python run.py train --jobs ancestry-conv6-56px-s0
python run.py summarize                   # mean and sd over seeds -> runs/summary.csv
python run.py genealogy                   # tree distance vs. glyph shape and transfer
```

A job is one model. It is trained, evaluated, and leaves `runs/<job id>/` with
`result.json` (scores, training history, environment), `train.log` and `checkpoint.pt`.
Interrupted work loses at most the job that was running; start the same command again.

| Group | Jobs | What it answers | Rough GPU time* |
|---|---:|---|---:|
| `encoders` | 30 | Which encoder works best? Six encoders on the ancestry split, plus every analysis of the main model | 10 h |
| `baselines` | 12 | How far do untrained and frozen pretrained encoders get? | 1 h |
| `resolution` | 50 | Input size, with a flattened and a globally pooled embedding | 15 h |
| `loso` | 35 | What does withholding a script from training cost? | 8 h |
| `single` | 35 | How well does training on one script transfer to each other script? | 6 h |
| `within` | 35 | Supervised per-script classifiers as a non-few-shot reference | 1.5 h |

\* on one mid-range GPU (RTX 3050, T4). All groups use five seeds (0 to 4).

### One protocol for everything

- **Training** (`lipi/train.py`): episodic ProtoNet, 5-way 5-shot with 10 queries per class,
  80 episodes per epoch, up to 100 epochs, SGD with step decay, early stopping on 100
  fixed validation episodes. The same recipe for every model.
- **Evaluation** (`lipi/episodes.py`): 1,000 episodes per setting. Support and query images
  always come from different glyph pairs, so a glyph and its augmented copy never meet
  across an episode. Episode seeds are fixed, so all models are scored on the same episodes.
- **Methods**: ProtoNet, SimpleShot and LaplacianShot (EasyFSL), the last two on centred,
  L2-normalised features.
- **Reporting**: mean and standard deviation over five training seeds.

The full list of runs is declared in `lipi/experiments.py`.

## Sharing the work

Jobs are independent, so they can be split between people and machines. See
[docs/DELEGATING.md](docs/DELEGATING.md) for the short version to send to a collaborator,
and `notebooks/remote_runner.ipynb` for Colab, Kaggle, Lightning or an Azure notebook.

## Layout

```
lipi/            the package
  data.py          files, splits, image loading
  models.py        encoders
  train.py         the training recipe
  episodes.py      episode sampling and few-shot evaluation
  jobs.py          run one job: train, evaluate, write the result
  experiments.py   every run of the paper
  degradation.py   synthetic damage
  genealogy.py     paleographic tree and the tests against it
  manuscripts.py   validation on crops from original manuscripts
run.py           command line
tests/           checks on data, samplers and methods
tools/           one-off utilities (augmentation script, crop ingestion, pair check)
data/            splits and metadata (images are downloaded separately)
runs/            results
```

## Citation

```bibtex
@misc{shrestha2026lipi,
  title        = {{Lipi: Handwritten Glyphs of Seven Historical Scripts of Nepal (Version 1)}},
  author       = {Shrestha, Ashim and Bajracharya, Sujal and Tuladhar, Yajjyu and Kunwar, Kushal and Regmi, Sunil and Bal, Bal Krishna},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24240673},
  url          = {https://doi.org/10.5281/zenodo.24240673}
}
```
