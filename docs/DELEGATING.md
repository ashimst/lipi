# Running a share of the experiments

You have been asked to run one group of jobs (or a slice of one). It takes five steps.
You need a machine with an NVIDIA GPU (4 GB is enough) or a free cloud notebook.

## On your own machine

```bash
# 1. get the code
git clone <repository-url> lipi && cd lipi
pip install torch torchvision            # GPU build: https://pytorch.org/get-started/locally/
pip install -r requirements.txt

# 2. get the images (link sent to you separately) and unpack them into data/
#    afterwards data/canonical/brahmi/ka/0025.png should exist

# 3. check that everything works (about two minutes)
python run.py smoke

# 4. run what you were assigned, for example
python run.py train --group loso --who sujal
#    or a slice of a group, if it is shared between three people:
python run.py train --group resolution --shard 2/3 --who sujal

# 5. send the results back
python run.py pack --who sujal           # writes lipi_runs_sujal_<date>.zip
```

Send the zip file. It contains, for every finished job, the scores, the training log and
the model weights (a few MB in total).

Things worth knowing:

- **You can stop and continue.** Close the terminal or press Ctrl+C whenever you need the
  machine. Running the same `train` command again skips what is finished and repeats only
  the job that was interrupted.
- **Progress:** `python run.py status`, or `python run.py list --group loso`.
- **If a job fails,** the others still run. The failed ones are listed at the end; run the
  command again, and if it fails again send `runs/<job id>/train.log`.
- **Do not change the code or the settings.** All runs must use the same recipe to be
  comparable. If something looks wrong, ask first.
- Results do not depend on who runs them: seeds are fixed, and every result file records
  the machine, library versions and code version it came from.

## On Colab, Kaggle, Lightning or Azure

Open `notebooks/remote_runner.ipynb` there, fill in the three values in the first cell
(repository, data link, what to run) and run all cells. The last cell produces the zip.

Free notebooks disconnect after a few hours. That is fine: run the notebook again and it
continues, provided the `runs` folder is kept. The notebook explains how to keep it on
Google Drive (Colab) or in the persistent workspace (Kaggle, Lightning, Azure).

## For the person collecting results

```bash
python run.py unpack lipi_runs_sujal_*.zip lipi_runs_yajjyu_*.zip   # never overwrites
python run.py status
python run.py summarize
```

A reasonable split for four people:

| Who | Command |
|---|---|
| A | `python run.py train --group encoders --group baselines` |
| B | `python run.py train --group loso --group within` |
| C | `python run.py train --group single` followed by `--group resolution --shard 1/2` |
| D | `python run.py train --group resolution --shard 2/2` |

`--seeds 0,1,2` restricts any command to some seeds, for a first pass.
