# Running the GRFP control sweep on the cluster

20 datasets x 18 strategies x 100 seeds = 36,000 array tasks, about 2,100
core-hours. `wine_white` is deliberately excluded: at n=4898 it is roughly
9,500 core-hours on its own, more than four times everything else combined.

Nothing here needs editing. The sbatch files carry no absolute paths — they
locate themselves from `$SLURM_SUBMIT_DIR`, and their log paths are relative,
so SLURM resolves them against the directory you submit from.

## 1. Check the environment

From the `Code/` directory:

```bash
python -c "import numpy, sklearn, scipy; print('core deps ok')"
python -c "import torch; print('torch', torch.__version__)"
```

`torch` is needed only for `WiGS (SAC)`. Without it that strategy fails and you
get 17 of 18; everything else is unaffected.

If `python` is not the right interpreter, set `PYTHON` instead of editing
anything:

```bash
export PYTHON=python3          # or a conda/module path
```

## 2. Smoke-test one task

TaskID 1501 is `WiGS (SAC)` at seed 0 — the strategy most likely to be missing
a dependency, so it is the useful one to try first:

```bash
python -m utils.GRFPControls.RunControlsCluster \
    --Data concrete_cs --TaskID 1501 --NReplications 100
```

TaskID maps as `selector_index = (TaskID-1) // NReplications` and
`seed = (TaskID-1) % NReplications`, in the order given by
`utils/GRFPControls/FullSuite.STRATEGIES`.

## 3. Submit

**From the `Code/` directory**, not from this one:

```bash
cd /path/to/Code
for f in Cluster/RunControls/controls_*.sbatch; do sbatch "$f"; done
```

Submitting from `Cluster/RunControls/` also works — the scripts walk up one
level — but the relative log paths assume `Code/`, so prefer that.

Tasks are idempotent: each writes one pickle and skips if it already exists, so
re-submitting after a partial failure costs only the missing tasks.

## 4. Collect and plot

```bash
python -m utils.GRFPControls.AggregateControls \
    --in ../results_grfp_controls/cluster \
    --out ../results_grfp_controls/full_suite

python -m utils.GRFPControls.ManuscriptFigures \
    --dirs ../results_grfp_controls/full_suite/full_pool \
    --metrics full_pool --out ../results_grfp_controls/figures --budget 0.70

python -m utils.GRFPControls.ManuscriptFigures \
    --dirs ../results_grfp_controls/full_suite/test \
    --metrics test --out ../results_grfp_controls/figures --budget 0.70
```

`AggregateControls` prints per-dataset completeness, so a partially returned
job array is visible immediately rather than silently producing figures with
missing seeds.

## Regenerating the sbatch files

Only needed to change seeds, protocol, or which datasets are included:

```bash
python -m utils.GRFPControls.CreateControlsSbatch --all          # 20 datasets
python -m utils.GRFPControls.CreateControlsSbatch --all --exclude  # include wine_white
```

It times a short probe per dataset to set wall times, so generation takes a
couple of minutes and the limits reflect the machine it runs on.
