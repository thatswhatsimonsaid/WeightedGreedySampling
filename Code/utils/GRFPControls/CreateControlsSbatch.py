"""
Generate SLURM sbatch files for the control sweep.

Follows `GenerateJobs.create_master_sbatch` in shape, with two differences
forced by measured cost:

* the array is over (selector, seed) rather than (model, seed), because a task
  covering all 18 strategies on `wine_white` would not fit a 24-hour limit;
* the wall time and memory per dataset are set from measured per-run cost
  rather than one global value.

Measured single-selector full-run times (iGS, paper protocol, one seed):
`wine_white` 82 min, `wine_red` 3.5 min, `concrete_4` 1.1 min,
`cps_wage` 0.3 min. Cross-validating selectors (the MAB variants, the random-arm
control, SAC) cost roughly twice that, and QBC carries a five-model committee,
so the limits below apply a 4x headroom factor.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.CreateControlsSbatch               # the 8 unrun datasets
    python -m utils.GRFPControls.CreateControlsSbatch --all         # all 21
"""

from utils.GRFPControls import _bootstrap  # noqa: F401

import argparse
import math
import os
import pickle
import stat

from utils.GRFPControls.FullSuite import STRATEGIES

### Datasets with no control results yet ###
UNRUN = ["wine_white", "wine_red", "concrete_4", "cps_wage", "dgp_new",
         "burbidge_correct", "burbidge_low_noise", "burbidge_misspecified"]

### Cluster parameters, matching CreateSimulationSbatch.py ###
PARTITION = "cpu-g2"
HEADROOM = 4.0          # cheap-selector time -> worst-selector time
MIN_MINUTES = 20


PROBE_ROUNDS = 150


def measured_minutes(dataset, protocol, standardize):
    """
    Time a short run and extrapolate, rather than trusting a closed-form model.

    Per-round cost grows with the labelled set, so total time over T rounds goes
    roughly as T^2 relative to a T0-round probe. Probing costs a few seconds per
    dataset and removes the guesswork: an earlier analytic fit predicted 210
    minutes for a wine_white run that actually measured 82.
    """
    import time
    from utils.GRFPControls.ControlLoop import RunControlIteration, _load_dataset
    n = len(_load_dataset(dataset))
    T = max(PROBE_ROUNDS + 1, int(round(n * 0.85)))
    t0 = time.time()
    RunControlIteration(dataset, Seed=0, SelectorType='GreedySamplingSelector',
                        selector_params={'strategy': 'iGS'},
                        MaxIterations=PROBE_ROUNDS,
                        Protocol=protocol, Standardize=bool(standardize))
    el = time.time() - t0
    return max(0.2, el * (T / PROBE_ROUNDS) ** 2 / 60.0), n


def sbatch_for(dataset, n_rows, n_feat, base_minutes, n_reps, code_dir, out_dir,
               protocol="paper", standardize=1):
    n_tasks = len(STRATEGIES) * n_reps
    per_run = base_minutes * HEADROOM
    minutes = max(MIN_MINUTES, int(math.ceil(per_run * 1.3)))
    hours, mins = divmod(min(minutes, 23 * 60 + 59), 60)
    time_limit = f"{hours:02d}:{mins:02d}:59"
    memory = "8000MB" if n_rows > 2000 else "4000MB"
    log_dir = "/".join([code_dir.rstrip("/\\"), "Cluster", "RunControls", "ClusterMessages"])

    content = f"""#!/bin/bash
#SBATCH --job-name=CTRL_{dataset}
#SBATCH --partition={PARTITION}
#SBATCH --array=1-{n_tasks}
#SBATCH --output={log_dir}/out/{dataset}_%A_%a.out
#SBATCH --error={log_dir}/error/{dataset}_%A_%a.err
#SBATCH --time={time_limit}
#SBATCH --mem-per-cpu={memory}
#SBATCH --cpus-per-task=1

cd "{code_dir}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -m utils.GRFPControls.RunControlsCluster \\
    --Data "{dataset}" \\
    --TaskID "$SLURM_ARRAY_TASK_ID" \\
    --NReplications {n_reps} \\
    --Protocol {protocol} \\
    --Standardize {standardize}
"""
    path = os.path.join(out_dir, f"controls_{dataset}.sbatch")
    with open(path, "w", newline="\n") as f:
        f.write(content)
    os.chmod(path, stat.S_IRWXU)
    return path, n_tasks, time_limit, per_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="Every processed dataset, not just the unrun ones.")
    ap.add_argument("--exclude", nargs="*", default=["wine_white"],
                    help="Datasets to skip. Defaults to wine_white, which at "
                         "n=4898 is about 9,460 core-hours on its own - 81%% of "
                         "the full sweep - because cost grows steeply with pool "
                         "size. Pass --exclude with no names to include it.")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--protocol", choices=["repo", "paper"], default="paper")
    ap.add_argument("--standardize", type=int, default=1)
    ap.add_argument("--code-dir", type=str, default=None,
                    help="Path to Code/ ON THE CLUSTER. The sbatch files cd here, "
                         "so a Windows path baked in from a local run will not "
                         "execute on a Linux node.")
    args = ap.parse_args()

    # cluster_dir is baked into the sbatch (where the compute node will cd);
    # local_dir is where the files get written now. Conflating the two writes
    # sbatch files into a path that only exists on the cluster.
    local_dir = os.getcwd()
    code_dir = (args.code_dir or local_dir).replace("\\", "/").rstrip("/")
    if len(code_dir) > 1 and code_dir[1] == ":":
        print(f"WARNING: --code-dir looks like a Windows path ({code_dir}).")
        print("         The sbatch files cd there, so pass the cluster path, e.g.")
        print("         --code-dir /gscratch/.../Code")
        print("         (under Git Bash, prefix the command with MSYS_NO_PATHCONV=1)\n")
    root = os.path.dirname(local_dir)
    data_dir = os.path.join(root, "Data", "processed")
    out_dir = os.path.join(local_dir, "Cluster", "RunControls")
    for sub in ("ClusterMessages/out", "ClusterMessages/error"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    names = sorted(f[:-4] for f in os.listdir(data_dir) if f.endswith(".pkl"))
    if not args.all:
        names = [n for n in names if n in UNRUN]
    skipped = [n for n in names if n in set(args.exclude)]
    names = [n for n in names if n not in set(args.exclude)]
    if skipped:
        print(f"excluded: {', '.join(skipped)}\n")

    print(f"{len(STRATEGIES)} strategies x {args.seeds} seeds per dataset "
          f"| protocol={args.protocol} standardize={bool(args.standardize)}\n")
    print(f"{'dataset':<24}{'n':>7}{'d':>4}{'tasks':>8}{'walltime':>11}{'est/run':>10}{'core-h':>9}")
    total_core_h = 0.0
    for name in names:
        with open(os.path.join(data_dir, name + ".pkl"), "rb") as fh:
            df = pickle.load(fh).dropna()
        n_rows, n_feat = len(df), df.shape[1] - 1
        base, _ = measured_minutes(name, args.protocol, args.standardize)
        path, n_tasks, tl, per_run = sbatch_for(
            name, n_rows, n_feat, base, args.seeds, code_dir, out_dir,
            args.protocol, args.standardize)
        core_h = n_tasks * per_run / 60.0
        total_core_h += core_h
        print(f"{name:<24}{n_rows:>7}{n_feat:>4}{n_tasks:>8}{tl:>11}"
              f"{per_run:>9.1f}m{core_h:>9.0f}")
    print(f"\n{'TOTAL':<24}{'':>7}{'':>4}{len(names)*len(STRATEGIES)*args.seeds:>8}"
          f"{'':>11}{'':>10}{total_core_h:>9.0f}")
    print(f"\nWrote {len(names)} sbatch files to {out_dir}")
    print("Submit with:  for f in Cluster/RunControls/controls_*.sbatch; do sbatch $f; done")


if __name__ == "__main__":
    main()
