R"""
Compute the hypervolume of the Pareto fronts stored in ``results/**/result.csv``.

Accuracies are maximized, so the objective vectors are negated before being handed
to pymoo (which minimizes). The reference point is the origin in accuracy space,
i.e. ``[0, ..., 0]`` accuracy, matching the hypervolume reported in the manuscript.

Usage::

    python scripts/compute_hypervolume.py results/clip_pareto_moe_ls/version_0 \
        results/clip_pareto_pm/version_20 --tasks SUN397 Cars
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def non_dominated(points: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset of ``points`` (larger is better)."""
    keep = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if not keep[i]:
            continue
        dominated_by_p = np.all(points <= p, axis=1) & np.any(points < p, axis=1)
        keep &= ~dominated_by_p
        keep[i] = True
    return points[keep]


def hypervolume(
    accuracies: np.ndarray, ref_point: Optional[np.ndarray] = None
) -> float:
    """Hypervolume dominated by ``accuracies`` w.r.t. ``ref_point`` (default: origin)."""
    if ref_point is None:
        ref_point = np.zeros(accuracies.shape[1])
    indicator = HV(ref_point=-np.asarray(ref_point, dtype=float))
    return float(indicator(-accuracies.astype(float)))


def load_front(csv_path: Path, tasks: Optional[List[str]], step: Optional[int]):
    df = pd.read_csv(csv_path)
    if step is not None and "step" in df.columns:
        df = df[df["step"] == step]
    if tasks is None:
        tasks = [
            c
            for c in df.columns
            if c != "step" and not c.startswith("ray_") and c != "Avg."
        ]
    accuracies = df[tasks].to_numpy(dtype=float)
    # accuracies are stored in [0, 1]; guard against files that store percentages
    if accuracies.max() > 1.5:
        accuracies = accuracies / 100.0
    return tasks, accuracies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for result_dir in args.result_dirs:
        csv_path = result_dir if result_dir.suffix == ".csv" else result_dir / "result.csv"
        if not csv_path.is_file():
            log.warning(f"skipping {csv_path}: not found")
            continue
        tasks, accuracies = load_front(csv_path, args.tasks, args.step)
        front = non_dominated(accuracies)
        rows.append(
            {
                "result": str(csv_path.parent.relative_to(Path("results")))
                if "results" in csv_path.parts
                else str(csv_path),
                "tasks": "-".join(tasks),
                "num_points": len(accuracies),
                "num_nondominated": len(front),
                "hypervolume": hypervolume(accuracies),
                "best_avg_acc": float(accuracies.mean(axis=1).max()),
            }
        )
        log.info(
            f"{rows[-1]['result']:60s} HV={rows[-1]['hypervolume']:.4f} "
            f"({len(front)}/{len(accuracies)} non-dominated)"
        )

    df = pd.DataFrame(rows)
    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv_out, index=False)
        log.info(f"written to {args.csv_out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
