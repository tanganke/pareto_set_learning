R"""
Plot the Pareto fronts of our method and the recent preference-aware merging
baselines for a pair of tasks, and write the figure into the manuscript's
``figure/`` directory.

Usage::

    python scripts/plot_pareto_front_comparison.py --tasks SUN397 Cars \
        --out ../../figure/pareto_front_comparison_sun397_cars.pdf
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

# (label, result csv, marker, colour, linestyle)
CURVES = [
    ("Task Arithmetic", "ViT-B-32/version_SUN397Cars/task_arithmetic.csv", "v", "#9467bd", "--"),
    ("Pareto Merging (Attn+MLP)", "clip_pareto_pm/version_21/result.csv", "D", "#ff7f0e", "-"),
    ("Ours-LS (MLP)", "clip_pareto_moe_ls/version_50/result.csv", "o", "#8c564b", "-"),
    ("Ours-LS (Attn+MLP)", "clip_pareto_moe_ls/version_51/result.csv", "o", "#1f77b4", "-"),
]
# single-point references
POINTS = [
    ("AdaMerging", "clip_adamerging/ViT-B-32/version_{pair}/result.csv", "*", "#d62728"),
]


def load(csv_path: Path, tasks):
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    missing = [t for t in tasks if t not in df.columns]
    if missing:
        return None
    values = df[list(tasks)].to_numpy(dtype=float)
    if values.max() > 1.5:
        values = values / 100.0
    return values * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs=2, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="ViT-B-32")
    args = parser.parse_args()

    task_a, task_b = args.tasks
    pair_key = f"{task_a}{task_b}"

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for label, rel, marker, colour, linestyle in CURVES:
        values = load(RESULTS_DIR / rel, args.tasks)
        if values is None:
            print(f"skip {label}: {rel} unavailable")
            continue
        order = values[:, 0].argsort()
        ax.plot(
            values[order, 0],
            values[order, 1],
            marker=marker,
            color=colour,
            linestyle=linestyle,
            markersize=4,
            linewidth=1.2,
            label=label,
        )
    for label, rel, marker, colour in POINTS:
        values = load(RESULTS_DIR / rel.format(pair=pair_key), args.tasks)
        if values is None:
            print(f"skip {label}: {rel} unavailable")
            continue
        ax.scatter(
            values[:, 0], values[:, 1], marker=marker, s=110, color=colour, label=label,
            zorder=5,
        )

    ax.set_xlabel(f"Accuracy on {task_a} (%)")
    ax.set_ylabel(f"Accuracy on {task_b} (%)")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=6.5, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
