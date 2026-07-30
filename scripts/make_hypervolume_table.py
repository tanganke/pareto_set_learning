R"""
Emit the LaTeX rows of the SUN397--Cars hypervolume table.

All hypervolumes are recomputed from the fronts of a single set of runs with the
origin of the accuracy space as the reference point, so the numbers are directly
comparable across methods.

Usage::

    python scripts/make_hypervolume_table.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_hypervolume import hypervolume, load_front, non_dominated  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
TASKS = ["SUN397", "Cars"]

# (label, relative csv path, "set" if the method returns a whole front else "single")
ROWS = [
    ("Task Arithmetic", "ViT-B-32/version_SUN397Cars/task_arithmetic.csv", "set"),
    ("AdaMerging", "clip_adamerging/ViT-B-32/version_SUN397Cars/result.csv", "single"),
    ("Pareto Merging (MLP)", "clip_pareto_pm/version_20/result.csv", "set"),
    ("Pareto Merging (Attn+MLP)", "clip_pareto_pm/version_21/result.csv", "set"),
    ("PaLoRA (MLP)", "clip_pareto_palora/version_20/result.csv", "set"),
    ("PaLoRA (Attn+MLP)", "clip_pareto_palora/version_21/result.csv", "set"),
    ("Ours-LS (MLP)", "clip_pareto_moe_ls/version_50/result.csv", "set"),
    ("Ours-LS (Attn+MLP)", "clip_pareto_moe_ls/version_51/result.csv", "set"),
    ("Ours-EPO (MLP)", "clip_pareto_moe_epo/version_52/result.csv", "set"),
    ("Ours-EPO (Attn+MLP)", "clip_pareto_moe_epo/version_53/result.csv", "set"),
]


def main():
    results = []
    for label, rel, kind in ROWS:
        csv_path = RESULTS_DIR / rel
        if not csv_path.is_file():
            print(f"% missing: {label} ({rel})")
            continue
        _, accuracies = load_front(csv_path, TASKS, step=None)
        front = non_dominated(accuracies)
        spread = (accuracies.max(axis=0) - accuracies.min(axis=0)) * 100.0
        results.append(
            {
                "label": label,
                "kind": kind,
                "hv": hypervolume(accuracies),
                "points": len(accuracies),
                "nondominated": len(front),
                "spread": spread,
            }
        )

    if not results:
        print("% no results yet")
        return
    best = max(r["hv"] for r in results)
    print("% ---- hypervolume table rows ----")
    print(f"% columns: Method & #models & HV & range({TASKS[0]}) & range({TASKS[1]})")
    for r in results:
        hv = f"\\textbf{{{r['hv']:.4f}}}" if r["hv"] == best else f"{r['hv']:.4f}"
        points = "1" if r["kind"] == "single" else str(r["points"])
        if r["kind"] == "single":
            ranges = "--- & ---"
        else:
            ranges = f"{r['spread'][0]:.1f} & {r['spread'][1]:.1f}"
        print(f"      {r['label']:<28} & {points:>2} & {hv} & {ranges} \\\\")
    print(f"% best HV = {best:.4f}")


if __name__ == "__main__":
    main()
