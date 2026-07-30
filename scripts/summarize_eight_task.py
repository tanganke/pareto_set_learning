R"""
Summarize the eight-task ``result.csv`` files produced with
``num_evaluation_samples=equal_weight+priority``.

For every run we report

* ``equal``   -- the eight accuracies obtained with the equal-weight preference,
* ``priority``-- the accuracy of task :math:`j` under the preference vector that puts
  weight 0.5 on task :math:`j` (the "priority preference" protocol of Pareto
  Merging / ReACT), i.e. the diagonal of the priority block,
* ``control`` -- how much prioritizing a task gains over the average of the runs
  that prioritize a *different* task. This measures how much preference control
  the method actually offers.

Usage::

    python scripts/summarize_eight_task.py --latex
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

TASKS = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]

# label -> result directory (relative to results/)
RUNS = {
    "Ours-LS (MLP)": "clip_pareto_moe_ls/version_10",
    "Ours-LS (Attn+MLP)": "clip_pareto_moe_ls/version_11",
    "Pareto Merging (MLP)": "clip_pareto_pm/version_10",
    "Pareto Merging (Attn+MLP)": "clip_pareto_pm/version_11",
    "PaLoRA (MLP)": "clip_pareto_palora/version_12",
    "PaLoRA (Attn+MLP)": "clip_pareto_palora/version_13",
    "Ours-LS (MLP, AM base)": "clip_pareto_moe_ls/version_30",
    "Ours-LS (Attn+MLP, AM base)": "clip_pareto_moe_ls/version_31",
    "Pareto Merging (Attn+MLP, AM base)": "clip_pareto_pm/version_31",
    "PaLoRA (Attn+MLP, AM base)": "clip_pareto_palora/version_33",
    "Ours-LS (MLP), ViT-L/14": "clip_pareto_moe_ls/version_40",
    "Ours-LS (Attn+MLP), ViT-L/14": "clip_pareto_moe_ls/version_41",
    "Pareto Merging (Attn+MLP), ViT-L/14": "clip_pareto_pm/version_41",
}


def summarize(csv_path: Path):
    df = pd.read_csv(csv_path)
    tasks = [t for t in TASKS if t in df.columns]
    num_objectives = len(tasks)
    ray_cols = [f"ray_{i}" for i in range(num_objectives)]
    rays = df[ray_cols].to_numpy(dtype=float)
    accuracies = df[tasks].to_numpy(dtype=float) * 100.0

    equal_mask = np.all(np.isclose(rays, 1.0 / num_objectives, atol=1e-3), axis=1)
    equal = accuracies[equal_mask][0] if equal_mask.any() else None

    # priority rows: the row whose maximum weight sits on task j
    priority = np.full(num_objectives, np.nan)
    off_priority = np.full(num_objectives, np.nan)
    priority_rows = np.where(~equal_mask)[0]
    owner = {int(rays[i].argmax()): i for i in priority_rows}
    for j in range(num_objectives):
        if j not in owner:
            continue
        priority[j] = accuracies[owner[j], j]
        others = [owner[k] for k in owner if k != j]
        if others:
            off_priority[j] = accuracies[others, j].mean()

    return {
        "tasks": tasks,
        "equal": equal,
        "priority": priority,
        "control": priority - off_priority,
    }


def fmt(values, bold_idx=None):
    out = []
    for i, v in enumerate(values):
        if np.isnan(v):
            out.append("--")
        elif bold_idx is not None and i == bold_idx:
            out.append(f"\\textbf{{{v:.1f}}}")
        else:
            out.append(f"{v:.1f}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    rows = []
    for label, rel in RUNS.items():
        csv_path = RESULTS_DIR / rel / "result.csv"
        if not csv_path.is_file():
            print(f"# missing: {label} ({rel})")
            continue
        s = summarize(csv_path)
        for kind in ("equal", "priority"):
            values = s[kind]
            if values is None:
                continue
            rows.append(
                {
                    "method": label,
                    "preference": kind,
                    **{t: v for t, v in zip(s["tasks"], values)},
                    "Avg.": float(np.nanmean(values)),
                }
            )
        rows.append(
            {
                "method": label,
                "preference": "control",
                **{t: v for t, v in zip(s["tasks"], s["control"])},
                "Avg.": float(np.nanmean(s["control"])),
            }
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(df.round(2).to_string(index=False))

    if args.latex:
        print("\n% ---- LaTeX rows ----")
        for _, r in df.iterrows():
            cells = [f"{r[t]:.1f}" if not pd.isna(r[t]) else "--" for t in TASKS if t in r]
            print(
                f"{r['method']} ({r['preference']}) & "
                + " & ".join(cells)
                + f" & {r['Avg.']:.1f} \\\\"
            )

    out = RESULTS_DIR / "_summary_eight_task.csv"
    df.to_csv(out, index=False)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
