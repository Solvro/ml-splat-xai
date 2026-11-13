"""Plot accuracy-vs-removal curves from summary_all_seeds.csv (or summary.csv).

    python perturbation/plot_removal.py ROOT/summary_all_seeds.csv --out ROOT/removal_curves
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"importance": "#c0392b", "random": "#2c3e50"}
XLABEL = {"splat": "splats removed [%]", "voxel": "voxels removed (top-k)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="acc", choices=["acc", "agree", "p_true"])
    a = ap.parse_args()
    df = pd.read_csv(a.summary)
    out = Path(a.out or Path(a.summary).with_suffix(""))
    modes = [m for m in ("splat", "voxel") if m in set(df["mode"])]
    fig, axes = plt.subplots(1, len(modes), figsize=(5.5 * len(modes), 4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        for strat, g in df[df["mode"] == mode].groupby("strategy"):
            g = g.sort_values("level")
            y = g[f"{a.metric}_mean"]
            ax.plot(g["level"], y, "o-", ms=3, color=COLORS.get(strat), label=strat)
            if a.metric == "acc":
                sd = g["acc_std_runs"].fillna(0)
                ax.fill_between(g["level"], y - sd, y + sd, color=COLORS.get(strat), alpha=0.15)
        ax.set_xlabel(XLABEL[mode])
        ax.set_ylabel({"acc": "test accuracy", "agree": "agreement with clean prediction",
                       "p_true": "p(true class)"}[a.metric])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title(f"{mode} removal")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}_{a.metric}.{ext}", dpi=150)
    print(f"saved {out}_{a.metric}.png/.pdf")


if __name__ == "__main__":
    main()
