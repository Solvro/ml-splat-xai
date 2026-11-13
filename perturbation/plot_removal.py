"""Plot accuracy-vs-removal curves from summary_all_seeds.csv (or summary.csv).

    python perturbation/plot_removal.py ROOT/summary_all_seeds.csv --out ROOT/removal_curves
    python perturbation/plot_removal.py ROOT/summary_all_seeds.csv --out ROOT/removal_curves --separate
        -> ROOT/removal_curves_{splat,voxel}_<metric>.{png,pdf}
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"importance": "#c0392b", "random": "#2c3e50"}
XLABEL = {"splat": "splats removed [%]", "voxel": "voxels removed (top-k)"}
YLABEL = {"acc": "test accuracy", "agree": "agreement with clean prediction", "p_true": "accuracy"}


def draw(ax, df, mode, metric):
    for strat, g in df[df["mode"] == mode].groupby("strategy"):
        g = g.sort_values("level")
        y = g[f"{metric}_mean"]
        ax.plot(g["level"], y, "o-", ms=3, color=COLORS.get(strat), label=strat)
        if metric == "acc":
            sd = g["acc_std_runs"].fillna(0)
            ax.fill_between(g["level"], y - sd, y + sd, color=COLORS.get(strat), alpha=0.15)
    ax.set_xlabel(XLABEL[mode])
    ax.set_ylabel(YLABEL[metric])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"{mode} removal")


def save(fig, stem):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    plt.close(fig)
    print(f"saved {stem}.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="acc", choices=["acc", "agree", "p_true"])
    ap.add_argument("--separate", action="store_true", help="one figure per mode instead of side-by-side")
    a = ap.parse_args()
    df = pd.read_csv(a.summary)
    out = Path(a.out or Path(a.summary).with_suffix(""))
    modes = [m for m in ("splat", "voxel") if m in set(df["mode"])]
    if a.separate:
        for mode in modes:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            draw(ax, df, mode, a.metric)
            save(fig, f"{out}_{mode}_{a.metric}")
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(5.5 * len(modes), 4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        draw(ax, df, mode, a.metric)
    save(fig, f"{out}_{a.metric}")


if __name__ == "__main__":
    main()
