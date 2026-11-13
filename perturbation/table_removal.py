"""Accuracy tables (mean ± std) per removal mode, importance vs random, from summary_all_seeds.csv.

    python perturbation/table_removal.py ROOT/summary_all_seeds.csv --out ROOT/removal_table
        -> ROOT/removal_table_{splat,voxel}.{csv,tex,md}

std = acc_std_runs, i.e. the same band as in plot_removal.py
(importance: across the 3 checkpoints; random: across 3 checkpoints x n_random draws).
"""
import argparse
from pathlib import Path

import pandas as pd

LEVEL = {"splat": "removed [%]", "voxel": "non-empty voxels removed [%]"}
LEVEL_VOXEL_COUNT = "voxels removed (k)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", default=None)
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--voxel_unit", choices=["percent", "count"], default="percent")
    a = ap.parse_args()
    if a.voxel_unit == "count":
        LEVEL["voxel"] = LEVEL_VOXEL_COUNT
    df = pd.read_csv(a.summary)
    out = Path(a.out or Path(a.summary).with_name("removal_table"))
    d = a.digits
    for mode in [m for m in ("splat", "voxel") if m in set(df["mode"])]:
        g = df[df["mode"] == mode]
        t = g.pivot(index="level", columns="strategy", values=["acc_mean", "acc_std_runs"]).sort_index()
        num = pd.DataFrame({
            LEVEL[mode]: t.index.astype(int),
            "importance_acc": t[("acc_mean", "importance")].values,
            "importance_std": t[("acc_std_runs", "importance")].values,
            "random_acc": t[("acc_mean", "random")].values,
            "random_std": t[("acc_std_runs", "random")].values,
        })
        num.round(d + 1).to_csv(f"{out}_{mode}.csv", index=False)
        pm = lambda m, s: f"{m:.{d}f} ± {s:.{d}f}"  # noqa: E731
        md = ["| " + LEVEL[mode] + " | importance | random |", "|---:|---:|---:|"]
        tex = [r"\begin{tabular}{rcc}", r"\toprule",
               LEVEL[mode].replace("%", r"\%") + r" & importance & random \\", r"\midrule"]
        for _, r in num.iterrows():
            lv = int(r[LEVEL[mode]])
            md.append(f"| {lv} | {pm(r.importance_acc, r.importance_std)} | {pm(r.random_acc, r.random_std)} |")
            tex.append(f"{lv} & ${r.importance_acc:.{d}f} \\pm {r.importance_std:.{d}f}$ & "
                       f"${r.random_acc:.{d}f} \\pm {r.random_std:.{d}f}$ \\\\")
        tex += [r"\bottomrule", r"\end{tabular}"]
        Path(f"{out}_{mode}.md").write_text("\n".join(md) + "\n")
        Path(f"{out}_{mode}.tex").write_text("\n".join(tex) + "\n")
        print(f"== {mode}\n" + "\n".join(md) + f"\nsaved {out}_{mode}.csv/.tex/.md")


if __name__ == "__main__":
    main()
