"""Deletion experiment: accuracy after removing the top-p% splats or the top-k voxels.

Run from the repo root (~/ml-splat-xai):
    python perturbation/run_removal.py --ckpt .../seed_777/stage2_compensated.pt --modes splat --out OUT
    python perturbation/run_removal.py --merge ROOT      # summarise every per_sample.csv under ROOT
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from perturbation.removal import (  # noqa: E402
    ShapeSplatTest, classify, importance, keep_splat_importance, keep_splat_random,
    keep_voxel_importance, keep_voxel_random, load_model, read_stage1_args,
)

DEFAULT_PERCENTS = "0,1,2,5,10,20,30,40,50,60,70,80,90"
DEFAULT_VOXEL_KS = "0,1,2,3,5,10,20,30,40,50,60,70,80,100"


def parse_list(s, typ):
    return [typ(x) for x in s.split(",") if x.strip() != ""]


def record(rows, base, strategy, rseed, level, s, keep, model, n_vox=None):
    n_left = int(keep.numel())
    if n_left == 0:
        pred, p_true = -1, 0.0
    else:
        logits = classify(model, s.subset(keep))
        pred = int(logits.argmax())
        p_true = float(torch.softmax(logits, 0)[s.label])
    rows.append({
        **base, "strategy": strategy, "rseed": rseed, "level": level,
        "n_removed": s.n - n_left, "n_left": n_left, "frac_left": n_left / s.n,
        "n_voxels_removed": n_vox, "pred": pred,
        "correct": int(pred == s.label), "agree_clean": int(pred == base["pred_clean"]),
        "p_true": p_true,
    })


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    a = read_stage1_args(args.ckpt)
    grid = args.grid_size or a.get("grid_size", 7)
    pooling = args.pooling or a.get("pooling", "max")
    stn_3d = a.get("stn_3d", True) if args.stn_3d is None else args.stn_3d
    stn_nd = a.get("stn_nd", True) if args.stn_nd is None else args.stn_nd
    head = a.get("head_size", 256)

    ds = ShapeSplatTest(args.data_dir, grid_size=grid, sampling=args.sampling,
                        num_points=args.num_points, seed=args.data_seed)
    if a.get("num_classes") not in (None, ds.num_classes):
        raise ValueError(f"ckpt has {a['num_classes']} classes, train dirs give {ds.num_classes}")
    model = load_model(args.ckpt, ds.num_classes, grid_size=grid, stn_3d=stn_3d, stn_nd=stn_nd,
                       pooling=pooling, head_size=head, device=device)

    percents = parse_list(args.percents, float)
    voxel_ks = parse_list(args.voxel_ks, int)
    modes = parse_list(args.modes, str)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {**vars(args), "grid_size": grid, "pooling": pooling, "stn_3d": stn_3d,
            "stn_nd": stn_nd, "head_size": head, "device": device, "n_test": len(ds),
            "num_classes": ds.num_classes,
            "notes": ("bs=1 exact deletion (mask=None); survivors keep original xyz_normalized and voxel_ids; "
                      "point scores = sum_{c in topK} w_c (U P)[c,n]; with max pooling U is applied after the "
                      "max, so point scores approximate each point's contribution")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    rows, max_diff, t0 = [], 0.0, time.time()
    for i in range(n):
        s = ds.get(i, device)
        imp = importance(model, s, args.topk_channels)
        if args.check:
            ref = classify(model, s)
            max_diff = max(max_diff, float((ref - imp.logits).abs().max()))
        base = {"ckpt_seed": args.ckpt_seed, "sample": i, "path": s.path, "label": s.label,
                "n_points": s.n, "pred_clean": imp.pred,
                "channels": " ".join(map(str, imp.channels.tolist()))}

        if "splat" in modes:
            for p in percents:
                keep = keep_splat_importance(imp, p)
                if args.check:
                    assert keep.numel() == s.n - round(s.n * p / 100)
                    assert torch.unique(keep).numel() == keep.numel()
                record(rows, {**base, "mode": "splat"}, "importance", -1, p, s, keep, model)
                for r in range(args.n_random):
                    rng = np.random.default_rng([args.random_seed, r, i, int(p * 100)])
                    keep = keep_splat_random(s.n, p, rng, device)
                    record(rows, {**base, "mode": "splat"}, "random", r, p, s, keep, model)
        if "voxel" in modes:
            for k in voxel_ks:
                keep, ke = keep_voxel_importance(imp, s, k)
                record(rows, {**base, "mode": "voxel"}, "importance", -1, k, s, keep, model, ke)
                for r in range(args.n_random):
                    rng = np.random.default_rng([args.random_seed, r, i, k])
                    keep, ke = keep_voxel_random(imp, s, k, rng)
                    record(rows, {**base, "mode": "voxel"}, "random", r, k, s, keep, model, ke)

        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"[{i + 1}/{n}] {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_sample.csv", index=False)
    summary(df).to_csv(out / "summary.csv", index=False)
    clean = df[df.level == 0].groupby("sample").first()
    msg = {"clean_acc": float((clean.pred_clean == clean.label).mean()), "n": int(len(clean))}
    if args.check:
        lvl0 = df[df.level == 0]
        msg["level0_all_equal_clean"] = bool((lvl0.pred == lvl0.pred_clean).all())
        msg["max_logit_diff_importance_vs_forward"] = max_diff
    print(json.dumps(msg, indent=2))
    (out / "check.json").write_text(json.dumps(msg, indent=2))


def summary(df: pd.DataFrame) -> pd.DataFrame:
    # accuracy per (ckpt seed, random seed) run, then mean/std over runs
    keys = ["mode", "strategy", "level"]
    per_run = (df.groupby(keys + ["ckpt_seed", "rseed"])
                 .agg(acc=("correct", "mean"), agree=("agree_clean", "mean"),
                      p_true=("p_true", "mean"), frac_left=("frac_left", "mean"),
                      n_voxels_removed=("n_voxels_removed", "mean"), n=("sample", "nunique"))
                 .reset_index())
    seed_level = per_run.groupby(keys + ["ckpt_seed"])["acc"].mean().reset_index()
    out = (per_run.groupby(keys)
             .agg(acc_mean=("acc", "mean"), acc_std_runs=("acc", "std"),
                  agree_mean=("agree", "mean"), p_true_mean=("p_true", "mean"),
                  frac_left_mean=("frac_left", "mean"),
                  n_voxels_removed_mean=("n_voxels_removed", "mean"),
                  n_runs=("acc", "size"), n_samples=("n", "max"))
             .reset_index())
    std_ckpt = seed_level.groupby(keys)["acc"].std().rename("acc_std_ckpt_seeds").reset_index()
    return out.merge(std_ckpt, on=keys).sort_values(keys)


def merge(root):
    files = sorted(Path(root).rglob("per_sample.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    s = summary(df)
    s.to_csv(Path(root) / "summary_all_seeds.csv", index=False)
    print(f"merged {len(files)} files, seeds={sorted(df.ckpt_seed.unique())}")
    print(s.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", type=str, default=None)
    ap.add_argument("--ckpt", type=str)
    ap.add_argument("--ckpt_seed", type=int, default=-1)
    ap.add_argument("--data_dir", type=str,
                    default="/lustre/pd03/hpc-danbor2008-1756464546/datasets/SHAPESPLAT/SHAPESPLAT_ply")
    ap.add_argument("--out", type=str)
    ap.add_argument("--modes", type=str, default="splat,voxel")
    ap.add_argument("--percents", type=str, default=DEFAULT_PERCENTS)
    ap.add_argument("--voxel_ks", type=str, default=DEFAULT_VOXEL_KS)
    ap.add_argument("--topk_channels", type=int, default=5)
    ap.add_argument("--n_random", type=int, default=5)
    ap.add_argument("--random_seed", type=int, default=0)
    ap.add_argument("--sampling", choices=["random", "original_size"], default="random")
    ap.add_argument("--num_points", type=int, default=8192)
    ap.add_argument("--data_seed", type=int, default=777)
    ap.add_argument("--grid_size", type=int, default=None)
    ap.add_argument("--pooling", type=str, default=None)
    ap.add_argument("--stn_3d", type=lambda x: x.lower() == "true", default=None)
    ap.add_argument("--stn_nd", type=lambda x: x.lower() == "true", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge(args.merge)
    else:
        run(args)


if __name__ == "__main__":
    main()
