import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict
import logging

def run_explain(
    python_exe: str,
    script: str,
    ply_path: str,
    output_path: str,
    num_prototypes: int,
    data_dir: str,
    pointnet_ckpt: str,
    save_viz: bool = False,
    config: str = None,
    grid_size: int = None,
    num_channels: int = None,
    head_size: int = None,
    sampling: str = None,
    num_samples: int = None,
    stn_3d: bool = False,
    stn_nd: bool = False,
) -> subprocess.CompletedProcess:
    cmd = [
        python_exe, script,
        "--ply_path", ply_path,
        "--output_path", output_path,
        "--num_prototypes", str(num_prototypes),
        "--data_dir", data_dir,
        "--pointnet_ckpt", pointnet_ckpt
    ]

    if config:
        cmd.extend(["--config", config])
    if grid_size is not None:
        cmd.extend(["--grid_size", str(grid_size)])
    if num_channels is not None:
        cmd.extend(["--num_channels", str(num_channels)])
    if head_size is not None:
        cmd.extend(["--head_size", str(head_size)])
    if sampling is not None:
        cmd.extend(["--sampling", sampling])
    if num_samples is not None:
        cmd.extend(["--num_samples", str(num_samples)])
    if stn_3d:
        cmd.append("--stn_3d")
    if stn_nd:
        cmd.append("--stn_nd")
    if save_viz:
        cmd.append("--save_viz")
    else:
        cmd.append("--no_viz")

    return subprocess.run(cmd, capture_output=False, text=False)

def collect_stats(explanation_root: Path) -> Dict[str, dict]:
    merged: Dict[str, dict] = {}
    for p in explanation_root.rglob("inference_stats.json"):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        # infer filename key from parent folder or from file path
        # parent name is expected to be the base filename (e.g. cup_001)
        key = p.parent.name
        # fallback: use filename without extension if parent is not informative
        if not key:
            key = p.stem
        merged[key] = data
    return merged

def main():
    parser = argparse.ArgumentParser(
        description="Batch explanation runner for XSPLAIN"
    )

    # Config file support
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file for explain.py (passed through)")

    # Data paths
    parser.add_argument("--data_root", type=str, default="data/test",
                        help="Root test folder containing subdirectories with PLY files")
    parser.add_argument("--data_dir", type=str, default="data/train",
                        help="Directory of reference samples for prototypes")
    parser.add_argument("--explanation_root", type=str, required=True,
                        help="Root directory where explanations are written")

    # Model checkpoint
    parser.add_argument("--pointnet_ckpt", type=str, required=True,
                        help="Path to trained PointNet + Disentangler checkpoint")

    # Model architecture (passed to explain.py)
    parser.add_argument("--grid_size", type=int, default=None,
                        help="Voxel grid size (must match training)")
    parser.add_argument("--num_channels", type=int, default=None,
                        help="Number of feature channels (must match training)")
    parser.add_argument("--head_size", type=int, default=None,
                        help="Size of classification head (must match training)")

    # Model architecture flags
    parser.add_argument("--stn_3d", action="store_true",
                        help="Enable 3D Spatial Transformer Network (must match training)")
    parser.add_argument("--stn_nd", action="store_true",
                        help="Enable feature-space STN (must match training)")

    # Data processing
    parser.add_argument("--sampling", type=str, default=None,
                        choices=["fps", "random", "original_size"],
                        help="Point sampling method")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of points to sample")

    # Explanation settings
    parser.add_argument("--num_prototypes", type=int, default=5,
                        help="Number of prototypes per channel")

    # Output settings
    parser.add_argument("--save_viz", action="store_true", default=True,
                        help="Save point cloud visualizations")
    parser.add_argument("--no_viz", action="store_true",
                        help="Disable visualization output")
    parser.add_argument("--merge_out", type=str, default="merged_inference_stats.json",
                        help="Output filename for merged stats")

    # Execution settings
    parser.add_argument("--script", type=str, default="explain.py",
                        help="Explanation script to run")
    parser.add_argument("--python_exe", type=str, default="python",
                        help="Python executable path")
    parser.add_argument("--max_per_dir", type=int, default=5,
                        help="Max number of files to process per subdirectory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print files without processing")

    args = parser.parse_args()

    # Handle --no_viz flag
    if args.no_viz:
        args.save_viz = False


    data_root = Path(args.data_root)
    script = args.script
    python_exe = args.python_exe
    num_prototypes = args.num_prototypes
    data_dir = args.data_dir
    save_viz = args.save_viz
    explanation_root = Path(args.explanation_root)
    pointnet_ckpt = Path(args.pointnet_ckpt)
    explanation_root.mkdir(parents=True, exist_ok=True)

    # gather first N .ply files from each subdir
    to_process = []
    for sub in sorted(os.listdir(data_root)):
        subp = data_root / sub
        if not subp.is_dir():
            continue
        ply_files = sorted([f for f in os.listdir(subp) if f.lower().endswith(".ply")])
        for fname in ply_files[: args.max_per_dir]:
            to_process.append(str(subp / fname))
            logging.info(str(fname))

    print(f"Found {len(to_process)} files to process (max {args.max_per_dir} per subdir).")

    for ply in to_process:
        print(f"Processing: {ply}")
        if args.dry_run:
            continue
        res = run_explain(
            python_exe=python_exe,
            script=script,
            ply_path=ply,
            output_path=str(explanation_root),
            num_prototypes=num_prototypes,
            data_dir=data_dir,
            pointnet_ckpt=str(pointnet_ckpt),
            save_viz=save_viz,
            config=args.config,
            grid_size=args.grid_size,
            num_channels=args.num_channels,
            head_size=args.head_size,
            sampling=args.sampling,
            num_samples=args.num_samples,
            stn_3d=args.stn_3d,
            stn_nd=args.stn_nd,
        )
        if res.returncode != 0:
            print(f"  Error running script for {ply}:")
            print(res.stderr)
        else:
            print(f"  OK: {ply}")

    merged = collect_stats(explanation_root)
    merge_out = os.path.join(args.explanation_root, args.merge_out)
    with open(merge_out, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    print(f"Merged {len(merged)} inference_stats.json into {merge_out}")

if __name__ == "__main__":
    main()
