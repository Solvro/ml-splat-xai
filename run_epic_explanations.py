import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict
import logging

def run_explain(python_exe: str, script: str, ply_path: str, output_path: str, num_prototypes: int, data_dir: str, save_viz: bool = False, pointnet_ckpt: str = 'pointnet_epic_compensated.pt') -> subprocess.CompletedProcess:
    cmd = [python_exe, script, "--ply_path", ply_path, "--output_path", output_path, "--num_prototypes", str(num_prototypes), "--data_dir", data_dir, "--pointnet_ckpt", pointnet_ckpt]
    if save_viz:
        cmd.append("--save_viz")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../archive/data/data/test",
                        help="Root test folder containing subdirectories (1,2,...)")
    parser.add_argument("--pointnet_ckpt", type=str, default="model_epic/CHINOL_kl_3-5_grid_7_1024-256_downsampled2/pointnet_epic_compensated.pt",
                        help="Path to pointnet")
    parser.add_argument('--num_prototypes', type=int, default=5,
                        help='number of prototypes to use')
    parser.add_argument('--data_dir', type=str, default='../archive/data/data/train',
                        help='directory of samples to choose from')
    parser.add_argument("--explanation_root", type=str, default="wcss/explaining/chinol/CHINOL_kl_3-5_grid_7_1024-256_downsampled2_VOXELIZATION_NO_EPIC",
                        help="Root directory where explanations (per-file dirs) are written")
    parser.add_argument('--save_viz', action='store_true', default=True,
                        help='Save point cloud visualizations')
    parser.add_argument("--script", type=str, default="./explain_epic.py",
                        help="Explanation script to run")
    parser.add_argument("--python_exe", type=str, default="python",
                        help="Python executable to run the script (Windows path shown)")
    parser.add_argument("--max_per_dir", type=int, default=5,
                        help="Max number of files to process per subdirectory")
    parser.add_argument("--merge_out", type=str, default="merged_inference_stats.json",
                        help="Output merged JSON file")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()


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
        res = run_explain(python_exe, script, ply, explanation_root, num_prototypes, data_dir, save_viz, pointnet_ckpt)
        if res.returncode != 0:
            print(f"  Error running script for {ply}:")
            print(res.stderr)
        else:
            print(f"  OK: {ply}")
            # optional: print stdout for debugging
            # print(res.stdout)

    # after all runs, collect inference_stats.json under explanation_root
    merged = collect_stats(explanation_root)
    merge_out = os.path.join(args.explanation_root, args.merge_out)
    with open(merge_out, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    print(f"Merged {len(merged)} inference_stats.json into {merge_out}")

if __name__ == "__main__":
    main()
