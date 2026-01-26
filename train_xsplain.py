import argparse
from pathlib import Path
import subprocess
import sys

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_stage_1_command(config: dict) -> list[str]:
    cmd = [sys.executable, "train_stage_1_backbone.py"]

    arg_mapping = {
        "data_dir": "--data_dir",
        "device": "--device",
        "epochs": "--epochs",
        "min_epochs": "--min_epochs",
        "batch_size": "--batch_size",
        "lr": "--lr",
        "val_split": "--val_split",
        "workers": "--workers",
        "accelerator": "--accelerator",
        "sampling": "--sampling",
        "num_points": "--num_points",
        "grid_size": "--grid_size",
        "model_save_path": "--model_save_path",
        "model_save_name": "--model_save_name",
        "count_penalty_type": "--count_penalty_type",
        "count_penalty_weight": "--count_penalty_weight",
        "count_penalty_beta": "--count_penalty_beta",
        "count_penalty_tau": "--count_penalty_tau",
        "pooling": "--pooling",
        "head_size": "--head_size"
    }

    bool_flags = {
        "stn_3d": "--stn_3d",
        "stn_nd": "--stn_nd",
        "stn_head_norm": "--stn_head_norm",
        "fast_dev_run": "--fast_dev_run",
    }

    for key, arg in arg_mapping.items():
        if key in config and config[key] is not None:
            cmd.extend([arg, str(config[key])])

    for key, arg in bool_flags.items():
        if config.get(key, False):
            cmd.append(arg)

    return cmd


def build_stage_2_command(config: dict, stage_1_checkpoint: str | None = None) -> list[str]:
    cmd = [sys.executable, "train_stage_2_disentangler.py"]

    checkpoint = stage_1_checkpoint or config.get("pointnet_ckpt")
    if checkpoint:
        cmd.extend(["--pointnet_ckpt", str(checkpoint)])

    arg_mapping = {
        "data_dir": "--data_dir",
        "val_split": "--val_split",
        "sampling": "--sampling",
        "num_samples": "--num_samples",
        "grid_size": "--grid_size",
        "epochs": "--epochs",
        "batch_size": "--batch_size",
        "num_workers": "--num_workers",
        "lr": "--lr",
        "accumulate_grad_batches": "--accumulate_grad_batches",
        "prototype_update_freq": "--prototype_update_freq",
        "initial_topk": "--initial_topk",
        "final_topk": "--final_topk",
        "num_channels": "--num_channels",
        "early_stopping_patience": "--early_stopping_patience",
        "output_dir": "--output_dir",
        "log_name": "--log_name",
        "viz_channels": "--viz_channels",
        "viz_topk": "--viz_topk",
    }

    for key, arg in arg_mapping.items():
        if key in config and config[key] is not None:
            cmd.extend([arg, str(config[key])])

    return cmd


def find_best_checkpoint(checkpoint_dir: str, model_name: str) -> str | None:
    checkpoint_path = Path(checkpoint_dir)

    exact_path = checkpoint_path / f"{model_name}.ckpt"
    if exact_path.exists():
        return str(exact_path)

    ckpt_files = list(checkpoint_path.glob("*.ckpt"))
    if not ckpt_files:
        return None

    return str(max(ckpt_files, key=lambda p: p.stat().st_mtime))


def run_command(cmd: list[str], stage_name: str) -> int:
    print(f"Starting {stage_name}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n{stage_name} failed with exit code {result.returncode}")
    else:
        print(f"\n{stage_name} completed successfully")

    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full training pipeline (Stage 1 + Stage 2)"
    )
    parser.add_argument(
        "--config_stage_1",
        type=str,
        default="config/stage_1.yaml",
        help="Path to stage 1 configuration file"
    )
    parser.add_argument(
        "--config_stage_2",
        type=str,
        default="config/stage_2.yaml",
        help="Path to stage 2 configuration file"
    )
    parser.add_argument(
        "--skip_stage_1",
        action="store_true",
        help="Skip stage 1 and use existing checkpoint"
    )
    parser.add_argument(
        "--skip_stage_2",
        action="store_true",
        help="Skip stage 2 (only run backbone training)"
    )
    parser.add_argument(
        "--stage_1_checkpoint",
        type=str,
        default=None,
        help="Override stage 1 checkpoint path for stage 2"
    )

    args = parser.parse_args()

    print("Loading configurations:")
    config_stage_1 = load_config(args.config_stage_1)
    config_stage_2 = load_config(args.config_stage_2)

    stage_1_checkpoint = args.stage_1_checkpoint

    if not args.skip_stage_1:
        cmd_stage_1 = build_stage_1_command(config_stage_1)
        exit_code = run_command(cmd_stage_1, "Stage 1: PointNet Backbone Training")

        if exit_code != 0:
            print("Pipeline aborted due to Stage 1 failure")
            sys.exit(exit_code)

        if stage_1_checkpoint is None:
            checkpoint_dir = config_stage_1.get("model_save_path", "checkpoints/stage_1")
            model_name = config_stage_1.get("model_save_name", "pointnet_backbone")
            stage_1_checkpoint = find_best_checkpoint(checkpoint_dir, model_name)

            if stage_1_checkpoint is None:
                print(f"Error: Could not find checkpoint in {checkpoint_dir}")
                sys.exit(1)

            print(f"Found Stage 1 checkpoint: {stage_1_checkpoint}")
    else:
        print("Skipping Stage 1 (--skip_stage_1 flag set)")
        if stage_1_checkpoint is None:
            stage_1_checkpoint = config_stage_2.get("pointnet_ckpt")

        if stage_1_checkpoint is None or not Path(stage_1_checkpoint).exists():
            print(f"Error: Stage 1 checkpoint not found: {stage_1_checkpoint}")
            return

    if not args.skip_stage_2:
        if "grid_size" in config_stage_1:
            config_stage_2["grid_size"] = config_stage_1["grid_size"]

        cmd_stage_2 = build_stage_2_command(config_stage_2, stage_1_checkpoint)
        exit_code = run_command(cmd_stage_2, "Stage 2: Disentangler Training")

        if exit_code != 0:
            print("Pipeline completed with Stage 2 failure")
            return
    else:
        print("Skipping Stage 2 (--skip_stage_2 flag set)")

    print("Pipeline completed successfully!")

    if not args.skip_stage_1:
        print()
        print("Stage 1 outputs:")
        print(f"Checkpoint: {stage_1_checkpoint}")

    if not args.skip_stage_2:
        output_dir = config_stage_2.get("output_dir", "checkpoints/stage_2")
        print()
        print("Stage 2 outputs:")
        print(f"Output directory: {output_dir}")
        print(f"Compensated model: {output_dir}/pointnet_disentangler_compensated.pt")
        print(f"Orthogonal matrix: {output_dir}/final_orthogonal_matrix.pt")


if __name__ == "__main__":
    main()
