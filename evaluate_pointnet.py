import argparse
import os
import json
import pathlib

import torch
from torch import set_float32_matmul_precision
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from pointnet.pointnet import PointNetLightning
from pointnet.dataset import GaussianDataModule, FEATURE_NAMES

set_float32_matmul_precision("medium")


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_split(
    lit_model: PointNetLightning,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str,
    max_print: int = 10,
    save_json: bool = False,      
    output_dir: str = ".",        
):
    if dataloader is None:
        print(f"[{split_name}] No dataloader – skipping.")
        return

    lit_model.eval()
    lit_model.to(device)

    all_preds = []
    all_labels = []

    n_samples = 0
    n_correct = 0

    printed = 0

    records = []

    print(f"\n=== Evaluating split: {split_name} ===")

    for batch_idx, batch in enumerate(dataloader):
        features = batch["gauss"].to(device)
        xyz = batch["xyz_normalized"].to(device)
        labels = batch["label"].to(device)
        mask = batch["mask"].to(device)

        # NOTE: model needs to return addionally `voxel_features` so that we can save them
        # modify PointNet classes accordingly
        logits, global_features, voxel_activations_3d, point_counts, indices, voxel_features = lit_model.model(
            features, xyz, mask
        )
        preds = torch.argmax(logits, dim=1)
        n_samples += labels.numel()
        n_correct += (preds == labels).sum().item()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())


        if save_json:  # NEW
            B = labels.shape[0]
            for i in range(B):
                rec = {
                    "batch_idx": int(batch_idx),
                    "label": int(labels[i].item()),
                    "logits": logits[i].detach().cpu().numpy().tolist(),
                    "global_features": global_features[i].detach().cpu().numpy().tolist(),
                    "voxel_activations_3d": voxel_activations_3d[i].detach().cpu().numpy().tolist(),
                    "point_counts": point_counts[i].detach().cpu().numpy().tolist(),
                    "indices": indices[i].detach().cpu().numpy().tolist(),
                    "voxel_features": voxel_features[i].detach().cpu().numpy().tolist(),
                }
                records.append(rec)

        if printed < max_print:
            for i in range(labels.shape[0]):
                if printed >= max_print:
                    break
                print(
                    f"[{split_name}] sample {batch_idx:04d}/{i:03d} | "
                    f"true={labels[i].item()} pred={preds[i].item()}, "
                    f"logits={logits[i].cpu().numpy()}"
                )
                printed += 1

    if n_samples == 0:
        print(f"[{split_name}] No samples in this split.")
        return

    acc = n_correct / float(n_samples)
    print(f"\n[{split_name}] Accuracy: {acc:.4f} ({n_correct}/{n_samples})")

    if save_json:
        os.makedirs(pathlib.Path(output_dir).parent, exist_ok=True)
        out_path = os.path.join(output_dir)
        payload = {
            "split": split_name,
            "num_samples": len(records),
            "accuracy": acc,
            "records": records,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[{split_name}] Saved activations to: {out_path}")


def main():
    pl.seed_everything(777)

    parser = argparse.ArgumentParser(description="Evaluate PointNet checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory with data")
    parser.add_argument("--device", type=str, default="auto", help="cpu / cuda / auto")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--sampling",
        type=str,
        default="random",
        choices=["fps", "random", "original_size"],
        help="Point sampling method",
    )
    parser.add_argument("--num_points", type=int, default=8192, help="Number of points if sampling is used")
    parser.add_argument("--grid_size", type=int, default=10, help="Size of the voxel grid for aggregation")
    parser.add_argument(
        "--stn_3d",
        action="store_true",
        help="Must match training: use 3D STN layer",
    )
    parser.add_argument(
        "--stn_nd",
        action="store_true",
        help="Must match training: use feature STN layer",
    )
    parser.add_argument(
        "--head_norm",
        action="store_true",
        help="Must match training: use head LayerNorm",
    )
    parser.add_argument(
        "--stn_head_norm",
        action="store_true",
        help="Must match training: use STN head norm",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="max",
        choices=["max", "avg"],
        help="Pooling used in voxel aggregation (must match training)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to .ckpt checkpoint file",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to evaluate",
    )
    parser.add_argument(
        "--max_print",
        type=int,
        default=20,
        help="How many predictions to print per split",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="If set, saves activations_{split}.json for each evaluated split",
    )
    parser.add_argument(
        "--json_output_dir",
        type=str,
        default=".",   # NEW
        help="Directory to store activations_{split}.json files",
    )

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    datamodule = GaussianDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        val_split=args.val_split,
        sampling=args.sampling,
        num_points=args.num_points,
    )
    datamodule.setup()  # creates train/val/test splits

    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    if args.checkpoint_path.endswith(".ckpt"):
        lit_model: PointNetLightning = PointNetLightning.load_from_checkpoint(
            args.checkpoint_path,
            in_dim=len(FEATURE_NAMES),
            num_classes=datamodule.num_classes,
            grid_size=args.grid_size,
            stn_3d=args.stn_3d,
            stn_nd=args.stn_nd,
            head_norm=args.head_norm,
            stn_head_norm=args.stn_head_norm,
            pooling=args.pooling,
        )

        print(f"Loaded checkpoint from: {args.checkpoint_path}")
        print(f"Model hyperparameters: {lit_model.hparams}")

    elif args.checkpoint_path.endswith(".pt"):
        pt_state_dict = torch.load(args.checkpoint_path)

        lit_model: PointNetLightning = PointNetLightning(
            in_dim=len(FEATURE_NAMES),
            num_classes=datamodule.num_classes,
            grid_size=args.grid_size,
            stn_3d=args.stn_3d,
            stn_nd=args.stn_nd,
            head_norm=args.head_norm,
            stn_head_norm=args.stn_head_norm,
            pooling=args.pooling,
        )
        num_channels = 256
        lit_model.model.attach_epic(num_channels)
        print(f"Loaded .pt PointNet state_dict from: {args.checkpoint_path}")
        print('loading model: Epic matrix attached: ')
        lit_model.model.load_state_dict(pt_state_dict['pointnet_state_dict'])
        print(lit_model.model.epic.get_weight())
        lit_model.eval()

    if "train" in args.splits:
        evaluate_split(
            lit_model,
            datamodule.train_dataloader(),
            device,
            split_name="train",
            max_print=args.max_print,
            save_json=args.save_json,
            output_dir=args.json_output_dir,
        )

    if "val" in args.splits:
        evaluate_split(
            lit_model,
            datamodule.val_dataloader(),
            device,
            split_name="val",
            max_print=args.max_print,
            save_json=args.save_json,
            output_dir=args.json_output_dir,
        )

    if "test" in args.splits:
        evaluate_split(
            lit_model,
            datamodule.test_dataloader(),
            device,
            split_name="test",
            max_print=args.max_print,
            save_json=args.save_json,
            output_dir=args.json_output_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
