import json
import argparse
import os
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pointnet.datasets.gaussian_point_cloud import FEATURE_NAMES, GaussianDataModule, collate_fn
from pointnet.pointnet import PointNetLightning
from pointnet.callbacks.disentangler_visualization import DisentanglerVisualizationCallback, load_and_preprocess_ply
from train_stage_2_disentangler import DisentanglerTrainer


def _compute_voxel_ids_np(xyz_normalized: np.ndarray, grid_size: int) -> np.ndarray:
    gs = grid_size
    coords = np.floor(xyz_normalized * gs).astype(np.int64)
    coords = np.clip(coords, 0, gs - 1)
    voxel_ids = coords[:, 0] * (gs * gs) + coords[:, 1] * gs + coords[:, 2]
    return voxel_ids


def topk_active_channels(dis_trainer, ply_path, ds, topk, device, grid_size, do_sample=False):
    data = load_and_preprocess_ply(Path(ply_path))
    gauss = data["gauss"]
    xyz_normalized = data["xyz_normalized"]
    if do_sample and hasattr(ds, "_random_sample"):
        ids = ds._random_sample(gauss)
    else:
        ids = np.arange(gauss.shape[0])

    gauss_sub = gauss[ids]                         # (M, F)
    xyz_norm_sub = xyz_normalized[ids]            # (M, 3)

    features = gauss_sub.unsqueeze(0).transpose(1, 2).to(device)   # (1, F, M)
    xyz_norm_t = xyz_norm_sub.unsqueeze(0).to(device)              # (1, M, 3)
    mask = data.get("mask", None)
    if mask is not None:
        mask = mask[ids].to(device)

    voxel_ids_np = _compute_voxel_ids_np(xyz_norm_sub.numpy(), grid_size)  # (M,)
    voxel_ids = torch.from_numpy(voxel_ids_np).long().unsqueeze(0).to(device)  # (1, M)

    with torch.no_grad():
        point_features, _ = dis_trainer.pointnet.extract_point_features(
            features, xyz_norm_t, mask
        )
        voxel_features, _, _ = dis_trainer.pointnet.voxel_agg(
            point_features, voxel_ids
        )
        voxel_features_flat = voxel_features.view(
            voxel_features.size(0),
            voxel_features.size(1),
            -1
        ) 
        voxel_features_flat = dis_trainer.disentangler(voxel_features_flat)

        global_features = F.adaptive_avg_pool1d(voxel_features_flat, 1).squeeze(-1)  # (B, C)
        logits = dis_trainer.pointnet.head(global_features)  # (B, num_classes)
        predicted_class = logits.argmax(dim=-1).item()
        print(f"Predicted class: {predicted_class} (logits: {logits[0].cpu().numpy()})")

        classifier_linear = dis_trainer.pointnet.head

        class_weights = classifier_linear.weight[predicted_class]  # (C,)

        weighted_global_features = class_weights * F.relu(global_features.squeeze(0))  # (C,)

        _, channels = torch.topk(weighted_global_features, topk, largest=True, sorted=True)

    return channels.tolist()


def explain_prediction(dis_trainer, ply_path, ds, topk, device, grid_size, do_sample=False):
    prototypes = getattr(dis_trainer, "last_val_prototypes", None)
    if prototypes is None:
        print("No stored prototypes found")
        prototypes = {}

    channels = topk_active_channels(dis_trainer, ply_path, ds, topk, device, grid_size, do_sample=do_sample)
    print(f"Max active channels are {channels}")
    new_prototypes = {c: [-1] for c in channels}
    for c in channels:
        indices_for_c = prototypes.get(c, [])[: topk]
        new_prototypes[c].extend(indices_for_c)

    print(f"New prototypes: {new_prototypes}")
    dis_trainer.last_val_prototypes = new_prototypes


def ammend_dataset_files(dataset, ply_path):
    cls_name = ply_path.replace("\\", "/").split("/")[-2]
    dataset.files.append((ply_path, dataset.class_to_idx[cls_name]))


def get_inference_stats(val_prototypes, dataset):
    info = {channel: {} for channel in val_prototypes}
    for channel in val_prototypes:
        samples = val_prototypes[channel][1:]
        info[channel]["samples"] = samples
        info[channel]["classes"] = [
            dataset.classes[dataset.files[sample_idx][1]] for sample_idx in samples
        ]
    return info


def save_inference_stats(info, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(info, f, indent=4)


def main(args):
    ply_path = args.ply_path
    data_dir = args.data_dir
    output_root = args.output_path
    num_prototypes = args.num_prototypes
    save_viz = args.save_viz
    do_sample = args.do_sample

    pointnet_ckpt = args.pointnet_ckpt
    grid_size = args.grid_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    sampling = args.sampling
    num_samples = args.num_samples
    num_channels = args.num_channels
    head_size = args.head_size
    stn_3d = args.stn_3d
    stn_nd = args.stn_nd

    print(f"Explaining: {ply_path}")
    print(f"Using checkpoint: {pointnet_ckpt}")
    print(f"Grid size: {grid_size}, Channels: {num_channels}, STN 3D: {stn_3d}, STN ND: {stn_nd}")

    output_dir = os.path.join(output_root, Path(ply_path).stem)

    dm = GaussianDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=0.0,
        sampling=sampling,
        num_points=num_samples,
        grid_size=grid_size,
    )
    dm.setup()

    dataset = dm.test_ds
    print(f"Dataset size: {len(dataset)}")

    pt_state_dict = torch.load(pointnet_ckpt, weights_only=False)
    pl_module = PointNetLightning(
        in_dim=len(FEATURE_NAMES),
        num_classes=dm.num_classes,
        grid_size=grid_size,
        head_size=head_size,
        stn_3d=stn_3d,
        stn_nd=stn_nd,
    )
    pl_module.model.attach_disentangler(num_channels)
    pl_module.model.load_state_dict(pt_state_dict['pointnet_state_dict'])
    pl_module.eval()

    dis_trainer = DisentanglerTrainer(
        pl_module.model,
        num_channels=num_channels
    )
    dis_trainer.hparams.batch_size = batch_size
    dis_trainer.hparams.num_workers = num_workers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dis_trainer.to(device=device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )

    dis_trainer.update_test_prototypes(
        loader,
        num_prototypes,
        batch_size,
        num_workers,
        device
    )
    ammend_dataset_files(dataset, ply_path)
    dis_viz_cb = DisentanglerVisualizationCallback(
        output_dir=output_dir,
        num_channels=num_channels,
        grid_size=grid_size,
        val_dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=data_dir,
        num_prototypes=num_prototypes + 1
    )

    explain_prediction(
        dis_trainer,
        ply_path,
        ds=dataset,
        topk=num_prototypes,
        device=device,
        grid_size=grid_size,
        do_sample=do_sample
    )
    if save_viz:
        dis_viz_cb.visualize_disentangler_prototypes(dis_trainer, is_first_explained=True)
    stats = get_inference_stats(dis_trainer.last_val_prototypes, dataset)
    save_inference_stats(stats, os.path.join(output_dir, "inference_stats.json"))


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="XSPLAIN: Explainable predictions for Gaussian Splat point clouds"
    )

    # Config file (optional, overrides defaults)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides command-line defaults)"
    )

    # Required arguments
    parser.add_argument(
        "--ply_path",
        type=str,
        required=True,
        help="Path to input PLY file to explain"
    )

    # Model and checkpoint
    parser.add_argument(
        "--pointnet_ckpt",
        type=str,
        default="checkpoints/stage_2/pointnet_disentangler_compensated.pt",
        help="Path to trained PointNet + Disentangler checkpoint"
    )

    # Data settings
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing reference samples for prototypes"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="explanations",
        help="Output directory for explanation results"
    )

    # Model architecture (should match training)
    parser.add_argument(
        "--grid_size",
        type=int,
        default=10,
        help="Voxel grid size (must match training)"
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=256,
        help="Number of feature channels (must match training)"
    )
    parser.add_argument(
        "--head_size",
        type=int,
        default=256,
        help="Size of classification head (must match training)"
    )
    parser.add_argument(
        "--stn_3d",
        action="store_true",
        help="Enable 3D Spatial Transformer Network (must match training)"
    )
    parser.add_argument(
        "--stn_nd",
        action="store_true",
        help="Enable feature-space STN (must match training)"
    )

    # Data processing
    parser.add_argument(
        "--sampling",
        type=str,
        default="original_size",
        choices=["fps", "random", "original_size"],
        help="Point sampling method"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8192,
        help="Number of points to sample"
    )

    # Explanation settings
    parser.add_argument(
        "--num_prototypes",
        type=int,
        default=5,
        help="Number of prototype examples per active channel"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Apply sampling to the input PLY file"
    )

    # Output settings
    parser.add_argument(
        "--save_viz",
        action="store_true",
        default=True,
        help="Save point cloud visualizations"
    )
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Disable visualization output"
    )

    # Runtime settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for prototype generation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of data loading workers"
    )

    args = parser.parse_args()

    # Load config file if provided and merge with args
    if args.config is not None:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    # Handle --no_viz flag
    if args.no_viz:
        args.save_viz = False

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
