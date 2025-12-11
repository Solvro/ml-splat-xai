import json
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pointnet.dataset import FEATURE_NAMES, GaussianDataModule, collate_fn, prepare_gaussian_cloud
from pointnet.pointnet import PointNetLightning
from train_epic_dislocated import EpicTrainer, EpicVisualizationCallback, load_and_preprocess_ply


def _compute_voxel_ids_np(xyz_normalized: np.ndarray, grid_size: int) -> np.ndarray:
    gs = grid_size
    coords = np.floor(xyz_normalized * gs).astype(np.int64)
    coords = np.clip(coords, 0, gs - 1)
    voxel_ids = coords[:, 0] * (gs * gs) + coords[:, 1] * gs + coords[:, 2]
    return voxel_ids


def topk_active_channels(epic_trainer, ply_path, ds, topk, device, grid_size, do_sample=False):
    data = load_and_preprocess_ply(Path(ply_path))
    pts = data["pts"]
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
        point_features, xyz_for_vox = epic_trainer.pointnet.extract_point_features(
            features, xyz_norm_t, mask
        )
        voxel_features, indices_vox, point_counts = epic_trainer.pointnet.voxel_agg(
            point_features, voxel_ids
        )
        voxel_features_flat = voxel_features.view(
            voxel_features.size(0),
            voxel_features.size(1),
            -1
        ) 
        voxel_features_flat = epic_trainer.epic(voxel_features_flat)

        global_features = F.adaptive_avg_pool1d(voxel_features_flat, 1).squeeze(-1)  # (B, C)
        logits = epic_trainer.pointnet.head(global_features)  # (B, num_classes)
        predicted_class = logits.argmax(dim=-1).item()
        print(f"Predicted class: {predicted_class} (logits: {logits[0].cpu().numpy()})")

        classifier_linear = None
        for m in epic_trainer.pointnet.head:
            if isinstance(m, torch.nn.Linear):
                classifier_linear = m
                break

        if classifier_linear is None:
            raise ValueError("Could not find Linear layer in classification head")

        class_weights = classifier_linear.weight[predicted_class]  # (C,)

        weighted_global_features = class_weights * F.relu(global_features.squeeze(0))  # (C,)

        _, channels = torch.topk(weighted_global_features, topk, largest=True, sorted=True)

    return channels.tolist()


def explain_prediction(epic_trainer, ply_path, ds, topk, device, grid_size, do_sample=False):
    prototypes = getattr(epic_trainer, "last_val_prototypes", None)
    if prototypes is None:
        print("No stored prototypes found")
        prototypes = {}

    channels = topk_active_channels(epic_trainer, ply_path, ds, topk, device, grid_size, do_sample=do_sample)
    print(f"Max active channels are {channels}")
    new_prototypes = {c: [-1] for c in channels}
    for c in channels:
        indices_for_c = prototypes.get(c, [])[: topk]
        new_prototypes[c].extend(indices_for_c)

    print(f"New prototypes: {new_prototypes}")
    epic_trainer.last_val_prototypes = new_prototypes

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

    pointnet_ckpt = "test_test/epic_dislocated_output_grid_10/pointnet_epic_compensated.pt"
    grid_size = 7
    data_dir = args.data_dir
    output_root = args.output_path
    num_prototypes = args.num_prototypes
    save_viz = args.save_viz
    do_sample = False # if True then explained point cloud is sampled

    pointnet_ckpt = args.pointnet_ckpt
    print('pointnet_ckpt in explain epic: ',pointnet_ckpt)
    grid_size = 7
    batch_size = 1
    num_workers = 2
    sampling = "original_size"
    num_samples = 8192
    num_channels = 256

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

    pt_state_dict = torch.load(pointnet_ckpt)
    pl_module = PointNetLightning(
        in_dim=len(FEATURE_NAMES),
        num_classes=dm.num_classes,
        grid_size=grid_size,
        head_norm=True
    )
    pl_module.model.attach_epic(num_channels)
    pl_module.model.load_state_dict(pt_state_dict['pointnet_state_dict'])
    pl_module.eval()

    epic_trainer = EpicTrainer(
        pl_module.model,
        num_channels=num_channels
    )
    epic_trainer.hparams.batch_size = batch_size
    epic_trainer.hparams.num_workers = num_workers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epic_trainer.to(device=device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )

    epic_trainer.update_test_prototypes(
        loader,
        num_prototypes,
        batch_size,
        num_workers,
        device
    )
    ammend_dataset_files(dataset, ply_path)
    epic_viz_cb = EpicVisualizationCallback(
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
        epic_trainer,
        ply_path,
        ds=dataset,
        topk=num_prototypes,
        device=device,
        grid_size=grid_size,
        do_sample=do_sample
    )
    if save_viz:
        epic_viz_cb.visualize_epic_prototypes(None, epic_trainer, is_first_explained=True)
    stats = get_inference_stats(epic_trainer.last_val_prototypes, dataset)
    save_inference_stats(stats, os.path.join(output_dir, "inference_stats.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Explaining EPIC PointNet (dislocated version)")
    parser.add_argument("--ply_path", type=str, required=True, help="path of input ply file")
    parser.add_argument("--output_path", type=str, default="./epic-visualization/", help="path of output directory")
    parser.add_argument("--num_prototypes", type=int, default=5, help="number of prototypes to use")
    parser.add_argument("--data_dir", type=str, default="../archive/toys_ds/data/", help="directory of samples to choose from")
    parser.add_argument("--pointnet_ckpt", type=str, default="pointnet_epic_compensated.pt", help="path to pointnet")
    parser.add_argument("--save_viz", action="store_true", default=True, help="Save point cloud visualizations")
    args = parser.parse_args()
    main(args)
