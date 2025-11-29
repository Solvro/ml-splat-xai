import json
import torch
from pointnet.dataset import FEATURE_NAMES, GaussianDataModule, collate_fn, prepare_gaussian_cloud
from pointnet.pointnet import PointNetLightning
from train_epic_dislocated import EpicTrainer
from train_epic_dislocated import EpicVisualizationCallback, load_and_preprocess_ply
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from plyfile import PlyData
import torch.nn.functional as F

def topk_active_channels(epic_trainer, ply_path, ds, topk, device, do_sample=False):
    data = load_and_preprocess_ply(ply_path)
    ids = ds._random_sample(data["pts"]) if do_sample else np.arange(data["pts"].shape[0])
    features = data["gauss"][ids].unsqueeze(0).transpose(1, 2).to(device)
    xyz_normalized = data["xyz_normalized"][ids].unsqueeze(0).to(device)
    mask = data.get("mask", None)
    if mask is not None:
        mask = mask[ids].to(device)
    print(f"{features.device=}")

    with torch.no_grad():
        point_features, xyz_for_vox = epic_trainer.pointnet.extract_point_features(features, xyz_normalized, mask)
        voxel_features, indices_vox, point_counts, _, _ = epic_trainer.pointnet.voxel_agg(point_features, xyz_for_vox, mask)

        voxel_features_flat = voxel_features.view(voxel_features.size(0), voxel_features.size(1), -1)  # (B, C, V)
        voxel_features_flat = epic_trainer.epic(voxel_features_flat)

        # classification_head.before_linear()
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

        # topk(class_weights * relu(before_linear(features)))
        weighted_global_features = class_weights * F.relu(global_features.squeeze(0))  # (C,)

        _, channels = torch.topk(weighted_global_features, topk, largest=True, sorted=True)

    return channels.tolist()


def explain_prediction(epic_trainer, ply_path, ds, topk, device, do_sample = False):
    prototypes = getattr(epic_trainer, "last_val_prototypes", None)
    if prototypes is None:
        print("No stored prototypes found")
        prototypes = {}

    channels = topk_active_channels(epic_trainer, ply_path, ds, topk, device, do_sample=do_sample)
    print(f"Max active channels are {channels}")
    new_prototypes = {c: [-1] for c in channels}
    for c in channels:
        indices_for_c = prototypes.get(c, [])[: topk]
        new_prototypes[c].extend(indices_for_c)

    print(f"New prototypes: {new_prototypes}")
    
    epic_trainer.last_val_prototypes = new_prototypes

def ammend_dataset_files(dataset, ply_path):
    dataset.files.append((ply_path, dataset.class_to_idx[ply_path.replace("\\", "/").split("/")[-2]])) # add cloud that is being explained

def get_inference_stats(val_prototypes, dataset):
    info = {channel : {} for channel in val_prototypes}
    for channel in val_prototypes:
        info[channel]["samples"] = val_prototypes[channel][1:]
        info[channel]["classes"] = [dataset.classes[dataset.files[sample_idx][1]] for sample_idx in val_prototypes[channel][1:]]
    return info 

def save_inference_stats(info, filename):
    with open(filename, "w") as f:
        json.dump(info, f, indent=4)



def main(args):
    ply_path = args.ply_path
    pointnet_ckpt = "pointnet_epic_compensated.pt"

    grid_size = 7
    data_dir = args.data_dir
    batch_size = 4
    num_workers = 2
    do_sample = False # if True then explained point cloud is sampled
    sampling = "original_size"
    num_samples = 8192
    save_viz = args.save_viz
    num_prototypes = args.num_prototypes
    output_dir = args.output_path
    output_dir = os.path.join(output_dir, Path(ply_path).stem)

    dm = GaussianDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=0.0,
        sampling=sampling,
        num_points=num_samples
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
    num_channels = 256
    pl_module.model.attach_epic(num_channels)
    pl_module.model.load_state_dict(pt_state_dict['pointnet_state_dict'])
    pl_module.eval()

    epic_trainer = EpicTrainer(
        pl_module.model,
        num_channels = num_channels
    )
    print(f"A_raw = {epic_trainer.epic.A_raw}")
    print(f"U = {epic_trainer.epic.get_weight()}")
    epic_trainer.hparams.batch_size = batch_size
    epic_trainer.hparams.num_workers = num_workers


    device = "cuda" if torch.cuda.is_available() else "cpu"
    epic_trainer.to(device=device)


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
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
        num_channels=256,
        grid_size=grid_size,
        val_dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=data_dir,
        num_prototypes=num_prototypes + 1
    )

    explain_prediction(epic_trainer, ply_path, ds=dataset, topk=num_prototypes, device=device, do_sample=do_sample)
    if save_viz:
        epic_viz_cb.visualize_epic_prototypes(None, epic_trainer, is_first_explained=True)
    stats = get_inference_stats(epic_trainer.last_val_prototypes, dataset)
    save_inference_stats(stats, os.path.join(output_dir, "inference_stats.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Explaining EPIC Pointnet')
    parser.add_argument('--ply_path', type=str, required=True, help='path of input ply file')
    parser.add_argument('--output_path', type=str, default="./epic-visualization/", help='path of output directory')
    parser.add_argument('--num_prototypes', type=int, default=5, help='number of prototypes to use')
    parser.add_argument('--data_dir', type=str, default='data/train', help='directory of samples to choose from')
    parser.add_argument('--save_viz', action='store_true', default=True, help='Save point cloud visualizations')
    args = parser.parse_args()
    main(args)
