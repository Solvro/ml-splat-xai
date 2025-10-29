import os
import torch
from pointnet.dataset import FEATURE_NAMES, GaussianDataModule, collate_fn
from pointnet.pointnet import PointNetLightning
from train_epic import load_and_preprocess_ply
from train_epic_dislocated import EpicTrainer, EpicVisualizationCallback
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData, PlyElement
from typing import Sequence
from pathlib import Path 

def color_points_by_voxel_activation(
    voxel_features_flat,
    indices_flat,
    xyz_raw,
    channel: int | None = None,
    agg: str = "max",           # "max" | "mean" | "sum"
    cmap_name: str = "viridis",
    eps: float = 1e-8,
):
    """
    Map per-voxel activation -> per-point RGBA colors.

    Args:
      voxel_features_flat: torch.Tensor with shape (1, C, V) or (C, V) after EPIC+ReLU.
      indices_flat: tensor of voxel indices for each point (shape (1, N) or (N,))
      xyz_raw: numpy array (N,3) of raw point coordinates (not used for color computation)
      channel: if int -> use activations from this channel for each voxel (vf[channel, :]).
               if None -> aggregate across channels using `agg`.
      agg: aggregation when channel is None. One of {"max","mean","sum"}.
      cmap_name: matplotlib colormap name
      eps: small value to avoid division by zero when normalizing
    Returns:
      colors: (N,4) numpy RGBA colors for each point
      voxel_vals: (V,) numpy array of activation values per voxel (used for coloring)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    vf = voxel_features_flat
    # normalize expected shape to (C, V)
    if vf.dim() == 3 and vf.size(0) == 1:
        vf = vf.squeeze(0)
    # vf: (C, V)
    C, V = vf.size()

    # choose per-voxel values according to `channel` / `agg`
    if channel is not None:
        if not (0 <= channel < C):
            raise ValueError(f"channel {channel} out of range [0, {C-1}]")
        voxel_vals = vf[channel, :].detach().cpu().numpy()
    else:
        if agg == "max":
            voxel_vals = vf.max(dim=0).values.detach().cpu().numpy()
        elif agg == "mean":
            voxel_vals = vf.mean(dim=0).detach().cpu().numpy()
        elif agg == "sum":
            voxel_vals = vf.sum(dim=0).detach().cpu().numpy()
        else:
            raise ValueError("agg must be one of {'max','mean','sum'}")

    # normalize activations -> 0..1 for colormap
    if voxel_vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanmin(voxel_vals))
        vmax = float(np.nanmax(voxel_vals))
    norm = (voxel_vals - vmin) / (vmax - vmin + eps)

    cmap = plt.get_cmap(cmap_name)
    voxel_colors = cmap(norm)  # shape (V,4)

    # get per-point voxel ids (ensure 1D numpy array)
    pvid = indices_flat
    if isinstance(pvid, torch.Tensor):
        pvid = pvid.detach().cpu().numpy()
    if pvid.ndim == 2 and pvid.shape[0] == 1:
        pvid = pvid[0]
    # allocate per-point colors
    N = pvid.shape[0]
    colors = np.zeros((N, 4), dtype=float)
    # mark valid points that map to a voxel index
    valid = (pvid >= 0) & (pvid < len(voxel_colors))
    if valid.any():
        colors[valid] = voxel_colors[pvid[valid]]
    # invalid points (not assigned to any voxel) -> gray
    colors[~valid] = (0.6, 0.6, 0.6, 1.0)

    return colors, voxel_vals


def plot_xyz_colored_by_voxel(xyz_raw, colors, figsize=(6,6), s=6, elev=20, azim=45):
    """
    Simple 3D scatter of points colored by provided RGBA colors.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz_raw[:, 0], xyz_raw[:, 1], xyz_raw[:, 2], c=colors, s=s)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


@staticmethod
def create_colored_ply(original_ply_path: str, output_path: str, colors: Sequence[Sequence[float]] | np.ndarray):
    """
    Read original PLY and write a new PLY where each vertex is assigned the color
    provided in `colors`.

    `colors` may be:
        - an (N,3) or (N,4) array-like (values in 0..1 or 0..255)
        - a single RGB(A) triplet/list -> applied to all vertices

    The function preserves existing vertex properties and adds (or overwrites)
    f_dc_0,f_dc_1,f_dc_2 fields (and leaves any other f_rest_* fields untouched).
    """
    try:
        plydata = PlyData.read(original_ply_path)
        vertices = plydata["vertex"]
        n = len(vertices)

        # normalize colors to numpy array shape (n,3) with float32 0..1
        col_arr = np.asarray(colors)
        if col_arr.ndim == 1 and col_arr.size in (3,4):
            col_arr = np.tile(col_arr[None, :3], (n, 1))
        elif col_arr.ndim == 2 and col_arr.shape[0] == 1 and col_arr.shape[1] in (3,4):
            col_arr = np.tile(col_arr[0, :3][None, :], (n, 1))
        elif col_arr.ndim == 2 and col_arr.shape[0] != n:
            raise ValueError(f"colors length {col_arr.shape[0]} does not match vertex count {n}")
        elif col_arr.ndim == 2 and col_arr.shape[1] not in (3,4):
            raise ValueError("colors must have 3 (RGB) or 4 (RGBA) channels per vertex")

        # convert 0..255 -> 0..1 if necessary
        if col_arr.dtype.kind in ("i", "u") or col_arr.max() > 1.0:
            col_arr = col_arr.astype(np.float32) / 255.0
        col_arr = col_arr.astype(np.float32)
        # keep only RGB
        col_arr = col_arr[:, :3]

        field_names = [prop.name for prop in vertices.properties]
        # build dtype: keep existing fields, add f_dc_0..2 if missing
        dtype = [(name, vertices.data[name].dtype) for name in field_names]
        if "f_dc_0" not in field_names:
            dtype.extend([("f_dc_0", np.float32), ("f_dc_1", np.float32), ("f_dc_2", np.float32)])

        new_vertices = np.zeros(n, dtype=dtype)
        # copy existing fields where present
        for name in field_names:
            new_vertices[name] = vertices[name]

        # assign colors (clamp to [0,1])
        col_clamped = np.clip(col_arr, 0.0, 1.0)
        new_vertices["f_dc_0"][:] = col_clamped[:, 0]
        new_vertices["f_dc_1"][:] = col_clamped[:, 1]
        new_vertices["f_dc_2"][:] = col_clamped[:, 2]

        new_ply = PlyData([PlyElement.describe(new_vertices, "vertex")], text=False)
        new_ply.write(output_path)
        return new_ply
    except Exception as e:
        print(f"Error creating colored PLY: {e}")
        return False

def get_path(ply_input_path, channel, agg):
    name = Path(ply_input_path).stem
    channel = "_" if channel is None else f"_ch={channel}_"
    return os.path.join(args.output_dir, f"{name}{channel}{agg}.ply")

def main(args):
    ply_input_path = args.ply_input_path
    pointnet_ckpt = "checkpoints/toys_pointnet_epic_dislocated_10x10\pointnet_epic_compensated.pt"
    grid_size = 10
    data_dir = "data/toys_ds_cleaned/train/"
    channel = None if args.channel < 0 else args.channel
    agg = args.agg
    ply_output_path = get_path(ply_input_path, channel, agg)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pt_state_dict = torch.load(pointnet_ckpt)
    pl_module = PointNetLightning(
        in_dim=len(FEATURE_NAMES),
        num_classes=len(os.listdir(data_dir)),
        grid_size=grid_size,
        head_norm=True
    )

    print("None" if pl_module.model.epic is None else "NOT none")

    pl_module.model.attach_epic()

    print(f"BEFORE A_raw = {pl_module.model.epic.A_raw}")

    pl_module.model.load_state_dict(pt_state_dict['pointnet_state_dict'])
    pl_module.eval()

    epic_trainer = EpicTrainer(
        pl_module.model
    )
    print(f"A_raw = {pl_module.model.epic.A_raw}")
    print(f"U = {epic_trainer.epic.get_weight()}")
    epic_trainer.to(device=device)

    features, xyz_normalized, raw_xyz, ply_min, ply_max = EpicVisualizationCallback._read_ply_to_tensors_with_raw(ply_input_path)

    with torch.no_grad():
        features = features.to(device)
        xyz_normalized = xyz_normalized.to(device)
        point_features, xyz_for_vox = epic_trainer.pointnet.extract_point_features(features, xyz_normalized, mask=None)
        voxel_features, indices_flat, point_counts = epic_trainer.pointnet.voxel_agg(point_features, xyz_for_vox, mask=None)

    voxel_features_flat = voxel_features.view(voxel_features.size(0), voxel_features.size(1), -1)
    voxel_features_flat = epic_trainer.epic(voxel_features_flat)
    voxel_features_flat = F.relu(voxel_features_flat)
    voxel_features_flat = voxel_features_flat.squeeze(0)

    
    stn_T = epic_trainer.pointnet.last_stn_T if epic_trainer.pointnet.last_stn_T is not None else torch.eye(3, device=device).unsqueeze(0)
    rescale_min = epic_trainer.pointnet.last_rescale_min if epic_trainer.pointnet.last_rescale_min is not None else torch.zeros(1,3,1, device=device)
    rescale_max = epic_trainer.pointnet.last_rescale_max if epic_trainer.pointnet.last_rescale_max is not None else torch.ones(1,3,1, device=device)

    xyz_raw = EpicVisualizationCallback.unit_to_raw(
        xyz_for_vox, stn_T, rescale_min, rescale_max, ply_min, ply_max
    )

    vf = F.relu(voxel_features_flat.view(1, voxel_features.size(1), -1))   # you already have similar
    colors, vals = color_points_by_voxel_activation(vf, indices_flat, xyz_raw, channel=channel, agg=agg)    

    _ = create_colored_ply(
        ply_input_path, ply_output_path, colors
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Visualize point cloud with voxel colors using EPIC Pointnet')
    parser.add_argument("--ply_input_path", type=str, help="Path to input PLY file")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument("--channel", type=int, default=-1, help="Which channel to explain, -1 if to use agg")
    parser.add_argument("--agg", type=str, default="max", choices=["max", "mean", "sum"], help="How to aggregate channels")
    parser.add_argument("--cmap_name", type=str, default="Reds", help="Colormap to use")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)

    