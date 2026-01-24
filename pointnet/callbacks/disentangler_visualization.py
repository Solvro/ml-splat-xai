import os
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import pytorch_lightning as pl
import torch

from pointnet.datasets.gaussian_point_cloud import FEATURE_NAMES, prepare_gaussian_cloud


def load_and_preprocess_ply(ply_path: Path):
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    pts = np.vstack([vertex[name] for name in FEATURE_NAMES]).T.astype(np.float32)

    gauss_np, xyz_normalized_np, xyz_min, xyz_max, mask = prepare_gaussian_cloud(pts)

    orig_indices = np.nonzero(mask)[0].astype(np.int64)  # (M,)

    gauss = torch.from_numpy(gauss_np)
    xyz_normalized = torch.from_numpy(xyz_normalized_np)
    gauss = torch.cat([xyz_normalized, gauss], dim=1)

    return {
        "pts": pts,
        "gauss": gauss,
        "xyz_normalized": xyz_normalized,
        "xyz_min": xyz_min,
        "xyz_max": xyz_max,
        "orig_indices": orig_indices,
    }


class DisentanglerVisualizationCallback(pl.Callback):
    def __init__(self, output_dir="disentangler_visualizations", num_channels=6, grid_size=10,
                 val_dataset=None, batch_size=4, num_workers=2, data_dir=None,
                 num_prototypes: int = 5):
        super().__init__()
        self.output_dir = output_dir
        self.num_channels = num_channels
        self.grid_size = grid_size
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.num_prototypes = num_prototypes
        os.makedirs(output_dir, exist_ok=True)

    def on_train_end(self, trainer, pl_module):
        if self.val_dataset is None:
            print("No validation dataset")
            return
        self.visualize_disentangler_prototypes(pl_module)

    @staticmethod
    def _base_dataset(ds):
        try:
            from torch.utils.data import Subset
            if isinstance(ds, Subset):
                return ds.dataset
        except Exception:
            pass
        return ds

    def find_ply_file(self, sample_idx, label=None):
        base = self._base_dataset(self.val_dataset)
        if hasattr(base, "files") and len(base.files) > sample_idx:
            file_entry = base.files[sample_idx]
            if isinstance(file_entry, tuple) and len(file_entry) > 0:
                path = str(file_entry[0])
                if os.path.exists(path):
                    return path
            elif isinstance(file_entry, str) and os.path.exists(file_entry):
                return file_entry

        if self.data_dir is not None:
            if label is not None and hasattr(base, "classes"):
                label_idx = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
                if 0 <= label_idx < len(base.classes):
                    class_name = base.classes[label_idx]
                    class_dir = os.path.join(self.data_dir, class_name)
                    if os.path.isdir(class_dir):
                        for fname in os.listdir(class_dir):
                            if fname.endswith(".ply") and str(sample_idx) in fname:
                                return os.path.join(class_dir, fname)
            for root, _, files in os.walk(self.data_dir):
                for fname in files:
                    if fname.endswith(".ply") and str(sample_idx) in fname:
                        return os.path.join(root, fname)
        return None

    @staticmethod
    def create_colored_ply(original_ply_path: str, output_path: str, highlight_vertex_ids: Sequence[int]):
        try:
            plydata = PlyData.read(original_ply_path)
            vertices = plydata["vertex"]
            n = len(vertices)

            field_names = [prop.name for prop in vertices.properties]
            dtype = [(name, vertices.data[name].dtype) for name in field_names]
            new_vertices = np.zeros(n, dtype=dtype)
            for name in field_names:
                new_vertices[name] = vertices[name]

            red = (1.0, 0.0, 0.0)
            gray = (0.5, 0.5, 0.5)

            ids = np.array([i for i in highlight_vertex_ids if 0 <= i < n], dtype=np.int64)

            def highlight_color(ids, color):
                nonlocal new_vertices
                if ids.size > 0:
                    new_vertices["f_dc_0"][ids] = color[0]
                    new_vertices["f_dc_1"][ids] = color[1]
                    new_vertices["f_dc_2"][ids] = color[2]
                    for j in range(45):
                        nm = f"f_rest_{j}"
                        if nm in field_names:
                            new_vertices[nm][ids] = 0.0

            highlight_color(ids, red)
            highlight_color(np.setdiff1d(np.arange(n), ids), gray)

            PlyData([PlyElement.describe(new_vertices, "vertex")], text=False).write(output_path)
            return True
        except Exception as e:
            print(f"Error creating colored PLY: {e}")
            return False

    @staticmethod
    def _read_ply_to_tensors_with_raw(ply_path: str) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data = load_and_preprocess_ply(Path(ply_path))
        # raw_xyz for the filtered points
        raw_xyz = data["pts"][data["orig_indices"], :3].astype(np.float32)

        features = data["gauss"].unsqueeze(0).transpose(1, 2)    # (1, C, M)
        xyz_norm = data["xyz_normalized"].unsqueeze(0)           # (1, M, 3)
        dataset_min = data["xyz_min"]
        dataset_max = data["xyz_max"]
        orig_indices = data["orig_indices"]                       # (M,)

        return features, xyz_norm, raw_xyz, dataset_min, dataset_max, orig_indices

    @staticmethod
    def undo_stn_transformation(xyz_after_stn: torch.Tensor, stn_T: torch.Tensor) -> torch.Tensor:
        T_inv = torch.linalg.pinv(stn_T)
        xyz_bt = xyz_after_stn.transpose(1, 2)  # (B, 3, N)
        xyz_pre_stn = torch.bmm(T_inv, xyz_bt).transpose(1, 2)  # (B, N, 3)
        return xyz_pre_stn

    @staticmethod
    def unit_to_raw(xyz_after_stn: torch.Tensor, stn_T: torch.Tensor, dataset_min: np.ndarray, dataset_max: np.ndarray) -> np.ndarray:
        xyz_pre_stn = DisentanglerVisualizationCallback.undo_stn_transformation(xyz_after_stn, stn_T)
        xyz_normalized = xyz_pre_stn[0].cpu().numpy()  # (N, 3)

        xyz_raw = xyz_normalized * (dataset_max - dataset_min) + dataset_min
        return xyz_raw.astype(np.float32)

    def _voxel_corners_unit(self, voxel_idx: int, G: int) -> np.ndarray:
        vx = voxel_idx // (G*G)
        vy = (voxel_idx // G) % G
        vz = voxel_idx % G
        x0, x1 = vx / G, (vx + 1) / G
        y0, y1 = vy / G, (vy + 1) / G
        z0, z1 = vz / G, (vz + 1) / G
        corners = np.array([
            [x0,y0,z0],[x1,y0,z0],[x0,y1,z0],[x1,y1,z0],
            [x0,y0,z1],[x1,y0,z1],[x0,y1,z1],[x1,y1,z1]
        ], dtype=np.float32)
        return corners

    def _compute_voxel_ids_np(self, xyz_normalized: np.ndarray) -> np.ndarray:
        gs = self.grid_size
        coords = np.floor(xyz_normalized * gs).astype(np.int64)
        coords = np.clip(coords, 0, gs - 1)
        voxel_ids = coords[:, 0] * (gs * gs) + coords[:, 1] * gs + coords[:, 2]
        return voxel_ids

    def _plot_panels_points(self, panels, title, out_path, isolated=False, is_first_explained=False):
        fig = plt.figure(figsize=(4*len(panels), 4))
        edges = [
            (0,1),(0,2),(1,3),(2,3),
            (4,5),(4,6),(5,7),(6,7),
            (0,4),(1,5),(2,6),(3,7)
        ]
        index_offset = 0 if is_first_explained else 1
        for j, (xyz_raw, mask_voxel, corners_raw, ptcl_name) in enumerate(panels, start=index_offset):
            ax = fig.add_subplot(1, len(panels), j + (1 - index_offset), projection='3d')
            if not isolated:
                ax.scatter(xyz_raw[:, 0], xyz_raw[:, 1], xyz_raw[:, 2], c='lightgray', s=1, alpha=0.3)
            vox_pts = xyz_raw[mask_voxel]
            if vox_pts.size > 0:
                ax.scatter(vox_pts[:, 0], vox_pts[:, 1], vox_pts[:, 2], c='crimson', s=6 if isolated else 4, alpha=0.95)
            for i_idx, j_idx in edges:
                xs, ys, zs = corners_raw[[i_idx, j_idx]].T
                ax.plot(xs, ys, zs, color='gold', lw=1.5, alpha=0.9)
            ax.set_title(f"target\n{ptcl_name}" if (is_first_explained and j == index_offset) else f"rank {j}\n{ptcl_name}")
            ax.set_box_aspect([1, 1, 1])
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    def get_point_cloud_name(self, ply_path: str):
        path = Path(ply_path)
        return path.stem

    def visualize_disentangler_prototypes(self, pl_module, is_first_explained=False):
        model = pl_module.pointnet
        device = pl_module.device
        model.eval().to(device)

        prototypes = getattr(pl_module, "last_val_prototypes", None)
        if prototypes is None:
            print("No stored prototypes found")
            prototypes = {}

        os.makedirs(self.output_dir, exist_ok=True)
        for c in range(self.num_channels):
            indices_for_c = prototypes.get(c, [])[: self.num_prototypes]
            if not indices_for_c:
                continue

            channel_dir = os.path.join(self.output_dir, f"channel_{c:04d}")
            os.makedirs(channel_dir, exist_ok=True)

            full_panels = []
            iso_panels = []

            for rank, sample_idx in enumerate(indices_for_c):
                ply_path = self.find_ply_file(sample_idx, label=None)
                if ply_path is None or not os.path.exists(ply_path):
                    print(f"channel {c} Could not find PLY for sample {sample_idx}, skipping.")
                    continue

                try:
                    full_features, full_xyz_norm, _, dataset_min, dataset_max, orig_indices = self._read_ply_to_tensors_with_raw(ply_path)

                except Exception as e:
                    print(f"channel {c} Error reading PLY {ply_path}: {e}")
                    continue

                with torch.no_grad():
                    full_features = full_features.to(device)
                    full_xyz_norm = full_xyz_norm.to(device)

                    point_features, xyz_for_vox = model.extract_point_features(full_features, full_xyz_norm)

                    xyz_norm_np = full_xyz_norm[0].cpu().numpy()  # (N, 3)
                    voxel_ids_np = self._compute_voxel_ids_np(xyz_norm_np)  # (N,)
                    voxel_ids = torch.from_numpy(voxel_ids_np).long().unsqueeze(0).to(device)  # (1, N)

                    voxel_features, indices_flat, _ = model.voxel_agg(point_features, voxel_ids)

                    voxel_features_flat = voxel_features.view(voxel_features.size(0), voxel_features.size(1), -1)  # (B, C, V)
                    voxel_features_flat = pl_module.disentangler(voxel_features_flat)

                    channel_activation = voxel_features_flat[:, c, :]  # (B, V)
                    _, max_idxs = torch.max(channel_activation, dim=1)  # (B,)
                    voxel_idx = int(max_idxs[0].item())

                    stn_T = model.last_stn_T if model.last_stn_T is not None else torch.eye(3, device=device).unsqueeze(0)

                    xyz_raw = self.unit_to_raw(
                        xyz_for_vox, stn_T, dataset_min, dataset_max
                    )

                    point_voxel_ids = indices_flat[0].detach().cpu().numpy()
                    mask_voxel = (point_voxel_ids == voxel_idx)
                    ptcl_name = self.get_point_cloud_name(ply_path)
                    corners_unit = self._voxel_corners_unit(voxel_idx, self.grid_size)  # (8, 3) in [0,1]
                    corners_raw = corners_unit * (dataset_max - dataset_min) + dataset_min  # (8, 3)

                    full_panels.append((xyz_raw, mask_voxel, corners_raw, ptcl_name))
                    iso_panels.append((xyz_raw[mask_voxel], np.ones(mask_voxel.sum(), dtype=bool), corners_raw, ptcl_name))

                    highlight_ids_proc = np.nonzero(mask_voxel)[0]  # (K,)
                    highlight_ids = orig_indices[highlight_ids_proc].tolist()


                    rank_offset = 0 if is_first_explained else 1
                    try:
                        full_out = os.path.join(channel_dir, f"rank{rank+rank_offset:02d}_full_colored.ply")
                        ok = self.create_colored_ply(ply_path, full_out, highlight_ids)
                        if not ok:
                            print(f"channel {c} failed to write colored full PLY for rank {rank+1}")
                        if highlight_ids:
                            try:
                                plydata = PlyData.read(ply_path)
                                nverts = len(plydata["vertex"])
                                safe_ids = [i for i in highlight_ids if 0 <= i < nverts]
                                if safe_ids:
                                    isolated_vertices = plydata["vertex"][safe_ids]
                                    PlyData([PlyElement.describe(isolated_vertices, "vertex")], text=False).write(
                                        os.path.join(channel_dir, f"rank{rank+rank_offset:02d}_isolated_voxel.ply")
                                    )
                            except Exception as e:
                                print(f"channel {c} error writing isolated voxel PLY for rank {rank+1}: {e}")
                    except Exception as e:
                        print(f"channel {c} PLY outputs error for rank {rank+1}: {e}")

            if full_panels:
                self._plot_panels_points(
                    full_panels,
                    title=f"Channel {c} – full cloud (raw space) – points only",
                    out_path=os.path.join(channel_dir, "prototypes_full_3d.png"),
                    isolated=False, 
                    is_first_explained=is_first_explained
                )
            if iso_panels:
                self._plot_panels_points(
                    iso_panels,
                    title=f"Channel {c} – isolated voxels (raw space) – points only",
                    out_path=os.path.join(channel_dir, "prototypes_isolated_3d.png"),
                    isolated=True,
                    is_first_explained=is_first_explained
                )

        print(f"Disentangler visualizations saved to {self.output_dir}")
