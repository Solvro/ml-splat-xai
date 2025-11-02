from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import os
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from typing import Sequence
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from tqdm import tqdm
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt

from pointnet.pointnet import PointNetLightning
from pointnet.dataset import GaussianDataModule, FEATURE_NAMES, collate_fn, prepare_gaussian_cloud
from pointnet.epic import EpicDisentangler


@torch.no_grad()
def generate_prototypes_pointnet(model, dataloader, num_channels, topk=5, device="cpu", U=None, debug=False):
    model.eval()
    model.to(device)
    if U is not None:
        U = U.to(device)

    top_acts = torch.full((topk, num_channels), -float("inf"), device=device)
    top_inds = torch.full((topk, num_channels), -1, dtype=torch.long, device=device)

    total_samples_seen = 0

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating prototypes"):
        features = batch["gauss"].to(device)
        xyz_normalized = batch["xyz_normalized"].to(device)
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(device)

        B = features.size(0)
        dataset_indices = torch.arange(total_samples_seen, total_samples_seen + B, device=device, dtype=torch.long)
        total_samples_seen += B

        if debug and batch_idx == 0:
            print(f"Batch {batch_idx}: Using dataset indices from {dataset_indices[0].item()} to {dataset_indices[-1].item()}")

        point_features, xyz_for_vox = model.extract_point_features(features, xyz_normalized, mask)
        voxel_features, voxel_indices, point_counts = model.voxel_agg(point_features, xyz_for_vox, mask)
        voxel_mask = (point_counts.squeeze(1) > 0)  # (B, V)
        voxel_features_flat = rearrange(voxel_features, 'b c x y z -> b c (x y z)')

        if U is not None:
            voxel_features_flat = torch.einsum("cd,bdn->bcn", U, voxel_features_flat)

        voxel_features_flat = F.relu(voxel_features_flat)
        voxel_features_flat = voxel_features_flat.masked_fill(
            ~voxel_mask.unsqueeze(1).expand_as(voxel_features_flat),
            -float("inf")
        )

        max_voxel_activations, _ = torch.max(voxel_features_flat, dim=2)  # (B, C)

        min_top_acts, _ = torch.min(top_acts, dim=0) # (C,)
        update_mask = max_voxel_activations > min_top_acts

        if not update_mask.any():
            continue

        combined_acts = torch.cat([top_acts, max_voxel_activations], dim=0)   # (topk+B, C)
        dataset_idx_exp = dataset_indices.view(B, 1).expand(B, num_channels)  # (B, C)
        combined_inds = torch.cat([top_inds, dataset_idx_exp], dim=0)        # (topk+B, C)

        channels_to_update = torch.where(update_mask.any(dim=0))[0]

        vals, idxs = torch.topk(combined_acts[:, channels_to_update], k=topk, dim=0)

        top_acts[:, channels_to_update] = vals
        top_inds[:, channels_to_update] = combined_inds[:, channels_to_update][idxs, torch.arange(len(channels_to_update))]

    prototypes_dict = {c: top_inds[:, c].cpu().numpy().tolist() for c in range(num_channels)}
    return prototypes_dict


def purity_argmax_voxel(voxel_features: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
    # voxel_features: (B, C, V)
    B, C, V = voxel_features.shape
    device = voxel_features.device

    acts_c = voxel_features[torch.arange(B, device=device), channels, :]  # (B, V)
    max_vals, max_indices = torch.max(acts_c, dim=1)  # (B,)

    vectors = voxel_features[torch.arange(B, device=device), :, max_indices]  # (B, C)
    vectors = torch.where(torch.isfinite(vectors), vectors, torch.zeros_like(vectors))

    target = vectors[torch.arange(B, device=device), channels]  # (B,)
    norms = vectors.norm(dim=1).clamp_min(1e-8)
    purity = target / norms

    valid = torch.isfinite(max_vals)
    purity = torch.where(valid, purity, purity.new_zeros(purity.shape))
    return purity


class PrototypesDataset(Dataset):
    def __init__(self, original_dataset, prototypes_dict):
        self.original_dataset = original_dataset
        self.prototypes_dict = prototypes_dict

        self.samples = []
        for channel, indices in prototypes_dict.items():
            for idx in indices:
                if idx < len(original_dataset):
                    self.samples.append((idx, channel))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        original_idx, channel = self.samples[idx]
        data = self.original_dataset[original_idx]
        return {
            "gauss": data["gauss"],
            "xyz_normalized": data["xyz_normalized"],
            "mask": data.get("mask", None),
            "indices": data.get("indices", None),
            "label": data.get("label", None),
            "channel": channel
        }


def collate_prototypes(batch):
    channels = [item["channel"] for item in batch]
    batch_without_channel = [{k: v for k, v in item.items() if k != "channel"} for item in batch]
    batch_dict = collate_fn(batch_without_channel)
    batch_dict["channel"] = torch.tensor(channels, dtype=torch.long)
    return batch_dict


class EpicTrainer(pl.LightningModule):
    def __init__(
        self,
        pointnet_model,
        num_channels=256,
        lr: float = 1e-4,
        initial_topk=40,
        final_topk=5,
        max_epochs=20,
        point_subset_ratio: float = 1.0,
        subset_seed: int | None = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pointnet_model"])
        self.pointnet = pointnet_model.eval()
        for p in self.pointnet.parameters():
            p.requires_grad_(False)

        self.epic = EpicDisentangler(C=num_channels)
        self.lr = lr
        self.initial_topk = initial_topk
        self.final_topk = final_topk
        self.max_epochs = max_epochs
        self.prototypes_loader = None
        self.val_prototypes_loader = None
        self.current_topk = initial_topk

        self.last_val_prototypes = None

    def training_step(self, batch, batch_idx):
        self.pointnet.eval()
        features = batch["gauss"]
        xyz_normalized = batch["xyz_normalized"]
        mask = batch.get("mask", None)
        channels = batch["channel"]

        with torch.no_grad():
            point_features, xyz_for_vox = self.pointnet.extract_point_features(features, xyz_normalized, mask)
            voxel_features, indices_vox, point_counts = self.pointnet.voxel_agg(point_features, xyz_for_vox, mask)

        voxel_features_flat = voxel_features.view(voxel_features.size(0), voxel_features.size(1), -1) # B 256 1000
        voxel_features_flat = self.epic(voxel_features_flat)
        voxel_features_flat = F.relu(voxel_features_flat)

        purity = purity_argmax_voxel(voxel_features_flat, channels)
        loss = -purity.mean()

        self.log("train/purity_mean", purity.mean(), prog_bar=True)
        self.log("train/purity_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.pointnet.eval()
        features = batch["gauss"]
        xyz_normalized = batch["xyz_normalized"]
        mask = batch.get("mask", None)
        channels = batch["channel"]

        with torch.no_grad():
            point_features, xyz_for_vox = self.pointnet.extract_point_features(features, xyz_normalized, mask)
            voxel_features, indices_vox, point_counts = self.pointnet.voxel_agg(point_features, xyz_for_vox, mask)

        voxel_features_flat = voxel_features.view(voxel_features.size(0), voxel_features.size(1), -1) # B 256 1000
        voxel_features_flat = self.epic(voxel_features_flat)
        voxel_features_flat = F.relu(voxel_features_flat)

        purity = purity_argmax_voxel(voxel_features_flat, channels)
        loss = -purity.mean()

        self.log("val/purity_mean", purity.mean(), prog_bar=True)
        self.log("val/epic_purity_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.epic.parameters(), lr=self.lr)
        return opt

    def train_dataloader(self):
        print("Get train_loader")
        return self.prototypes_loader

    def val_dataloader(self):
        print("Get val_loader")
        return self.val_prototypes_loader

    def test_step(self, batch, batch_idx):
        self.pointnet.eval()
        features = batch["gauss"]
        xyz_normalized = batch["xyz_normalized"]
        mask = batch.get("mask", None)
        channels = batch["channel"]

        with torch.no_grad():
            point_features, xyz_for_vox = self.pointnet.extract_point_features(features, xyz_normalized, mask)
            voxel_features, indices_vox, point_counts = self.pointnet.voxel_agg(point_features, xyz_for_vox, mask)

        voxel_features_flat = voxel_features.view(voxel_features.size(0), voxel_features.size(1), -1) # B 256 1000
        voxel_features_flat = self.epic(voxel_features_flat)
        voxel_features_flat = F.relu(voxel_features_flat)

        purity = purity_argmax_voxel(voxel_features_flat, channels)
        loss = -purity.mean()

        self.log("test/purity_mean", purity.mean(), prog_bar=True)
        self.log("test/epic_purity_loss", loss, prog_bar=True)
        return loss

    def test_dataloader(self):
        return self.test_prototypes_loader

    @torch.no_grad()
    def update_test_prototypes(self, test_loader, n_prototypes, batch_size, num_workers, device):
        U = self.epic.get_weight().to(device)

        print(f"Generating {n_prototypes} prototypes per channel...")
        test_dataset = test_loader.dataset
        prototypes = generate_prototypes_pointnet(
            self.pointnet,
            test_loader,
            num_channels=self.hparams.num_channels,
            topk=n_prototypes,
            device=device,
            U=U,
            debug=True
        )

        self.last_val_prototypes = prototypes
        prototypes_dataset = PrototypesDataset(test_dataset, prototypes)

        self.test_prototypes_loader = DataLoader(
            prototypes_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_prototypes
        )

    @torch.no_grad()
    def update_prototypes(self, train_loader, val_loader, batch_size, num_workers, device):
        progress = min(self.current_epoch / max(1, self.max_epochs), 1.0)
        self.current_topk = int(self.initial_topk - progress * (self.initial_topk - self.final_topk))

        U = self.epic.get_weight().to(device)

        print(f"Generating {self.current_topk} prototypes per channel...")
        prototypes = generate_prototypes_pointnet(
            self.pointnet,
            train_loader,
            num_channels=self.hparams.num_channels,
            topk=self.current_topk,
            device=device,
            U=U,
            debug=True
        )

        val_prototypes = generate_prototypes_pointnet(
            self.pointnet,
            val_loader,
            num_channels=self.hparams.num_channels,
            topk=max(5, self.current_topk // 2),
            device=device,
            U=U,
            debug=True
        )

        self.last_val_prototypes = val_prototypes

        def debug_prototypes(prototypes_dict, dataset, name="dataset"):
            total_indices = sum(len(indices) for indices in prototypes_dict.values())
            valid_indices = sum(
                1 for indices in prototypes_dict.values()
                for idx in indices if idx < len(dataset)
            )
            print(f"{name} stats: Total indices: {total_indices}, Valid indices: {valid_indices}, Dataset size: {len(dataset)}")

        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset

        debug_prototypes(prototypes, train_dataset, "Training")
        debug_prototypes(val_prototypes, val_dataset, "Validation")

        prototypes_dataset = PrototypesDataset(train_dataset, prototypes)
        val_proto_dataset = PrototypesDataset(val_dataset, val_prototypes)

        print(f"Train prototypes dataset size: {len(prototypes_dataset)}")
        print(f"Val prototypes dataset size: {len(val_proto_dataset)}")

        self.prototypes_loader = DataLoader(
            prototypes_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_prototypes
        )

        self.val_prototypes_loader = DataLoader(
            val_proto_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_prototypes
        )


class PrototypeUpdateCallback(pl.Callback):
    def __init__(self, update_freq, train_loader, val_loader, batch_size, num_workers, device):
        self.update_freq = update_freq
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def on_train_epoch_end(self, trainer, pl_module):
        if not (trainer.current_epoch + 1) % self.update_freq:
            print("Updating datasets ...")
            pl_module.update_prototypes(
                self.train_loader,
                self.val_loader,
                self.batch_size,
                self.num_workers,
                self.device
            )


def load_and_preprocess_ply(ply_path: Path):
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    pts = np.vstack([vertex[name] for name in FEATURE_NAMES]).T.astype(np.float32)

    gauss, xyz_normalized, xyz_min, xyz_max, mask = prepare_gaussian_cloud(pts)
    gauss = torch.from_numpy(gauss)
    xyz_normalized = torch.from_numpy(xyz_normalized)
    gauss = torch.cat([xyz_normalized, gauss], dim=1)

    return {
        "pts": pts,
        "gauss": gauss,
        "xyz_normalized": xyz_normalized,
        "xyz_min": xyz_min,
        "xyz_max": xyz_max
    }


class EpicVisualizationCallback(pl.Callback):
    def __init__(self, output_dir="epic_visualizations", num_channels=6, grid_size=10,
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
        self.visualize_epic_prototypes(trainer, pl_module)

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
    def _read_ply_to_tensors_with_raw(ply_path: str) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        data = load_and_preprocess_ply(Path(ply_path))
        raw_xyz = data["pts"][:, :3].astype(np.float32)
        features = data["gauss"].unsqueeze(0).transpose(1, 2)
        xyz_norm = data["xyz_normalized"].unsqueeze(0)
        raw_min = data["xyz_min"]
        raw_max = data["xyz_max"]
        return features, xyz_norm, raw_xyz, raw_min, raw_max

    @staticmethod
    def undo_stn_transformation(xyz_unit: torch.Tensor, stn_T: torch.Tensor, rescale_min: torch.Tensor, rescale_max: torch.Tensor) -> torch.Tensor:
        min_used = rescale_min.permute(0, 2, 1)
        max_used = rescale_max.permute(0, 2, 1)
        xyz_rescaled = xyz_unit * (max_used - min_used) + min_used
        T_inv = torch.linalg.pinv(stn_T)
        xyz_bt = xyz_rescaled.transpose(1, 2)
        xyz_pre_stn = torch.bmm(T_inv, xyz_bt).transpose(1, 2)
        return xyz_pre_stn

    @staticmethod
    def unit_to_raw(xyz_unit: torch.Tensor, stn_T: torch.Tensor, rescale_min: torch.Tensor, rescale_max: torch.Tensor,
                    ply_min: np.ndarray, ply_max: np.ndarray) -> np.ndarray:
        xyz_pre_stn = EpicVisualizationCallback.undo_stn_transformation(xyz_unit, stn_T, rescale_min, rescale_max)
        xyz_pre = xyz_pre_stn[0].cpu().numpy()
        raw = xyz_pre * (ply_max - ply_min)[None, :] + ply_min[None, :]
        return raw.astype(np.float32)

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

    def visualize_epic_prototypes(self, trainer, pl_module, is_first_explained=False):
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
                    full_features, full_xyz_norm, raw_xyz, ply_min, ply_max = self._read_ply_to_tensors_with_raw(ply_path)
                except Exception as e:
                    print(f"channel {c} Error reading PLY {ply_path}: {e}")
                    continue

                with torch.no_grad():
                    full_features = full_features.to(device)
                    full_xyz_norm = full_xyz_norm.to(device)

                    point_features, xyz_for_vox = model.extract_point_features(full_features, full_xyz_norm)
                    point_features = pl_module.epic(point_features)
                    voxel_features, indices_flat, _ = model.voxel_agg(point_features, xyz_for_vox)

                    vf = F.relu(voxel_features.view(1, voxel_features.size(1), -1))
                    max_vals, max_idxs = torch.max(vf[:, c, :], dim=1)
                    voxel_idx = int(max_idxs[0].item())

                    stn_T = model.last_stn_T if model.last_stn_T is not None else torch.eye(3, device=device).unsqueeze(0)
                    rescale_min = model.last_rescale_min if model.last_rescale_min is not None else torch.zeros(1,3,1, device=device)
                    rescale_max = model.last_rescale_max if model.last_rescale_max is not None else torch.ones(1,3,1, device=device)

                    xyz_raw = self.unit_to_raw(
                        xyz_for_vox, stn_T, rescale_min, rescale_max, ply_min, ply_max
                    )

                    point_voxel_ids = indices_flat[0].detach().cpu().numpy()
                    mask_voxel = (point_voxel_ids == voxel_idx)

                    corners_unit = self._voxel_corners_unit(voxel_idx, self.grid_size)
                    corners_unit_t = torch.from_numpy(corners_unit).float().unsqueeze(0).to(device)
                    corners_raw = self.unit_to_raw(
                        corners_unit_t, stn_T, rescale_min, rescale_max, ply_min, ply_max
                    )

                    ptcl_name = self.get_point_cloud_name(ply_path)
                    full_panels.append((xyz_raw, mask_voxel, corners_raw, ptcl_name))
                    iso_panels.append((xyz_raw[mask_voxel], np.ones(mask_voxel.sum(), dtype=bool), corners_raw, ptcl_name))

                    highlight_ids = np.nonzero(mask_voxel)[0].tolist()
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

        print(f"EPIC visualizations saved to {self.output_dir}")



def main():
<<<<<<< HEAD
    pointnet_ckpt = 'checkpoints/kl_3-5_grid_10_1024-256_downsampled/model.ckpt'
    data_dir = 'data/toys_ds_cleaned'
    batch_size = 2
    num_workers = 2
    epochs = 1
=======
    pointnet_ckpt = 'models_pointnet/kl_3-5_grid_7_1024-256/kl_3-5_grid_7_1024-256_downsampled/model.ckpt'
    data_dir = '../archive/new_dataset/toys_ds_cleaned'
    batch_size = 2
    num_workers = 2
    epochs = 8
>>>>>>> d6ab86b (bug fix and the code for correct results)
    lr = 1e-3
    prototype_update_freq = 2
    sampling = "random"
    num_samples = 8192
    initial_topk = 15
    final_topk = 3
<<<<<<< HEAD
    output_dir = "checkpoints/toys_pointnet_epic_265_8192"
    num_channel = 256
    grid_size=10
=======
    output_dir = "test_to_remove"
    num_channel = 256
>>>>>>> d6ab86b (bug fix and the code for correct results)
    dm = GaussianDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=0.1,
        sampling=sampling,
        num_points=num_samples,
    )
    dm.setup()

    train_dataset = dm.train_ds
    val_dataset = dm.val_ds

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    pl_model = PointNetLightning.load_from_checkpoint(
        pointnet_ckpt,
        in_dim=len(FEATURE_NAMES),
        num_classes=dm.num_classes,
<<<<<<< HEAD
        grid_size=grid_size
=======
        grid_size=7
>>>>>>> d6ab86b (bug fix and the code for correct results)
    )
    pointnet_model = pl_model.model
    pointnet_model.eval()

    epic_trainer = EpicTrainer(
        pointnet_model,
        num_channels=num_channel,
        lr=lr,
        initial_topk=initial_topk,
        final_topk=final_topk,
        max_epochs=epochs,
        point_subset_ratio=1.0,
        subset_seed=42
    )
    epic_trainer.hparams.batch_size = batch_size
    epic_trainer.hparams.num_workers = num_workers

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )

    epic_trainer.update_prototypes(
        train_loader,
        val_loader,
        batch_size,
        num_workers,
        device
    )

    print(f"Train prototypes loader size: {len(epic_trainer.prototypes_loader)}")
    print(f"Val prototypes loader size: {len(epic_trainer.val_prototypes_loader)}")
    if len(epic_trainer.val_prototypes_loader) == 0:
        print("Validation dataset is empty")

    os.makedirs(output_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="epic-{epoch:02d}-{val/epic_purity_loss:.4f}",
        monitor="val/epic_purity_loss",
        mode="min",
        save_top_k=3
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger = TensorBoardLogger(output_dir, name="epic_logs")

    prototype_callback = PrototypeUpdateCallback(
        prototype_update_freq,
        train_loader,
        val_loader,
        batch_size,
        num_workers,
        device
    )

    epic_viz_cb = EpicVisualizationCallback(
        output_dir=os.path.join(output_dir, "epic_visualizations"),
        num_channels=6,
<<<<<<< HEAD
        grid_size=grid_size,
=======
        grid_size=7,
>>>>>>> d6ab86b (bug fix and the code for correct results)
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=data_dir,
        num_prototypes=5
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, lr_monitor, prototype_callback, epic_viz_cb],
        log_every_n_steps=10,
        logger=logger,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        accumulate_grad_batches=4,
        reload_dataloaders_every_n_epochs=prototype_update_freq
    )

    trainer.fit(epic_trainer)

    final_matrix = epic_trainer.epic.get_weight()
    torch.save(final_matrix, os.path.join(output_dir, "final_orthogonal_matrix.pt"))

    pointnet_model.attach_epic(num_channel)
    pointnet_model.epic.load_state_dict(epic_trainer.epic.state_dict())
    pointnet_model.apply_classifier_compensation()

    torch.save({
        "pointnet_state_dict": pointnet_model.state_dict(),
        "epic_matrix": final_matrix
    }, os.path.join(output_dir, "pointnet_epic_compensated.pt"))


if __name__ == "__main__":
    main()
