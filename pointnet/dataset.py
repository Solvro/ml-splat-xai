from pathlib import Path

import numpy as np
from numpy.random import default_rng
import torch
from plyfile import PlyData
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from sklearn.preprocessing import normalize 


# ---- FEATURE NAMES ----

BASE_FEATURE_NAMES: list[str] = [
    "x", "y", "z",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "opacity",
]

COLOR_FEATURE_NAMES: list[str] = BASE_FEATURE_NAMES + [
    "f_dc_0", "f_dc_1", "f_dc_2",
]

# This is the one your training code imports
FEATURE_NAMES: list[str] = BASE_FEATURE_NAMES


OPACITY_THRESHOLD = 0.005


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prepare_gaussian_cloud(
    pts: np.ndarray,
    has_color: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    pts: (N, D) where D = 11 (no color) or 14 (with color)
    Returns:
        gauss: (M, F)   F=8 (no color) or 11 (with color)
        xyz_normalized: (M, 3)
        xyz_min: (3,)
        xyz_max: (3,)
        mask: (N,) bool mask applied to input pts
    """
    pts = pts.copy()

    # opacity sigmoid + threshold
    pts[:, 10] = sigmoid(pts[:, 10])
    mask = pts[:, 10] >= OPACITY_THRESHOLD
    pts = pts[mask]

    feat_dim = 8 + (3 if has_color else 0)  # scale(3) + rot(4) + opacity(1) [+ color(3)]

    if pts.shape[0] == 0:
        # no valid points after opacity filtering
        return (
            np.zeros((0, feat_dim), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(mask.shape[0], dtype=bool),
        )

    # normalize quaternion
    q = pts[:, 6:10]
    q_norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    pts[:, 6:10] = q / q_norm

    # normalize scales
    pts[:, 3:6] = normalize(pts[:, 3:6])

    # normalize color if present
    if has_color:
        pts[:, 11:14] = normalize(pts[:, 11:14])

    xyz = pts[:, :3]
    gauss = pts[:, 3:]  # includes scale, rot, opacity, and optionally color

    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    xyz_normalized = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-8)

    return (
        gauss.astype(np.float32),
        xyz_normalized.astype(np.float32),
        xyz_min.astype(np.float32),
        xyz_max.astype(np.float32),
        mask,
    )


class GaussianPointCloud(Dataset):
    def __init__(
        self,
        root: Path | str,
        num_points: int = 2048,
        sampling_method: str | None = "random",  # "random", "original_size"
        random_seed: int | None = None,
        has_color: bool = False,
        grid_size: int | None = None,
    ):
        """
        has_color:
            - False: expects BASE_FEATURE_NAMES in PLY
            - True:  expects COLOR_FEATURE_NAMES in PLY

        grid_size:
            - if not None, voxel_ids are computed on full (opacity-filtered) cloud,
              then sub-sampled together with points returned by __getitem__.
        """
        self.root = Path(root)
        self.num_points = num_points
        self.sampling_method: str | None = sampling_method
        self.random_seed = random_seed
        self.rng = default_rng(self.random_seed) if self.random_seed else None
        self.pt_generator = torch.Generator() if random_seed else None
        if random_seed:
            self.pt_generator.manual_seed(self.random_seed)

        self.has_color = has_color
        self.feature_names = COLOR_FEATURE_NAMES if has_color else BASE_FEATURE_NAMES
        self.grid_size = grid_size

        self.files: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        self.class_to_idx = {}

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            self.class_to_idx[class_name] = len(self.class_to_idx)
            self.classes.append(class_name)
            for ply_path in class_dir.glob("*.ply"):
                self.files.append((ply_path, self.class_to_idx[class_name]))

    def _read_ply(self, path: Path) -> np.ndarray:
        plydata = PlyData.read(str(path))
        vertex = plydata["vertex"]
        data = np.vstack([vertex[name] for name in self.feature_names]).T
        return data.astype(np.float32)

    def _random_sample(self, num_points: int) -> np.ndarray:
        """
        Sample indices from [0, num_points).
        Uses self.num_points as the target count.
        """
        N = num_points
        if N == 0:
            return np.zeros(0, dtype=np.int64)

        if N >= self.num_points:
            if self.rng is None:
                idx = np.random.choice(N, self.num_points, replace=False)
            else:
                idx = self.rng.choice(N, self.num_points, replace=False)
        else:
            if self.rng is None:
                idx = np.random.choice(N, self.num_points, replace=True)
            else:
                idx = self.rng.choice(N, self.num_points, replace=True)
        return idx.astype(np.int64)

    def _sample_index(self, num_points: int) -> np.ndarray:
        if self.sampling_method == "random":
            return self._random_sample(num_points)
        elif self.sampling_method == "original_size":
            return np.arange(num_points, dtype=np.int64)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

    def _compute_voxel_ids(self, xyz_normalized: np.ndarray) -> np.ndarray:
        """
        xyz_normalized: (M, 3) in [0, 1]
        returns voxel_ids: (M,) in [0, grid_size^3 - 1]
        """
        if self.grid_size is None:
            return np.zeros(xyz_normalized.shape[0], dtype=np.int64)

        gs = self.grid_size
        # convert [0,1] to [0, gs-1]
        coords = np.floor(xyz_normalized * gs).astype(np.int64)
        coords = np.clip(coords, 0, gs - 1)

        # linearize 3D voxel index
        voxel_ids = coords[:, 0] * (gs * gs) + coords[:, 1] * gs + coords[:, 2]
        return voxel_ids

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, label = self.files[idx]

        # 1. Read FULL cloud from ply
        pts_full = self._read_ply(path)  # (N_full, D)

        # 2. Apply opacity threshold, normalize scales/rot/color on full cloud
        gauss_all, xyz_norm_all, xyz_min, xyz_max, mask_full = prepare_gaussian_cloud(
            pts_full,
            has_color=self.has_color,
        )
        M = gauss_all.shape[0]  # number of valid (opacity-filtered) points

        # 3. Compute voxel IDs for ALL valid points (NOT only the sampled ones)
        voxel_ids_all = self._compute_voxel_ids(xyz_norm_all)  # (M,)

        # 4. Map valid points to original PLY indices
        original_valid_indices = np.nonzero(mask_full)[0].astype(np.int64)  # (M,)

        # 5. Sample indices among valid points
        if self.sampling_method is None:
            # no sampling: use all valid points
            sampled_valid_indices = np.arange(M, dtype=np.int64)
        else:
            sampled_valid_indices = self._sample_index(M)  # indices into [0..M-1]

        # 6. Subselect everything according to sampled_valid_indices
        gauss = gauss_all[sampled_valid_indices]                    # (K, F)
        xyz_normalized = xyz_norm_all[sampled_valid_indices]        # (K, 3)
        voxel_ids = voxel_ids_all[sampled_valid_indices]            # (K,)
        orig_indices = original_valid_indices[sampled_valid_indices]  # (K,)

        # 7. Build final feature tensor: [xyz_normalized, gauss]
        gauss_tensor = torch.from_numpy(gauss)
        xyz_norm_tensor = torch.from_numpy(xyz_normalized)
        gauss_cat = torch.cat([xyz_norm_tensor, gauss_tensor], dim=1)  # (K, D)

        return {
            "gauss": gauss_cat,                                    # (N, D)
            "xyz_normalized": xyz_norm_tensor,                     # (N, 3)
            "label": torch.tensor(label, dtype=torch.long),
            "indices": torch.from_numpy(orig_indices).long(),      # (N,)
            "voxel_ids": torch.from_numpy(voxel_ids).long(),       # (N,)
            "sample_idx": torch.tensor(idx, dtype=torch.long),
        }


def collate_fn(batch):
    # max number of points in batch
    max_points = max(item["gauss"].shape[0] for item in batch)

    padded_features = []
    padded_xyz_normalized = []
    padded_indices = []
    padded_voxel_ids = []
    labels = []
    masks = []
    sample_idxs = []

    for item in batch:
        features = item["gauss"]               # (Ni, D)
        xyz_normalized = item["xyz_normalized"]  # (Ni, 3)
        indices = item["indices"]              # (Ni,)
        voxel_ids = item["voxel_ids"]          # (Ni,)
        num_points = features.shape[0]

        # mask of valid points
        mask = torch.zeros(max_points, dtype=torch.bool)
        mask[:num_points] = True
        masks.append(mask)

        padding_size = max_points - num_points

        if padding_size > 0:
            # pad features
            feature_padding = torch.zeros(
                (padding_size, features.shape[1]),
                dtype=features.dtype,
            )
            features = torch.cat([features, feature_padding], dim=0)

            # pad xyz
            xyz_padding = torch.zeros(
                (padding_size, 3),
                dtype=xyz_normalized.dtype,
            )
            xyz_normalized = torch.cat([xyz_normalized, xyz_padding], dim=0)

            # pad indices
            indices_padding = torch.full(
                (padding_size,),
                -1,
                dtype=torch.long,
            )
            indices = torch.cat([indices, indices_padding], dim=0)

            # pad voxel_ids
            voxel_ids_padding = torch.full(
                (padding_size,),
                -1,
                dtype=torch.long,
            )
            voxel_ids = torch.cat([voxel_ids, voxel_ids_padding], dim=0)

        padded_features.append(features)
        padded_xyz_normalized.append(xyz_normalized)
        padded_indices.append(indices)
        padded_voxel_ids.append(voxel_ids)
        labels.append(item["label"])
        sample_idxs.append(item["sample_idx"])

    return {
        "gauss": torch.stack(padded_features).transpose(1, 2),      # (B, D, N)
        "xyz_normalized": torch.stack(padded_xyz_normalized),       # (B, N, 3)
        "label": torch.stack(labels),                               # (B,)
        "mask": torch.stack(masks),                                 # (B, N)
        "indices": torch.stack(padded_indices),                     # (B, N)
        "voxel_ids": torch.stack(padded_voxel_ids),                 # (B, N)
        "sample_idx": torch.stack(sample_idxs),                     # (B,)
    }


class GaussianDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        sampling: str | None = "random",  # "random", "original_size", or None
        num_points: int = 4096,
        seed: int = 42,
        has_color: bool = False,
        grid_size: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_ds, self.val_ds = None, None
        self.num_classes, self.in_dim = 0, 0
        self.data_dir = data_dir
        self.sampling = sampling

    def setup(self, stage: str | None = None):
        root_path = Path(self.data_dir)
        if (root_path / "train").exists() and (root_path / "test").exists():
            train_path = root_path / "train"
            test_path = root_path / "test"
        else:
            train_path = root_path
            test_path = root_path
        if self.hparams.has_color:
            FEATURE_NAMES[:] = COLOR_FEATURE_NAMES  # modify in-place
        else:
            FEATURE_NAMES[:] = BASE_FEATURE_NAMES
        # test dataset
        self.test_ds = GaussianPointCloud(
            test_path,
            num_points=self.hparams.num_points,
            sampling_method=self.hparams.sampling,
            random_seed=self.hparams.seed,
            has_color=self.hparams.has_color,
            grid_size=self.hparams.grid_size,
        )

        # full train dataset (before split)
        dataset = GaussianPointCloud(
            train_path,
            num_points=self.hparams.num_points,
            sampling_method=self.hparams.sampling,
            random_seed=self.hparams.seed,
            has_color=self.hparams.has_color,
            grid_size=self.hparams.grid_size,
        )

        self.num_classes = len(dataset.classes)

        # infer input dim (D) from one sample: gauss is (N, D)
        if len(dataset) > 0:
            sample = dataset[0]
            self.in_dim = sample["gauss"].shape[1]
        else:
            self.in_dim = 0

        # split train/val
        n_val = int(len(dataset) * self.hparams.val_split)
        n_train = len(dataset) - n_val
        self.train_ds, self.val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
        )
