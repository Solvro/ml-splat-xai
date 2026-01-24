from pathlib import Path

import numpy as np
from numpy.random import default_rng
import torch
from plyfile import PlyData
import pytorch_lightning as pl
from sklearn.preprocessing import normalize 
from torch.utils.data import Dataset, DataLoader, random_split


FEATURE_NAMES: list[str] = [
    "x", "y", "z",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "opacity",
]
OPACITY_THRESHOLD = 0.005


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prepare_gaussian_cloud(
    pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = pts.copy()
    pts[:, 10] = sigmoid(pts[:, 10])
    mask = pts[:, 10] >= OPACITY_THRESHOLD
    pts = pts[mask]

    if pts.shape[0] == 0:
        return (np.zeros((0, 8), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(0, dtype=bool))

    q = pts[:, 6:10]
    q_norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    pts[:, 6:10] = q / q_norm
    pts[:, 3:6] = normalize(pts[:, 3:6])

    xyz = pts[:, :3]
    gauss = pts[:, 3:]

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
        grid_size: int | None = None,
    ):
        self.root = Path(root)
        self.num_points = num_points
        self.sampling_method: str | None = sampling_method
        self.random_seed = random_seed
        self.rng = default_rng(self.random_seed) if self.random_seed else None
        self.pt_generator = torch.Generator() if random_seed else None
        if random_seed:
            self.pt_generator.manual_seed(self.random_seed)
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
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .ply files found under {self.root}. "
                "Check data_dir/train (or data_dir) and file extensions."
            )

    @staticmethod
    def _read_ply(path: Path) -> np.ndarray:
        plydata = PlyData.read(str(path))
        vertex = plydata["vertex"]
        data = np.vstack([vertex[name] for name in FEATURE_NAMES]).T
        return data.astype(np.float32)

    def _random_sample(self, pts: np.ndarray) -> np.ndarray:
        N = pts.shape[0]
        if N >= self.num_points:
            idx = np.random.choice(N, self.num_points, replace=False) if self.rng is None else self.rng.choice(N, self.num_points, replace=False)
        else:
            idx = np.random.choice(N, self.num_points, replace=True) if self.rng is None else self.rng.choice(N, self.num_points, replace=True)
        return idx

    def _sample_index(self, pts: np.ndarray) -> np.ndarray:
        if self.sampling_method == "random":
            return self._random_sample(pts)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

    def _compute_voxel_ids(self, xyz_normalized: np.ndarray) -> np.ndarray:
        if self.grid_size is None:
            return np.zeros(xyz_normalized.shape[0], dtype=np.int64)
        gs = self.grid_size
        coords = np.floor(xyz_normalized * gs).astype(np.int64)
        coords = np.clip(coords, 0, gs - 1)
        voxel_ids = coords[:, 0] * (gs * gs) + coords[:, 1] * gs + coords[:, 2]
        return voxel_ids

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, label = self.files[idx]
        pts = self._read_ply(path)
        gauss_all, xyz_norm_all, xyz_min, xyz_max, mask_full = prepare_gaussian_cloud(pts)
        M = gauss_all.shape[0]
        voxel_ids_all = self._compute_voxel_ids(xyz_norm_all)
        original_valid_indices = np.nonzero(mask_full)[0].astype(np.int64)  # (M,)
        if self.sampling_method == "original_size" :
            sampled_valid_indices = np.arange(M, dtype=np.int64)
        else:
            sampled_valid_indices = self._sample_index(gauss_all)  # indices into [0..M-1]
        gauss = gauss_all[sampled_valid_indices]                    # (K, F)
        xyz_normalized = xyz_norm_all[sampled_valid_indices]        # (K, 3)
        voxel_ids = voxel_ids_all[sampled_valid_indices]            # (K,)
        orig_indices = original_valid_indices[sampled_valid_indices]  # (K,)

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
    max_points = max(item["gauss"].shape[0] for item in batch)

    padded_features = []
    padded_xyz_normalized = []
    padded_indices = []
    padded_voxel_ids = []
    labels = []
    masks = []
    sample_idxs = []

    for item in batch:
        features = item["gauss"]
        xyz_normalized = item["xyz_normalized"]
        indices = item["indices"]
        voxel_ids = item["voxel_ids"]
        num_points = features.shape[0]

        mask = torch.zeros(max_points, dtype=torch.bool)
        mask[:num_points] = True
        masks.append(mask)

        padding_size = max_points - num_points

        if padding_size > 0:
            feature_padding = torch.zeros(
                (padding_size, features.shape[1]),
                dtype=features.dtype,
            )
            features = torch.cat([features, feature_padding], dim=0)

            xyz_padding = torch.zeros(
                (padding_size, 3),
                dtype=xyz_normalized.dtype,
            )
            xyz_normalized = torch.cat([xyz_normalized, xyz_padding], dim=0)
            indices_padding = torch.full(
                (padding_size,),
                -1,
                dtype=torch.long,
            )
            indices = torch.cat([indices, indices_padding], dim=0)
            voxel_ids_padding = torch.zeros(
                (padding_size,),
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
        "gauss": torch.stack(padded_features).transpose(1, 2),
        "xyz_normalized": torch.stack(padded_xyz_normalized),
        "label": torch.stack(labels),
        "mask": torch.stack(masks),
        "indices": torch.stack(padded_indices),                
        "voxel_ids": torch.stack(padded_voxel_ids),
        "sample_idx": torch.stack(sample_idxs),
    }


class GaussianDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        sampling: str = "random",  # "random", "original_size", or None
        num_points: int = 4096,
        seed: int = 42,
        grid_size: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_ds, self.val_ds = None, None
        self.num_classes, self.in_dim = 0, len(FEATURE_NAMES)
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

        self.test_ds = GaussianPointCloud(
            test_path,
            num_points=self.hparams.num_points,
            sampling_method=self.hparams.sampling,
            random_seed=self.hparams.seed,
            grid_size=self.hparams.grid_size,
        )
        dataset = GaussianPointCloud(
            train_path,
            num_points=self.hparams.num_points,
            sampling_method=self.hparams.sampling,
            random_seed=self.hparams.seed,
            grid_size=self.hparams.grid_size,
        )
        self.num_classes = len(dataset.classes)
        n_val = int(len(dataset) * self.hparams.val_split)
        n_train = len(dataset) - n_val
        self.train_ds, self.val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(self.hparams.seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
        )
