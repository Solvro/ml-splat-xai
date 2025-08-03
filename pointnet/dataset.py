from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy.random import default_rng
import torch
from plyfile import PlyData
from torch.utils.data import Dataset
from pointnet.pointnet2 import exists
from pointnet.utils import farthest_point_sampling


FEATURE_NAMES: list[str] = [
    "x", "y", "z",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "opacity",
]

class GaussianPointCloud(Dataset):
    def __init__(
        self,
        root: Path,
        num_points: int = 2048,
        sampling_method: str | None = "random",  # choices: "random", "fps", "original_size"
        random_seed: int | None = None 
    ):
        self.root = Path(root)
        self.num_points = num_points
        self.sampling_method: str | None = sampling_method
        self.random_seed = random_seed
        self.rng = default_rng(self.random_seed) if exists(self.random_seed) else None 
        self.pt_generator = torch.Generator() if exists(random_seed) else None  
        if exists(random_seed):
            self.pt_generator.manual_seed(self.random_seed)

        self.files: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        class_to_idx = {}

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            class_to_idx[class_name] = len(class_to_idx)
            self.classes.append(class_name)
            for ply_path in class_dir.glob("*.ply"):
                self.files.append((ply_path, class_to_idx[class_name]))

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
        return pts[idx]
    
    def _sample(self, pts: np.ndarray) -> np.ndarray:
        if self.sampling_method == "random":
            return self._random_sample(pts)
        elif self.sampling_method == "fps":
            pts_tensor = torch.from_numpy(pts).float().unsqueeze(0)
            idx = farthest_point_sampling(pts_tensor, self.num_points, self.pt_generator).squeeze(0)
            return pts[idx.numpy()]
        elif self.sampling_method == "original_size": # only works when batch size is 1 (for debugging purposes)
            if pts.shape[0] >= self.num_points:
                return pts
            else:
                raise ValueError("Original size sampling requires at least num_points points in the point cloud.")
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        pts = self._read_ply(path)
        pts = self._sample(pts)

        xyz = pts[:, :3]
        gauss = pts[:, 3:]

        q = gauss[:, 3:7]
        q_norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
        gauss[:, 3:7] = q / q_norm

        xyz = torch.from_numpy(xyz).float()
        gauss = torch.from_numpy(gauss).float()
        label = torch.tensor(label, dtype=torch.long)

        return {
            "xyz": xyz,
            "gauss": gauss,
            "label": label,
        }


def collate_fn(batch):
    xyz = torch.stack([item["xyz"] for item in batch])
    gauss = torch.stack([item["gauss"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return xyz, gauss, labels
