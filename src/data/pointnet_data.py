import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from plyfile import PlyData


class GaussianPointCloudDataset(Dataset):
    def __init__(self, root: str, num_points: int = 2048):
        self.root = Path(root)
        self.num_points = num_points
        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}
        
        for class_dir in sorted(self.root.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.classes)
                    self.classes.append(class_name)
                for ply_file in class_dir.glob("*.ply"):
                    self.samples.append((ply_file, self.class_to_idx[class_name]))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if not isinstance(idx, int):
            raise TypeError(f"Index must be integer, got {type(idx)}")
        
        ply_path, label = self.samples[idx]
        data = self._read_ply(ply_path)
        data = self._random_sample(data)
        
        xyz = data[:, :3]
        gauss = data[:, 3:]
        
        if gauss.shape[1] >= 4:
            rotations = gauss[:, :4]
            rotations = rotations / (np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-8)
            gauss[:, :4] = rotations
        
        return (
            torch.from_numpy(xyz).float(),
            torch.from_numpy(gauss).float(),
            torch.tensor(label, dtype=torch.long),
            idx
        )

    def _read_ply(self, path: Path) -> np.ndarray:
        plydata = PlyData.read(path)
        vertex = plydata["vertex"]
        
        property_names = [p.name for p in vertex.properties]
        
        desired_props = [
            "x", "y", "z", 
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3",
            "opacity"
        ]
        
        data = []
        for prop in desired_props:
            if prop in property_names:
                data.append(np.asarray(vertex[prop]))
            else:
                data.append(np.zeros(vertex.count))
                
        return np.stack(data, axis=1)
    
    def _random_sample(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[0]
        if n >= self.num_points:
            indices = np.random.choice(n, self.num_points, replace=False)
        else:
            indices = np.random.choice(n, self.num_points, replace=True)
        return data[indices]


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xyz_list = []
    gauss_list = []
    labels_list = []
    indices_list = []
    
    for xyz, gauss, label, idx in batch:
        xyz_list.append(xyz)
        gauss_list.append(gauss)
        labels_list.append(label)
        indices_list.append(idx)
    
    return (
        torch.stack(xyz_list),
        torch.stack(gauss_list),
        torch.stack(labels_list),
        torch.tensor(indices_list)
    )


def create_indexed_dataloader(
    datapath: str,
    num_points: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = False
) -> DataLoader:
    dataset = GaussianPointCloudDataset(datapath, num_points)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


def prototypes_collate_fn(batch):
    xyz_list = []
    gauss_list = []
    point_indices_list = []
    channels_list = []
    sample_indices_list = []
    
    for item in batch:
        xyz_list.append(item[0])
        gauss_list.append(item[1])
        point_indices_list.append(item[2])
        channels_list.append(item[3])
        sample_indices_list.append(item[4])
    
    return (
        torch.stack(xyz_list),  # (B, num_points, 3)
        torch.stack(gauss_list),  # (B, num_points, num_gauss_features)
        torch.tensor(point_indices_list),  # (B,)
        torch.stack(channels_list),  # (B,)
        torch.tensor(sample_indices_list)  # (B,)
    )


class PrototypesDataset(Dataset):
    def __init__(self, prototypes, orig_dataset):
        self.prototypes_list = []
        self.orig_dataset = orig_dataset
        
        for c, prototypes_list in prototypes.items():
            for proto in prototypes_list:
                self.prototypes_list.append({
                    "sample_index": proto["sample_index"],
                    "point_idx": proto["point_idx"],
                    "channel": c
                })
    
    def __len__(self):
        return len(self.prototypes_list)
    
    def __getitem__(self, idx):
        proto = self.prototypes_list[idx]
        sample_idx = proto["sample_index"]
        point_idx = proto["point_idx"]
        c = proto["channel"]
        
        xyz, gauss, _, _ = self.orig_dataset[sample_idx]
        
        return (
            xyz,
            gauss,
            point_idx,
            torch.tensor(c),
            sample_idx
        )


def create_prototype_dataloader(
    positive_prototypes: dict[int, list[dict[str, any]]],
    orig_dataloader: DataLoader,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = True
) -> DataLoader:
    
    dataset = PrototypesDataset(positive_prototypes, orig_dataloader.dataset)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=prototypes_collate_fn
    )
