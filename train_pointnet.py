import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from pointnet.pointnet2 import PointNet2ClsMSG, PointNet2ClsSSG

FEATURE_NAMES: list[str] = [
    "x", "y", "z",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "opacity",
]

class GaussianPointCloud(Dataset):
    def __init__(self, root: Path, num_points: int = 2048):
        self.root = Path(root)
        self.num_points = num_points

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
            idx = np.random.choice(N, self.num_points, replace=False)
        else:
            idx = np.random.choice(N, self.num_points, replace=True)
        return pts[idx]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        pts = self._read_ply(path)
        pts = self._random_sample(pts)

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


def train_one_epoch(
        model: PointNet2ClsSSG,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str) -> tuple[float, float]:
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for xyz, gauss, labels in tqdm(loader, desc="train", leave=False):
        xyz, gauss, labels = xyz.to(device), gauss.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(torch.cat([xyz, gauss], dim=-1).transpose(1,2), xyz.transpose(1,2))
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def eval_one_epoch(
        model: PointNet2ClsSSG | PointNet2ClsMSG,
        loader: DataLoader,
        device: str) -> tuple[float, float]:
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for xyz, gauss, labels in tqdm(loader, desc="eval", leave=False):
            xyz, gauss, labels = xyz.to(device), gauss.to(device), labels.to(device)
            out = model(torch.cat([xyz, gauss], dim=-1).transpose(1,2), xyz.transpose(1,2))
            loss = F.cross_entropy(out, labels)
            running_loss += loss.item() * labels.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train PointNet++ on Gaussian Splatting point clouds")
    parser.add_argument("--data_dir", type=str, default="data", help="root directory with class subfolders")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = GaussianPointCloud(Path(args.data_dir), num_points=args.num_points)
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    num_classes = len(dataset.classes)

    model = PointNet2ClsMSG(in_dim=11, out_dim=num_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, args.device)

        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "acc": best_acc}, "best_model_msg.pt")

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
