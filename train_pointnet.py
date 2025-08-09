import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from pointnet.pointnet import PointNetCls
from pointnet.dataset import GaussianPointCloud, collate_fn, FEATURE_NAMES

def train_one_epoch(
        model: PointNetCls,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str) -> tuple[float, float]:
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        features = batch["gauss"].to(device)
        xyz_normalized = batch["xyz_normalized"].to(device)
        labels = batch["label"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        out = model(features, xyz_normalized, mask)
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(
        model: PointNetCls,
        loader: DataLoader,
        device: str) -> tuple[float, float]:
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    for batch in tqdm(loader, desc="eval", leave=False):
        features = batch["gauss"].to(device)
        xyz_normalized = batch["xyz_normalized"].to(device)
        labels = batch["label"].to(device)
        mask = batch["mask"].to(device)

        out, _ = model(features, xyz_normalized, mask)
        loss = F.cross_entropy(out, labels)
        running_loss += loss.item() * labels.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train Interpretable PointNet on Gaussian Point Clouds")
    parser.add_argument("--data_dir", type=str, default="test_data/train", help="Root directory with class subfolders")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sampling", type=str, default="fps", choices=["fps", "random", "original_size"], help="Point sampling method")
    parser.add_argument("--num_points", type=int, default=512, help="Number of points if sampling is used")
    parser.add_argument("--grid_size", type=int, default=10, help="Size of the voxel grid for aggregation")
    parser.add_argument("--model_save_path", type=str, default="best_model_interpret43535.pt")
    parser.add_argument("--use_stn", action="store_true", help="Use STN layers")
    args = parser.parse_args()

    dataset = GaussianPointCloud(Path(args.data_dir), num_points=args.num_points, sampling_method=args.sampling)
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    num_classes = len(dataset.classes)
    in_dim = len(FEATURE_NAMES)

    model = PointNetCls(
        in_dim=in_dim, 
        out_dim=num_classes, 
        grid_size=args.grid_size,
        stn_3d=args.use_stn,
        stn_nd=args.use_stn,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, args.device)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "acc": best_acc,
                "config": vars(args)
            }, args.model_save_path)

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()