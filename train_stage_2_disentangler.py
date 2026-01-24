import argparse
import os

import yaml
from einops import rearrange
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from pointnet.callbacks.disentangler_visualization import DisentanglerVisualizationCallback
from pointnet.callbacks.prototype_update import PrototypeUpdateCallback
from pointnet.datasets.gaussian_point_cloud import GaussianDataModule, FEATURE_NAMES, collate_fn
from pointnet.datasets.prototypes import PrototypesDataset, collate_prototypes
from pointnet.orthogonal import OrthogonalDisentangler
from pointnet.pointnet import PointNetLightning


@torch.no_grad()
def generate_prototypes_pointnet(model, dataloader, num_channels, topk=5, device="cpu", U=None, debug=False):
    model.eval()
    model.to(device)
    if U is not None:
        U = U.to(device)

    top_acts = torch.full((topk, num_channels), -float("inf"), device=device)
    top_inds = torch.full((topk, num_channels), -1, dtype=torch.long, device=device)

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating prototypes"):
        features = batch["gauss"].to(device)
        xyz_normalized = batch["xyz_normalized"].to(device)
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(device)

        voxel_ids = batch["voxel_ids"].to(device)  # (B, N)
        dataset_indices = batch.get("sample_idx").to(device) # B,

        B = features.size(0)
        if debug and batch_idx == 0:
            print(f"Batch {batch_idx}: Using dataset indices {dataset_indices.tolist()}")
            print(f"Batch {batch_idx}: voxel_ids shape {voxel_ids.shape}, dtype={voxel_ids.dtype}")

        point_features, _ = model.extract_point_features(features, xyz_normalized, mask)

        voxel_features, _, point_counts = model.voxel_agg(point_features, voxel_ids, mask)
        voxel_mask = (point_counts.squeeze(1) > 0)  # (B, V)
        voxel_features_flat = rearrange(voxel_features, 'b c x y z -> b c (x y z)')

        if U is not None:
            voxel_features_flat = torch.einsum("cd,bdn->bcn", U, voxel_features_flat)

        voxel_features_flat = voxel_features_flat.masked_fill(
            ~voxel_mask.unsqueeze(1).expand_as(voxel_features_flat),
            0.0
        )

        vals, _ = voxel_features_flat.max(dim=2)  # (B, C)

        min_top_acts, _ = torch.min(top_acts, dim=0)  # (C,)
        update_mask = vals > min_top_acts

        if not update_mask.any():
            continue

        combined_acts = torch.cat([top_acts, vals], dim=0)   # (topk+B, C)
        dataset_idx_exp = dataset_indices.view(B, 1).expand(B, num_channels)  # (B, C)
        combined_inds = torch.cat([top_inds, dataset_idx_exp], dim=0)        # (topk+B, C)

        channels_to_update = torch.where(update_mask.any(dim=0))[0]

        vals, idxs = torch.topk(combined_acts[:, channels_to_update], k=topk, dim=0)

        top_acts[:, channels_to_update] = vals
        top_inds[:, channels_to_update] = combined_inds[:, channels_to_update][idxs, torch.arange(len(channels_to_update))]

    prototypes_dict = {c: top_inds[:, c].cpu().numpy().tolist() for c in range(num_channels)}
    return prototypes_dict


def purity_argmax_voxel(voxel_features: torch.Tensor, channels: torch.Tensor, voxel_mask: torch.Tensor | None = None) -> torch.Tensor:
    B, _, _ = voxel_features.shape
    device = voxel_features.device

    if voxel_mask is not None:
        voxel_features_masked = voxel_features.masked_fill(
            ~voxel_mask.unsqueeze(1).expand_as(voxel_features),
            -float("inf")
        )
    else:
        voxel_features_masked = voxel_features

    acts_c = voxel_features_masked[torch.arange(B, device=device), channels, :]  # (B, V)
    max_vals, max_indices = torch.max(acts_c, dim=1)  # (B,)

    vectors = voxel_features[torch.arange(B, device=device), :, max_indices]  # (B, C)
    vectors = torch.where(torch.isfinite(vectors), vectors, torch.zeros_like(vectors))

    target = vectors[torch.arange(B, device=device), channels]  # (B,)

    norms = vectors.norm(dim=1).clamp_min(1e-8)
    purity = target / norms

    valid = torch.isfinite(max_vals)
    purity = torch.where(valid, purity, purity.new_zeros(purity.shape))
    return purity


class DisentanglerTrainer(pl.LightningModule):
    def __init__(
        self,
        pointnet_model,
        num_channels=256,
        lr: float = 1e-4,
        initial_topk=40,
        final_topk=5,
        max_epochs=20,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["pointnet_model"])
        self.pointnet = pointnet_model.eval()
        for p in self.pointnet.parameters():
            p.requires_grad_(False)

        if self.pointnet.disentangler is None:
            self.disentangler = OrthogonalDisentangler(C=num_channels)
        else:
            self.disentangler = self.pointnet.disentangler
        self.lr = lr
        self.initial_topk = initial_topk
        self.final_topk = final_topk
        self.max_epochs = max_epochs
        self.prototypes_loader = None
        self.val_prototypes_loader = None
        self.current_topk = initial_topk
        self.last_val_prototypes = None

    def common_step(self, batch, _):
        self.pointnet.eval()
        features = batch["gauss"]
        xyz_normalized = batch["xyz_normalized"]
        mask = batch.get("mask", None)
        channels = batch["channel"]

        voxel_ids = batch["voxel_ids"]

        with torch.no_grad():
            point_features, _ = self.pointnet.extract_point_features(features, xyz_normalized, mask)
            voxel_features, _, point_counts = self.pointnet.voxel_agg(point_features, voxel_ids, mask)

        voxel_features_flat = voxel_features.view(voxel_features.size(0), voxel_features.size(1), -1) # (B, C, V)
        voxel_features_flat = self.disentangler(voxel_features_flat)

        voxel_mask = (point_counts.squeeze(1) > 0)  # (B, V)
        return voxel_features_flat, channels, voxel_mask

    def training_step(self, batch, batch_idx):
        voxel_features_flat, channels, voxel_mask = self.common_step(batch, batch_idx)
        purity = purity_argmax_voxel(voxel_features_flat, channels, voxel_mask)
        loss = -purity.mean()

        self.log("train/purity_mean", purity.mean(), prog_bar=True)
        self.log("train/purity_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        voxel_features_flat, channels, voxel_mask = self.common_step(batch, batch_idx)
        purity = purity_argmax_voxel(voxel_features_flat, channels, voxel_mask)
        loss = -purity.mean()

        self.log("val/purity_mean", purity.mean(), prog_bar=True)
        self.log("val/purity_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.disentangler.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(opt, T_max=self.max_epochs)
        return [opt], [scheduler]

    def train_dataloader(self):
        return self.prototypes_loader

    def val_dataloader(self):
        return self.val_prototypes_loader

    def test_step(self, batch, batch_idx):
        voxel_features_flat, channels, voxel_mask = self.common_step(batch, batch_idx)
        purity = purity_argmax_voxel(voxel_features_flat, channels, voxel_mask)
        loss = -purity.mean()

        self.log("test/purity_mean", purity.mean(), prog_bar=True)
        self.log("test/purity_loss", loss, prog_bar=True)
        return loss

    def test_dataloader(self):
        return self.test_prototypes_loader

    @torch.no_grad()
    def update_test_prototypes(self, test_loader, n_prototypes, batch_size, num_workers, device):
        U = self.disentangler.get_weight().to(device)

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

        U = self.disentangler.get_weight().to(device)

        print(f"Generating {self.current_topk} prototypes per channel:")
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


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: Train disentangler on frozen PointNet"
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides defaults)")

    parser.add_argument("--pointnet_ckpt", type=str, default=None,
                        help="Path to PointNet checkpoint from stage 1")

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--sampling", type=str, default="original_size",
                        choices=["fps", "random", "original_size"])
    parser.add_argument("--num_samples", type=int, default=8192)
    parser.add_argument("--grid_size", type=int, default=7)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)

    parser.add_argument("--prototype_update_freq", type=int, default=2)
    parser.add_argument("--initial_topk", type=int, default=40)
    parser.add_argument("--final_topk", type=int, default=5)
    parser.add_argument("--num_channels", type=int, default=256)

    parser.add_argument("--early_stopping_patience", type=int, default=15)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--log_name", type=str, default="disentangler_logs")

    parser.add_argument("--viz_channels", type=int, default=6)
    parser.add_argument("--viz_topk", type=int, default=5)

    args = parser.parse_args()

    if args.config is not None:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                if getattr(args, key) is None or getattr(args, key) == parser.get_default(key):
                    setattr(args, key, value)

    return args


def main() -> None:
    args = parse_args()

    print(f"Checkpoint: {args.pointnet_ckpt}")

    dm = GaussianDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.1,
        sampling=args.sampling,
        num_points=args.num_samples,
        grid_size=args.grid_size,

    )
    dm.setup()

    train_dataset = dm.train_ds
    val_dataset = dm.val_ds

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    pl_model = PointNetLightning.load_from_checkpoint(
        args.pointnet_ckpt,
        in_dim=len(FEATURE_NAMES),
        num_classes=dm.num_classes,
        grid_size=args.grid_size
    )
    pointnet_model = pl_model.model
    pointnet_model.eval()

    disentangler_trainer = DisentanglerTrainer(
        pointnet_model,
        num_channels=args.num_channels,
        lr=args.lr,
        initial_topk=args.initial_topk,
        final_topk=args.final_topk,
        max_epochs=args.epochs,
    )
    disentangler_trainer.hparams.batch_size = args.batch_size
    disentangler_trainer.hparams.num_workers = args.num_workers

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )

    disentangler_trainer.update_prototypes(
        train_loader,
        val_loader,
        args.batch_size,
        args.num_workers,
        device
    )

    print(f"Train prototypes loader size: {len(disentangler_trainer.prototypes_loader)}")
    print(f"Val prototypes loader size: {len(disentangler_trainer.val_prototypes_loader)}")
    if len(disentangler_trainer.val_prototypes_loader) == 0:
        print("Validation dataset is empty")

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="disentangler-{epoch:02d}-{val/purity_loss:.4f}",
        monitor="val/purity_loss",
        mode="min",
        save_top_k=3
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger = TensorBoardLogger(args.output_dir, name="disentangler_logs")

    prototype_callback = PrototypeUpdateCallback(
        args.prototype_update_freq,
        train_loader,
        val_loader,
        args.batch_size,
        args.num_workers,
        device
    )

    dis_viz_cb = DisentanglerVisualizationCallback(
        output_dir=os.path.join(args.output_dir, "disentangler_visualizations"),
        num_channels=6,
        grid_size=args.grid_size,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        num_prototypes=5
    )

    stopping_callback = EarlyStopping(
        monitor="val/purity_loss",
        patience=args.early_stopping_patience,
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, lr_monitor, prototype_callback, dis_viz_cb, stopping_callback],
        log_every_n_steps=10,
        logger=logger,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        accumulate_grad_batches=2,
        reload_dataloaders_every_n_epochs=args.prototype_update_freq
    )

    trainer.fit(disentangler_trainer)

    final_matrix = disentangler_trainer.disentangler.get_weight()
    torch.save(final_matrix, os.path.join(args.output_dir, "final_orthogonal_matrix.pt"))

    pointnet_model.attach_disentangler(args.num_channels)
    pointnet_model.disentangler.load_state_dict(disentangler_trainer.disentangler.state_dict())
    pointnet_model.apply_classifier_compensation()

    torch.save({
        "pointnet_state_dict": pointnet_model.state_dict(),
        "disentangler_matrix": final_matrix
    }, os.path.join(args.output_dir, "pointnet_disentangler_compensated.pt"))


if __name__ == "__main__":
    main()
