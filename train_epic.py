import argparse
import os
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pointnet.pointnet import PointNetLightning
from pointnet.dataset import GaussianDataModule, FEATURE_NAMES
from pointnet.epic import EpicDisentangler


def purity_argmax_3d_voxel(voxel_features):
    B, C, Gx, Gy, Gz = voxel_features.shape
    flat = voxel_features.view(B, C, -1)  # (B, C, S)

    # 1) argmax index per (b,c): (B, C)
    idx = flat.argmax(dim=2)  # (B, C)

    # 2) permute to (B, S, C)
    flat_perm = flat.permute(0, 2, 1)  # (B, S, C)

    device = flat.device
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, C).reshape(-1)  # (B*C,)
    n_idx = idx.reshape(-1)  # (B*C,)

    # 4) gather: result shape (B*C, C)
    vals = flat_perm[b_idx, n_idx, :]  # (B*C, C)

    # 5) reshape to (B, C, C): all_acts[b, c, :] is vector v(b,c)
    all_acts = vals.view(B, C, C)  # (B, C, C)

    # 6) diagonal -> (B, C): value of channel c in its own argmax location
    diag = torch.diagonal(all_acts, dim1=1, dim2=2)  # (B, C)

    # 7) norms over channel-dim -> (B, C)
    norms = torch.linalg.norm(all_acts, dim=2)  # (B, C)

    purity = diag / (norms + 1e-8)  # (B, C)
    return purity


# def purity_argmax_3d_voxel_corrected(voxel_features):
#     B, C, Gx, Gy, Gz = voxel_features.shape
#     voxel_features_flat = voxel_features.view(B, C, -1)  # (B, C, G*G*G)
#     
#     max_vals, max_indices = torch.max(voxel_features_flat, dim=2)  # (B, C)
#     
#     all_activations = torch.zeros(B, C, C, device=voxel_features.device)
#     for b in range(B):
#         for c in range(C):
#             all_activations[b, :, c] = voxel_features_flat[b, :, max_indices[b, c]]
#     
#     norms = torch.norm(all_activations, dim=1)  # (B, C)
#     
#     purity = max_vals / (norms + 1e-8)  # (B, C)
#     
#     return purity


class EpicTrainer(pl.LightningModule):
    def __init__(self, pointnet_model, lr: float = 1e-5):  # Zmniejszony learning rate
        super().__init__()
        self.save_hyperparameters(ignore=["pointnet_model"])
        self.pointnet = pointnet_model.eval()
        for p in self.pointnet.parameters():
            p.requires_grad_(False)

        self.epic = EpicDisentangler(C=1024)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        features = batch["gauss"]
        xyz_normalized = batch["xyz_normalized"]
        mask = batch.get("mask", None)

        point_features, xyz_for_vox = self.pointnet.extract_point_features(features, xyz_normalized, mask)
        
        self.log("train/pf_mean", point_features.mean(), prog_bar=False)
        self.log("train/pf_std", point_features.std(), prog_bar=False)
        
        point_features = self.epic(point_features)
        
        self.log("train/pf_after_epic_mean", point_features.mean(), prog_bar=False)
        self.log("train/pf_after_epic_std", point_features.std(), prog_bar=False)
        
        voxel_features, indices, point_counts = self.pointnet.voxel_agg(point_features, xyz_for_vox, mask)
        
        self.log("train/voxel_mean", voxel_features.mean(), prog_bar=False)
        self.log("train/voxel_std", voxel_features.std(), prog_bar=False)
        
        purity = purity_argmax_3d_voxel(voxel_features)
        loss = -purity.mean()

        self.log("train/purity_mean", purity.mean(), prog_bar=True)
        self.log("train/purity_min", purity.min(), prog_bar=False)
        self.log("train/purity_max", purity.max(), prog_bar=False)
        self.log("train/epic_purity_loss", loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        features = batch["gauss"]
        xyz_normalized = batch["xyz_normalized"]
        mask = batch.get("mask", None)

        point_features, xyz_for_vox = self.pointnet.extract_point_features(features, xyz_normalized, mask)
        point_features = self.epic(point_features)
        voxel_features, indices, point_counts = self.pointnet.voxel_agg(point_features, xyz_for_vox, mask)
        
        purity = purity_argmax_3d_voxel(voxel_features)
        loss = -purity.mean()
    
        self.log("val/epic_purity_loss", loss, prog_bar=True)
        self.log("val/purity_mean", purity.mean(), prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.epic.parameters(), lr=self.lr, weight_decay=1e-5)
        return opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointnet_ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)  # Więcej epoch
    parser.add_argument("--lr", type=float, default=1e-5)   # Mniejszy learning rate
    parser.add_argument("--out", type=str, default="orthogonal.pt")
    parser.add_argument("--logging_out", type=str, default="logger")
    parser.add_argument("--sampling", type=str, default="original_size")
    parser.add_argument("--max_samples", type=int, default=8192)
    args = parser.parse_args()

    dm = GaussianDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
                            sampling=args.sampling, num_points=args.max_samples)
    dm.setup()

    pl_base = PointNetLightning.load_from_checkpoint(args.pointnet_ckpt,
                                                     in_dim=len(FEATURE_NAMES),
                                                     num_classes=dm.num_classes,
                                                     grid_size=10)
    pointnet_model = pl_base.model
    pointnet_model.eval()
    for p in pointnet_model.parameters():
        p.requires_grad_(False)

    epic_trainer = EpicTrainer(pointnet_model, lr=args.lr)

    model_save_dir = args.logging_out
    os.makedirs(model_save_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        monitor="val/epic_purity_loss",
        mode="min",
        save_top_k=1,
        filename="epic-{epoch:02d}-{val/epic_purity_loss:.4f}",
        every_n_epochs=1
    )
    
    lr_mon = LearningRateMonitor(logging_interval="epoch")
    early_stop = EarlyStopping(
        monitor="val/epic_purity_loss",
        patience=10,
        mode="min",
        min_delta=0.001
    )
    
    logger = TensorBoardLogger(model_save_dir, name="epic_logger")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[ckpt_cb, lr_mon, early_stop],
        log_every_n_steps=10,
        logger=logger
    )

    trainer.fit(epic_trainer, datamodule=dm)

    W = epic_trainer.epic.get_weight()
    torch.save(W, args.out)
    print("Saved orthogonal matrix tensor to:", args.out)

if __name__ == "__main__":
    main()
