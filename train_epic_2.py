import argparse
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import os
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pointnet.pointnet import PointNetLightning
from pointnet.dataset import GaussianDataModule, FEATURE_NAMES, collate_fn
from pointnet.epic import EpicDisentangler
from tqdm import tqdm
import numpy as np


def generate_prototypes_pointnet(model, dataloader, num_channels, topk=5, device="cpu", U=None):
    model.eval()
    model.to(device)

    top_activations = torch.full((topk, num_channels), -float("inf"), device=device)
    top_indices = torch.full((topk, num_channels), -1, dtype=torch.long, device=device)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating prototypes"):
            features = batch["gauss"].to(device)
            xyz_normalized = batch["xyz_normalized"].to(device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device)

            # Użyj prawdziwych indeksów z datasetu
            dataset_indices = batch["indices"].to(device)

            # Przejdź przez pełną forward pass, aby dostać aktywacje wokseli
            point_features, _ = model.extract_point_features(features, xyz_normalized, mask)
            
            # Zastosuj macierz disentanglującą jeśli podana
            if U is not None:
                U = U.to(point_features.device)
                point_features = torch.einsum("dc,bdn->bcn", U, point_features)
            
            # Przejdź przez agregację wokselową, aby dostać aktywacje wokseli
            voxel_features, voxel_indices, point_counts = model.voxel_agg(point_features, 
                                                                         xyz_normalized, 
                                                                         mask)
            
            # voxel_features: (B, C, G, G, G)
            voxel_features_flat = rearrange(voxel_features, 'b c x y z -> b c (x y z)')
            max_voxel_activations, _ = torch.max(voxel_features_flat, dim=2)  # (B, C)
            
            for c in range(num_channels):
                activations = max_voxel_activations[:, c]  # (B,)
                
                combined_activations = torch.cat([top_activations[:, c], activations])
                combined_indices = torch.cat([top_indices[:, c], dataset_indices])
                
                new_topk_vals, new_topk_indices = torch.topk(combined_activations, topk, largest=True)
                
                top_activations[:, c] = new_topk_vals
                top_indices[:, c] = combined_indices[new_topk_indices]

    prototypes_dict = {}
    for c in range(num_channels):
        prototypes_dict[c] = top_indices[:, c].cpu().numpy().tolist()
    
    return prototypes_dict


def purity_argmax_point(feature_map, channels, mask=None):
    B, C, N = feature_map.shape
    device = feature_map.device

    if mask is not None:
        mask_exp = mask.unsqueeze(1).expand(B, C, N)
        feature_map = feature_map.masked_fill(~mask_exp, float("-inf"))

    max_vals, max_indices = torch.max(feature_map, dim=2)  # (B, C)

    feature_map_transposed = feature_map.transpose(1, 2)  # (B, N, C)
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, C)  # (B, C)
    
    activations_at_max = feature_map_transposed[batch_indices, max_indices, :]  # (B, C, C)

    target_activations = activations_at_max[torch.arange(B), :, channels]  # (B, C)

    norms = torch.norm(activations_at_max, dim=2, p=2)  # (B, C)
    purity = target_activations / (norms + 1e-8)
    
    return purity[torch.arange(B), channels]  # (B,)


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
    batch_without_channel = [
        {k: v for k, v in item.items() if k != "channel"} 
        for item in batch
    ]
    
    batch_dict = collate_fn(batch_without_channel)
    
    batch_dict["channel"] = torch.tensor(channels, dtype=torch.long)
    return batch_dict


class EpicTrainer(pl.LightningModule):
    def __init__(self, pointnet_model, num_channels=1024, lr: float = 1e-4, initial_topk=40, final_topk=5, max_epochs=20):
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

    def training_step(self, batch, batch_idx):
        self.pointnet.eval()

        features = batch["gauss"]
        xyz_normalized = batch["xyz_normalized"]
        mask = batch.get("mask", None)
        channels = batch["channel"]
        
        with torch.no_grad():
            point_features, _ = self.pointnet.extract_point_features(features, xyz_normalized, mask)
        
        point_features = self.epic(point_features)
        
        purity = purity_argmax_point(point_features, channels, mask)
        loss = -purity.mean()

        self.log("train/purity_mean", purity.mean(), prog_bar=True)
        self.log("train/epic_purity_loss", loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        self.pointnet.eval()

        features = batch["gauss"]
        xyz_normalized = batch["xyz_normalized"]
        mask = batch.get("mask", None)
        channels = batch["channel"]
        
        with torch.no_grad():
            point_features, _ = self.pointnet.extract_point_features(features, xyz_normalized, mask)
        
        point_features = self.epic(point_features)
        purity = purity_argmax_point(point_features, channels, mask)
        loss = -purity.mean()
    
        self.log("val/purity_mean", purity.mean(), prog_bar=True)
        self.log("val/epic_purity_loss", loss, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.epic.parameters(), lr=self.lr)
        return opt

    def train_dataloader(self):
        return self.prototypes_loader

    def val_dataloader(self):
        return self.val_prototypes_loader

    def update_prototypes(self, train_dataloader, val_dataloader, device):
        progress = min(self.current_epoch / self.max_epochs, 1.0)
        self.current_topk = int(self.initial_topk - progress * (self.initial_topk - self.final_topk))
        
        # Get current U matrix
        U = self.epic.get_weight().detach()
        
        # Generate new prototypes with current U matrix
        print(f"Generating {self.current_topk} prototypes per channel...")
        prototypes = generate_prototypes_pointnet(
            self.pointnet,
            train_dataloader,
            num_channels=self.hparams.num_channels,
            topk=self.current_topk,
            device=device,
            U=U
        )
        
        val_prototypes = generate_prototypes_pointnet(
            self.pointnet,
            val_dataloader,
            num_channels=self.hparams.num_channels,
            topk=max(5, self.current_topk // 2),  # At least 5 prototypes for validation
            device=device,
            U=U
        )
        
        # Update dataloaders
        prototypes_dataset = PrototypesDataset(train_dataloader.dataset, prototypes)
        self.prototypes_loader = DataLoader(
            prototypes_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_prototypes
        )
        
        val_dataset = PrototypesDataset(val_dataloader.dataset, val_prototypes)
        self.val_prototypes_loader = DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_prototypes
        )


def main():
    parser = argparse.ArgumentParser(description="Train EPIC disentanglement matrix for PointNet")
    parser.add_argument("--pointnet_ckpt", type=str, required=True, help="Path to PointNet checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")  # 20 epochs as in paper
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--prototype_update_freq", type=int, default=2, help="Update prototypes every N epochs")
    parser.add_argument("--sampling", type=str, default="data", help="Sampling")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples")
    parser.add_argument("--initial_topk", type=int, default=40, help="Number of initial prototypes in training")
    parser.add_argument("--final_topk", type=int, default=5, help="Number of final prototypes in training")
    parser.add_argument("--output_dir", type=str, default="epic_output", help="Output directory")
    args = parser.parse_args()

    dm = GaussianDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.1,
        sampling=args.sampling,
        num_points=args.num_samples,
    )
    dm.setup()
    
    pl_model = PointNetLightning.load_from_checkpoint(
        args.pointnet_ckpt,
        in_dim=len(FEATURE_NAMES),
        num_classes=dm.num_classes,
        grid_size=10
    )
    pointnet_model = pl_model.model
    pointnet_model.eval()
    
    # Initialize EPIC trainer
    epic_trainer = EpicTrainer(
        pointnet_model, 
        num_channels=1024, 
        lr=args.lr,
        initial_topk=args.initial_topk,
        final_topk=args.final_topk,
        max_epochs=args.epochs,
    )
    epic_trainer.hparams.batch_size = args.batch_size
    epic_trainer.hparams.num_workers = args.num_workers
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initial prototype generation
    epic_trainer.update_prototypes(dm.train_dataloader(), dm.val_dataloader(), device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="epic-{epoch:02d}-{val/epic_purity_loss:.4f}",
        monitor="val/epic_purity_loss",
        mode="min",
        save_top_k=3
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    logger = TensorBoardLogger(args.output_dir, name="epic_logs")
    
    # Custom callback to update prototypes during training
    class PrototypeUpdateCallback(pl.Callback):
        def __init__(self, update_freq, train_dataloader, val_dataloader, device):
            self.update_freq = update_freq
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            self.device = device
            
        def on_train_epoch_start(self, trainer, pl_module):
            # Update prototypes every N epochs
            if trainer.current_epoch % self.update_freq == 0:
                pl_module.update_prototypes(self.train_dataloader, self.val_dataloader, self.device)
    
    prototype_callback = PrototypeUpdateCallback(
        args.prototype_update_freq,
        dm.train_dataloader(),
        dm.val_dataloader(),
        device
    )
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, lr_monitor, prototype_callback],
        log_every_n_steps=10,
        logger=logger,
        check_val_every_n_epoch=1
    )
    
    trainer.fit(epic_trainer)
    
    final_matrix = epic_trainer.epic.get_weight()
    torch.save(final_matrix, os.path.join(args.output_dir, "final_orthogonal_matrix.pt"))
    
    pointnet_model.attach_epic()
    pointnet_model.epic.load_state_dict(epic_trainer.epic.state_dict())
    pointnet_model.apply_classifier_compensation()
    
    torch.save({
        "pointnet_state_dict": pointnet_model.state_dict(),
        "epic_matrix": final_matrix
    }, os.path.join(args.output_dir, "pointnet_epic_compensated.pt"))
    
    print(f"Training complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
