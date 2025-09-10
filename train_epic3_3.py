# train epic
import argparse
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import os
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from tqdm import tqdm
import numpy as np


import torch
import torch.nn.functional as F
from tqdm import tqdm

from pointnet.pointnet import PointNetLightning
from pointnet.dataset import GaussianDataModule, FEATURE_NAMES, collate_fn
from pointnet.epic import EpicDisentangler

def generate_prototypes_pointnet(model, dataloader, num_channels, topk=5, device="cpu", U=None):
    model.eval()
    model.to(device)
    if U is not None:
        U = U.to(device)

    top_acts = torch.full((topk, num_channels), -float("inf"), device=device)
    top_inds = torch.full((topk, num_channels), -1, dtype=torch.long, device=device)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating prototypes"):
            features = batch["gauss"].to(device)                
            xyz_normalized = batch["xyz_normalized"].to(device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device)                         
            dataset_indices = batch["indices"].to(device)       
            B = features.size(0)

            point_features, _ = model.extract_point_features(features, xyz_normalized, mask)

            if U is not None:
                point_features = torch.einsum("cd,bdn->bcn", U, point_features)
                point_features = F.relu(point_features)
            if mask is not None:
                mask_exp = mask.unsqueeze(1).expand(B, point_features.size(1), point_features.size(2))
                point_features = point_features.masked_fill(~mask_exp, float("-inf"))

            vals, argidxs = torch.max(point_features, dim=2) 

            batch_idx_ar = torch.arange(B, device=device).unsqueeze(1) 
            current_inds = dataset_indices[batch_idx_ar, argidxs] 

            invalid_mask = (vals == float("-inf"))
            if invalid_mask.any():
                current_inds = current_inds.clone()
                current_inds[invalid_mask] = -1

            combined_acts = torch.cat([top_acts, vals], dim=0)      # (topk + B, C)
            combined_inds = torch.cat([top_inds, current_inds], dim=0)  # (topk + B, C)

            vals_topk, idxs = torch.topk(combined_acts, k=topk, dim=0, largest=True, sorted=True)  # (topk, C)
            rows = idxs  # (topk, C)
            cols = torch.arange(num_channels, device=device).unsqueeze(0).expand(topk, num_channels)  # (topk, C)

            new_top_inds = combined_inds[rows, cols]  # (topk, C)

            top_acts = vals_topk
            top_inds = new_top_inds

    prototypes_dict = {c: top_inds[:, c].cpu().numpy().tolist() for c in range(num_channels)}
    return prototypes_dict



def purity_argmax_point(feature_map, channels, mask=None):
    B, C, N = feature_map.shape
    device = feature_map.device
    mask = None # change

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.max_epochs,   
            eta_min=1e-5             
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # update co epokę
            }
        }

    def train_dataloader(self):
        return self.prototypes_loader

    def val_dataloader(self):
        return self.val_prototypes_loader

    def update_prototypes(self, train_dataloader, val_dataloader, device):
        progress = min(self.current_epoch / self.max_epochs, 1.0)
        self.current_topk = int(self.initial_topk - progress * (self.initial_topk - self.final_topk))
        
        U = self.epic.get_weight().detach()
        
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
        self.last_val_prototypes = val_prototypes
        
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


    @torch.no_grad()
    def on_train_epoch_end(self):
        if self.val_prototypes_loader is None:
            return

        all_purities = []
        device = next(self.parameters()).device
        for batch in self.val_prototypes_loader:
            features = batch["gauss"].to(device)
            xyz = batch["xyz_normalized"].to(device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device)
            channels = batch["channel"].to(device)

            point_features, _ = self.pointnet.extract_point_features(features, xyz, mask)
            point_features = self.epic(point_features)

            purity = purity_argmax_point(point_features, channels, mask)
            all_purities.append(purity.cpu())

        all_purities = torch.cat(all_purities)
        mean_val_purity = all_purities.mean()
        self.log("val/purity_mean", mean_val_purity, prog_bar=True)
        print(f"Epoch {self.current_epoch}: val/purity_mean = {mean_val_purity:.4f}")



def main():

    # pointnet_ckpt = '/kaggle/input/pointnet_wcss/pytorch/default/1/model_wcss_kl_2.ckpt'
    # data_dir = '/kaggle/input/gaussy-sigma/data/data'
    import os
    pointnet_ckpt = os.environ.get("POINTNET_CKPT", "/pointnet_toys_kl_3-5.ckpt")
    data_dir = os.environ.get("DATA_DIR", "/new_dataset/new_dataset")
    batch_size = 4
    num_workers = 4
    epochs = 50
    lr = 1e-4 
    prototype_update_freq = 2
    sampling = "original_size"
    num_samples = 75000
    initial_topk = 50
    final_topk = 5
    output_dir = os.path.join(os.environ.get("HOME"), "ml-splat-xai", "experiments")
    os.makedirs(output_dir, exist_ok=True)


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
        grid_size=10
    )
    pointnet_model = pl_model.model
    pointnet_model.eval()
    
    # Initialize EPIC trainer
    epic_trainer = EpicTrainer(
        pointnet_model, 
        num_channels=1024, 
        lr=lr,
        initial_topk=initial_topk,
        final_topk=final_topk,
        max_epochs=epochs,
    )
    epic_trainer.hparams.batch_size = batch_size
    epic_trainer.hparams.num_workers = num_workers
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initial prototype generation
    epic_trainer.update_prototypes(dm.train_dataloader(), dm.val_dataloader(), device)
    
    val_prototypes_dict = generate_prototypes_pointnet(
        epic_trainer.pointnet,
        dm.val_dataloader(),
        num_channels=epic_trainer.hparams.num_channels,
        topk=5,
        device=device,
        U=epic_trainer.epic.get_weight().detach()
    )
    
    val_prototypes_dataset = PrototypesDataset(dm.val_dataloader().dataset, val_prototypes_dict)
    val_prototypes_loader = DataLoader(
        val_prototypes_dataset,
        batch_size=epic_trainer.hparams.batch_size,
        shuffle=False,
        num_workers=epic_trainer.hparams.num_workers,
        collate_fn=collate_prototypes
    )

    epic_trainer.val_prototypes_loader = val_prototypes_loader
    
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
        prototype_update_freq,
        dm.train_dataloader(),
        dm.val_dataloader(),
        device
    )
    
    # epic_viz_cb = EpicVisualizationCallback(
    #     output_dir=os.path.join(output_dir, "epic_visualizations"),
    #     num_channels=6,
    #     grid_size=10,
    #     val_dataset=val_dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     data_dir=data_dir,
    #     num_prototypes=5
    # )
    
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, lr_monitor, prototype_callback],
        log_every_n_steps=10,
        logger=logger,
        check_val_every_n_epoch=1
    )
        
    trainer.fit(epic_trainer)
    
    final_matrix = epic_trainer.epic.get_weight()
    torch.save(final_matrix, os.path.join(output_dir, "final_orthogonal_matrix.pt"))
    
    pointnet_model.attach_epic()
    pointnet_model.epic.load_state_dict(epic_trainer.epic.state_dict())
    pointnet_model.apply_classifier_compensation()
    
    torch.save({
        "pointnet_state_dict": pointnet_model.state_dict(),
        "epic_matrix": final_matrix
    }, os.path.join(output_dir, "pointnet_epic_compensated.pt"))
    
    print(f"Training complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
