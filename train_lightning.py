import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch import set_float32_matmul_precision

set_float32_matmul_precision('medium')

from pointnet.pointnet import PointNetLightning
from pointnet.dataset import GaussianDataModule, FEATURE_NAMES

def main():
    pl.seed_everything(777)

    parser = argparse.ArgumentParser(description="Train PointNet")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory with data")
    parser.add_argument("--device", type=str, default="auto") # "auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator")
    parser.add_argument("--sampling", type=str, default="random", choices=["fps", "random", "original_size"], help="Point sampling method")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points if sampling is used")
    parser.add_argument("--grid_size", type=int, default=10, help="Size of the voxel grid for aggregation")
    parser.add_argument("--model_save_path", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--model_save_name", type=str, default="model", help="Save model checkpoints")
    parser.add_argument("--stn_3d", action="store_true", help="Use 3D STN layer")
    parser.add_argument("--stn_nd", action="store_true", help="Use feature STN layer")    
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()

    datamodule = GaussianDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        val_split=args.val_split,
        sampling=args.sampling,
        num_points=args.num_points,
    )
    datamodule.setup()

    model = PointNetLightning(
        in_dim=len(FEATURE_NAMES),
        num_classes=datamodule.num_classes,
        grid_size=args.grid_size,
        stn_3d=args.stn_3d,
        stn_nd=args.stn_nd,
        lr=args.lr,
    )

    model_save_dir = f"./experiments/{args.model_save_path}"
    os.makedirs(model_save_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        filename=args.model_save_name,
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger(model_save_dir, name=args.model_save_name)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.device,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
        logger=logger
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
