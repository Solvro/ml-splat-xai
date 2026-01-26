import argparse
import os

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch import set_float32_matmul_precision

set_float32_matmul_precision("medium")

from pointnet.pointnet import PointNetLightning
from pointnet.datasets.gaussian_point_cloud import GaussianDataModule, FEATURE_NAMES
from pointnet.callbacks.prototype_visualization import PrototypeVisualizationCallback


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Train PointNet Backbone")

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides defaults)")

    parser.add_argument("--data_dir", type=str, default="data", help="Root directory with data")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--sampling", type=str, default="random",
                        choices=["fps", "random", "original_size"], help="Point sampling method")
    parser.add_argument("--num_samples", type=int, default=8192, help="Number of points if sampling is used")
    parser.add_argument("--grid_size", type=int, default=7, help="Size of the voxel grid for aggregation")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--stn_3d", action="store_true", help="Use 3D STN layer")
    parser.add_argument("--stn_nd", action="store_true", help="Use feature STN layer")
    parser.add_argument("--head_size", type=int, default=256, help="Size of classification head")
    parser.add_argument("--stn_head_norm", action="store_true", help="Use STN head normalization")
    parser.add_argument("--pooling", type=str, default="max", choices=["max", "avg"], help="Pooling type")

    parser.add_argument("--count_penalty_type", type=str, default="kl_to_counts",
                        choices=["softmax", "ratio", "kl_to_counts"], help="Count penalty type")
    parser.add_argument("--count_penalty_weight", type=float, default=3.5)
    parser.add_argument("--count_penalty_beta", type=float, default=1.0)
    parser.add_argument("--count_penalty_tau", type=float, default=1.0)

    parser.add_argument("--model_save_path", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--model_save_name", type=str, default="model", help="Checkpoint filename")

    parser.add_argument("--fast_dev_run", action="store_true")

    args = parser.parse_args()

    if args.config is not None:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                if isinstance(value, bool):
                    setattr(args, key, value)
                elif getattr(args, key) == parser.get_default(key):
                    setattr(args, key, value)

    return args


def main():
    pl.seed_everything(777)

    args = parse_args()

    datamodule = GaussianDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        val_split=args.val_split,
        sampling=args.sampling,
        num_points=args.num_samples,
        grid_size=args.grid_size,
    )
    datamodule.setup()

    model = PointNetLightning(
        in_dim=len(FEATURE_NAMES),
        num_classes=datamodule.num_classes,
        grid_size=args.grid_size,
        stn_3d=args.stn_3d,
        stn_nd=args.stn_nd,
        lr=args.lr,
        head_size=args.head_size,
        stn_head_norm=args.stn_head_norm,
        count_penalty_weight=args.count_penalty_weight,
        count_penalty_type=args.count_penalty_type,
        count_penalty_beta=args.count_penalty_beta,
        count_penalty_tau=args.count_penalty_tau,
        pooling=args.pooling,
    )

    model_save_dir = args.model_save_path
    os.makedirs(model_save_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        filename=args.model_save_name,
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    viz_cb = PrototypeVisualizationCallback(
        grid_size=args.grid_size,
        top_k=3,
        log_every_n_epochs=1,
        figure_dpi=160
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger(model_save_dir, name=args.model_save_name)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.device,
        callbacks=[checkpoint_callback, lr_monitor, viz_cb],
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
        logger=logger,
        # overfit_batches=1,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)

if __name__ == "__main__":
    main()
