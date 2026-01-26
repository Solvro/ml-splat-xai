import pytorch_lightning as pl


class PrototypeUpdateCallback(pl.Callback):
    def __init__(self, update_freq, train_loader, val_loader, batch_size, num_workers, device) -> None:
        self.update_freq = update_freq
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def on_train_epoch_end(self, trainer, pl_module):
        if not (trainer.current_epoch + 1) % self.update_freq:
            print("Updating datasets:")
            pl_module.update_prototypes(
                self.train_loader,
                self.val_loader,
                self.batch_size,
                self.num_workers,
                self.device
            )
