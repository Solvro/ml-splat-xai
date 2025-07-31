import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from data.pointnet_data import create_prototype_dataloader
from prototypes.purity_3d import get_prototypes_purity
from prototypes.selection_3d import generate_3d_prototypes


class PointNetTrainer:
    def __init__(
        self,
        model,
        dataloader,
        disentanglement_matrix,
        num_channels=512,
        num_epochs_per_prototypes=10,
        batch_size=512,
        lr=0.001,
        device="cpu",
        purity_fn=None,
        num_prototypes_scheduler=None
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.purity_fn = purity_fn
        self.num_epochs_per_prototypes = num_epochs_per_prototypes
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.disentanglement_matrix = disentanglement_matrix
        self.optimizer = torch.optim.Adam(
            self.disentanglement_matrix.parameters(), lr=lr, weight_decay=1e-5
        )
        self.avg_purities = []
        self.num_prototypes_scheduler = num_prototypes_scheduler

    def loss_fn(self, P, k):
        if self.disentanglement_matrix is not None:
            U = self.disentanglement_matrix()
            P = torch.einsum("ij,bjn->bin", U, P)
        
        normalized_purity = self.purity_fn(P, k)
        return -normalized_purity.mean()

    def train_prototypes(self, prototype_dataloader):
        for epoch in range(self.num_epochs_per_prototypes):
            avg_purity = 0.0
            for batch in tqdm(prototype_dataloader, desc=f"Epoch {epoch+1}"):
                xyz, gauss, point_indices, channels, _ = batch
                xyz = xyz.to(self.device)
                gauss = gauss.to(self.device)
                channels = channels.to(self.device)
                point_indices = point_indices.to(self.device)

                inputs = torch.cat([xyz, gauss], dim=-1).transpose(1, 2)
                
                with torch.no_grad():
                    feature_map = self.model(inputs, xyz.transpose(1, 2))  # (B, C, N)
                
                loss = self.loss_fn(feature_map, channels)
                avg_purity += -loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            avg_purity = avg_purity / len(prototype_dataloader)
            self.avg_purities.append(avg_purity)
            print(f"Epoch [{epoch+1}/{self.num_epochs_per_prototypes}], Avg Purity: {avg_purity:.4f}")

    def train(self):
        for n in self.num_prototypes_scheduler:
            print(f"Starting training for n={n} prototypes")
            positive_prototypes = generate_3d_prototypes(
                self.model,
                self.dataloader,
                self.num_channels,
                n,
                self.device,
                self.disentanglement_matrix() if self.disentanglement_matrix else None
            )
            
            prototype_dataloader = create_prototype_dataloader(
                positive_prototypes,
                self.dataloader,
                batch_size=self.batch_size,
                shuffle=True,
            )
            
            purity = get_prototypes_purity(
                self.model,
                prototype_dataloader,
                device=self.device,
                purity_fn=self.purity_fn,
                U=self.disentanglement_matrix() if self.disentanglement_matrix else None
            )
            self.avg_purities.append(purity)
            
            print(f"Training with {n} prototypes, initial purity: {purity:.4f}")
            self.train_prototypes(prototype_dataloader)

    def plot_purity_over_epochs(self, save_file):
        epochs = np.arange(1, len(self.avg_purities) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.avg_purities, "o-", label="Purity during training")
        plt.xlabel("Epoch")
        plt.ylabel("Average Purity")
        plt.title("Change in Average Purity Over Training")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_file, bbox_inches="tight")
        plt.close()
        print(f"Purity plot saved to {save_file}")
