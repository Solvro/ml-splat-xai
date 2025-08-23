import os
from typing import Sequence

import torch
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl


class PrototypeVisualizationCallback(pl.Callback):
    def __init__(self, grid_size: int, top_k: int = 3, log_every_n_epochs: int = 1, figure_dpi: int = 160):
        super().__init__()
        self.grid_size = grid_size
        self.top_k = top_k
        self.log_every_n_epochs = log_every_n_epochs
        self.figure_dpi = figure_dpi

    @staticmethod
    def _flat_to_xyz(flat_idx: int, G: int):
        x = flat_idx // (G * G)
        y = (flat_idx // G) % G
        z = flat_idx % G
        return int(x), int(y), int(z)

    @staticmethod
    def _add_colorbar(im, ax):
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)

    def _plot_volume_slices(self, vol_3d: torch.Tensor, title: str):
        G = vol_3d.shape[0]
        mid = G // 2
        xy = vol_3d[:, :, mid].cpu().numpy()
        xz = vol_3d[:, mid, :].cpu().numpy()
        yz = vol_3d[mid, :, :].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=self.figure_dpi)
        ims = []
        ims.append(axes[0].imshow(xy, cmap="Purples", origin="lower", aspect="equal"))
        axes[0].set_title("Slice XY (z=mid)")
        ims.append(axes[1].imshow(xz, cmap="Oranges", origin="lower", aspect="equal"))
        axes[1].set_title("Slice XZ (y=mid)")
        ims.append(axes[2].imshow(yz, cmap="Blues", origin="lower", aspect="equal"))
        axes[2].set_title("Slice YZ (x=mid)")

        for ax, im in zip(axes, ims):
            ax.set_xlabel("u")
            ax.set_ylabel("v")
            self._add_colorbar(im, ax)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    def _plot_xy_projection(self, xyz_norm: torch.Tensor, top_voxels: Sequence[int], G: int, title: str):
        x = xyz_norm[:, 0].cpu().numpy()
        y = xyz_norm[:, 1].cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=self.figure_dpi)
        ax.scatter(x, y, s=2, c="lightgray", alpha=0.65, label="points")

        for i in range(1, G):
            ax.axvline(i / G, color="k", lw=0.2, alpha=0.2)
            ax.axhline(i / G, color="k", lw=0.2, alpha=0.2)

        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(top_voxels))))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("X (normalized)")
        ax.set_ylabel("Y (normalized)")
        return fig, ax, colors

    def _plot_topk_bars(self, top_vals: torch.Tensor, top_counts: torch.Tensor, title: str):
        vals = top_vals.detach().cpu().numpy()
        cnts = top_counts.detach().cpu().numpy()
        k = len(vals)
        fig, ax1 = plt.subplots(1, 1, figsize=(max(4, 0.6 * k + 2), 3.5), dpi=self.figure_dpi)
        x = np.arange(k)
        ax1.bar(x, vals, color="tomato", alpha=0.85, label="activation")
        ax1.set_ylabel("Activation")
        ax1.set_xlabel("Top-K voxels (rank)")
        ax1.set_title(title)
        ax1.grid(axis="y", alpha=0.3)
        for i, c in enumerate(cnts):
            ax1.text(x[i], vals[i], f"n={int(c)}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        return fig

    def _plot_counts_hist(self, counts_nonzero: torch.Tensor, title: str):
        c = counts_nonzero.detach().cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), dpi=self.figure_dpi)
        ax.hist(c, bins=min(50, int(c.max()) + 1), color="slateblue", alpha=0.85)
        ax.set_xlabel("Points per non-empty voxel")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_hexbin_counts_vs_act(self, counts: torch.Tensor, acts_pos: torch.Tensor, title: str):
        c = counts.detach().cpu().numpy()
        a = acts_pos.detach().cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=self.figure_dpi)
        hb = ax.hexbin(c, a, gridsize=30, cmap="PuBu", bins="log")
        ax.set_xlabel("Points per voxel")
        ax.set_ylabel("Activation (ReLU)")
        ax.set_title(title)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("log10(N)")
        fig.tight_layout()
        return fig
    
    def _plot_logits(self, logits: torch.Tensor, title: str):
        logits_np = logits.detach().cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=self.figure_dpi)
        ax.bar(range(len(logits_np)), logits_np, color="skyblue", alpha=0.85)
        ax.set_xlabel("Class index")
        ax.set_ylabel("Logit value")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx, dataloader_idx: int = 0):
        if batch_idx != 0 or (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        logger = trainer.logger
        if logger is None:
            return
        writer = logger.experiment

        model = pl_module.model
        if not hasattr(model, "last_voxel_activations") or model.last_voxel_activations is None:
            return

        G = self.grid_size
        vol = model.last_voxel_activations[0].squeeze(0).detach().cpu()  # (G, G, G) -> take the 1st batch item
        indices_b0 = model.last_indices[0].detach().cpu()                # (N,)
        xyz_b0 = model.last_xyz_for_vox[0].detach().cpu()                # (N, 3)

        flat = vol.flatten()
        K = min(self.top_k, flat.numel())
        top_vals, top_idx = torch.topk(flat, K, largest=True)
        top_idx_list = [int(i) for i in top_idx.detach().cpu().tolist()]

        # liczności
        counts = torch.bincount(indices_b0, minlength=G**3).float()  # (G^3,) -> how many points in each voxel
        nonempty = counts[counts > 0]
        occupancy_ratio = (counts > 0).float().mean().item() # fraction of the whole grid that is occupied
        mean_pts_per_nonempty = float(nonempty.mean().item()) if nonempty.numel() > 0 else 0.0
        pts_in_topk = int(counts[top_idx].sum().item())
        frac_pts_in_topk = pts_in_topk / max(1, int(counts.sum().item()))
        topk_mask = torch.zeros_like(indices_b0, dtype=torch.bool)

        # penalty-specific: korelacja, udział masy aktywacji w małych kubełkach
        acts_pos = torch.relu(flat)
        # Pearson r (na niepustych, żeby uniknąć trzymania zera/zera)
        mask_nonzero = (counts > 0)
        if mask_nonzero.any():
            c_np = counts[mask_nonzero].numpy()
            a_np = acts_pos[mask_nonzero].numpy()
            if c_np.std() > 0 and a_np.std() > 0:
                corr = float(np.corrcoef(c_np, a_np)[0, 1])
            else:
                corr = 0.0
        else:
            corr = 0.0

        mass = acts_pos.sum().item() if acts_pos.numel() > 0 else 1.0
        def mass_le(thr: int):
            m = acts_pos[(counts <= thr)].sum().item()
            return m / max(mass, 1e-8)
        mass_le_1 = mass_le(1)
        mass_le_2 = mass_le(2)
        mass_le_3 = mass_le(3)

        writer.add_scalar("viz/occupancy_ratio_b0", float(occupancy_ratio), trainer.current_epoch)
        writer.add_scalar("viz/mean_points_per_nonempty_voxel_b0", float(mean_pts_per_nonempty), trainer.current_epoch)
        writer.add_scalar("viz/fraction_points_in_topk_b0", float(frac_pts_in_topk), trainer.current_epoch)
        writer.add_scalar("penalty/corr_counts_activation_b0", corr, trainer.current_epoch)
        writer.add_scalar("penalty/act_mass_le1_b0", mass_le_1, trainer.current_epoch)
        writer.add_scalar("penalty/act_mass_le2_b0", mass_le_2, trainer.current_epoch)
        writer.add_scalar("penalty/act_mass_le3_b0", mass_le_3, trainer.current_epoch)

        fig_slices = self._plot_volume_slices(vol, title=f"Voxel activations – epoch {trainer.current_epoch}") # activations before passing thru classification head
        writer.add_figure("viz/volume_slices_b0", fig_slices, trainer.current_epoch)
        plt.close(fig_slices)

        fig_xy, ax_xy, colors = self._plot_xy_projection(xyz_b0, top_idx_list, G, title=f"Points (post-STN normalized) – epoch {trainer.current_epoch}")
        ax_xy.scatter(xyz_b0[:, 0], xyz_b0[:, 1], s=2, c="lightgray", alpha=0.65, label="all points")
        for r, flat_idx in enumerate(top_idx_list):
            mask = (indices_b0 == flat_idx).numpy()
            topk_mask = topk_mask | mask
            pts = xyz_b0[mask]
            if pts.numel() == 0:
                continue
            ax_xy.scatter(pts[:, 0], pts[:, 1], s=8, color=colors[r], alpha=0.95, label=f"voxel#{r+1} (idx={flat_idx}, n={int(counts[flat_idx].item())})")
            x_idx, y_idx, z_idx = self._flat_to_xyz(flat_idx, G)
            x0, x1 = x_idx / G, (x_idx + 1) / G
            y0, y1 = y_idx / G, (y_idx + 1) / G
            ax_xy.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=colors[r], lw=2, alpha=0.9)
        ax_xy.legend(loc="upper right", fontsize=8, framealpha=0.8)
        fig_xy.tight_layout()
        writer.add_figure("viz/xy_projection_b0", fig_xy, trainer.current_epoch)
        plt.close(fig_xy)

        # Top-K: aktywacja + adnotacja o liczności
        top_counts = counts[top_idx]
        fig_bars = self._plot_topk_bars(top_vals, top_counts, title=f"Top-{K} voxel activations – epoch {trainer.current_epoch}")
        writer.add_figure("penalty/topk_activation_with_counts_b0", fig_bars, trainer.current_epoch)
        plt.close(fig_bars)

        # Histogram liczności niepustych
        if nonempty.numel() > 0:
            fig_hist = self._plot_counts_hist(nonempty, title=f"Counts per non-empty voxel – epoch {trainer.current_epoch}")
            writer.add_figure("penalty/counts_hist_b0", fig_hist, trainer.current_epoch)
            plt.close(fig_hist)

        # Hexbin: counts vs activation
        if acts_pos.sum().item() > 0:
            fig_hex = self._plot_hexbin_counts_vs_act(counts, acts_pos, title=f"Counts vs Activation – epoch {trainer.current_epoch}")
            writer.add_figure("penalty/counts_vs_activation_hexbin_b0", fig_hex, trainer.current_epoch)
            plt.close(fig_hex)

        # Logits
        logits = outputs
        fig_logits = self._plot_logits(logits[0], title=f"Logits – epoch {trainer.current_epoch}")
        writer.add_figure("viz/logits_b0", fig_logits, trainer.current_epoch)
        plt.close(fig_logits)
