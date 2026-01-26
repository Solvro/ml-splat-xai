from einops import rearrange, repeat
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from pointnet.orthogonal import OrthogonalDisentangler


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class STN(nn.Module):
    def __init__(self, in_dim=3, out_nd=None, head_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_nd = default(out_nd, in_dim)
        self.last_transform = None

        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, 1, bias=False),
        )

        norm = nn.BatchNorm1d if head_norm else nn.Identity
        self.norm = norm(256)
        self.act = nn.GELU()

        self.head = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            norm(128),
            nn.GELU(),
            nn.Linear(128, 64, bias=False),
            norm(64),
            nn.GELU(),
            nn.Linear(64, self.out_nd**2),
        )

        self.head[-1].weight.data.zero_()
        self.head[-1].bias.data.copy_(
            torch.eye(self.out_nd, dtype=torch.float).flatten()
        )

    def forward(self, x):
        # x: (b, d, n)
        x = self.net(x)
        x = torch.max(x, dim=-1, keepdim=False)[0]
        x = self.act(self.norm(x))
        x = self.head(x)
        x = rearrange(x, "b (x y) -> b x y", x=self.out_nd, y=self.out_nd)
        self.last_transform = x.detach()
        return x


class VoxelAggregation(nn.Module):
    def __init__(self, grid_size: int = 10, pooling: str = "max"):
        super().__init__()
        self.grid_size = grid_size
        self.num_voxels = grid_size**3
        self.pooling = pooling

    def forward(
        self,
        features: torch.Tensor,
        voxel_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        # features: (B, D, N)
        B, D, _ = features.shape
        V = self.num_voxels

        if mask is not None:
            features = features * mask.unsqueeze(1).float()
        point_counts_raw = torch.zeros(
            B, V, device=features.device, dtype=torch.long
        )
        ones = torch.ones_like(voxel_ids, dtype=torch.long)
        if mask is not None:
            ones = ones * mask.long()
        point_counts_raw.scatter_add_(1, voxel_ids, ones)  # (B, V)

        voxel_ids_expanded = repeat(voxel_ids, "b n -> b d n", d=D)

        if self.pooling == "avg":
            voxel_features_sum = torch.zeros(
                B, D, V, device=features.device, dtype=features.dtype
            )
            voxel_features_sum.scatter_add_(
                2, voxel_ids_expanded, features
            )  # (B, D, V)

            point_counts = point_counts_raw.unsqueeze(1)  # (B,1,V)
            voxel_features = voxel_features_sum / (point_counts.clamp(min=1))  # (B, D, V)
        else:  # "max"
            voxel_features = torch.zeros(
                B, D, V, device=features.device, dtype=features.dtype
            )
            voxel_features.scatter_reduce_(
                2,
                voxel_ids_expanded,
                features,
                reduce="amax",
                include_self=False,
            )
            point_counts = point_counts_raw.unsqueeze(1)  # (B,1,V)

        G = self.grid_size
        voxel_features_3d = rearrange(
            voxel_features, "b d (x y z) -> b d x y z", x=G, y=G, z=G
        )
        return voxel_features_3d, voxel_ids, point_counts


class PointNetCls(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        grid_size: int = 10,
        stn_3d=True,
        stn_nd: bool = True,
        stn_head_norm=False,
        pooling: str = "max",
        head_size: int = 256,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size

        self.stn_3d = (
            STN(in_dim=3, head_norm=stn_head_norm) if stn_3d else nn.Identity()
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.stn_nd = (
            STN(in_dim=64, head_norm=stn_head_norm) if stn_nd else nn.Identity()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, head_size, 1, bias=False),
            nn.BatchNorm1d(head_size),
        )

        self.voxel_agg = VoxelAggregation(grid_size, pooling=pooling)
        self.voxel_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Linear(head_size, out_dim, bias=False)

        self.disentangler: OrthogonalDisentangler | None = None

        self.last_stn_T = None
        self.last_voxel_min = None
        self.last_voxel_max = None
        self.last_xyz_for_vox = None
        self.last_voxel_activations = None
        self.last_point_counts = None
        self.last_indices = None

    def attach_disentangler(self, C: int = 1024) -> None:
        if self.disentangler is None:
            self.disentangler = OrthogonalDisentangler(C)

    @torch.no_grad()
    def apply_classifier_compensation(self) -> None:
        if self.disentangler is None:
            return
        invU = (
            self.disentangler.inverse()
            .to(self.head.weight.device)
            .to(self.head.weight.dtype)
        )
        W = self.head.weight.data
        with torch.no_grad():
            newW = W @ invU
            self.head.weight.copy_(newW)

    def extract_point_features(
        self,
        features: torch.Tensor,
        xyz_normalized: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        xyz_transposed = rearrange(xyz_normalized, "b n d -> b d n")

        transform_3d = self.stn_3d(xyz_transposed)
        if isinstance(self.stn_3d, STN):
            xyz_transposed = torch.bmm(transform_3d, xyz_transposed)
            self.last_stn_T = transform_3d.detach()
        else:
            self.last_stn_T = None

        self.last_rescale_min = None
        self.last_rescale_max = None

        features_cat = torch.cat([xyz_transposed, features[:, 3:, :]], dim=1)
        x = self.conv1(features_cat)
        transform_nd = self.stn_nd(x)
        if isinstance(self.stn_nd, STN):
            x = torch.bmm(transform_nd, x)
        point_features = self.conv2(x)
        xyz_for_vox = rearrange(xyz_transposed, "b d n -> b n d")
        return point_features, xyz_for_vox

    def extract_voxel_features(
        self,
        features: torch.Tensor,
        xyz_normalized: torch.Tensor,
        voxel_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        point_features, xyz_for_vox = self.extract_point_features(
            features, xyz_normalized, mask
        )

        voxel_features, _, _ = self.voxel_agg(
            point_features, voxel_ids, mask
        )

        self.last_voxel_min = None
        self.last_voxel_max = None

        if self.disentangler is not None:
            _, _, X, Y, Z = voxel_features.shape
            flat_vox = rearrange(voxel_features, "b c x y z -> b c (x y z)")
            rotated_vox = self.disentangler(flat_vox)
            voxel_features = rearrange(
                rotated_vox, "b c (x y z) -> b c x y z", x=X, y=Y, z=Z
            )

        return voxel_features, xyz_for_vox

    def forward(
        self,
        features: torch.Tensor,
        xyz_normalized: torch.Tensor,
        voxel_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        point_features, xyz_for_vox = self.extract_point_features(
            features, xyz_normalized, mask
        )

        voxel_features, indices, point_counts = self.voxel_agg(
            point_features, voxel_ids, mask
        )

        self.last_voxel_min = None
        self.last_voxel_max = None

        if self.disentangler is not None:
            _, _, X, Y, Z = voxel_features.shape
            flat_vox = rearrange(voxel_features, "b c x y z -> b c (x y z)")
            rotated_vox = self.disentangler(flat_vox)
            voxel_features = rearrange(
                rotated_vox, "b c (x y z) -> b c x y z", x=X, y=Y, z=Z
            )

        voxel_features_flat = rearrange(voxel_features, "b d x y z -> b d (x y z)")
        global_features = self.voxel_avg_pool(voxel_features_flat).squeeze(-1)

        voxel_activations_3d = voxel_features.norm(dim=1)
        voxel_activations_3d = voxel_activations_3d.reshape(
            -1, self.grid_size, self.grid_size, self.grid_size
        )

        logits = self.head(global_features)

        self.last_xyz_for_vox = xyz_for_vox.detach()
        self.last_voxel_activations = voxel_activations_3d.detach()
        self.last_point_counts = point_counts.detach()
        self.last_indices = indices.detach()

        return logits, global_features, voxel_activations_3d, point_counts, indices


class PointNetLightning(pl.LightningModule):
    def __init__(
        self,
        in_dim,
        num_classes,
        grid_size=10,
        stn_3d=True,
        stn_nd=True,
        lr=1e-3,
        weight_decay=1e-4,
        stn_head_norm=False,
        head_size=256,
        count_penalty_weight: float | None = None,
        count_penalty_type: str = "softmax",
        count_penalty_beta: float = 1.0,
        count_penalty_tau: float = 1.0,
        pooling: str = "max",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = PointNetCls(
            in_dim,
            num_classes,
            grid_size,
            stn_3d,
            stn_nd,
            stn_head_norm=stn_head_norm,
            pooling=pooling,
        )
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self._penalty_weight = float(count_penalty_weight or 0.0)

    def forward(self, features, xyz_normalized, voxel_ids, mask=None):
        return self.model(features, xyz_normalized, voxel_ids, mask)

    def _count_penalty(
        self, voxel_activations: torch.Tensor, point_counts: torch.Tensor
    ) -> torch.Tensor:
        if self._penalty_weight is None or self._penalty_weight <= 0:
            return voxel_activations.new_zeros(())
        eps = 1e-8
        B = voxel_activations.shape[0]
        G3 = (
            voxel_activations.shape[1]
            * voxel_activations.shape[2]
            * voxel_activations.shape[3]
        )
        acts = voxel_activations.reshape(B, G3)
        acts_pos = F.relu(acts)
        counts = point_counts.reshape(B, G3).float()
        beta = float(self.hparams.count_penalty_beta)
        tau = float(self.hparams.count_penalty_tau)
        if self.hparams.count_penalty_type == "ratio":
            denom = counts.pow(beta) + eps
            pen = (acts_pos / denom).mean()
        elif self.hparams.count_penalty_type == "kl_to_counts":
            p = F.softmax(acts_pos / max(tau, 1e-4), dim=1)
            q_unnorm = counts.pow(beta) + eps
            q = q_unnorm / q_unnorm.sum(dim=1, keepdim=True)
            pen = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=1).mean()
        else:
            p = F.softmax(acts_pos / max(tau, 1e-4), dim=1)
            inv_counts = 1.0 / (counts.pow(beta) + eps)
            pen = (p * inv_counts).sum(dim=1).mean()
        return pen

    def _calculate_and_log_metrics(self, features, prefix):
        features_cpu = features.detach().float().cpu()

        features_cpu = features_cpu - features_cpu.mean(dim=0, keepdim=True)

        cov_matrix = torch.cov(features_cpu.T)
        eigenvalues = torch.linalg.eigvalsh(cov_matrix)

        sorted_eigenvalues, _ = torch.sort(eigenvalues, descending=True)

        self.log(f"{prefix}/eigenvalue_max", sorted_eigenvalues[0])
        self.log(f"{prefix}/eigenvalue_mean", eigenvalues.mean())
        self.log(f"{prefix}/eigenvalue_min", sorted_eigenvalues[-1])

        try:
            singular_values = torch.linalg.svdvals(features_cpu)
            singular_values_normalized = singular_values / singular_values.sum()
            entropy = -torch.sum(
                singular_values_normalized
                * torch.log(singular_values_normalized + 1e-8)
            )
            rankme_score = torch.exp(entropy)
            self.log(f"{prefix}/rankme", rankme_score)
        except torch.linalg.LinAlgError:
            self.log(f"{prefix}/rankme", 0.0)

    def training_step(self, batch, batch_idx):
        features, xyz_normalized, labels, mask, voxel_ids = (
            batch["gauss"],
            batch["xyz_normalized"],
            batch["label"],
            batch["mask"],
            batch["voxel_ids"],
        )
        logits, global_features, voxel_activations, point_counts, _ = self.model(
            features, xyz_normalized, voxel_ids, mask
        )
        cls_loss = F.cross_entropy(logits, labels)
        pen = self._count_penalty(voxel_activations, point_counts)
        total = cls_loss + self._penalty_weight * pen
        self.train_acc(logits, labels)

        self.log(
            "train/classification_loss",
            cls_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/count_penalty", pen, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log("train/total_loss", total, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        )

        self._calculate_and_log_metrics(global_features, "train")
        return total

    def validation_step(self, batch, batch_idx):
        features, xyz_normalized, labels, mask, voxel_ids = (
            batch["gauss"],
            batch["xyz_normalized"],
            batch["label"],
            batch["mask"],
            batch["voxel_ids"],
        )
        logits, global_features, voxel_activations, point_counts, _ = self.model(
            features, xyz_normalized, voxel_ids, mask
        )
        cls_loss = F.cross_entropy(logits, labels)
        pen = self._count_penalty(voxel_activations, point_counts)
        total = cls_loss + self._penalty_weight * pen
        self.val_acc(logits, labels)
        self.log("val/classification_loss", cls_loss, on_epoch=True, prog_bar=False)
        self.log("val/count_penalty", pen, on_epoch=True, prog_bar=False)
        self.log("val_loss", total, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self._calculate_and_log_metrics(global_features, "val")
        return logits

    def test_step(self, batch, batch_idx):
        features, xyz_normalized, labels, mask, voxel_ids = (
            batch["gauss"],
            batch["xyz_normalized"],
            batch["label"],
            batch["mask"],
            batch["voxel_ids"],
        )
        logits, global_features, _, _, _ = self.model(
            features, xyz_normalized, voxel_ids, mask
        )
        cls_loss = F.cross_entropy(logits, labels)
        self.test_acc(logits, labels)
        self.log("test/classification_loss", cls_loss, on_epoch=True, prog_bar=False)
        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)

        self._calculate_and_log_metrics(global_features, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
        return [optimizer], [scheduler]
