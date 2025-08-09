import torch
import torch.nn as nn
from einops import rearrange, repeat


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class STN(nn.Module):
    # perform spatial transformation in n-dimensional space

    def __init__(self, in_dim=3, out_nd=None, head_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_nd = default(out_nd, in_dim)

        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 1024, 1, bias=False),
        )

        norm = nn.BatchNorm1d if head_norm else nn.Identity
        self.norm = norm(1024)
        self.act = nn.GELU()

        self.head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            norm(512),
            nn.GELU(),
            nn.Linear(512, 256, bias=False),
            norm(256),
            nn.GELU(),
            nn.Linear(256, self.out_nd ** 2),
        )

        self.head[-1].weight.data.zero_()
        self.head[-1].bias.data.copy_(torch.eye(self.out_nd, dtype=torch.float).flatten())

    def forward(self, x):
        # x: (b, d, n)
        x = self.net(x)
        x = torch.max(x, dim=-1, keepdim=False)[0]
        x = self.act(self.norm(x))

        x = self.head(x)
        x = rearrange(x, "b (x y) -> b x y", x=self.out_nd, y=self.out_nd)
        return x


class VoxelAggregation(nn.Module):
    def __init__(self, grid_size: int = 10):
        super().__init__()
        self.grid_size = grid_size
        self.num_voxels = grid_size ** 3

    def forward(self, features: torch.Tensor, xyz_normalized: torch.Tensor, mask: torch.Tensor | None = None):
        B, D, N = features.shape
        G = self.grid_size

        voxel_indices = (xyz_normalized * G).long().clamp(0, G - 1)
        voxel_indices_flat = voxel_indices[..., 2] * (G * G) + voxel_indices[..., 1] * G + voxel_indices[..., 0]

        if mask is not None:
            features = features * mask.unsqueeze(1).float()

        voxel_features = torch.zeros(B, D, self.num_voxels, device=features.device, dtype=features.dtype)
        voxel_indices_flat_expanded = repeat(voxel_indices_flat, "b n -> b d n", d=D)
        voxel_features.scatter_add_(2, voxel_indices_flat_expanded, features)

        point_counts = torch.zeros(B, self.num_voxels, device=features.device, dtype=torch.long)
        ones = torch.ones_like(voxel_indices_flat, dtype=torch.long)
        if mask is not None:
            ones = ones * mask.long()
        point_counts.scatter_add_(1, voxel_indices_flat, ones)

        point_counts = point_counts.clamp(min=1).unsqueeze(1)
        voxel_features_avg = voxel_features / point_counts

        return voxel_features_avg


class PointNetCls(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            grid_size: int = 10,
            stn_3d=True,
            stn_nd: bool = True,
            head_norm=True,
            dropout=0.3,
    ):
        super().__init__()
        self.grid_size = grid_size

        # if using stn, put other features behind xyz
        self.stn_3d = STN(in_dim=3) if stn_3d else nn.Identity()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.stn_nd = STN(in_dim=64, head_norm=head_norm) if stn_nd else nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
        )

        self.voxel_agg = VoxelAggregation(grid_size)
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=1, padding='same', bias=False)

        norm = nn.BatchNorm1d if head_norm else nn.Identity
        
        head_size = self.grid_size ** 3
        self.head = nn.Sequential(
            norm(head_size),
            nn.GELU(),
            nn.Linear(head_size, 512, bias=False),
            norm(512),
            nn.GELU(),
            nn.Linear(512, 256, bias=False),
            norm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, features: torch.Tensor, xyz_normalized: torch.Tensor, mask: torch.Tensor | None = None):
        # features (B, D, N)
        # xyz_normalized (B, N, 3)
        # mask (B, N).

        xyz_features = features[:, :3, :]

        transform_3d = self.stn_3d(xyz_features)
        if isinstance(self.stn_3d, STN):
            transformed_features = torch.bmm(transform_3d, xyz_features)
            features = torch.cat([transformed_features, features[:, 3:, :]], dim=1)

        x = self.conv1(features)

        transform_nd = self.stn_nd(x)
        if isinstance(self.stn_nd, STN):
            x = torch.bmm(transform_nd, x)

        point_features = self.conv2(x) # Shape: (B, 1024, N)

        voxel_features = self.voxel_agg(point_features, xyz_normalized, mask) # (B, 1024, G^3)
        
        global_feature = self.conv3(voxel_features)  # (B, 1, G^3)
        global_feature = global_feature.squeeze(1) # (B, G^3)

        logits = self.head(global_feature) # (B, out_dim)

        if self.training:
            return logits
        else:
            voxel_activations, _ = torch.max(voxel_features, dim=1) # (B, G^3)
            return logits, voxel_activations
