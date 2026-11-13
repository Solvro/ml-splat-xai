"""Importance-based splat / voxel removal for XSPLAIN (PointNet + voxels + orthogonal disentangler).

Channel importance is the same as in explain.py:70-72:
    w = head.weight[pred] * relu(global_features);   K = topk(w, k)
Point / voxel scores project the top-K weighted channels back onto the disentangled
per-point features (U @ P) or per-voxel features (U @ voxel_agg(P)).

Removal is exact: points are sliced out of the tensors (batch size 1, mask=None);
features are never zeroed, and survivors keep their original xyz_normalized / voxel_ids.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pointnet.datasets.gaussian_point_cloud import (
    FEATURE_NAMES,
    GaussianPointCloud,
    prepare_gaussian_cloud,
)
from pointnet.pointnet import PointNetLightning


# --------------------------------------------------------------------------- model
def read_stage1_args(ckpt_path: Path) -> dict:
    """Architecture args stored next to the revision checkpoints (metrics_stage1.json)."""
    p = Path(ckpt_path).parent / "metrics_stage1.json"
    if not p.exists():
        return {}
    with open(p) as f:
        m = json.load(f)
    a = dict(m.get("args", {}))
    a["num_classes"] = m.get("num_classes")
    a["in_dim"] = m.get("in_dim")
    return a


def load_model(ckpt_path, num_classes, grid_size=7, stn_3d=True, stn_nd=True,
               pooling="max", head_size=256, in_dim=len(FEATURE_NAMES),
               num_channels=256, device="cpu"):
    pl_module = PointNetLightning(
        in_dim=in_dim, num_classes=num_classes, grid_size=grid_size,
        stn_3d=stn_3d, stn_nd=stn_nd, head_size=head_size, pooling=pooling,
    )
    model = pl_module.model
    model.attach_disentangler(num_channels)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["pointnet_state_dict"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# --------------------------------------------------------------------------- data
@dataclass
class Sample:
    idx: int
    path: str
    label: int
    feat: torch.Tensor   # (1, D, N)
    xyz: torch.Tensor    # (1, N, 3)
    vids: torch.Tensor   # (1, N)

    @property
    def n(self) -> int:
        return self.feat.shape[-1]

    def subset(self, keep: torch.Tensor) -> "Sample":
        return Sample(self.idx, self.path, self.label,
                      self.feat[:, :, keep], self.xyz[:, keep], self.vids[:, keep])


class ShapeSplatTest:
    """Test split labelled with the *train* class indices.

    SHAPESPLAT_ply/test has only 43 of the 55 train class dirs, so GaussianPointCloud(test)
    would assign wrong labels (that is why the stage-1 test_acc is about 0.12, but 0.875 after relabelling).
    """

    def __init__(self, data_dir, grid_size=7, sampling="random", num_points=8192, seed=777):
        root = Path(data_dir)
        self.classes = sorted(d.name for d in (root / "train").iterdir() if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # reuse the dataset's ply reading / voxel-id code; its own labels are ignored
        self._ds = GaussianPointCloud(root / "test", num_points=num_points,
                                      sampling_method="original_size", grid_size=grid_size)
        self.files = sorted(
            (str(p), self.class_to_idx[p.parent.name])
            for p in (root / "test").glob("*/*.ply")
        )
        self.sampling = sampling
        self.num_points = num_points
        self.seed = seed

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self):
        return len(self.files)

    def get(self, idx: int, device="cpu") -> Sample:
        path, label = self.files[idx]
        pts = self._ds._read_ply(Path(path))
        gauss, xyz_norm, _, _, _ = prepare_gaussian_cloud(pts)
        vids = self._ds._compute_voxel_ids(xyz_norm)
        M = gauss.shape[0]
        if self.sampling == "random":
            rng = np.random.default_rng([self.seed, idx])  # deterministic per sample
            sel = rng.choice(M, self.num_points, replace=M < self.num_points)
        else:
            sel = np.arange(M)
        xyz_t = torch.from_numpy(xyz_norm[sel])
        feat = torch.cat([xyz_t, torch.from_numpy(gauss[sel])], dim=1)  # (N, 11)
        return Sample(
            idx, path, label,
            feat.T.unsqueeze(0).contiguous().to(device),
            xyz_t.unsqueeze(0).to(device),
            torch.from_numpy(vids[sel]).long().unsqueeze(0).to(device),
        )


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def classify(model, s: Sample) -> torch.Tensor:
    """Logits (C,) for a single, already-reduced cloud. Uses the model's own forward."""
    logits, *_ = model(s.feat, s.xyz, s.vids, None)
    return logits[0]


@dataclass
class Importance:
    logits: torch.Tensor        # (classes,)
    pred: int
    channels: torch.Tensor      # (k,)
    weights: torch.Tensor       # (C,) weighted_global_features
    point_scores: torch.Tensor  # (N,)
    voxel_scores: torch.Tensor  # (V,)  -inf on empty voxels
    point_counts: torch.Tensor  # (V,)


@torch.no_grad()
def importance(model, s: Sample, topk_channels: int = 5) -> Importance:
    P, _ = model.extract_point_features(s.feat, s.xyz, None)          # (1, C, N)
    vox, _, counts = model.voxel_agg(P, s.vids, None)                 # (1, C, G,G,G), (1,1,V)
    Z = model.disentangler(vox.flatten(2))                            # (1, C, V)
    g = Z.mean(dim=2)                                                 # (1, C)  == avg pool
    logits = model.head(g)[0]
    pred = int(logits.argmax())
    w = model.head.weight[pred] * F.relu(g[0])                        # explain.py:70
    _, ch = torch.topk(w, topk_channels, largest=True, sorted=True)   # explain.py:72

    wk = w[ch].unsqueeze(1)                                           # (k, 1)
    Zp = model.disentangler(P)                                        # (1, C, N)  "unfolded"
    point_scores = (wk * Zp[0, ch]).sum(0)                            # (N,)
    counts = counts.view(-1)
    voxel_scores = (wk * Z[0, ch]).sum(0)                             # (V,)
    voxel_scores = voxel_scores.masked_fill(counts == 0, float("-inf"))
    return Importance(logits, pred, ch, w, point_scores, voxel_scores, counts)


# --------------------------------------------------------------------------- removal
def n_to_remove(n: int, percent: float) -> int:
    return int(round(n * percent / 100.0))


def keep_splat_importance(imp: Importance, percent: float) -> torch.Tensor:
    n = imp.point_scores.numel()
    r = n_to_remove(n, percent)
    order = torch.argsort(imp.point_scores, descending=True, stable=True)
    return order[r:].sort().values


def keep_splat_random(n: int, percent: float, rng: np.random.Generator, device) -> torch.Tensor:
    r = n_to_remove(n, percent)
    keep = np.sort(rng.permutation(n)[r:])
    return torch.from_numpy(keep).long().to(device)


def _keep_outside(vids: torch.Tensor, removed_voxels: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(~torch.isin(vids[0], removed_voxels), as_tuple=False).view(-1)


def keep_voxel_importance(imp: Importance, s: Sample, k: int):
    nonempty = int((imp.point_counts > 0).sum())
    k = min(k, nonempty)
    top = torch.topk(imp.voxel_scores, k).indices if k > 0 else imp.voxel_scores.new_zeros(0, dtype=torch.long)
    return _keep_outside(s.vids, top), k


def keep_voxel_random(imp: Importance, s: Sample, k: int, rng: np.random.Generator):
    nonempty = torch.nonzero(imp.point_counts > 0, as_tuple=False).view(-1).cpu().numpy()
    k = min(k, len(nonempty))
    chosen = torch.from_numpy(rng.choice(nonempty, k, replace=False)).long().to(s.vids.device)
    return _keep_outside(s.vids, chosen), k
