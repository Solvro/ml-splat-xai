import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement

from pointnet.pointnet import PointNetLightning, PointNetCls
from pointnet.dataset import FEATURE_NAMES, prepare_gaussian_cloud


def read_ply_and_prepare(ply_path: Path):
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    pts = np.vstack([vertex[name] for name in FEATURE_NAMES]).T.astype(np.float32)  # (N, D)
    original_xyz = pts[:, :3].copy()

    gauss, xyz_normalized, xyz_min, xyz_max = prepare_gaussian_cloud(pts.copy())

    return {
        "gauss": torch.from_numpy(gauss).float(),                # (N, D)
        "xyz_normalized": torch.from_numpy(xyz_normalized).float(),  # (N, 3)
        "original_xyz": torch.from_numpy(original_xyz).float(),  # (N, 3)
        "xyz_min": torch.from_numpy(xyz_min).float(),            # (3,)
        "xyz_max": torch.from_numpy(xyz_max).float(),            # (3,)
    }


def flat_to_xyz(flat_idx: int, G: int):
    x = flat_idx // (G * G)
    y = (flat_idx // G) % G
    z = flat_idx % G
    return int(x), int(y), int(z)


def draw_parallelepiped(ax, corners_original: torch.Tensor, color="blue", alpha=0.9, linewidth=1.5):
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    c = corners_original.detach().cpu().numpy()
    for a, b in edges:
        seg = np.vstack([c[a], c[b]])
        ax.plot(seg[:,0], seg[:,1], seg[:,2], color=color, alpha=alpha, linewidth=linewidth)


def invert_voxel_cell_to_original(x_idx, y_idx, z_idx, G: int,
                                  T_3x3: torch.Tensor, min_used: torch.Tensor, max_used: torch.Tensor,
                                  xyz_min: torch.Tensor, xyz_max: torch.Tensor):
    device = T_3x3.device
    dtype = T_3x3.dtype

    lows = torch.tensor([x_idx / G, y_idx / G, z_idx / G], device=device, dtype=dtype)
    highs = torch.tensor([(x_idx + 1) / G, (y_idx + 1) / G, (z_idx + 1) / G], device=device, dtype=dtype)
    corners_unit = torch.stack([
        torch.tensor([lows[0], lows[1], lows[2]], device=device, dtype=dtype),
        torch.tensor([highs[0], lows[1], lows[2]], device=device, dtype=dtype),
        torch.tensor([highs[0], highs[1], lows[2]], device=device, dtype=dtype),
        torch.tensor([lows[0], highs[1], lows[2]], device=device, dtype=dtype),
        torch.tensor([lows[0], lows[1], highs[2]], device=device, dtype=dtype),
        torch.tensor([highs[0], lows[1], highs[2]], device=device, dtype=dtype),
        torch.tensor([highs[0], highs[1], highs[2]], device=device, dtype=dtype),
        torch.tensor([lows[0], highs[1], highs[2]], device=device, dtype=dtype),
    ], dim=0)  # (8,3)

    pre_rescale = min_used.view(3) + corners_unit * (max_used.view(3) - min_used.view(3))  # (8,3)

    # undo STN
    try:
        T_inv = torch.linalg.inv(T_3x3)
    except RuntimeError:
        T_inv = torch.linalg.pinv(T_3x3)
    coords_norm = (T_inv @ pre_rescale.T).T  # (8,3)

    scale = (xyz_max - xyz_min).to(coords_norm)
    corners_original = coords_norm * scale + xyz_min.to(coords_norm)

    return corners_original  # (8,3)


def build_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    hparams = ckpt.get("hyper_parameters", {})

    lit = PointNetLightning(
        in_dim=hparams.get("in_dim", len(FEATURE_NAMES)),
        num_classes=hparams.get("num_classes", 1),
        grid_size=hparams.get("grid_size", 10),
        stn_3d=hparams.get("stn_3d", True),
        stn_nd=hparams.get("stn_nd", True),
        lr=hparams.get("lr", 1e-3),
        weight_decay=hparams.get("weight_decay", 1e-4),
        head_norm=hparams.get("head_norm", False),
        stn_head_norm=hparams.get("stn_head_norm", False),
        pooling=hparams.get("pooling", "max"),
    ).to(device)
    lit.load_state_dict(ckpt["state_dict"], strict=True)
    lit.eval()
    return lit


def save_ply_subset_topk(src_ply_path: Path, topk_mask_np: np.ndarray, out_path: Path):
    ply = PlyData.read(str(src_ply_path))
    vert = ply["vertex"].data
    if topk_mask_np.dtype != np.bool_:
        topk_mask_np = topk_mask_np.astype(bool)
    sel = vert[topk_mask_np]
    vertex_el = PlyElement.describe(sel, "vertex")
    PlyData([vertex_el], text=False).write(str(out_path))


def save_ply_highlight_full(src_ply_path: Path, topk_mask_np: np.ndarray, out_path: Path,
                            dc_scale: float = 4.0, opacity_to: float | None = None):
    ply = PlyData.read(str(src_ply_path))
    vert = ply["vertex"].data.copy()
    names = vert.dtype.names or ()

    for c in (0, 1, 2):
        key = f"f_dc_{c}"
        if key in names:
            vert[key][topk_mask_np] = vert[key][topk_mask_np] * float(dc_scale)

    if opacity_to is not None and "opacity" in names:
        val = float(max(0.0, min(1.0, opacity_to)))
        vert["opacity"][topk_mask_np] = val

    vertex_el = PlyElement.describe(vert, "vertex")
    PlyData([vertex_el], text=False).write(str(out_path))


def visualize(ply_file: str, ckpt_path: str, top_k: int = 3, device_str: str = "cuda",
              output_prefix: str = "prototype_visualization",
              save_topk_ply: bool = False,
              save_highlight_ply: bool = False,
              highlight_scale: float = 4.0,
              highlight_opacity: float | None = None):
    device = torch.device(device_str if (device_str.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    lit = build_model_from_ckpt(ckpt_path, device)
    model: PointNetCls = lit.model
    model.eval()

    src_ply_path = Path(ply_file)
    data = read_ply_and_prepare(src_ply_path)

    features = data["gauss"].unsqueeze(0).transpose(1, 2).to(device)   # (1, D, N)
    xyz_norm = data["xyz_normalized"].unsqueeze(0).to(device)           # (1, N, 3)
    original_xyz = data["original_xyz"].to(device)                      # (N, 3)

    with torch.no_grad():
        logits, voxel_acts, _, indices = model(features, xyz_norm, mask=None)
    pred_idx = logits.argmax(dim=1).item()

    if getattr(model, "last_stn_T", None) is None or getattr(model, "last_rescale_min", None) is None or getattr(model, "last_rescale_max", None) is None:
        raise RuntimeError("Brak zapisanych STN/rescale affine w modelu. Upewnij się, że stn_3d=True i że forward zapisuje last_stn_T/min/max.")

    T = model.last_stn_T[0].detach()                 # (3,3)
    min_used = model.last_rescale_min[0].detach().view(3)  # (3,)
    max_used = model.last_rescale_max[0].detach().view(3)  # (3,)

    G = model.grid_size
    vol = voxel_acts[0].detach().cpu()  # (G, G, G)

    indices_b0 = indices[0].detach().cpu().numpy()  # (N,)
    counts_np = np.bincount(indices_b0, minlength=G**3)  # (V,)
    nonempty_mask_np = counts_np > 0
    num_nonempty = int(nonempty_mask_np.sum())

    flat_acts = vol.flatten().cpu()  # (V,)
    flat_acts_masked = flat_acts.clone()
    flat_acts_masked[~torch.from_numpy(nonempty_mask_np)] = -float('inf')

    K = min(top_k, num_nonempty)
    top_vals, top_idx = torch.topk(flat_acts_masked, K, largest=True)
    top_idx = top_idx.cpu()

    orig_np = original_xyz.detach().cpu().numpy()

    topk_vox_np = top_idx.numpy()
    topk_mask_np = np.isin(indices_b0, topk_vox_np)

    fig_ctx, axes_ctx = plt.subplots(1, K, figsize=(6*K, 6), subplot_kw={"projection": "3d"})
    fig_iso, axes_iso = plt.subplots(1, K, figsize=(6*K, 6), subplot_kw={"projection": "3d"})
    if K == 1:
        axes_ctx = [axes_ctx]
        axes_iso = [axes_iso]

    for i in range(K):
        ax_c = axes_ctx[i]
        ax_i = axes_iso[i]

        flat = int(top_idx[i].item())
        x_idx, y_idx, z_idx = flat_to_xyz(flat, G)

        mask_pts = (indices_b0 == flat)
        pts_in = orig_np[mask_pts]
        num_pts_in = pts_in.shape[0]

        corners_original = invert_voxel_cell_to_original(
            x_idx, y_idx, z_idx, G,
            T_3x3=T, min_used=min_used, max_used=max_used,
            xyz_min=data["xyz_min"], xyz_max=data["xyz_max"]
        )

        ax_c.scatter(orig_np[:,0], orig_np[:,1], orig_np[:,2], s=0.5, c="gray", alpha=0.15)
        if num_pts_in > 0:
            ax_c.scatter(pts_in[:,0], pts_in[:,1], pts_in[:,2], s=4, c="red", alpha=0.9)
            ax_i.scatter(pts_in[:,0], pts_in[:,1], pts_in[:,2], s=6, c="red", alpha=0.9)
        draw_parallelepiped(ax_c, corners_original, color="blue", alpha=0.9, linewidth=1.5)
        draw_parallelepiped(ax_i, corners_original, color="blue", alpha=0.9, linewidth=1.5)

        title = f"Voxel ({x_idx},{y_idx},{z_idx})  act={top_vals[i].item():.3f}  points={num_pts_in}"
        ax_c.set_title(title)
        ax_i.set_title(title)

        for ax in (ax_c, ax_i):
            mins = orig_np.min(axis=0)
            maxs = orig_np.max(axis=0)
            ranges = maxs - mins + 1e-6
            ax.set_box_aspect(ranges.tolist())
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_proj_type("ortho")
            ax.view_init(elev=25, azim=40)

    fig_ctx.suptitle(f"Top-{K} prototypes (pred={pred_idx}) – context", fontsize=14)
    fig_iso.suptitle(f"Top-{K} prototypes (pred={pred_idx}) – isolated", fontsize=14)
    fig_ctx.tight_layout()
    fig_iso.tight_layout()

    out_ctx = f"{output_prefix}_context.png"
    out_iso = f"{output_prefix}_prototypes_only.png"
    fig_ctx.savefig(out_ctx, dpi=200)
    fig_iso.savefig(out_iso, dpi=200)
    plt.close(fig_ctx)
    plt.close(fig_iso)

    print(f"Saved:\n - {out_ctx}\n - {out_iso}")
    print(f"Predicted class index: {pred_idx}")

    if save_topk_ply:
        out_topk = Path(f"{output_prefix}_top{K}_voxels_only.ply")
        save_ply_subset_topk(src_ply_path, topk_mask_np, out_topk)
        print(f"Saved Top-K subset PLY: {out_topk}")

    if save_highlight_ply:
        out_high = Path(f"{output_prefix}_highlight_top{K}.ply")
        save_ply_highlight_full(src_ply_path, topk_mask_np, out_high,
                                dc_scale=highlight_scale,
                                opacity_to=highlight_opacity)
        print(f"Saved highlighted full PLY: {out_high}")


def main():
    ap = argparse.ArgumentParser(description="Visualize top-K voxel prototypes in original space and export PLYs")
    ap.add_argument("--ply_file", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_prefix", type=str, default="prototype_visualization")
    ap.add_argument("--save_topk_ply", action="store_true")
    ap.add_argument("--save_highlight_ply", action="store_true")
    ap.add_argument("--highlight_scale", type=float, default=4.0)
    ap.add_argument("--highlight_opacity", type=float, default=-1.0)

    args = ap.parse_args()
    highlight_op = None if args.highlight_opacity is None or args.highlight_opacity < 0 else args.highlight_opacity

    visualize(
        ply_file=args.ply_file,
        ckpt_path=args.model_path,
        top_k=args.top_k,
        device_str=args.device,
        output_prefix=args.output_prefix,
        save_topk_ply=args.save_topk_ply,
        save_highlight_ply=args.save_highlight_ply,
        highlight_scale=args.highlight_scale,
        highlight_opacity=highlight_op,
    )


if __name__ == "__main__":
    main()
