import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData

from pointnet.pointnet import PointNetCls
from pointnet.dataset import GaussianPointCloud, FEATURE_NAMES

def flat_to_3d_index(idx_flat: int, grid_size: int) -> list[int]:
    z = idx_flat // (grid_size * grid_size)
    remainder = idx_flat % (grid_size * grid_size)
    y = remainder // grid_size
    x = remainder % grid_size
    return [x, y, z]

def draw_voxel_cube(ax, voxel_idx, grid_size, xyz_min, xyz_max):
    bbox_size = xyz_max - xyz_min
    voxel_size = bbox_size / grid_size
    start_point = xyz_min + np.array(voxel_idx) * voxel_size

    p = start_point
    s = voxel_size
    vertices = np.array([
        p, p + [s[0], 0, 0], p + [s[0], s[1], 0], p + [0, s[1], 0],
        p + [0, 0, s[2]], p + [s[0], 0, s[2]], p + [s[0], s[1], s[2]], p + [0, s[1], s[2]]
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for edge in edges:
        points = np.vstack([vertices[edge[0]], vertices[edge[1]]])
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="blue", alpha=0.8, linewidth=1.5)

def load_and_preprocess_ply(ply_path: Path):
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    pts = np.vstack([vertex[name] for name in FEATURE_NAMES]).T.astype(np.float32)

    q = pts[:, 7:11]
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    pts[:, 7:11] = q / q_norm

    xyz = pts[:, :3]
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    xyz_normalized = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-8)

    return {
        "gauss": torch.from_numpy(pts).float(),
        "xyz_normalized": torch.from_numpy(xyz_normalized).float(),
        "original_xyz": torch.from_numpy(xyz).float(),
        "xyz_min": xyz_min,
        "xyz_max": xyz_max,
    }

def visualize_prototypes(
    ply_path: Path,
    model: PointNetCls,
    top_k: int = 3,
    device: str = "cpu",
    output_prefix: str = "prototype_visualization"
):
    model.eval()
    model.to(device)

    data = load_and_preprocess_ply(ply_path)
    features = data["gauss"].unsqueeze(0).transpose(1, 2).to(device)
    xyz_normalized = data["xyz_normalized"].unsqueeze(0).to(device)
    original_xyz = data["original_xyz"].numpy()
    xyz_min, xyz_max = data["xyz_min"], data["xyz_max"]

    with torch.no_grad():
        logits, voxel_activations = model(features, xyz_normalized)
    
    pred_class_idx = logits.argmax(dim=1).item()
    voxel_activations = voxel_activations.squeeze(0)

    top_activations, top_indices_flat = torch.topk(voxel_activations, k=top_k)
    top_indices_flat = top_indices_flat.cpu().numpy()

    print(f"File: {ply_path.name}, Predicted class index: {pred_class_idx} | True class index: {ply_path.parent.name}")

    G = model.grid_size
    voxel_indices = (xyz_normalized.squeeze(0) * G).long().clamp(0, G - 1)
    voxel_indices_flat = voxel_indices[..., 2] * (G * G) + voxel_indices[..., 1] * G + voxel_indices[..., 0]
    voxel_indices_flat = voxel_indices_flat.cpu().numpy()

    fig_context, axes_context = plt.subplots(1, top_k, figsize=(18, 6), subplot_kw={"projection": "3d"})
    fig_context.suptitle(f"Top {top_k} Prototypes in Context for {ply_path.name}", fontsize=16)

    fig_isolated, axes_isolated = plt.subplots(1, top_k, figsize=(18, 6), subplot_kw={"projection": "3d"})
    fig_isolated.suptitle(f"Top {top_k} Isolated Prototypes for {ply_path.name}", fontsize=16)

    fig_activations, axis_activations = plt.subplots(1, 1, figsize=(10, 6))
    fig_activations.suptitle(f"Voxel activations for {ply_path.name}", fontsize=16)


    for i in range(top_k):
        ax_context = axes_context[i]
        ax_isolated = axes_isolated[i]

        prototype_flat_idx = top_indices_flat[i]
        prototype_3d_idx = flat_to_3d_index(prototype_flat_idx, G)
        
        point_mask = (voxel_indices_flat == prototype_flat_idx)
        
        ax_context.scatter(original_xyz[:, 0], original_xyz[:, 1], original_xyz[:, 2], s=0.5, c="gray", alpha=0.2)
        
        num_points_in_voxel = np.sum(point_mask)
        if num_points_in_voxel > 0:
            prototype_points = original_xyz[point_mask]
            ax_context.scatter(prototype_points[:, 0], prototype_points[:, 1], prototype_points[:, 2], s=2, c="red")
            ax_isolated.scatter(prototype_points[:, 0], prototype_points[:, 1], prototype_points[:, 2], s=5, c="red")

        print(f"Prototype #{i+1}: Voxel {prototype_3d_idx} (Index {prototype_flat_idx}) | Gaussians: {num_points_in_voxel}")

        draw_voxel_cube(ax_context, prototype_3d_idx, G, xyz_min, xyz_max)
        draw_voxel_cube(ax_isolated, prototype_3d_idx, G, xyz_min, xyz_max)

        for ax in [ax_context, ax_isolated]:
            ax.set_title(f"Voxel: {prototype_3d_idx}\nActivation: {top_activations[i]:.2f}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=30, azim=45)

    axis_activations.bar(range(len(voxel_activations)), voxel_activations.cpu().numpy(), color='blue', alpha=0.7)
    axis_activations.set_title("Voxel Activations")
    axis_activations.set_xlabel("Voxel Index")
    axis_activations.set_ylabel("Activation Value")
    axis_activations.set_xticks(range(0, G**3, G**2))
    axis_activations.set_xticklabels([f"{i}" for i in range(0, G**3, G**2)])
    axis_activations.grid(True)
    path_activations = f"{output_prefix}_activations.png"
    fig_activations.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_activations.savefig(path_activations)

    path_context = f"{output_prefix}_context.png"
    fig_context.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_context.savefig(path_context)
    
    path_isolated = f"{output_prefix}_prototypes_only.png"
    fig_isolated.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_isolated.savefig(path_isolated)
    
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Visualize PointNet Prototypes")
    parser.add_argument("--ply_file", type=str, required=True, help="Path to the .ply file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top prototypes to visualize.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_prefix", type=str, default="prototype_visualization", help="Prefix for the output PNG files.")
    args = parser.parse_args()

    checkpoint = torch.load(args.model_path, map_location=args.device)
    config = checkpoint["config"]
    
    ply_path = Path(args.ply_file)
    data_root = ply_path.parent.parent
    dataset = GaussianPointCloud(data_root, sampling_method="none")
    num_classes = len(dataset.classes)
    in_dim = len(FEATURE_NAMES)

    use_stn_from_config = config.get("use_stn", False)

    model = PointNetCls(
        in_dim=in_dim,
        out_dim=num_classes,
        grid_size=config["grid_size"],
        stn_3d=use_stn_from_config,
        stn_nd=use_stn_from_config
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])

    output_prefix = args.output_prefix.removesuffix(".png")

    visualize_prototypes(
        ply_path=ply_path,
        model=model,
        top_k=args.top_k,
        device=args.device,
        output_prefix=output_prefix
    )

if __name__ == "__main__":
    main()
