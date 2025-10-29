# language: python
#!/usr/bin/env python3
"""
Simple Open3D point cloud visualizer.

Usage examples (Windows PowerShell):
  python visualize_pc.py --path data/model.ply
  python visualize_pc.py --path data/points.npy
  python visualize_pc.py            # shows a random demo cloud

Requirements:
  pip install open3d numpy
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


def load_point_cloud_from_file(path: Path) -> o3d.geometry.PointCloud:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".ply", ".pcd"}:
        pcd = o3d.io.read_point_cloud(str(path))
        return pcd
    elif path.suffix.lower() in {".npy", ".npz"}:
        arr = np.load(str(path))
        if isinstance(arr, np.lib.npyio.NpzFile):  # .npz
            # try common keys
            arr = arr.get("points", arr[list(arr.files)[0]])
        return point_cloud_from_numpy(arr)
    elif path.suffix.lower() in {".txt", ".csv"}:
        arr = np.loadtxt(str(path), delimiter=",")
        return point_cloud_from_numpy(arr)
    else:
        raise ValueError("Unsupported file type: " + path.suffix)


def point_cloud_from_numpy(arr: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Accepts Nx3 (xyz) or Nx6 (xyzrgb) numpy array and returns an Open3D PointCloud.
    Colors must be in [0,1] range; if 0-255 they will be scaled automatically.
    """
    if arr.ndim != 2 or arr.shape[1] not in (3, 6):
        raise ValueError("Expected Nx3 or Nx6 array, got shape " + str(arr.shape))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:, :3].astype(np.float64))
    if arr.shape[1] == 6:
        colors = arr[:, 3:6].astype(np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def create_demo_cloud(n=2048):
    xyz = np.random.uniform(-0.5, 0.5, size=(n, 3))
    rgb = (xyz - xyz.min()) / (xyz.max() - xyz.min())
    arr = np.hstack([xyz, rgb])
    return point_cloud_from_numpy(arr)


def main():
    parser = argparse.ArgumentParser(description="Visualize a point cloud with Open3D")
    parser.add_argument("--path", type=str, default=None, help="Path to .ply/.pcd/.npy/.txt file")
    parser.add_argument("--show_normals", action="store_true", help="Estimate & show normals")
    parser.add_argument("--point_size", type=float, default=10.0, help="Point size in pixels")
    args = parser.parse_args()

    if args.path:
        pcd = load_point_cloud_from_file(Path(args.path))
    else:
        pcd = create_demo_cloud()

    if args.show_normals:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.normalize_normals()

    print("Points:", np.asarray(pcd.points).shape)
    if pcd.has_colors():
        print("Colors present")
        print("First 5 colors:\n", np.asarray(pcd.colors))

    # use Visualizer so we can set point size
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud", width=1280, height=720)
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = args.point_size  # set point radius (in pixels)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()