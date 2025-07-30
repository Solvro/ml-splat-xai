import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_3d_prototypes(prototypes, dataset, output_dir, prefix="", max_cols=3):
    os.makedirs(output_dir, exist_ok=True)
    
    for c, proto_list in prototypes.items():
        if not proto_list:
            continue
        
        n_prototypes = len(proto_list)
        n_rows = (n_prototypes + max_cols - 1) // max_cols
        n_cols = min(max_cols, n_prototypes)
        
        fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
        fig.suptitle(f"Prototypes for Channel {c}", fontsize=16)
        
        for i, proto in enumerate(proto_list):
            ax = fig.add_subplot(n_rows, n_cols, i+1, projection="3d")
            
            sample_idx = proto["sample_index"]
            xyz, _, _, _ = dataset[sample_idx]
            
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="lightgray", alpha=0.3, s=5)
            
            proto_point = proto["point"]
            ax.scatter(proto_point[0], proto_point[1], proto_point[2], 
                       c="red", s=100, marker="o", edgecolors="black")
            
            ax.text(proto_point[0], proto_point[1], proto_point[2], 
                    f"Prototype {i}", fontsize=10, color="black", weight="bold")
            
            ax.set_title(f"Sample {sample_idx}, Point {proto['point_idx']}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            
            if len(xyz) > 0:
                max_range = np.array([
                    xyz[:, 0].max()-xyz[:, 0].min(), 
                    xyz[:, 1].max()-xyz[:, 1].min(), 
                    xyz[:, 2].max()-xyz[:, 2].min()
                ]).max() / 2.0
                
                mid_x = (xyz[:, 0].max()+xyz[:, 0].min()) * 0.5
                mid_y = (xyz[:, 1].max()+xyz[:, 1].min()) * 0.5
                mid_z = (xyz[:, 2].max()+xyz[:, 2].min()) * 0.5
                
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
        plt.savefig(os.path.join(output_dir, f"{prefix}_channel_{c}_prototypes.png"), dpi=150)
        plt.close(fig)


def visualize_3d_explanations(
    image_prototypes,
    point_prototypes,
    dataset,
    output_path,
    name="3d_explanation"
):
    num_prototypes = len(point_prototypes)
    if num_prototypes == 0:
        return
    
    fig = plt.figure(figsize=(5 * (num_prototypes + 1), 5))
    fig.suptitle("3D Explanation", fontsize=16)
    
    ax1 = fig.add_subplot(1, num_prototypes + 1, 1, projection="3d")
    xyz = image_prototypes["xyz"]
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="lightgray", alpha=0.3, s=5)
    ax1.set_title("Input Point Cloud")
    
    if len(xyz) > 0:
        max_range = calculate_max_range(xyz)
        set_axis_limits(ax1, xyz, max_range)
    
    colors = plt.cm.tab10.colors
    for i, (c, proto) in enumerate(point_prototypes.items()):
        ax = fig.add_subplot(1, num_prototypes + 1, i+2, projection="3d")
        
        sample_idx = proto["sample_index"]
        xyz_proto, _, _, _ = dataset[sample_idx]
        
        ax.scatter(xyz_proto[:, 0], xyz_proto[:, 1], xyz_proto[:, 2], 
                  c="lightgray", alpha=0.3, s=5)
        
        proto_point = proto["point"]
        ax.scatter([proto_point[0]], [proto_point[1]], [proto_point[2]],
                  c=[colors[i % len(colors)]], s=150, marker="o", edgecolors="black")
        
        ax.text(proto_point[0], proto_point[1], proto_point[2],
               f"P{c}", fontsize=12, color="black", weight="bold")
        
        ax.set_title(f"Prototype for Channel {c}")
        
        if len(xyz_proto) > 0:
            max_range = calculate_max_range(xyz_proto)
            set_axis_limits(ax, xyz_proto, max_range)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{name}.png"), dpi=150)
    plt.close(fig)

def calculate_max_range(xyz):
    x_range = xyz[:, 0].max() - xyz[:, 0].min()
    y_range = xyz[:, 1].max() - xyz[:, 1].min()
    z_range = xyz[:, 2].max() - xyz[:, 2].min()
    return max(x_range, y_range, z_range) / 2.0

def set_axis_limits(ax, xyz, max_range):
    mid_x = (xyz[:, 0].max() + xyz[:, 0].min()) * 0.5
    mid_y = (xyz[:, 1].max() + xyz[:, 1].min()) * 0.5
    mid_z = (xyz[:, 2].max() + xyz[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
