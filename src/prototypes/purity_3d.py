import torch
from tqdm import tqdm

def purity_argmax3d(feature_map, c):
    batch_size = feature_map.size(0)
    channel_activation = feature_map[torch.arange(batch_size), c]  # (B, N)
    max_idx = torch.argmax(channel_activation, dim=1)  # (B,)
    
    all_channel_values = feature_map[torch.arange(batch_size), :, max_idx]  # (B, C)
    
    l2_norm = torch.norm(all_channel_values, dim=1, keepdim=True).clamp_min(1e-8)  # (B, 1)
    purity = all_channel_values[torch.arange(batch_size), c] / l2_norm.squeeze(1)  # (B,)
    
    return purity

PURITY_FUNCTIONS_3D = {
    "argmax3d": purity_argmax3d
}

def get_3d_purity_fn(type):
    if type in PURITY_FUNCTIONS_3D:
        return PURITY_FUNCTIONS_3D[type]
    else:
        raise ValueError(f"Unknown 3D purity function type: {type}")


@torch.no_grad()
def get_prototypes_purity(
    feature_model,
    prototypes_dataloader,
    device="cpu",
    U=None,
    purity_fn=purity_argmax3d,
):
    total_purity = 0.0
    total_count = 0

    feature_model.eval()
    
    for batch in tqdm(prototypes_dataloader, desc="Calculating prototypes purity"):
        xyz_full, gauss_full, point_indices, channels, sample_indices = batch
        
        xyz_full = xyz_full.to(device, non_blocking=True)
        gauss_full = gauss_full.to(device, non_blocking=True)
        point_indices = point_indices.to(device, non_blocking=True)
        channels = channels.to(device, non_blocking=True)
        
        inputs_full = torch.cat([xyz_full, gauss_full], dim=-1).transpose(1, 2)  # (B, num_features, num_points)
        
        feature_map_full = feature_model(inputs_full, xyz_full.transpose(1, 2))  # (B, C, num_points)
        
        if U is not None:
            feature_map_full = torch.einsum("ij,bjn->bin", U, feature_map_full)
        
        batch_size = feature_map_full.size(0)
        feature_map_proto = torch.stack([
            feature_map_full[i, :, point_indices[i]] for i in range(batch_size)
        ]).unsqueeze(-1)  # (B, C, 1)
        
        purity = purity_fn(feature_map_proto, channels)
        
        total_purity += purity.sum().item()
        total_count += purity.numel()

    return total_purity / total_count if total_count > 0 else 0.0
