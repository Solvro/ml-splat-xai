import torch
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def generate_3d_prototypes(
    feature_model, 
    dataloader, 
    num_channels, 
    N=5, 
    device="cpu", 
    U=None
):
    feature_model.eval()
    activations_pos = torch.full((N, num_channels), -np.inf, device=device)
    prototypes = {c: [] for c in range(num_channels)}
    global_sample_index = 0

    for batch_idx, (xyz, gauss, _, indices) in tqdm(enumerate(dataloader), total=len(dataloader)):
        xyz = xyz.to(device, non_blocking=True)
        gauss = gauss.to(device, non_blocking=True)
        
        inputs = torch.cat([xyz, gauss], dim=-1).transpose(1, 2)
        
        feature_map = feature_model(inputs, xyz.transpose(1, 2))
        
        if U is not None:
            feature_map = torch.einsum("ij,bjn->bin", U, feature_map)
        
        batch_activations = feature_map.max(dim=-1)[0]  # (B, C)

        for b in range(batch_activations.size(0)):
            for c in range(num_channels):
                activation_val = batch_activations[b, c].item()
                
                min_val, min_idx = activations_pos[:, c].min(dim=0)
                
                if activation_val > min_val.item():
                    activations_pos[min_idx, c] = activation_val
                    
                    point_idx = feature_map[b, c].argmax().item()
                    prototype_info = {
                        "sample_index": indices[b].item(),
                        "point_idx": point_idx,
                        "activation": activation_val,
                        "point": xyz[b, point_idx].cpu().numpy(),
                        "gauss": gauss[b, point_idx].cpu().numpy()
                    }
                    
                    if len(prototypes[c]) >= N:
                        min_proto = min(prototypes[c], key=lambda x: x["activation"])
                        prototypes[c].remove(min_proto)
                    
                    prototypes[c].append(prototype_info)
        
        global_sample_index += xyz.size(0)
    
    for c in range(num_channels):
        prototypes[c] = sorted(prototypes[c], key=lambda x: x["activation"], reverse=True)[:N]
    
    return prototypes


@torch.no_grad()
def topk_active_channels_3d(
    feature_model,
    classification_head,
    inputs,
    xyz,
    k=4,
    device="cpu",
    U=None
):
    feature_model.eval()
    classification_head.eval()
    
    feature_map = feature_model(inputs, xyz)
    
    if U is not None:
        feature_map = torch.einsum("ij,bjn->bin", U, feature_map)
    
    global_feature = feature_map.max(dim=-1)[0]  # (1, C)
    
    output = classification_head(inputs, xyz)
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output
    
    pred_class = logits.argmax(dim=1).item()
    
    last_linear = None
    for module in classification_head.modules():
        if isinstance(module, torch.nn.Linear) and module.in_features == global_feature.size(1):
            last_linear = module
            break
    
    if last_linear is None:
        for module in classification_head.head.modules():
            if isinstance(module, torch.nn.Linear) and module.in_features == global_feature.size(1):
                last_linear = module
                break
    
    if last_linear is None:
        raise RuntimeError("No linear layer found in the classification model")
    
    class_weights = last_linear.weight[pred_class]  # (out_features,)
    
    channel_importance = class_weights * global_feature.squeeze(0)  # (C,)
    
    _, topk_channels = torch.topk(channel_importance, k, largest=True)
    return topk_channels.tolist()
