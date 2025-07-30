from dataclasses import astuple, dataclass

import torch

from models.registry import model_list

from pointnets.pointnet2 import PointNet2ClsSSG, PointNet2ClsMSG
from .backbones_3d import create_3d_backbone


@dataclass
class ModelBundle:
    base_model: torch.nn.Module
    backbone: torch.nn.Module
    transform: callable
    num_channels: int

    def __iter__(self):
        return iter(astuple(self))


def create_backbone_model(
    model_name, device="cuda", custom_weights_path=None, num_classes=1000
):
    if model_name not in model_list:
        raise ValueError(
            f"Invalid model name '{model_name}'. Supported models: {', '.join(model_list.keys())}"
        )

    model_fn, weights, num_channels = model_list[model_name]

    if "pointnet" in model_name:
        if model_name == "pointnet2_ssg":
            model_fn = PointNet2ClsSSG
            num_channels = 1024
        elif model_name == "pointnet2_msg":
            model_fn = PointNet2ClsMSG
            num_channels = 1280
            
        base_model = model_fn(
            in_dim=11,  # XYZ + gauss
            out_dim=num_classes
        ).to(device)
        
        if custom_weights_path:
            state_dict = torch.load(custom_weights_path, map_location=device)
            base_model.load_state_dict(state_dict["model_state_dict"])
        
        backbone = create_3d_backbone(model_name, base_model)
        transform = None
        return ModelBundle(base_model, backbone, transform, num_channels)
