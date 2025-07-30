import torch

class PointNetBackbone(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.feature_extractor = model
        
    def forward(self, x, xyz):
        _, features = self.feature_extractor(x, xyz)
        return features[-1]

BACKBONE_CLASSES_3D = {
    "pointnet2_ssg": PointNetBackbone,
    "pointnet2_msg": PointNetBackbone,
}

def create_3d_backbone(model_name, base_model):
    for prefix, cls in BACKBONE_CLASSES_3D.items():
        if prefix in model_name:
            return cls(base_model)
    raise ValueError(f"No backbone implementation for model '{model_name}'")
