import os
import random
import numpy as np
import torch
from omegaconf import DictConfig
from models.factory import create_backbone_model
from data.pointnet_data import create_indexed_dataloader
from matrix.factory import create_matrix
from prototypes.selection_3d import generate_3d_prototypes, topk_active_channels_3d
from prototypes.visualization_3d import visualize_3d_explanations


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def explain_predictions(config: DictConfig):
    set_seeds(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_model, feature_model, _, num_channels = create_backbone_model(
        config.model.name,
        device,
        config.model.custom_weights_path,
        config.model.num_classes
    )
    feature_model = feature_model.to(device).eval()
    print(f"Feature model {config.model.name} initialized.")

    dataloader_val = create_indexed_dataloader(
        config.dataset.datapath_val,
        config.dataset.num_points,
        config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        shuffle=config.dataloader.shuffle,
    )

    disentanglement_matrix = create_matrix(config.matrix.type, num_channels, device)
    disentanglement_matrix.load_state(
        os.path.join(config.output_path, f"{config.matrix.type}.pt"),
        map_location=device,
    )

    positive_prototypes = generate_3d_prototypes(
        feature_model,
        dataloader_val,
        num_channels,
        config.visualization.num_prototypes,
        device,
        disentanglement_matrix(),
    )
    print("Prototypes created.")

    for repeat in range(config.visualization.num_explanations):
        idx = np.random.randint(0, len(dataloader_val.dataset))
        xyz, gauss, label, _ = dataloader_val.dataset[idx]
        xyz = xyz.unsqueeze(0).to(device)
        gauss = gauss.unsqueeze(0).to(device)
        inputs = torch.cat([xyz, gauss], dim=-1).transpose(1, 2)

        channels = topk_active_channels_3d(
            feature_model=feature_model,
            classification_head=base_model,
            inputs=inputs,
            xyz=xyz.transpose(1,2),
            k=config.visualization.num_prototypical_channels,
            device=device,
            U=disentanglement_matrix(),
        )

        image_prototypes = {
            "xyz": xyz.squeeze(0).cpu().numpy(),
        }
        point_prototypes = {}
        for c in channels:
            if c in positive_prototypes and len(positive_prototypes[c]) > 0:
                point_prototypes[c] = positive_prototypes[c][0]

        explanation_path = os.path.join(config.output_path, "Explanations")
        os.makedirs(explanation_path, exist_ok=True)
        
        visualize_3d_explanations(
            image_prototypes,
            point_prototypes,
            dataloader_val.dataset,
            explanation_path,
            f"explanation_{repeat}"
        )
