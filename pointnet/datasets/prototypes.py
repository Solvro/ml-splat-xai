import torch
from torch.utils.data import Dataset, Subset

from pointnet.datasets.gaussian_point_cloud import collate_fn


class PrototypesDataset(Dataset):
    def __init__(self, original_dataset, prototypes_dict):
        self.original_dataset = original_dataset.dataset if isinstance(original_dataset, Subset) else original_dataset
        self.prototypes_dict = prototypes_dict

        self.samples = []
        for channel, indices in prototypes_dict.items():
            for idx in indices:
                if idx < len(self.original_dataset):
                    self.samples.append((idx, channel))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        original_idx, channel = self.samples[idx]
        data = self.original_dataset[original_idx]
        return {
            "gauss": data["gauss"],
            "xyz_normalized": data["xyz_normalized"],
            "mask": data.get("mask", None),
            "indices": data.get("indices", None),
            "label": data.get("label", None),
            "sample_idx": data.get("sample_idx", None),
            "voxel_ids": data["voxel_ids"],
            "channel": channel
        }


def collate_prototypes(batch):
    channels = [item["channel"] for item in batch]
    batch_without_channel = [{k: v for k, v in item.items() if k != "channel"} for item in batch]
    batch_dict = collate_fn(batch_without_channel)
    batch_dict["channel"] = torch.tensor(channels, dtype=torch.long)
    return batch_dict
