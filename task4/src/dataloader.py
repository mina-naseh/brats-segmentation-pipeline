import os
from typing import List, Dict, Optional
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
)
from monai.data import CacheDataset, DataLoader, Dataset
from monai.config import print_config
from monai.transforms.transform import MapTransform

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    Reference: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
    
def get_dataloader(
    data_dicts: List[Dict[str, str]],
    batch_size: int,
    augmentations: Optional[List] = None,
    cache_rate: float = 0.5,
    num_workers: int = 4,
    shuffle: bool = True,
):
    """
    Create a DataLoader for the BraTS dataset.

    Args:
        data_dicts (List[Dict[str, str]]): List of data dictionaries with "image" and "label" keys.
        batch_size (int): Batch size for DataLoader.
        augmentations (Optional[List]): List of additional augmentations.
        cache_rate (float): Percentage of data to cache in memory.
        num_workers (int): Number of workers for data loading.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
        *(augmentations if augmentations else []),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    dataset = CacheDataset(
        data=data_dicts,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader