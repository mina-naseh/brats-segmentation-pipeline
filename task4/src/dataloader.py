import os
from glob import glob
from typing import List, Dict
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, EnsureTyped
from monai.data import CacheDataset, DataLoader

def get_patient_data(base_dir: str) -> List[Dict[str, str]]:
    """
    Generate a list of dictionaries with paths to modalities and labels for each patient.

    Args:
        base_dir (str): Path to the dataset directory.

    Returns:
        List[Dict[str, str]]: List of patient data dictionaries.
    """
    patients = sorted(glob(os.path.join(base_dir, "BraTS2021_*")))
    data_dicts = []

    for patient in patients:
        patient_id = os.path.basename(patient)  # Get folder name like "BraTS2021_00000"
        data_dicts.append({
            "FLAIR": os.path.join(patient, f"{patient_id}_flair.nii.gz"),
            "T1": os.path.join(patient, f"{patient_id}_t1.nii.gz"),
            "T1CE": os.path.join(patient, f"{patient_id}_t1ce.nii.gz"),
            "T2": os.path.join(patient, f"{patient_id}_t2.nii.gz"),
            "label": os.path.join(patient, f"{patient_id}_seg.nii.gz"),
        })

    return data_dicts


def get_dataloader(
    data_dicts: List[Dict[str, str]],
    batch_size: int,
    cache_rate: float = 0.5,
    num_workers: int = 4,
    shuffle: bool = True,
):
    """
    Create a DataLoader for the BraTS dataset.

    Args:
        data_dicts (List[Dict[str, str]]): List of patient data dictionaries.
        batch_size (int): Batch size for DataLoader.
        cache_rate (float): Percentage of data to cache in memory.
        num_workers (int): Number of workers for data loading.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    transforms = Compose([
        LoadImaged(keys=["FLAIR", "T1", "T1CE", "T2", "label"]),
        EnsureChannelFirstd(keys=["FLAIR", "T1", "T1CE", "T2", "label"]),
        ScaleIntensityd(keys=["FLAIR", "T1", "T1CE", "T2"]),
        EnsureTyped(keys=["FLAIR", "T1", "T1CE", "T2", "label"]),
    ])

    dataset = CacheDataset(
        data=data_dicts,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
