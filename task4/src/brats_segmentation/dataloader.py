import os
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np

class BraTSDataset(Dataset):
    """
    Custom PyTorch Dataset for BraTS data.
    """
    def __init__(self, patient_dirs, transform=None):
        """
        Args:
            patient_dirs (list): List of patient directories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.patient_dirs = patient_dirs
        self.transform = transform

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        """
        Load data and labels for a given patient.
        """
        patient_dir = self.patient_dirs[idx]
        sample = {}

        # Load the four MRI modalities
        modalities = ['t1', 't1ce', 't2', 'flair']
        for modality in modalities:
            file_path = os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_{modality}.nii.gz")
            sample[modality] = nib.load(file_path).get_fdata()

        # Load the segmentation label
        label_path = os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_seg.nii.gz")
        sample['label'] = nib.load(label_path).get_fdata()

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_patient_data(base_path, limit=20):
    """
    List patient directories.

    Args:
        base_path (str): Path to the dataset directory.
        limit (int): Number of patients to load (default is 20).

    Returns:
        list: List of patient directories.
    """
    all_patients = [os.path.join(base_path, patient) for patient in os.listdir(base_path) if patient.startswith("BraTS")]
    return all_patients[:limit]  # Limit to 20 patients for now


def get_dataloader(patient_dirs, batch_size=2, shuffle=True, num_workers=0, transform=None):
    """
    Create a PyTorch DataLoader for the BraTS dataset.

    Args:
        patient_dirs (list): List of patient directories.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker threads for data loading.
        transform (callable, optional): Data transformations.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    dataset = BraTSDataset(patient_dirs, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
