import glob
import os
from typing import Literal
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.widgets import Slider
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from monai.transforms import Compose, RandFlip, RandRotate, ScaleIntensity


class BraTSDataset(Dataset):
    """
    Custom PyTorch Dataset for BraTS data.
    """

    def __init__(self, base_path: str | os.PathLike, transform=None, limit=None):
        """
        Args:
            base_path (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        all_patients = list(glob.glob(os.path.join(base_path, "BraTS*")))
        if limit:
            self.patient_dirs = all_patients[:limit]  # Limit to 20 patients for now
        self.patient_dirs = all_patients

        self.transform = transform

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(
        self, idx
    ) -> dict[Literal["t1", "t1ce", "t2", "flair", "label"], np.ndarray]:
        """
        Load data and labels for a given patient.
        """
        patient_dir = self.patient_dirs[idx]
        sample = {}

        # Load the four MRI modalities
        modalities = ["t1", "t1ce", "t2", "flair"]
        print(patient_dir)
        for modality in modalities:
            file_path = os.path.join(
                patient_dir, f"{os.path.basename(patient_dir)}_{modality}.nii.gz"
            )
            sample[modality] = nib.load(file_path).get_fdata()

        # Load the segmentation label
        label_path = os.path.join(
            patient_dir, f"{os.path.basename(patient_dir)}_seg.nii.gz"
        )
        sample["label"] = nib.load(label_path).get_fdata()

        if self.transform:
            for modality in modalities:
                sample[modality] = self.transform(sample[modality])

        return sample


def get_dataloader(
    base_path: str | os.PathLike,
    batch_size: int = 2,
    limit: int = None,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """
    Create a PyTorch DataLoader for the BraTS dataset.

    Args:
        base_path (str): Path to the dataset directory.
        batch_size (int): Number of samples per batch.
        limit (int, optional): Limit the number of patients to use.
        shuffle (bool, optional): Whether to shuffle the data.
        num_workers (int, optional): Number of workers to use for

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    dataset = BraTSDataset(base_path, limit=limit)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


if __name__ == "__main__":
    transform = Compose(
        [
            ScaleIntensity(),
            RandFlip(spatial_axis=[0, 1]),
            RandRotate(range_x=10, range_y=10, range_z=10),
        ]
    )
    # TODO: we have a BIG problems here:
    # 1. We can not apply same transformation to imags and labels (we may be able to use two different Compose)
    # 2. if we apply two transformations, how we can make sure that the same transformation is applied to both images and labels due to the random nature of the transformations!

    dataset = BraTSDataset("../../../dataset_sample/", transform=transform)

    label_color = {1: [0, 0, 1], 2: [0, 1, 0], 4: [1, 0, 0]}
    label_text = {1: "Necrotic", 2: "Edema", 4: "Enhancing"}

    sample = dataset[0]
    for i in sample:
        print(i, sample[i].shape, sample[i].dtype)

    image = sample["t1"]
    segmentation = sample["label"]
    initial_slice = image.shape[2] // 2
    print(np.unique(image), image.max(), image.min())

    # 3D visualization :')
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    img_display = ax.imshow(image[:, :, initial_slice], cmap="gray")
    ax.set_title(f"Slice {initial_slice}")

    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])  # [left, bottom, width, height]
    slider = Slider(
        ax_slider, "Slice", 0, image.shape[2] - 1, valinit=initial_slice, valstep=1
    )

    def update(_):
        slice_index = int(slider.val)
        img = image[:, :, slice_index]
        # img = (img / img.max() * 255).astype(np.uint8)
        seg = segmentation[:, :, slice_index]
        img = np.stack([img, img, img], axis=-1)

        for label, color in label_color.items():
            img[seg == label] = color

        img_display.set_data(img)
        ax.set_title(f"Slice {slice_index}")
        fig.canvas.draw_idle()

    cmap = ListedColormap(list(label_color.values()))
    ax.legend(
        handles=[Patch(color=cmap(i - 1), label=label_text[i]) for i in label_text],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    slider.on_changed(update)
    plt.show()
