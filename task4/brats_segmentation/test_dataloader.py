# from dataloader import ConvertToMultiChannelBasedOnBratsClassesd
# import os
# import torch


# # Example label tensor
# example_label = torch.tensor([
#     [[0, 1, 2, 4],
#      [0, 1, 0, 0],
#      [4, 2, 1, 0]]
# ]).unsqueeze(0)  # Shape [1, H, W]

# # Apply the transform
# transform = ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"])
# data = {"label": example_label}
# transformed = transform(data)

# # Print results
# print("Original Label:")
# print(example_label)
# print("Transformed Label:")
# for i, channel in enumerate(transformed["label"]):
#     print(f"Channel {i} (unique values): {torch.unique(channel)}")
#     print(channel)


# def test_transform():
#     example_label = torch.tensor([
#         [[0, 1, 2, 4],
#          [0, 1, 0, 0],
#          [4, 2, 1, 0]]
#     ]).unsqueeze(0)  # Shape: [1, H, W]

#     transform = ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"])
#     data = {"label": example_label}
#     transformed = transform(data)

#     # Extract channels and squeeze singleton dimensions
#     tc = transformed["label"][0].squeeze()  # Tumor Core
#     wt = transformed["label"][1].squeeze()  # Whole Tumor
#     et = transformed["label"][2].squeeze()  # Enhancing Tumor

#     # Assertions with squeezed expected tensors
#     assert torch.equal(tc, torch.tensor([
#         [0, 1, 0, 1],
#         [0, 1, 0, 0],
#         [1, 0, 1, 0]
#     ]).float()), "Tumor Core (TC) is incorrect"

#     assert torch.equal(wt, torch.tensor([
#         [0, 1, 1, 1],
#         [0, 1, 0, 0],
#         [1, 1, 1, 0]
#     ]).float()), "Whole Tumor (WT) is incorrect"

#     assert torch.equal(et, torch.tensor([
#         [0, 0, 0, 1],
#         [0, 0, 0, 0],
#         [1, 0, 0, 0]
#     ]).float()), "Enhancing Tumor (ET) is incorrect"

#     print("All tests passed!")

# # Run the test
# test_transform()




# def test_transform_edge_cases():
#     # Case 1: All zeros
#     label_zeros = torch.zeros((1, 3, 3)).unsqueeze(0)
#     transformed = ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"])({"label": label_zeros})
#     for channel in transformed["label"]:
#         assert torch.equal(channel, torch.zeros_like(channel)), "Failed on all-zero input"

#     # Case 2: Single label (1)
#     label_single_1 = torch.ones((1, 3, 3)).unsqueeze(0)
#     transformed = ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"])({"label": label_single_1})
#     assert torch.equal(transformed["label"][0], label_single_1.float()), "TC incorrect for single-label (1)"
#     assert torch.equal(transformed["label"][1], label_single_1.float()), "WT incorrect for single-label (1)"
#     assert torch.equal(transformed["label"][2], torch.zeros_like(label_single_1)), "ET incorrect for single-label (1)"

#     # Case 3: Random large tensor
#     random_labels = torch.randint(0, 5, (1, 256, 256)).unsqueeze(0)
#     transformed = ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"])({"label": random_labels})
#     print("Edge case tests passed successfully!")

# # Run edge case tests
# test_transform_edge_cases()

import os
from dataloader import get_dataloaders, get_transforms
from monai.data import CacheDataset, DataLoader
from matplotlib import pyplot as plt
import json

import os

if __name__ == "__main__":
    # Define paths, ROI size, batch size, and number of workers
    split_dir = "./splits/split3"
    roi_size = (128, 128, 128)
    batch_size = 1
    num_workers = 4

    # Create data loaders
    train_loader, val_loader, test_loader = get_dataloaders(split_dir, roi_size, batch_size, num_workers)

    # Visualize one example from validation dataset
    val_transform = get_transforms(roi_size, augment=False)
    val_ds = CacheDataset(data=json.load(open(f"{split_dir}/validation.txt")), 
                          transform=val_transform, 
                          cache_rate=1.0, 
                          num_workers=num_workers)

    # Pick one example for visualization
    val_data_example = val_ds[2]
    print(f"Image shape: {val_data_example['image'].shape}")  # Expected: [4, H, W, D]
    print(f"Label shape: {val_data_example['label'].shape}")  # Expected: [3, H, W, D]

    # Create output folder
    output_folder = "visualization_nonzero_slices"
    os.makedirs(output_folder, exist_ok=True)

    # Find slices with non-zero labels
    non_zero_slices = []
    for slice_idx in range(val_data_example["label"].shape[-1]):  # Iterate over depth (D-axis)
        if val_data_example["label"][:, :, :, slice_idx].sum() > 0:
            non_zero_slices.append(slice_idx)
    print(f"Non-zero label slices: {non_zero_slices}")

    # Iterate over non-zero slices and save visualizations
    for slice_idx in non_zero_slices:
        # Save the image channels for this slice
        plt.figure("Image", (24, 6))
        for i in range(4):  # Assuming 4 channels for T1, T1ce, T2, FLAIR
            plt.subplot(1, 4, i + 1)
            plt.title(f"Image Channel {i}")
            plt.imshow(val_data_example["image"][i, :, :, slice_idx].detach().cpu(), cmap="gray")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/image_slice_{slice_idx:03d}.png")  # Save the image channels plot
        plt.close()  # Close the figure to free memory

        # Save the label channels for this slice
        plt.figure("Label", (18, 6))
        for i in range(3):  # Assuming 3 labels: TC, WT, ET
            plt.subplot(1, 3, i + 1)
            plt.title(f"Label Channel {i}")
            # Use .squeeze() to remove singleton dimension
            plt.imshow(val_data_example["label"][i, :, :, slice_idx].squeeze().detach().cpu(), cmap="viridis")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/label_slice_{slice_idx:03d}.png")  # Save the label channels plot
        plt.close()  # Close the figure to free memory

    print(f"All non-zero label slices have been visualized and saved to the folder: {output_folder}")
