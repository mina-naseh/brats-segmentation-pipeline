# import torch
# from dataloader import get_dataloaders

# def test_dataloader():
#     # Parameters for testing
#     split_dir = "./splits/split1"
#     roi_size = [128, 128, 128]  # Region of interest size
#     batch_size = 2  # Batch size for testing
#     num_workers = 4  # Number of workers for data loading

#     # Get the data loaders
#     train_loader, val_loader, test_loader = get_dataloaders(
#         split_dir=split_dir, roi_size=roi_size, batch_size=batch_size, num_workers=num_workers
#     )

#     # Test the training data loader
#     print("Testing the training DataLoader...")
#     for batch in train_loader:
#         print(f"Train - T1 shape: {batch['t1'].shape}")
#         print(f"Train - T1CE shape: {batch['t1ce'].shape}")
#         print(f"Train - T2 shape: {batch['t2'].shape}")
#         print(f"Train - FLAIR shape: {batch['flair'].shape}")
#         print(f"Train - Label shape: {batch['label'].shape}")
#         break  # Inspect the first batch only

#     # Test the validation data loader
#     print("Testing the validation DataLoader...")
#     for batch in val_loader:
#         print(f"Validation - T1 shape: {batch['t1'].shape}")
#         print(f"Validation - T1CE shape: {batch['t1ce'].shape}")
#         print(f"Validation - T2 shape: {batch['t2'].shape}")
#         print(f"Validation - FLAIR shape: {batch['flair'].shape}")
#         print(f"Validation - Label shape: {batch['label'].shape}")
#         break  # Inspect the first batch only

#     # Test the testing data loader
#     print("Testing the test DataLoader...")
#     for batch in test_loader:
#         print(f"Test - T1 shape: {batch['t1'].shape}")
#         print(f"Test - T1CE shape: {batch['t1ce'].shape}")
#         print(f"Test - T2 shape: {batch['t2'].shape}")
#         print(f"Test - FLAIR shape: {batch['flair'].shape}")
#         print(f"Test - Label shape: {batch['label'].shape}")
#         break  # Inspect the first batch only

# if __name__ == "__main__":
#     test_dataloader()


# from dataloader import get_dataloaders
# import torch

# def verify_label_format():
#     # Parameters for testing
#     split_dir = "./splits/split1"
#     roi_size = [128, 128, 128]
#     batch_size = 2
#     num_workers = 4

#     # Get data loaders
#     train_loader, val_loader, test_loader = get_dataloaders(
#         split_dir=split_dir, roi_size=roi_size, batch_size=batch_size, num_workers=num_workers
#     )

#     # Test the training DataLoader
#     print("Verifying labels in the training DataLoader...")
#     for batch in train_loader:
#         labels = batch["label"]
#         print(f"Label shape: {labels.shape}")
#         print(f"Unique values in label tensor: {torch.unique(labels)}")
#         print(f"Label example (first sample): \n{labels[0, :, :, :, 0]}")  # Slicing for a quick look
#         break

#     # Test the validation DataLoader
#     print("Verifying labels in the validation DataLoader...")
#     for batch in val_loader:
#         labels = batch["label"]
#         print(f"Label shape: {labels.shape}")
#         print(f"Unique values in label tensor: {torch.unique(labels)}")
#         print(f"Label example (first sample): \n{labels[0, :, :, :, 0]}")
#         break

#     # Test the test DataLoader
#     print("Verifying labels in the test DataLoader...")
#     for batch in test_loader:
#         labels = batch["label"]
#         print(f"Label shape: {labels.shape}")
#         print(f"Unique values in label tensor: {torch.unique(labels)}")
#         print(f"Label example (first sample): \n{labels[0, :, :, :, 0]}")
#         break

# if __name__ == "__main__":
#     verify_label_format()


import nibabel as nib
import numpy as np
import json
import os

def check_unique_values_in_test_labels(test_file_path):
    # Load the test.txt file
    with open(test_file_path, "r") as f:
        test_data = json.load(f)

    print(f"Checking unique values in {len(test_data)} label files...")

    # Iterate through all the label paths
    for i, entry in enumerate(test_data):
        label_path = entry["label"]

        # Check if the file exists
        if not os.path.exists(label_path):
            print(f"File not found: {label_path}")
            continue

        try:
            # Load the label file
            label_data = nib.load(label_path).get_fdata()

            # Get unique values
            unique_values = np.unique(label_data)
            print(f"File {i + 1}: {label_path}")
            print(f"  Unique values: {unique_values}")
        except Exception as e:
            print(f"Error processing {label_path}: {e}")

if __name__ == "__main__":
    test_file_path = "./splits/split1/test.txt"  # Path to your test.txt
    check_unique_values_in_test_labels(test_file_path)
