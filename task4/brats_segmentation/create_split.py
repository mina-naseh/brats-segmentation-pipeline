import os
import random
import json
import nibabel as nib
import numpy as np
from collections import defaultdict


def remap_labels(label_data):
    """
    Remap labels from [0, 1, 2, 4] to [0, 1, 2, 3].
    """
    remap_dict = {0: 0, 1: 1, 2: 2, 4: 3}
    remapped = np.zeros_like(label_data, dtype=np.uint8)
    for old_value, new_value in remap_dict.items():
        remapped[label_data == old_value] = new_value
    return remapped


# Helper function to generate file paths and count modalities/labels
def generate_file_list(files):
    file_list = []
    modality_counters = defaultdict(int)  # Counts for each modality
    label_value_counters = defaultdict(int)  # Counts for remapped label values

    for patient_dir in files:
        entry = {
            "t1": os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_t1.nii.gz"),
            "t1ce": os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_t1ce.nii.gz"),
            "t2": os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_t2.nii.gz"),
            "flair": os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_flair.nii.gz"),
            "label": os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_seg.nii.gz"),
        }

        # Validate paths and count modalities
        for key, path in entry.items():
            if os.path.exists(path):
                modality_counters[key] += 1
            else:
                raise FileNotFoundError(f"Missing file for {key}: {path}")

        # Validate label file format and count remapped unique values
        label_path = entry["label"]
        if os.path.exists(label_path):
            label_data = nib.load(label_path).get_fdata()
            remapped_label_data = remap_labels(label_data)  # Remap labels
            unique_values, counts = np.unique(remapped_label_data, return_counts=True)
            for value, count in zip(unique_values, counts):
                label_value_counters[int(value)] += count

        file_list.append(entry)

    return file_list, modality_counters, label_value_counters


def create_splits(data_dir, output_dir, train_ratio=0.7, val_ratio=0.2, seed=42, prefix="BraTS2021_"):
    # Set the random seed for reproducibility
    random.seed(seed)

    # List all patient directories
    patients = [os.path.join(data_dir, p) for p in os.listdir(data_dir) if p.startswith(prefix)]
    random.shuffle(patients)

    # Compute split sizes
    train_end = int(train_ratio * len(patients))
    val_end = train_end + int(val_ratio * len(patients))

    # Split into train, val, test
    train_files = patients[:train_end]
    val_files = patients[train_end:val_end]
    test_files = patients[val_end:]
    print(f"Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")

    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate splits and count modalities/label values
    train_data, train_modality_counts, train_label_counts = generate_file_list(train_files)
    val_data, val_modality_counts, val_label_counts = generate_file_list(val_files)
    test_data, test_modality_counts, test_label_counts = generate_file_list(test_files)

    # Save splits
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        json.dump(train_data, f, indent=2)
        print(f"Saved {len(train_data)} entries to train.txt")
        print(f"Train Modality Counts: {dict(train_modality_counts)}")
        print(f"Train Label Value Counts: {dict(train_label_counts)}")

    with open(os.path.join(output_dir, "validation.txt"), "w") as f:
        json.dump(val_data, f, indent=2)
        print(f"Saved {len(val_data)} entries to validation.txt")
        print(f"Validation Modality Counts: {dict(val_modality_counts)}")
        print(f"Validation Label Value Counts: {dict(val_label_counts)}")

    with open(os.path.join(output_dir, "test.txt"), "w") as f:
        json.dump(test_data, f, indent=2)
        print(f"Saved {len(test_data)} entries to test.txt")
        print(f"Test Modality Counts: {dict(test_modality_counts)}")
        print(f"Test Label Value Counts: {dict(test_label_counts)}")


if __name__ == "__main__":
    create_splits("/work/projects/ai_imaging_class/dataset", "./splits/split1")
