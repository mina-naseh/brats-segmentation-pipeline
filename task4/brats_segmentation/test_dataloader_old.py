import time
from brats_segmentation import get_dataloader


def log(message):
    """
    Simple logging function to print messages with timestamps.
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def test_dataloader():
    base_path = (
        "/work/projects/ai_imaging_class/dataset"  # Adjust to your dataset location
    )
    limit = 20  # Use 20 patients for now

    # Log the start of the test
    log("Starting dataloader test...")

    # Step 1: Get patient directories
    # log(f"Reading patient directories from {base_path}")
    # patient_dirs = get_patient_data(base_path, limit=limit)
    # log(f"Found {len(patient_dirs)} patient directories (limit = {limit})")

    # Step 2: Initialize DataLoader
    batch_size = 2
    dataloader = get_dataloader(
        base_path, batch_size=batch_size, shuffle=False, num_workers=0, limit=limit
    )
    log(
        f"Initialized DataLoader with batch_size={batch_size}, shuffle=False, num_workers=0"
    )

    # Step 3: Iterate through DataLoader
    log("Starting to load batches...")
    for i, batch in enumerate(dataloader):
        log(f"Loading batch {i + 1}")
        for modality in ["t1", "t1ce", "t2", "flair"]:
            assert modality in batch, f"Missing modality: {modality}"
            log(f"  Modality '{modality}' shape: {batch[modality].shape}")
        assert "label" in batch, "Missing label"
        log(f"  Label shape: {batch['label'].shape}")
        log(f"Batch {i + 1} loaded successfully")
        break  # Stop after loading the first batch to keep the test quick

    # Log the end of the test
    log("Dataloader test completed successfully.")


if __name__ == "__main__":
    test_dataloader()
