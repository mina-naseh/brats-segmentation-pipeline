from src.dataloader import get_patient_data, get_dataloader

def main():
    # Path to your dataset
    dataset_path = "/work/projects/ai_imaging_class/dataset"

    # Load patient data
    data_dicts = get_patient_data(dataset_path)

    # Print the number of patients and example paths
    print(f"Found {len(data_dicts)} patients.")
    print("Example patient:", data_dicts[0])

    # Create DataLoader
    dataloader = get_dataloader(data_dicts, batch_size=2, shuffle=False)

    # Iterate through the DataLoader
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("FLAIR shape:", batch["FLAIR"].shape)
        print("Label shape:", batch["label"].shape)
        break  # Test only the first batch

if __name__ == "__main__":
    main()
