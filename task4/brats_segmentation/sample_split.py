import os
import json
import random

def sample_split_files(input_dir, output_dir, fraction=0.1, seed=42):
    """
    Randomly sample a fraction of data from split files in input_dir and save to output_dir.
    
    Args:
        input_dir (str): Path to the source split folder (e.g., split1).
        output_dir (str): Path to the target split folder (e.g., split2).
        fraction (float): Fraction of the data to sample (e.g., 0.1 for 1/10th).
        seed (int): Random seed for reproducibility.
    """
    # Set the random seed
    random.seed(seed)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each split file
    for split_file in ["train.txt", "validation.txt", "test.txt"]:
        input_path = os.path.join(input_dir, split_file)
        output_path = os.path.join(output_dir, split_file)

        # Load the data
        with open(input_path, "r") as f:
            data = json.load(f)

        # Sample the data
        sampled_data = random.sample(data, int(len(data) * fraction))

        # Save the sampled data to the output directory
        with open(output_path, "w") as f:
            json.dump(sampled_data, f, indent=2)

        print(f"Sampled {len(sampled_data)} entries from {split_file} and saved to {output_path}")

if __name__ == "__main__":
    input_split_dir = "./splits/split1"
    output_split_dir = "./splits/split2"
    sample_fraction = 0.1  # Choose 1/10th of the data

    sample_split_files(input_split_dir, output_split_dir, fraction=sample_fraction)
