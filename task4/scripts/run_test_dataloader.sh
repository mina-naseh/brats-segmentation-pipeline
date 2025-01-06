#!/bin/bash -l
#SBATCH --job-name=test_dataloader   # Job name
#SBATCH --output=logs/test_dataloader.out  # Standard output log
#SBATCH --error=logs/test_dataloader.err   # Standard error log
#SBATCH --time=00:10:00              # Time limit (hh:mm:ss)
#SBATCH --partition=batch            # Use the 'batch' partitions
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                    # Memory per node

# # Load required modules
# module load python/3.8

# Activate your Python virtual environment
source ../.venv/bin/activate

# Navigate to the project directory and run the test
python tests/test_dataloader.py
