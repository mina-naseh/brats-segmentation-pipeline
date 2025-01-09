#!/bin/bash -l
#SBATCH --job-name=test_dataloader   # Job name
#SBATCH --output=logs/test_dataloader.out  # Standard output log
#SBATCH --error=logs/test_dataloader.err   # Standard error log
#SBATCH --time=00:10:00              # Time limit (hh:mm:ss)
#SBATCH --partition=batch            # Use the 'batch' partitions
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                    # Memory per node

# Set PYTHONPATH to include src
export PYTHONPATH=$(pwd):$PYTHONPATH

# Activate your Python virtual environment
# source /home/users/mnaseh/brats-segmentation-pipeline/.venv/bin/activate

# Run the test
# python tests/test_dataloader.py
# alternatively, you can run this command (suggested, it will run all the tests in the tests directory, gives better output, and runs faster):
pytest
