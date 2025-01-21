#!/bin/bash -l

#SBATCH --job-name=brats_train       
#SBATCH --output=logs/%x_%j.out      
#SBATCH --error=logs/%x_%j.err          
#SBATCH --cpus-per-task=1            
#SBATCH -G 1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH -p gpu


# module load lang/Python

# Activate the virtual environment
# source /home/users/$USER/brats-segmentation-pipeline/.venv/bin/activate
conda activate imaging
python -c "import sys; print(sys.executable)"

# Change to the directory where the code is located
# cd /home/users/(mnaseh)/brats-segmentation-pipeline/task4/brats_segmentation
python3 train.py
