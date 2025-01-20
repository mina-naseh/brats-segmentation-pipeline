#!/bin/bash -l

#SBATCH --job-name=brats_train       
#SBATCH --output=logs/%x_%j.out      
#SBATCH --error=logs/%x_%j.err       
          
#SBATCH --cpus-per-task=1            
#SBATCH --mem=24G                    
#SBATCH --time=00:30:00             


module load lang/Python

# Activate the virtual environment
source /home/users/mnaseh/brats-segmentation-pipeline/.venv/bin/activate

# Change to the directory where the code is located
cd /home/users/mnaseh/brats-segmentation-pipeline/task4/brats_segmentation

python3 train.py
