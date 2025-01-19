#!/bin/bash -l

#SBATCH --job-name=brats_train       
#SBATCH --output=logs/%x_%j.out      
#SBATCH --error=logs/%x_%j.err       
          
#SBATCH --cpus-per-task=1            
#SBATCH --mem=4G                    
#SBATCH --time=00:15:00             


module load lang/Python

source /home/users/mnaseh/brats-segmentation-pipeline/.venv/bin/activate

cd /home/users/mnaseh/brats-segmentation-pipeline/task4/brats_segmentation

python3 train.py
