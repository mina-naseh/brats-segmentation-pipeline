#!/bin/bash -l

#SBATCH --job-name=brats_train       
#SBATCH --output=logs/%x_%j.out      
#SBATCH --error=logs/%x_%j.err       
#SBATCH --partition=gpu              
#SBATCH --gres=gpu:1                 
#SBATCH --cpus-per-task=2            
#SBATCH --mem=4G                    
#SBATCH --time=00:30:00             


module load cuda/11.7.0              
module load python/3.10            

source /home/users/mnaseh/brats-segmentation-pipeline/.venv/bin/activate

cd /home/users/mnaseh/brats-segmentation-pipeline/task4/brats_segmentation

python3 train.py
