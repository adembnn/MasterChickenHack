#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --output=output/%j.out

# Adapter le nom du script ci-dessous (sample_submission.py, submission.py ...)
cd "$PROJECT/$USER/Hackathon"
source .venv/bin/activate
srun python sample_submission.py
