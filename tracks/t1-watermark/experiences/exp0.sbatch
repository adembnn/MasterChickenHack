#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=16
#SBATCH --partition=dc-gpu
#SBATCH --time=00:30:00
#SBATCH --job-name=t1x0
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
cd /p/scratch/training2666/destors1/MasterChickenHack/Hackathon_Watermark_Localisation_Task || exit 1
source .venv/bin/activate
srun python adem/exp0.py
