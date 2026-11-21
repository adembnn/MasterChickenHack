#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=96
#SBATCH --partition=dc-gpu
#SBATCH --time=01:00:00
#SBATCH --job-name=t1x1
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
cd /p/scratch/training2666/destors1/MasterChickenHack/Hackathon_Watermark_Localisation_Task || exit 1
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python adem/exp1.py --method pixelseal &
CUDA_VISIBLE_DEVICES=1 python adem/exp1.py --method trustmark &
CUDA_VISIBLE_DEVICES=2 python adem/exp1.py --method stegastamp &
wait
