#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=96
#SBATCH --partition=dc-gpu
#SBATCH --time=02:00:00
#SBATCH --job-name=t1seg2
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
T=/p/scratch/training2666/destors1/MasterChickenHack
cd $T/Hackathon_Watermark_Localisation_Task || exit 1
export HF_HOME=$T/.cache/huggingface TORCH_HOME=$T/.cache/torch HF_HUB_OFFLINE=1
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python adem/seg2.py --method pixelseal --arch dec --n 6000 --epochs 14 --tag v2dec &
CUDA_VISIBLE_DEVICES=1 python adem/seg2.py --method trustmark --arch dec --n 6000 --epochs 14 --tag v2dec &
CUDA_VISIBLE_DEVICES=2 python adem/seg2.py --method stegastamp --arch unet --n 6000 --epochs 14 --tag v2unet &
CUDA_VISIBLE_DEVICES=3 python adem/seg2.py --method trustmark --arch unet --n 6000 --epochs 14 --tag v2unet &
wait
