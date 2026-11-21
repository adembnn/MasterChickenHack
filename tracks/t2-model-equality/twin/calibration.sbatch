#!/bin/bash
#SBATCH --job-name=t2calib
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
source ~/.bashrc 2>/dev/null
cd /p/scratch/training2666/destors1/MasterChickenHack/t2-twin
source .venv/bin/activate
export HF_HUB_OFFLINE=1
echo "host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
python calibration.py --n 200 --max_new 64 --bs 16 --lora_steps 150 --out calib_results.json
