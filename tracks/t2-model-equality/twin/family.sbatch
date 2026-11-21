#!/bin/bash
#SBATCH --job-name=t2family
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
source ~/.bashrc 2>/dev/null
cd /p/scratch/training2666/destors1/MasterChickenHack/t2-twin
source .venv/bin/activate
export HF_HUB_OFFLINE=1
echo "host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
python family_detection.py --n_open 100 --max_new 64 --max_new_closed 16 --bs 16 --dtype bfloat16 \
  --models olmo,qwen,smol --out results_family_a100.json
