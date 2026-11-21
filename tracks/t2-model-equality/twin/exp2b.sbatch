#!/bin/bash
#SBATCH --job-name=t2exp2b
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --output=%x-%j.out
source ~/.bashrc 2>/dev/null
cd /p/scratch/training2666/destors1/MasterChickenHack/t2-twin
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
python exp2b_power.py --n 300 --max_new 96 --bs 16 --exp2_outputs results_exp2_a100_outputs.json --out results_exp2b_a100.json
