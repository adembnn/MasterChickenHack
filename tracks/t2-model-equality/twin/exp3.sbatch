#!/bin/bash
#SBATCH --job-name=t2exp3
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
source ~/.bashrc 2>/dev/null
cd /p/scratch/training2666/destors1/MasterChickenHack/t2-twin
source .venv/bin/activate
export HF_HUB_OFFLINE=1
echo "host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES=0 python exp3_margin.py --model allenai/OLMo-2-0425-1B-Instruct --foreign Qwen/Qwen2.5-1.5B-Instruct --n 300 --noise_bs 1,8,32 --a100_outputs results_exp2_h200_outputs.json --out results_exp3_olmo_a100.json > exp3_olmo_a100.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python exp3_margin.py --model Qwen/Qwen2.5-1.5B-Instruct --foreign allenai/OLMo-2-0425-1B-Instruct --n 300 --noise_bs 1,8,32 --out results_exp3_qwen_a100.json > exp3_qwen_a100.log 2>&1 &
wait
echo done
