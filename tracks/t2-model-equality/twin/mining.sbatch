#!/bin/bash
#SBATCH --job-name=t2mining
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
source ~/.bashrc 2>/dev/null
cd /p/scratch/training2666/destors1/MasterChickenHack/t2-twin
source .venv/bin/activate
export HF_HUB_OFFLINE=1
echo "host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
python prompt_mining.py --n_pool 2000 --k_margin 300 --n_stability 150 --m_final 100 \
  --max_new 20 --bs 32 --dtype bfloat16 --lora_steps 150 --n_temp_samples 5 \
  --out discriminating_prompts.json
