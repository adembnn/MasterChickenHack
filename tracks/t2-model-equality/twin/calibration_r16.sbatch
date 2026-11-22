#!/bin/bash
#SBATCH --job-name=t2calib16
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
# Variante d'architecture LoRA : rang 16, alpha 32, q/k/v/o, autres prompts de test
python calibration.py --n 200 --max_new 64 --bs 16 --lora_steps 150 --rank 16 --alpha 32 \
  --targets q_proj,k_proj,v_proj,o_proj --test_seed 7 --out calib_results_r16.json
