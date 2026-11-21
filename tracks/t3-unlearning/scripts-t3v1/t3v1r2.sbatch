#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --time=00:50:00
#SBATCH --job-name=t3v1r2
#SBATCH --output=/p/scratch/training2666/destors1/MasterChickenHack/unlearning/adem/t3v1r2_%j.log
D=/p/scratch/training2666/destors1/MasterChickenHack/unlearning
PY=$D/.venv/bin/python; cd $D/adem; S=t3v1
declare -a N=(S1_r2_A85_l30_lr1e-4 S1_r2_A85_l10_lr1e-4 S1_r2_A85_l30_lr5e-5 S1_r2_A92_l30_lr1e-4)
declare -a C=("0.85 1e-4 3000 30" "0.85 1e-4 3000 10" "0.85 5e-5 3000 30" "0.92 1e-4 3000 30")
for i in 0 1 2 3; do ( CUDA_VISIBLE_DEVICES=$i $PY $S/train_s1.py ${N[$i]} ${C[$i]} > log_r2_$i.txt 2>&1; CUDA_VISIBLE_DEVICES=$i $PY $S/diag.py ${N[$i]} $D/adem/candidates/${N[$i]}.pt > log_diag_${N[$i]}.txt 2>&1 ) & done
wait; echo ALL_DONE
