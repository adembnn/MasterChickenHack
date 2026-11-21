#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=t3v1
#SBATCH --output=/p/scratch/training2666/destors1/MasterChickenHack/unlearning/adem/t3v1_%j.log
D=/p/scratch/training2666/destors1/MasterChickenHack/unlearning
PY=$D/.venv/bin/python; cd $D/adem
S="t3v1"; LR=5e-5; ST=800
nvidia-smi --query-gpu=index,name --format=csv
# --- entrainement : un lot de configs par GPU ---
( export CUDA_VISIBLE_DEVICES=0; for t in 1.5 2 3; do $PY $S/train_s0.py $t 10; done; $PY $S/train_s1.py S1_A85_bc 0.85 $LR $ST 10 1.0 4 0.15 z_bc ) > log_g0.txt 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; $PY $S/train_s1.py S1_A75 0.75 $LR $ST 10; $PY $S/train_s1.py S1_A75_l30 0.75 $LR $ST 30 ) > log_g1.txt 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; $PY $S/train_s1.py S1_A85 0.85 $LR $ST 10; $PY $S/train_s1.py S1_A85_l30 0.85 $LR $ST 30 ) > log_g2.txt 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; $PY $S/train_s1.py S1_A92 0.92 $LR $ST 10; $PY $S/train_s1.py S1_A92_l30 0.92 $LR $ST 30 ) > log_g3.txt 2>&1 &
wait
echo TRAIN_DONE
# --- diagnostics : orig + tous les candidats, repartis sur les 4 GPU ---
i=0
for n in orig S0_tau1.5 S0_tau2 S0_tau3 S1_A75 S1_A75_l30 S1_A85 S1_A85_l30 S1_A92 S1_A92_l30 S1_A85_bc; do
  p=$D/adem/candidates/$n.pt; [ $n = orig ] && p=$D/model.pt
  CUDA_VISIBLE_DEVICES=$((i % 4)) $PY $S/diag.py $n $p > log_diag_$n.txt 2>&1 &
  i=$((i+1)); if [ $((i % 4)) = 0 ]; then wait; fi
done
wait
echo ALL_DONE
