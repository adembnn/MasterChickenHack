#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=t3m_r1
#SBATCH --output=/p/scratch/training2666/destors1/MasterChickenHack/unlearning/adem/t3-methodes/logs/r1_%j.log
export T3BASE=/p/scratch/training2666/destors1/MasterChickenHack/unlearning
export T3ME=$T3BASE/adem/t3-methodes
PY=$T3BASE/.venv/bin/python; cd $T3ME; C=$T3ME/code; export OMP_NUM_THREADS=8
if [ ! -f cache/held_mild_orig_logits.npy ]; then CUDA_VISIBLE_DEVICES=0 $PY $C/cache.py > logs/cache.log 2>&1; fi
cmp cache/split.pt cache/split_h200.pt && echo SPLIT_IDENTIQUE
T="$PY $C/train.py"
( export CUDA_VISIBLE_DEVICES=0; $T --name N0 --fset ab; $T --name N1 --fset ab --anchor mix --lam 10; $T --name N2 --fset ab --anchor mix --lam 30; $T --name NF0 --fset ab --tmode noflip ) > logs/r1_g0.txt 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; $T --name Xa --fset a --aset b --anchor real --lam 10; $T --name Xa_k2 --fset a --aset b --anchor real --lam 10 --blocks 2; $T --name Xa_nf --fset a --aset b --anchor real --lam 10 --tmode noflip; $T --name Xa_l30 --fset a --aset b --anchor real --lam 30 ) > logs/r1_g1.txt 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; $T --name Xb --fset b --aset a --anchor real --lam 10; $T --name Xb_k2 --fset b --aset a --anchor real --lam 10 --blocks 2; $T --name Xb_nf --fset b --aset a --anchor real --lam 10 --tmode noflip; $T --name Xb_l30 --fset b --aset a --anchor real --lam 30 ) > logs/r1_g2.txt 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; $T --name N0_k2 --fset ab --blocks 2; $T --name N0_k1 --fset ab --blocks 1; $T --name N0_k12 --fset ab --blocks 12 --lr 3e-5; $T --name N0_A92 --fset ab --A 0.92 ) > logs/r1_g3.txt 2>&1 &
wait; echo TRAIN_DONE
E="$PY $C/dash.py --fset ab --rset c --alphas 0.5,0.75,1"
CUDA_VISIBLE_DEVICES=0 $E --combo N0=N0 --combo N1=N1 --combo N2=N2 --combo NF0=NF0 > diag/r1_d0.md 2>&1 &
CUDA_VISIBLE_DEVICES=1 $E --combo X=Xa,Xb --combo X_k2=Xa_k2,Xb_k2 --combo X_nf=Xa_nf,Xb_nf --combo X_l30=Xa_l30,Xb_l30 > diag/r1_d1.md 2>&1 &
CUDA_VISIBLE_DEVICES=2 $E --combo N0_k2=N0_k2 --combo N0_k1=N0_k1 --combo N0_k12=N0_k12 --combo N0_A92=N0_A92 > diag/r1_d2.md 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY $C/dash.py --fset a --rset b --tag ab_ --combo Xa_seul=Xa --combo Xa_l30_seul=Xa_l30 > diag/r1_d3.md 2>&1 &
wait; echo ALL_DONE
