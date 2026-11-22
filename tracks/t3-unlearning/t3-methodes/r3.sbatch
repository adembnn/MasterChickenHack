#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=t3m_r3
#SBATCH --output=/p/scratch/training2666/destors1/MasterChickenHack/unlearning/adem/t3-methodes/logs/r3_%j.log
export T3BASE=/p/scratch/training2666/destors1/MasterChickenHack/unlearning
export T3ME=$T3BASE/adem/t3-methodes
PY=$T3BASE/.venv/bin/python; cd $T3ME; C=$T3ME/code; export OMP_NUM_THREADS=8
M="$PY $C/mk_tgt.py"; P85=0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise; P70=0.6:crop,0.4:mask,0.6:noise,0.6:color,0.8:blur; P48=0.8:crop,0.8:mask,0.8:noise,0.8:color,1.0:blur; P34=1.0:crop,1.0:mask,1.0:noise,1.0:color,1.0:blur
( $M T85_f20_l48 --pool $P85 --f 0.20 --llo 4 --lhi 8; $M T70_f30_l48 --pool $P70 --f 0.30 --llo 4 --lhi 8; $M T48_f50_l48 --pool $P48 --f 0.50 --llo 4 --lhi 8; $M T34_f70_l48 --pool $P34 --f 0.70 --llo 4 --lhi 8
  $M T48_g1 --pool $P48 --f 0 --gam 1; $M T34_g1 --pool $P34 --f 0 --gam 1 ) > logs/r3_tgt.txt 2>&1
T="$PY $C/train.py --steps 1500 --wce 1 --lam 3"; D="$PY $C/dash.py"
tr() { n=$1; t=$2; shift 2; $T --name $n --tgt $t --fset all "$@" && $D --fset all --tgt $t --tag all_ --combo $n=$n --save $n | grep "^| $n" > diag/row_$n.md; }
sp() { n=$1; t=$2; shift 2; $T --name $n --tgt $t --fset ab "$@" && $D --fset ab --rset c --tgt $t --tag ab_ --combo $n=$n | grep "^| $n" > diag/row_$n.md; }
DG="--anchor deg,mix --dlo 0.6 --dhi 1.0 --fviews halves"
( export CUDA_VISIBLE_DEVICES=0; sp Jab_f50_d5 T48_f50_l48 $DG --lamd 5; sp Jab_f50_d20 T48_f50_l48 $DG --lamd 20; tr J_f50_d5 T48_f50_l48 $DG --lamd 5; tr J_f50_d20 T48_f50_l48 $DG --lamd 20 ) > logs/r3_g0.txt 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; sp Jab_f30_d5 T70_f30_l48 $DG --lamd 5; sp Jab_f70_d10 T34_f70_l48 $DG --lamd 10; tr J_f30_d5 T70_f30_l48 $DG --lamd 5; tr J_f70_d10 T34_f70_l48 $DG --lamd 10 ) > logs/r3_g1.txt 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; sp Jab_f50_d5lo4 T48_f50_l48 --anchor deg,mix --dlo 0.4 --dhi 1.0 --fviews halves --lamd 5; sp Jab_g1_48_d5 T48_g1 $DG --lamd 5; tr J_g1_48_d5 T48_g1 $DG --lamd 5; tr J_g1_34_d5 T34_g1 $DG --lamd 5 ) > logs/r3_g2.txt 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; sp Jab_f20_mix T85_f20_l48 --anchor mix; sp Jab_f30_mix T70_f30_l48 --anchor mix; sp Jab_f20_d5 T85_f20_l48 $DG --lamd 5; tr J_f20_d5 T85_f20_l48 $DG --lamd 5 ) > logs/r3_g3.txt 2>&1 &
wait; echo ALL_DONE
