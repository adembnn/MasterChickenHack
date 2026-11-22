#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=t3m_r2
#SBATCH --output=/p/scratch/training2666/destors1/MasterChickenHack/unlearning/adem/t3-methodes/logs/r2_%j.log
export T3BASE=/p/scratch/training2666/destors1/MasterChickenHack/unlearning
export T3ME=$T3BASE/adem/t3-methodes
PY=$T3BASE/.venv/bin/python; cd $T3ME; C=$T3ME/code; export OMP_NUM_THREADS=8
M="$PY $C/mk_tgt.py"; P85=0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise; P92=0.2:crop,0.4:noise,0.4:blur,0.2:color,0.2:mask; P70=0.6:crop,0.4:mask,0.6:noise,0.6:color,0.8:blur
( $M T85 --pool $P85 --f 0.15; $M T85_f0 --pool $P85 --f 0; $M T85_f08 --pool $P85 --f 0.08; $M T85_f25 --pool $P85 --f 0.25; $M T85_l13 --pool $P85 --f 0.15 --llo 1 --lhi 3
  $M T85_l48 --pool $P85 --f 0.15 --llo 4 --lhi 8; $M T85_soft6 --pool $P85 --f 0.15 --soft 0.6; $M T85_s1 --pool $P85 --f 0.15 --rank s1; $M T92 --pool $P92 --f 0.08 --llo 2 --lhi 5
  $M T70 --pool $P70 --f 0.30 --llo 2 --lhi 7; $M T70s_f15 --pool $P70 --f 0.15 ) > logs/r2_tgt.txt 2>&1
T="$PY $C/train.py --anchor mix --steps 1200"; D="$PY $C/dash.py"
tr() { n=$1; t=$2; shift 2; $T --name $n --tgt $t --fset all --lam 3 "$@" && $D --fset all --tgt $t --tag all_ --combo $n=$n --save $n | grep "^| $n" > diag/row_$n.md; }
sp() { n=$1; t=$2; shift 2; $T --name $n --tgt $t --fset ab --lam 3 "$@" && $D --fset ab --rset c --tgt $t --tag ab_ --combo $n=$n | grep "^| $n" > diag/row_$n.md; }
( export CUDA_VISIBLE_DEVICES=0; tr PG_T92 T92; tr PG_T85_f0 T85_f0; tr PG_T85_f08 T85_f08; tr PG_T85_f25 T85_f25 ) > logs/r2_g0.txt 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; tr PG_T85_h T85 --steps 2400 --lr 1.5e-4 --w1 2; tr PG_T85_l48 T85_l48; tr PG_T85_l13 T85_l13; tr PG_T85_s1 T85_s1 ) > logs/r2_g1.txt 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; tr PG_T70s_f15 T70s_f15; tr PG_T85_soft6 T85_soft6; tr PG_T85_lam10 T85 --lam 10; tr PG_T85_k2 T85 --blocks 2; tr PG_T70 T70 ) > logs/r2_g2.txt 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; sp PGab_T85_h T85 --steps 2400 --lr 1.5e-4 --w1 2; sp PGab_T70 T70; sp PGab_T85_k2 T85 --blocks 2; sp PGab_T85_lam10 T85 --lam 10; sp PGab_T85_f0 T85_f0
  $D --fset a --rset c --tag ac_ --combo Xa_reserve=Xa | grep "^| Xa" > diag/row_Xa_reserve.md ) > logs/r2_g3.txt 2>&1 &
wait; echo ALL_DONE
