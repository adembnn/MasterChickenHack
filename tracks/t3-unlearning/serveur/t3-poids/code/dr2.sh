cd ~/eac/t3/t3-poids/code; export T3ME=~/eac/t3/t3-poids OMP_NUM_THREADS=4; PY=~/eac/t3/.venv/bin/python; DASH=~/eac/t3/t3-methodes/code/dash.py; C=~/eac/t3/t3-poids/candidates
$PY $DASH --fset ab --rset c --tgt T85 --tag r12_ --sd npoR2_m0.5=$C/ab_npoR2_m0.5.pt --sd npoR2_m0.8=$C/ab_npoR2_m0.8.pt --sd npoR2_m1.2=$C/ab_npoR2_m1.2.pt --sd npoR2_fin=$C/ab_npoR2.pt
echo FINDR2
