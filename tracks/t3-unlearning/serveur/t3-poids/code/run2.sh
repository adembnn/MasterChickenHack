cd ~/eac/t3/t3-poids/code; PY=~/eac/t3/.venv/bin/python; export T3ME=~/eac/t3/t3-poids OMP_NUM_THREADS=4; DASH=~/eac/t3/t3-methodes/code/dash.py
$PY tv.py --name tvE_ce_all --fset all --mode exact --loss ce --opt adamw --lr 2e-5 --epochs 3
$PY $DASH --fset all --rset none --tgt T85 --tag all_ --combo NEGce_all=tvE_ce_all:-1 --alphas 1,2 --save NEGce_all_a2
echo FIN2a
$PY mkdelta.py orth tvE_mg tvA_mg mgO
$PY mkdelta.py blocks tvE_ce -1 6 ce_lo; $PY mkdelta.py blocks tvE_ce 6 13 ce_hi; $PY mkdelta.py blocks tvE_ce 9 13 ce_top
$PY $DASH --fset ab --rset c --tgt T85 --combo NEGce=tvE_ce:-1 --alphas 3
$PY $DASH --fset ab --rset c --tgt T85 --combo NEGmgO=mgO:-1 --combo DIFFmg=tvE_mg:-1,tvA_mg:1 --alphas 0.5,1,2
$PY $DASH --fset ab --rset c --tgt T85 --combo NEGce_lo=ce_lo:-1 --combo NEGce_hi=ce_hi:-1 --combo NEGce_top=ce_top:-1 --alphas 2,4
echo FIN2b
$PY loc.py ablate
$PY loc.py fisher
echo FIN2c
