cd ~/eac/t3/t3-poids/code; export T3ME=~/eac/t3/t3-poids OMP_NUM_THREADS=4; PY=~/eac/t3/.venv/bin/python; DASH=~/eac/t3/t3-methodes/code/dash.py; C=~/eac/t3/t3-poids/candidates
ln -sf ~/eac/t3/t3-methodes/deltas/PG_T85.pt ../deltas/PG_T85.pt
$PY $DASH --fset all --rset none --tgt T85 --tag all_ --combo PG85+NEGhi4_all=PG_T85:1,ce_all_hi:-4 --alphas 1 --save PG85_NEGhi4_all
$PY chk.py $C/PG85_NEGhi4_all.pt
echo FINSUM
