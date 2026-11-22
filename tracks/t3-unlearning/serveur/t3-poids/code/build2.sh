cd ~/eac/t3/t3-poids/code; export T3ME=~/eac/t3/t3-poids OMP_NUM_THREADS=4; PY=~/eac/t3/.venv/bin/python; DASH=~/eac/t3/t3-methodes/code/dash.py; C=~/eac/t3/t3-poids/candidates
until ! pgrep -u adem -f "sim_neg.py" > /dev/null; do sleep 5; done
$PY mkdelta.py blocks tvE_ce_all 8 13 ce_all_t8
$PY $DASH --fset all --rset none --tgt T85 --tag all_ --combo NEGce_all_hi=ce_all_hi:-1 --alphas 6 --save NEGce_all_hi_a6
$PY $DASH --fset all --rset none --tgt T85 --tag all_ --combo NEGce_all_t8=ce_all_t8:-1 --alphas 8 --save NEGce_all_t8_a8
$PY $DASH --fset all --rset none --tgt T85 --tag all_ --combo NEGce_all_hi=ce_all_hi:-1 --alphas 8 --save NEGce_all_hi_a8
$PY $DASH --fset all --rset none --tgt T85 --tag all_ --combo NEGce_all_top=ce_all_top:-1 --alphas 12 --save NEGce_all_top_a12
$PY chk.py $C/NEGce_all_hi_a6.pt $C/NEGce_all_t8_a8.pt $C/NEGce_all_hi_a8.pt $C/NEGce_all_top_a12.pt
echo FINB2
