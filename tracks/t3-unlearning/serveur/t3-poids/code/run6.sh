cd ~/eac/t3/t3-poids/code; PY=~/eac/t3/.venv/bin/python; export T3ME=~/eac/t3/t3-poids OMP_NUM_THREADS=4; DASH=~/eac/t3/t3-methodes/code/dash.py
until grep -q FIN2c ../logs/run2.log; do sleep 5; done
$PY mkdelta.py blocks tvE_ce 8 13 ce_t8; $PY mkdelta.py blocks tvE_ce 10 13 ce_t10; $PY mkdelta.py blocks tvE_ce 11 13 ce_t11; $PY mkdelta.py blocks tvE_ce 12 13 ce_head
$PY $DASH --fset ab --rset c --tgt T85 --tag r6_ --combo NEGce_top=ce_top:-1 --alphas 8,12,16
$PY $DASH --fset ab --rset c --tgt T85 --tag r6_ --combo NEGce_hi=ce_hi:-1 --combo NEGce_t8=ce_t8:-1 --alphas 6,8
$PY $DASH --fset ab --rset c --tgt T85 --tag r6_ --combo NEGce_t10=ce_t10:-1 --combo NEGce_t11=ce_t11:-1 --combo NEGce_head=ce_head:-1 --alphas 8,16
echo FIN6
