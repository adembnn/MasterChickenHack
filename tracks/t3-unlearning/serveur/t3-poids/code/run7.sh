cd ~/eac/t3/t3-poids/code; PY=~/eac/t3/.venv/bin/python; export T3ME=~/eac/t3/t3-poids OMP_NUM_THREADS=4; DASH=~/eac/t3/t3-methodes/code/dash.py; C=~/eac/t3/t3-poids/candidates
until grep -q FIN6 ../logs/run6.log; do sleep 5; done
$PY mkdelta.py blocks tvE_ce_all 6 13 ce_all_hi; $PY mkdelta.py blocks tvE_ce_all 9 13 ce_all_top
$PY $DASH --fset all --rset none --tgt T85 --tag all_ --combo NEGce_all_hi=ce_all_hi:-1 --alphas 4 --save NEGce_all_hi_a4
$PY $DASH --fset all --rset none --tgt T85 --tag all_ --combo NEGce_all_top=ce_all_top:-1 --alphas 8 --save NEGce_all_top_a8
$PY chk.py $C/NEGce_all_hi_a4.pt $C/NEGce_all_top_a8.pt
echo FIN7a
$PY tv.py --name tvE_ce_s1 --mode exact --loss ce --opt adamw --lr 2e-5 --epochs 3 --seed 1
$PY tv.py --name tvE_ce_s2 --mode exact --loss ce --opt adamw --lr 2e-5 --epochs 3 --seed 2
$PY mkdelta.py cos tvE_ce tvE_ce_s1; $PY mkdelta.py cos tvE_ce_s1 tvE_ce_s2
$PY mkdelta.py blocks tvE_ce_s1 6 13 ce_hi_s1; $PY mkdelta.py blocks tvE_ce_s2 6 13 ce_hi_s2
$PY $DASH --fset ab --rset c --tgt T85 --tag r7_ --combo NEGmean3_hi=ce_hi:-0.3333,ce_hi_s1:-0.3333,ce_hi_s2:-0.3333 --alphas 4,8
echo FIN7
