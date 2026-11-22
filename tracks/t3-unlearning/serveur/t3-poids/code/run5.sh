cd ~/eac/t3/t3-poids/code; PY=~/eac/t3/.venv/bin/python; export T3ME=~/eac/t3/t3-poids OMP_NUM_THREADS=4; DASH=~/eac/t3/t3-methodes/code/dash.py
until grep -q FIN9 ../logs/run9.log; do sleep 5; done
# la selectivite vient-elle de la perte (CE ponderee par 1-p) ou de l'optimiseur (AdamW contre SGD) ?
$PY tv.py --name tvE_mgA --mode exact --loss margin --opt adamw --lr 2e-5 --epochs 3
$PY tv.py --name tvE_ceS --mode exact --loss ce --opt sgd --lr 5e-2 --epochs 3
$PY tv.py --name tvE_cetA --mode exact --loss cet --T 4 --opt adamw --lr 2e-5 --epochs 3
$PY mkdelta.py cos tvE_mgA tvE_ce; $PY mkdelta.py cos tvE_mgA tvE_mg; $PY mkdelta.py cos tvE_ceS tvE_ce; $PY mkdelta.py norms tvE_ceS
$PY $DASH --fset ab --rset c --tgt T85 --tag r5_ --combo NEGmgA=tvE_mgA:-1 --combo NEGcetA=tvE_cetA:-1 --alphas 0.5,1,2
$PY $DASH --fset ab --rset c --tgt T85 --tag r5_ --combo NEGceS=tvE_ceS:-1 --alphas 1,4,16
echo FIN5
