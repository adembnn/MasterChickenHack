cd ~/eac/t3/t3-poids/code; PY=~/eac/t3/.venv/bin/python; export T3ME=~/eac/t3/t3-poids OMP_NUM_THREADS=4
$PY tv.py --name tvE_ce --mode exact --loss ce --opt adamw --lr 2e-5 --epochs 3
$PY tv.py --name tvE_mg --mode exact --loss margin --opt sgd --lr 5e-4 --epochs 2
$PY tv.py --name tvA_mg --mode aug --loss margin --opt sgd --lr 5e-4 --epochs 2
$PY tv.py --name tvA_ce --mode aug --loss ce --opt adamw --lr 2e-5 --epochs 3
$PY mkdelta.py cos tvE_ce tvA_ce; $PY mkdelta.py cos tvE_mg tvA_mg; $PY mkdelta.py cos tvE_ce tvE_mg; $PY mkdelta.py norms tvE_ce; $PY mkdelta.py norms tvE_mg
$PY ~/eac/t3/t3-methodes/code/dash.py --fset ab --rset c --tgt T85 --combo NEGce=tvE_ce:-1 --combo NEGmg=tvE_mg:-1 --alphas 0.5,1,2,4
echo FIN1
