cd ~/eac/t3/t3-poids/code; PY=~/eac/t3/.venv/bin/python; export T3ME=~/eac/t3/t3-poids OMP_NUM_THREADS=4; DASH=~/eac/t3/t3-methodes/code/dash.py
until grep -q FIN7 ../logs/run7.log; do sleep 5; done
ln -sf ~/eac/t3/t3-methodes/deltas/PGab_T85.pt ../deltas/PGab_T85.pt
# somme de deux familles : cibles adoucies de t3-methodes (PGab_T85, appris sur a+b) + negation du vecteur de tache (blocs hauts)
$PY $DASH --fset ab --rset c --tgt T85 --tag r8_ --combo PGab=PGab_T85:1 --combo PG+NEGhi2=PGab_T85:1,ce_hi:-2 --combo PG+NEGhi4=PGab_T85:1,ce_hi:-4 --combo PG+NEGtop6=PGab_T85:1,ce_top:-6 --alphas 1
echo FIN8
