cd ~/eac/t3/t3-poids/code; PY=~/eac/t3/.venv/bin/python; export T3ME=~/eac/t3/t3-poids OMP_NUM_THREADS=4; DASH=~/eac/t3/t3-methodes/code/dash.py; C=~/eac/t3/t3-poids/candidates
until grep -q FIN8 ../logs/run8.log; do sleep 5; done
# version iteree : NPO (bornee) ou montee CE, blocs 8-11 + norme + tete, ancre KL sur melanges + vues fortes ; arret a une loss moyenne cible
$PY unl.py --name ab_npo8_m1 --forget npo:1 --wk 1 --ka 1 --lo 8 --steps 600 --lr 2e-5 --stop_m 1.0
$PY unl.py --name ab_npo8_m2 --forget npo:1 --wk 1 --ka 1 --lo 8 --steps 900 --lr 2e-5 --stop_m 2.0
$PY unl.py --name ab_ga8_m1 --forget ga --wk 1 --ka 1 --lo 8 --steps 600 --lr 2e-5 --stop_m 1.0
$PY unl.py --name ab_npo8_m1_noanc --forget npo:1 --lo 8 --steps 600 --lr 2e-5 --stop_m 1.0
$PY $DASH --fset ab --rset c --tgt T85 --tag r9_ --sd npo8_m1=$C/ab_npo8_m1.pt --sd npo8_m2=$C/ab_npo8_m2.pt --sd ga8_m1=$C/ab_ga8_m1.pt --sd npo8_m1_noanc=$C/ab_npo8_m1_noanc.pt
echo FIN9
