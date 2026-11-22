cd ~/eac/t3/t3-poids/code; PY=~/eac/t3/.venv/bin/python; export T3ME=~/eac/t3/t3-poids OMP_NUM_THREADS=4; DASH=~/eac/t3/t3-methodes/code/dash.py; C=~/eac/t3/t3-poids/candidates
until grep -q FIN5 ../logs/run5.log; do sleep 5; done
# piste 4 : montee controlee (charniere sur la marge / NPO / montee brute) sur exactes + vues legeres, ancre KL sur melanges + vues fortes
$PY unl.py --name ab_cap6 --forget cap:6 --wk 1 --ka 1 --steps 300 --lr 1e-5
$PY unl.py --name ab_npo --forget npo:1 --wk 1 --ka 1 --steps 300 --lr 1e-5
$PY unl.py --name ab_ga --forget ga --wf 0.3 --wk 1 --ka 1 --steps 150 --lr 5e-6
$PY unl.py --name ab_ga_noanc --forget ga --wf 0.3 --steps 150 --lr 5e-6
# piste 3 : bruit relatif sur les 4 derniers blocs puis reparation par distillation d'orig sur melanges + vues fortes
$PY unl.py --name ab_noise3 --init noise:0.3:8:12 --wk 1 --ka 1 --lo 8 --steps 500 --lr 5e-5
$PY unl.py --name ab_noise6 --init noise:0.6:8:12 --wk 1 --ka 1 --lo 8 --steps 500 --lr 5e-5
$PY unl.py --name ab_reinit11 --init reinit:11:12 --wk 1 --ka 1 --lo 10 --steps 800 --lr 1e-4
$PY $DASH --fset ab --rset c --tgt T85 --tag p34_ --sd cap6=$C/ab_cap6.pt --sd npo=$C/ab_npo.pt --sd ga=$C/ab_ga.pt --sd ga_noanc=$C/ab_ga_noanc.pt --sd noise3=$C/ab_noise3.pt --sd noise6=$C/ab_noise6.pt --sd reinit11=$C/ab_reinit11.pt
echo FIN3
