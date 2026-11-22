cd ~/eac/t3/t3-poids/code; PY=~/eac/t3/.venv/bin/python; export T3ME=~/eac/t3/t3-poids OMP_NUM_THREADS=4; DASH=~/eac/t3/t3-methodes/code/dash.py; C=~/eac/t3/t3-poids/candidates
until grep -q FIN3 ../logs/run3.log; do sleep 5; done
$PY loc.py fisher ab
$PY ssd.py ab 10:1:both 30:1:both 100:1:both 30:0.3:both 30:1:both:8 100:0.1:both
$PY $DASH --fset ab --rset c --tgt T85 --tag ssd_ --sd ssdA10=$C/ssd_ab_A10_L1_both.pt --sd ssdA30=$C/ssd_ab_A30_L1_both.pt --sd ssdA100=$C/ssd_ab_A100_L1_both.pt --sd ssdA30L.3=$C/ssd_ab_A30_L0.3_both.pt --sd ssdA30lo8=$C/ssd_ab_A30_L1_both_lo8.pt --sd ssdA100L.1=$C/ssd_ab_A100_L0.1_both.pt
echo FIN4
