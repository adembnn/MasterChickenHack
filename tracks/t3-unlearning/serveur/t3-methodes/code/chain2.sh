#!/bin/bash
# seconde vague sur h200e : attend les 6 entrainements de la premiere
cd ~/eac/t3/t3-methodes
until [ $(cat logs/PD_f30_l814.log logs/PD_f30_l814ab.log logs/PD_f40.log logs/PD_f40ab.log logs/PD_f20p10.log logs/PD_f20p10ab.log 2>/dev/null | grep -c "^saved") -ge 6 ]; do sleep 10; done
code/wave.sh "PD_f40_l814 T85_f40_l814 D" "PD_f50 T85_f50_l48 D" "PG_f20p10c T85_f20p10 M"
