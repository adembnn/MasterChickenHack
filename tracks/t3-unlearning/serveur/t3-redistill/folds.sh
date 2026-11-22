#!/bin/bash
# 5 plis hors echantillon, tete + 4 blocs, melanges : chaque image n est predite que par un modele qui ne l a vue sous aucune forme
cd ~/eac/t3/t3-redistill
while kill -0 27319 2>/dev/null || kill -0 27321 2>/dev/null; do sleep 10; done
for i in 0 1 2 3; do ./run.sh F4_f$i --k 4 --steps 1500 --hold f$i/5; done
while kill -0 14837 2>/dev/null; do sleep 10; done
./run.sh F4_f4 --k 4 --steps 1500 --hold f4/5
