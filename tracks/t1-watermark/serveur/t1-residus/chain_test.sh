#!/bin/bash
# Attend la fin des 4 tranches de pleine.py sur le test, puis calcule les cartes z-score et effaceur du test.
cd ~/eac/t1/t1-residus; export T1_ROOT=/home/adem/eac/t1 T1_OUT=$PWD PYTHONUNBUFFERED=1; source ../.venv/bin/activate
while [ "$(cat log_pleine_test_?.txt | grep -a -c '^FIN PLEINE')" != 4 ]; do sleep 20; done
SRC=maps_trustmark_test_pleine_s0of4.npz,maps_trustmark_test_pleine_s1of4.npz,maps_trustmark_test_pleine_s2of4.npz,maps_trustmark_test_pleine_s3of4.npz
python pz.py --split test --src $SRC > log_pz_test.txt 2>&1
python rm.py --split test --src $SRC > log_rm_test.txt 2>&1
echo CHAINE_FINIE >> log_rm_test.txt
