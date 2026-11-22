#!/bin/bash
# h200d : essai à blanc (venv du kit en lecture seule + bibliothèque de segmentation dans ~/eac/t1/pylib)
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=4; PY=/home/hermes/eac/kit/.venv/bin/python; R=t1-reel
$PY -c "import segmentation_models_pytorch as s; print('smp', s.__version__)" 2>&1 | tail -1
$PY $R/reel5.py cache 2>&1 | tail -1
$PY $R/reel5.py gen --method trustmark --suffix _smoke --small 1 --ncoco 0 --nrect 30 --padlo 10 --padhi 16 --rpadlo 2 --rpadhi 6 2>&1 | grep -v -i warn | tail -2
$PY $R/reel5.py train --method trustmark --suffix _smoke --lazy 1 --fold -1 --nreal 0 --tag smk --steps 30 --evalevery 15 --test 0 --st 1 --hp 2 --s1 1 --bs 8 2>&1 | grep -v -i warn | tail -2 | cut -c1-300
rm -rf $R/runs/smk $R/data_trustmark_smoke; nvidia-smi --query-gpu=memory.used --format=csv,noheader; echo FINI
