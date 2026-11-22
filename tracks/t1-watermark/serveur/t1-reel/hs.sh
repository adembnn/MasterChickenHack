#!/bin/bash
cd ~/eac/t1 && source .venv/bin/activate; export T1_THREADS=4; R=t1-reel
python $R/reel5.py train --method trustmark --suffix _p --lazy 1 --fold -1 --nreal 0 --tag smk5 --steps 40 --evalevery 20 --test 0 --st 1 --hp 2 --bs 16 2>&1 | grep -v -i warn | tail -3 | cut -c1-330
python $R/reel5.py train --method trustmark --suffix _p --lazy 1 --fold -1 --nreal 0 --tag smk6 --steps 20 --evalevery 20 --test 0 --st 1 --hp 2 --s1 1 --bs 8 2>&1 | grep -v -i warn | tail -2 | cut -c1-330
rm -rf $R/runs/smk5 $R/runs/smk6; echo FINI
