#!/bin/bash
cd ~/eac/t1 && source .venv/bin/activate; export T1_THREADS=3; R=t1-reel
python $R/reel6.py train --method trustmark --suffix _c0 --lazy 1 --fold -1 --nreal 0 --tag smkO --steps 6 --evalevery 6 --test 0 --s1 1 --bs 16 --pfull 0.0 --pshift 0.15 --pblob 0.15 --objsam /tmp/eac-t1-partage --dump /tmp/eac-t1-partage/lot_objets.png 2>&1 | grep -v -i warn | tail -3 | cut -c1-160; rm -rf $R/runs/smkO; ls -la /tmp/eac-t1-partage/lot_objets.png
