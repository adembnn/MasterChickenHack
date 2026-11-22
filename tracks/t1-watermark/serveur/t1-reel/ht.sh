#!/bin/bash
cd ~/eac/t1 && source .venv/bin/activate
for m in trustmark pixelseal stegastamp; do T1_THREADS=4 python -u t1-reel/tmean.py $m 8 2>&1 | grep -v -i -E "warn|state_dict|Model loaded" | cut -c1-420; done
echo FINI
