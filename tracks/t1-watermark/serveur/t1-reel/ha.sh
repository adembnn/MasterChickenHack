#!/bin/bash
cd ~/eac/t1 && source .venv/bin/activate
T1_THREADS=4 python -u t1-reel/agree.py 40 trustmark pixelseal 2>&1 | grep -v -i -E "warn|state_dict|Model loaded" | cut -c1-420
echo FINI
