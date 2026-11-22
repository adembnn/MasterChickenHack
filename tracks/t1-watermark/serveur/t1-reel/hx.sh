#!/bin/bash
cd ~/eac/t1 && source .venv/bin/activate
T1_THREADS=4 python -u t1-reel/xcorr.py trustmark 14 8 2>&1 | grep -v -i -E "warn|state_dict" | cut -c1-520
echo FINI
