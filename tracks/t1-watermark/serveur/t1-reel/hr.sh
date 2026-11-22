#!/bin/bash
cd ~/eac/t1 && source .venv/bin/activate
T1_THREADS=4 python -u t1-reel/rectdet.py 68 hors 2>&1 | grep -v -i -E "warn|state_dict|torch.from_numpy" | cut -c1-420
echo FINI
