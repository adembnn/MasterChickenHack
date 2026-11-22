#!/bin/bash
cd ~/eac/t1 && source .venv/bin/activate; T1_THREADS=4 python -u t1-reel/band.py stegastamp trustmark pixelseal 2>&1 | grep -v -i -E "warn|state_dict|Model loaded|wm.logits" | cut -c1-420; echo FINI
