#!/bin/bash
pkill -f "reel6.py train.*tag g_c3"; pkill -f "reel6.py train.*tag g_s1W"
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; /home/hermes/eac/kit/.venv/bin/python -u t1-reel/alpha.py trustmark g_s1 _p 96 2>&1 | grep -v -i -E "warn|torch.load" | cut -c1-230
/home/hermes/eac/kit/.venv/bin/python -u t1-reel/alpha.py stegastamp g_s1 _p 96 2>&1 | grep -v -i -E "warn|torch.load" | cut -c1-230
