#!/bin/bash
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=4; PY=/home/hermes/eac/kit/.venv/bin/python
[ -f t1-reel/stats_trustmark.json ] || $PY t1-reel/stats2.py > /dev/null 2>&1
$PY -u t1-reel/rmgrad.py 30 2>&1 | grep -v -i -E "warn|state_dict" | cut -c1-300; echo FINI
