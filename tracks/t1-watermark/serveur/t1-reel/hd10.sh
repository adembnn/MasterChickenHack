#!/bin/bash
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel7.py"
$PY agg --tag f_c4,f_s1,f_c4+f_s1 --write 1 2>&1 | grep -E "^\| (tag|f_).*stegastamp" | cut -c1-225
$PY agg --tag f_c4+f_c1L,f_c1L --write 1 2>&1 | grep -E "^\| (tag|f_).*trustmark" | cut -c1-225
for t in "f_c4 stegastamp" "f_c4+f_c1L trustmark"; do /home/hermes/eac/kit/.venv/bin/python t1-reel/holes.py $t 2>&1 | grep -E "##" | cut -c1-170; done
