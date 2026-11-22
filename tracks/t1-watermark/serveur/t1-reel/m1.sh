#!/bin/bash
# usage : m1.sh <methode> <reserve0> <reserve1> <graine0> <graine1> "<options gen>"  -> prépare la machine : cache puis 2 réserves d'exemples identiques à celles de h200g (mêmes graines)
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=4; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel7.py"; R=t1-reel
$PY cache
G="--ncoco 0 --nvar 3 --canvas real --rpadlo 2 --rpadhi 6 $6"
$PY gen --method $1 --suffix $2 --gseed $4 $G > $R/logs/gen$2_$1.log 2>&1 &
$PY gen --method $1 --suffix $3 --gseed $5 $G > $R/logs/gen$3_$1.log 2>&1 &
wait; tail -qn 1 $R/logs/gen*_$1.log; echo PRET
