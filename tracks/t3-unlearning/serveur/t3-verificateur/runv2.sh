#!/bin/bash
cd ~/eac/t3/t3-verificateur; export T3ROOT=$HOME/eac/t3 LOWMEM=1; S=$HOME/eac/t3
../.venv/bin/python vlogits.py PG85_NEGhi4_all=$S/t3-poids/candidates/PG85_NEGhi4_all.pt NEGce_all_top_a8=$S/t3-poids/candidates/NEGce_all_top_a8.pt G10_PG_T70=$S/t3-sondes/cand/G10_PG_T70.pt "$@" > out/vlogits3.log 2>&1
