#!/bin/bash
set -euo pipefail
cd /home/adem/eac/t3/t3-methodes
for pair in PG_RDt24_85_l1218:RDt24_85_l1218 PG_RDt24_90_l1218:RDt24_90_l1218 PG_RDt24_85_l1117:RDt24_85_l1117; do
  n=${pair%%:*}
  t=${pair##*:}
  echo "START_DASH $n $(date -Is)"
  ../.venv/bin/python code/dash.py --fset all --tgt "$t" --tag all_ --combo "$n=$n" --save "$n" > "logs/dash_$n.log" 2>&1
  echo "DONE_DASH $n $(date -Is)"
  tail -30 "logs/dash_$n.log"
done
