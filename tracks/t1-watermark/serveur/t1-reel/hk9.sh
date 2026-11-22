#!/bin/bash
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; V=/home/hermes/eac/kit/.venv/bin/python
for t in 8 2 1; do echo "symétries moyennées : $t"; $V t1-reel/predict.py --method trustmark --tags f_s1 --train 1 --tta $t --out t1-reel/verif_f_s1_tta$t 2>&1 | grep -v -i -E "warn|All keys|torch.load" | tail -2 | cut -c1-170; done
grep "\"step\"" t1-reel/logs/g_c4_trustmark.log | tail -n 1 | cut -c1-120
