#!/bin/bash
# vérification du livrable : predict.py doit redonner, depuis les poids sauvés, le score de réserve de l'entraînement
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY=/home/hermes/eac/kit/.venv/bin/python
$PY t1-reel/predict.py --method trustmark --tags f_s1a --train 1 --out t1-reel/verif_f_s1a 2>&1 | grep -v -i -E "warn|torch.load|All keys" | tail -3
$PY -c "
import numpy as np
a = np.load('t1-reel/verif_f_s1a/pred_test_trustmark.npz'); b = np.load('t1-reel/runs/f_s1a/trustmark/pred_test_trustmark.npz')
print('mêmes noms :', bool((a['names'] == b['names']).all()), '| écart max des logits test :', float(np.abs(a['logits'].astype(np.float32) - b['logits'].astype(np.float32)).max()), '| masques identiques à', float(((a['logits'] > 0) == (b['logits'] > 0)).mean()))
"
grep f_s1a t1-reel/logs/agg_f_s1a_trustmark.txt | cut -c1-200
