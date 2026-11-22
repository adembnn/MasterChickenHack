cd ~/eac/t3/t3-methodes
PY=../.venv/bin/python; P85=0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise
grep -h -o "\-\-full [0-9,]*" logs/*.log code/*.sh *.sh 2>/dev/null | sort | uniq -c | head -5
$PY - <<'PYE'
import numpy as np, torch
lg = np.load('cache/exact_orig_logits.npy'); y = lg.argmax(1); c = np.bincount(y, minlength=100)
o = np.argsort(-c); print('classes les plus presentes :', [(int(k), int(c[k])) for k in o[:14]]); print('nb classes presentes', int((c>0).sum()), 'total', len(y))
full = [int(k) for k in o if c[k] >= 129]; print('FULL=' + ','.join(map(str, full)), sum(int(c[k]) for k in full))
open('cache/full9.txt','w').write(','.join(map(str, full)))
PYE
FULL=$(cat cache/full9.txt)
mk() { T=$1; shift; [ -s cache/tgt_$T.pt ] || $PY code/mk_tgt.py $T --pool $P85 --full $FULL "$@" 2>&1 | tail -1 | cut -c1-200; }
mk C9_f10 --f 0.10 --llo 8 --lhi 14 --lfull 10 --sfull 3
mk C9_f25 --f 0.25 --llo 8 --lhi 14 --lfull 10 --sfull 3
mk C9_f40 --f 0.40 --llo 8 --lhi 14 --lfull 10 --sfull 3
mk C9L_f25 --f 0.25 --llo 8 --lhi 14 --lfull 13 --sfull 2
mk C9_f00 --f 0.0 --llo 8 --lhi 14 --lfull 10 --sfull 3
run() { N=$1; T=$2; shift 2
  (nohup $PY code/train.py --anchor mix --wce 1 --lam 3 --steps 1500 --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &); echo "lance $N"; }
for t in C9_f10 C9_f25 C9_f40 C9L_f25 C9_f00; do run PG_${t}c $t; done
sleep 30; nvidia-smi --query-gpu=memory.used --format=csv,noheader; tail -c 200 logs/PG_C9_f25c.log | tr '\r' '\n' | tail -1 | cut -c1-120
