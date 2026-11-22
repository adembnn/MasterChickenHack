#!/bin/bash
# h200d : arrêt des deux variantes TrustMark négatives (canal force attendue, deux étages sans réduction) ; lancement de 3 questions : plus long, plus gros, StegaStamp en pleine résolution
pkill -f "reel6.py train.*tag g_s1e"; pkill -f "reel6.py train.*tag g_s2"; sleep 2
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=2; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel6.py"; R=t1-reel
C="--lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0"
(setsid nohup $PY train --method trustmark --suffix _p,_g1,_g2,_g3 $C --tag g_s1L --s1 1 --bs 16 --steps 16000 > $R/logs/g_s1L_trustmark.log 2>&1 < /dev/null &)
(setsid nohup $PY train --method trustmark --suffix _p,_g1,_g2,_g3 $C --tag g_s1r50 --s1 1 --bs 16 --steps 8000 --enc resnet50 > $R/logs/g_s1r50_trustmark.log 2>&1 < /dev/null &)
(setsid nohup $PY train --method stegastamp --suffix _p $C --tag g_s1 --s1 1 --bs 16 --steps 8000 > $R/logs/g_s1_stegastamp.log 2>&1 < /dev/null &)
sleep 3; ps -eo etimes,cmd | grep "reel[56].py train" | grep -v grep | sed 's/.*train --method //' | cut -c1-90; nvidia-smi --query-gpu=memory.used --format=csv,noheader; tail -n 2 $R/logs/d7.log
