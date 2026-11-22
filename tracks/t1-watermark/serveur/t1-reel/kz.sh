#!/bin/bash
# arrêt de tous les entraînements t1-reel encore en cours sur cette carte
pkill -f "t1-reel/c[123].sh"; pkill -f "t1-reel/[a-z0-9]*\.sh"; pkill -f "t1-reel/reel1[12].py train"; sleep 3
ps -eo args | grep "reel1[12].py" | grep -v grep | grep -o "\-\-tag [a-z0-9_]*" | tr "\n" " "; echo; nvidia-smi --query-gpu=memory.used --format=csv,noheader
