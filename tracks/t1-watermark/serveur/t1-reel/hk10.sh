#!/bin/bash
# h200b : arrêt de tous mes entraînements de l ancienne génération (reel6, anciennes toiles)
pkill -f "bash hb"; pkill -f "t1-reel/reel6.py"; pkill -f "t1-reel/reel5.py"; sleep 2; ps -eo cmd | grep "t1-reel/reel" | grep -v grep | wc -l; nvidia-smi --query-gpu=memory.used --format=csv,noheader
