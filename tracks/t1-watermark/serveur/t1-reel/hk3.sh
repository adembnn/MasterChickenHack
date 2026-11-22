#!/bin/bash
pkill -f "reel4.py train"
sleep 2
ps -eo etimes,cmd | grep "t1-reel" | grep -v grep | cut -c1-150
nvidia-smi --query-gpu=memory.used --format=csv,noheader
cat ~/eac/t1/t1-reel/runs/g_r34/trustmark/fold-1/log.jsonl | cut -c1-200
