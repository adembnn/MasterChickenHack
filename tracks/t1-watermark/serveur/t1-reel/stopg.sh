#!/bin/bash
pkill -f "bash t1-reel/g8.sh"; pkill -f "t1-reel/c2.sh pixelseal"; pkill -f "reel1[12].py train"; sleep 4
ps -eo args | grep "reel1[12].py" | grep -v grep | wc -l; nvidia-smi --query-gpu=memory.used --format=csv,noheader
