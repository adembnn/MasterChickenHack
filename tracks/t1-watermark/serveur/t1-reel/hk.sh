#!/bin/bash
pkill -f "python -u t1-reel/rectdet.py"
cd ~/eac/t1/t1-reel && (setsid nohup bash hx.sh > logs/xcorr.log 2>&1 < /dev/null &)
sleep 1; echo ok; tail -n 8 logs/tmean.log | cut -c1-300
