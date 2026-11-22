#!/bin/bash
pkill -f "bash hg.sh"
sleep 1
cd ~/eac/t1/t1-reel && (setsid nohup bash hg2.sh > logs/hg2.log 2>&1 < /dev/null &)
sleep 1; ps -eo etimes,cmd | grep "t1-reel" | grep -v grep | cut -c1-110
