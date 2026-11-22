#!/bin/bash
pkill -f "hd16.sh"; sleep 1; cd ~/eac/t1
(setsid nohup bash t1-reel/c3.sh pixelseal _w4,_w5 16000 xfu "--s1 1 --bs 16 --aux 1.0 --seed 10 --pfull 0.9" "--realpop full" > t1-reel/logs/c3_xfu.log 2>&1 < /dev/null &)
sleep 2; echo "xfu lancé"
