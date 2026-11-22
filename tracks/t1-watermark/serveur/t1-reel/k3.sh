#!/bin/bash
# remplace x10st (réplique simple, lancée il y a 10 min) par un spécialiste des partielles
pkill -f "c2.sh pixelseal _w0,_w1 16000 x10st"; pkill -f "tag g_x10st"; sleep 3; rm -rf ~/eac/t1/t1-reel/runs/g_x10st ~/eac/t1/t1-reel/logs/g_x10st_pixelseal.log; ps -eo etimes,args | grep "reel1[12].py train" | grep -v grep | awk '{print $1, $4, $6, $8, $9}'
