#!/bin/bash
# arrêt des chaînes StegaStamp et PixelSeal sur h200g (elles reprennent ailleurs), TrustMark continue
pkill -f "bash g1.sh"; sleep 1; pkill -f "reel7.py train --method stegastamp"; pkill -f "reel7.py train --method pixelseal"; sleep 3
cd ~/eac/t1/t1-reel && (setsid nohup bash g2.sh > logs/g2.log 2>&1 < /dev/null &)
for m in stegastamp pixelseal; do ls -la runs/g_x1/$m/fold-1/ckpt.pt | awk "{print \$5, \$9}"; md5sum runs/g_x1/$m/fold-1/ckpt.pt | cut -c1-32; grep "\"step\"" logs/g_x1_$m.log | tail -n 1 | cut -c1-60; done; ps -eo cmd | grep "reel7.py train" | grep -v grep | wc -l
