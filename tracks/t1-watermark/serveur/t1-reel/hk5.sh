#!/bin/bash
pkill -f "reel6.py train --method pixelseal --suffix _p "
sleep 2
ps -eo etimes,cmd | grep "reel6.py" | grep -v grep | sed "s/python t1-reel.reel6.py //" | cut -c1-80
