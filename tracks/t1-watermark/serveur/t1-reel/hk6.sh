#!/bin/bash
pkill -f "reel6.py train.*tag g_c1F"
sleep 1; ps -eo etimes,cmd | grep "reel6.py train" | grep -v grep | wc -l
