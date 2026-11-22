#!/bin/bash
pkill -f "reel5.py train.*tag g_st "
sleep 1; ps -eo etimes,cmd | grep "reel5.py train" | grep -v grep | cut -c1-110
