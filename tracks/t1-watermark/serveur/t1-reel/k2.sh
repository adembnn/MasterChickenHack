#!/bin/bash
# arrête la mesure TrustMark des paires contrefactuelles, lue à 8000 pas
pkill -f "hd13.sh"; pkill -f "tag g_c5cf"; sleep 3; ps -eo pid,etimes,args | grep "reel" | grep -v grep | cut -c1-110
