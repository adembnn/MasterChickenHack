#!/bin/bash
# arrête les deux mesures TrustMark lues à 8000 pas (placement des trous, petits trous) pour rendre la carte à PixelSeal
pkill -f "hd12.sh"; pkill -f "tag g_c5s"; pkill -f "tag g_c5h"; sleep 3; ps -eo pid,etimes,args | grep "reel" | grep -v grep | cut -c1-120
