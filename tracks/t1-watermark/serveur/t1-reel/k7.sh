#!/bin/bash
# remplace un généraliste PixelSeal par un modèle dédié aux trous des pleines : 90 % de pleines fabriquées aux vraies formes de trous, affinage sur les vraies pleines seulement, canaux passe-haut (--hp 2 : sur TrustMark, IoU du trou en réserve 0,468 -> 0,519)
# usage : k7.sh <tag à arrêter> <filigrane> <réserves> <tag nouveau> <graine>
pkill -f "c2.sh pixelseal [^ ]* [0-9]* $1 "; pkill -f "tag g_$1 "; pkill -f "tag g_$1\$"; sleep 3; cd ~/eac/t1
(setsid nohup bash t1-reel/c3.sh $2 $3 16000 $4 "--s1 1 --bs 16 --aux 1.0 --seed $5 --hp 2 --pfull 0.9" "--realpop full" > t1-reel/logs/c3_$4.log 2>&1 < /dev/null &)
sleep 80; ps -eo args | grep "reel1[12].py train" | grep -v grep | grep -o "\-\-tag [a-z0-9_]*" | tr "\n" " "; echo; grep -c Traceback t1-reel/logs/g_$4_$2.log
