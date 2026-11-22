#!/bin/bash
# pshole (PixelSeal, canaux passe-haut) est en dessous du généraliste au même pas : arrêt ; la carte va à un 2e modèle TrustMark dédié aux trous (recette de tmhole, 10 000 pas pour finir avant 11 h 15)
pkill -f "16000 pshole"; pkill -f "tag g_pshole"; sleep 3; cd ~/eac/t1
(setsid nohup bash t1-reel/c3.sh trustmark _c0,_c1,_c2,_c3 10000 tmhole2 "--s1 1 --bs 16 --aux 1.0 --seed 32 --hp 2 --pfull 0.9" "--realpop full" > t1-reel/logs/c3_tmhole2.log 2>&1 < /dev/null &)
sleep 80; ps -eo args | grep "reel1[12].py train" | grep -v grep | grep -o "\-\-tag [a-z0-9_]*" | tr "\n" " "; echo; grep -c Traceback t1-reel/logs/g_tmhole2_trustmark.log
