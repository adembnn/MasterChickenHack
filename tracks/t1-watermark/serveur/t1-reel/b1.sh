#!/bin/bash
# h200b : la carte TrustMark dst (gains TrustMark non retrouvés au public) laisse la place à deux spécialistes des pleines PixelSeal, à pas égal (16 000), mêmes exemples (_w8,_w9) :
#  xfu : 90 % d'exemples fabriqués pleins, affinage sur les vraies pleines seulement
#  xfh : idem + 40 % de petits trous (0,5 à 4 % de l'image)
pkill -f "16000 dst"; pkill -f "tag g_dst"; sleep 3
cd ~/eac/t1; sed -i "s#/home/hermes/eac/kit/.venv/bin/python#$HOME/eac/t1/.venv/bin/python#g" t1-reel/c3.sh
(setsid nohup bash t1-reel/c3.sh pixelseal _w8,_w9 16000 xfu "--s1 1 --bs 16 --aux 1.0 --seed 10 --pfull 0.9" "--realpop full" > t1-reel/logs/c3_xfu.log 2>&1 < /dev/null &)
(setsid nohup bash t1-reel/c3.sh pixelseal _w8,_w9 16000 xfh "--s1 1 --bs 16 --aux 1.0 --seed 10 --pfull 0.9 --hsmall 0.4" "--realpop full" > t1-reel/logs/c3_xfh.log 2>&1 < /dev/null &)
sleep 100; for t in xfu xfh; do f=t1-reel/logs/g_${t}_pixelseal.log; echo "$t : $(grep -c "Traceback\|Error" $f) erreur(s) ; $(grep -v "Warning\|G = lambda\|torch.load" $f | tail -n 1 | cut -c1-110)"; done; nvidia-smi --query-gpu=memory.used --format=csv,noheader
