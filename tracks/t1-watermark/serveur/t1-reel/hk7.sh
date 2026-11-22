#!/bin/bash
pkill -f "reel6.py train.*tag g_c1H"
sleep 1; cd ~/eac/t1/t1-reel; ps -eo etimes,cmd | grep "reel6.py train" | grep -v grep | sed "s/.*--tag //" | cut -c1-40 | tr "
" ";"; echo; cat logs/agg_f_s1_pixelseal.txt 2>/dev/null | cut -c1-220; for v in g_c2_trustmark g_c3_trustmark g_s1W_pixelseal g_decW_pixelseal; do echo "$v $(grep "\"step\"" logs/$v.log 2>/dev/null | tail -n 1 | cut -c1-200)"; done
