#!/bin/bash
# arrête les chaînes des spécialistes des pleines tombées par manque de mémoire graphique, et montre qui occupe la carte
pkill -f "16000 xfu"; pkill -f "16000 xfh"; pkill -f "tag f_xfu"; pkill -f "tag f_xfh"; pkill -f "e4.sh"; sleep 3
rm -rf ~/eac/t1/t1-reel/runs/f_xfu ~/eac/t1/t1-reel/runs/f_xfh ~/eac/t1/t1-reel/runs/g_xfu ~/eac/t1/t1-reel/runs/g_xfh ~/eac/t1/t1-reel/logs/*_xfu_* ~/eac/t1/t1-reel/logs/*_xfh_* ~/eac/t1/t1-reel/logs/agg_f_xf*
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read p m; do echo "$(ps -o user=,etimes= -p $p 2>/dev/null) $m $(ps -o args= -p $p 2>/dev/null | grep -o "\-\-tag [a-z0-9_]*\|[a-zA-Z0-9_./-]*\.py" | head -n 2 | tr "\n" " ")"; done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
