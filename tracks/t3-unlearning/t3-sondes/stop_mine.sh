#!/bin/bash
# Arrete les processus de la session t3-sondes sur cette machine (motifs propres a mes scripts ; lance depuis un fichier pour ne pas se viser soi-meme).
PAT='run_ab2?\.sh|train\.py --name AB_|dump_logits\.py|mk_affine\.py|mk_bank\.py|mk_temp\.py|mk_zero\.py|mk_held\.py|mk_from_delta\.py AB_'
echo "avant :"; pgrep -af "$PAT" | grep -v stop_mine | cut -c1-150
for p in $(pgrep -f "$PAT"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
sleep 4
for p in $(pgrep -f "$PAT"); do [ "$p" != "$$" ] && kill -9 "$p" 2>/dev/null; done
sleep 1; echo "apres :"; pgrep -af "$PAT" | grep -v stop_mine | cut -c1-150; echo "(fin de liste)"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
