#!/bin/bash
# Usage : ./run_real.sh '<glob des journaux>' <tag> [k_bc]
# Pour chaque jumeau candidat : journal -> raw -> reread (GPU, lectures locales) -> decide (garde jumeau + score_practice).
set -u
cd ~/eac-t2/titouan-v4
export HF_HUB_OFFLINE=1
P=$PWD/.venv/bin/python3
J="$1"; TAG="$2"; K="${3:-}"
CANDS="HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM2-360M-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct allenai/OLMo-2-0425-1B-Instruct Qwen/Qwen2.5-0.5B-Instruct"
mkdir -p real_$TAG
for m in $CANDS; do
  t=$(basename $m)
  ( $P method_v4.py from_journal --journal "$J" --model $m --out real_$TAG/raw_$t.json --answers practice_answers.csv > real_$TAG/from_$t.log 2>&1 \
    && $P method_v4.py reread --raw real_$TAG/raw_$t.json --out real_$TAG/reread_$t.json > real_$TAG/reread_$t.log 2>&1 \
    && $P method_v4.py decide --raw real_$TAG/raw_$t.json --reread real_$TAG/reread_$t.json --submission real_$TAG/sub_$t.csv \
         --score_dir . ${K:+--k_bc $K} --log real_$TAG/decide_$t.log > /dev/null 2>&1 ) &
done
wait
echo "=== sélection du jumeau par la garde (t le plus bas = jumeau retenu) ==="
for m in $CANDS; do t=$(basename $m); echo "$t : $(grep -h "Sélecteur" real_$TAG/decide_$t.log | sed "s/.*T=1 = //;s/ (.*//") | $(grep -h "AUCUN\|Practice score" real_$TAG/decide_$t.log | head -1)"; done
