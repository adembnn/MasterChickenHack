#!/bin/bash
# Usage : ./run_ep.sh '<glob journaux ep>' <tag>
# Décision fixée AVANT les données (voir reports/titouan/t2-method-v4.md, « Règle pré-enregistrée pour les ep ») :
#  1. famille = jumeau candidat de NLL/token la plus basse (assign) ;
#  2. dans chaque famille : relecture locale par ce jumeau puis decide, référence = plancher local (pas --pool_ref :
#     sur les vrais ref, plancher local 0,273 contre pool 0,000) ;
#  3. PRINCIPAL (mis à jour avant les données ep, #4) : règle « lecteurs » = un endpoint est une variante si un lecteur
#     (fp8 e4m3 par tenseur, W8A8 par tenseur, LoRA estimée si la famille est SmolLM2-360M) donne z apparié >= 3
#     contre les autres endpoints de la famille de T* < 1,0, référence élaguée itérativement des endpoints déjà
#     isolés (LoRA estimée à l'échelle 1,0) ; REPLI = règle v4 précédente (2.) ;
#  4. merge_ep -> CSV (Fi / FiS, FiT1...) -> validate_t2.py, pour les deux.
set -u
cd ~/eac-t2/titouan-v4
export HF_HUB_OFFLINE=1
P=$PWD/.venv/bin/python3
J="$1"; TAG="$2"
CANDS="${CANDS:-HuggingFaceTB/SmolLM2-135M-Instruct,HuggingFaceTB/SmolLM2-360M-Instruct,HuggingFaceTB/SmolLM2-1.7B-Instruct,HuggingFaceTB/SmolLM3-3B,Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct,allenai/OLMo-2-0425-1B-Instruct,allenai/OLMo-2-1124-7B-Instruct,microsoft/Phi-3.5-mini-instruct,mistralai/Mistral-7B-Instruct-v0.3,unsloth/Llama-3.2-1B-Instruct,unsloth/Llama-3.2-3B-Instruct}"
D=ep_$TAG; mkdir -p $D/fallback
$P method_v4.py assign --journal "$J" --candidates "$CANDS" --out $D/assign.json 2>&1 | grep -v "Loading weights" | tee $D/assign.log
$P family_table.py --ep_dir $D --out $D/familles.json > /dev/null && echo "TABLE DES FAMILLES (sans T*) : $D/familles.json"
for m in $($P -c "import json;print(' '.join(sorted(set(json.load(open('$D/assign.json'))['family_twin'].values()))))"); do
  t=$(basename $m)
  mem=$($P -c "import json;a=json.load(open('$D/assign.json'))['family_twin'];print(','.join(sorted(e for e,v in a.items() if v=='$m')))")
  n=$(echo $mem | tr ',' '\n' | wc -l)
  pool=""   # pré-enregistré : plancher local (MCC réel ref 0,273) plutôt que référence pool (0,000)
  LORA_EST="${LORA_EST-$HOME/casser-fp8/lora_est_bloc0}"; LORA_SCALE="${LORA_SCALE:-1.0}"
  lora=""; [ -n "${LORA_EST:-}" ] && [ "$t" = "SmolLM2-360M-Instruct" ] && lora="--lora $LORA_EST --lora_scale ${LORA_SCALE:-1.0}"
  ( $P method_v4.py from_journal --journal "$J" --endpoints $mem --model $m --out $D/raw_$t.json --answers practice_answers.csv > $D/from_$t.log 2>&1 \
    && $P method_v4.py reread --raw $D/raw_$t.json --out $D/reread_$t.json > $D/reread_$t.log 2>&1 \
    && $P method_v4.py decide --raw $D/raw_$t.json --reread $D/reread_$t.json --submission $D/fallback/sub_$t.csv --score_dir . $pool \
         --log $D/fallback/decide_$t.log --dump $D/fallback/dump_$t.json > /dev/null 2>&1 \
    && $P method_v4.py readers --raw $D/raw_$t.json $lora --out $D/readers_$t.json > $D/readers_$t.log 2>&1 \
    && $P method_v4.py decide --raw $D/raw_$t.json --reread $D/reread_$t.json --readers $D/readers_$t.json --submission $D/sub_$t.csv \
         --score_dir . --log $D/decide_$t.log --dump $D/dump_$t.json > /dev/null 2>&1 ) &
done
wait
$P family_table.py --ep_dir $D --out $D/familles.json > /dev/null && echo "TABLE DES FAMILLES (avec T*) : $D/familles.json"
echo "=== PRINCIPAL (règle « lecteurs » : fp8 par tenseur + LoRA estimée si famille SmolLM2-360M) ==="
$P method_v4.py merge_ep --assign $D/assign.json --dir $D --out $D/submission_ep.csv
$P validate_t2.py $D/submission_ep.csv
echo "=== REPLI (règle v4 précédente) ==="
$P method_v4.py merge_ep --assign $D/assign.json --dir $D/fallback --out $D/submission_ep_fallback.csv
$P validate_t2.py $D/submission_ep_fallback.csv
