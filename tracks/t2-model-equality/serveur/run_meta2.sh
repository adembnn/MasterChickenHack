#!/bin/bash
# attend la fin des mondes v3 puis génère les sondes d extraction supplémentaires
cd ~/eac/t2
while pgrep -u adem -f "[p]robes_v3.jsonl" > /dev/null; do sleep 10; done
for fam in olmo qwen; do
 if [ $fam = olmo ]; then M=allenai/OLMo-2-0425-1B-Instruct; else M=Qwen/Qwen2.5-1.5B-Instruct; fi
 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python adem/bench/world.py --model $M --fam $fam --regime G --probes adem/probes/probes_meta2.jsonl --lora lora_$fam --bs 32 --out world3/${fam}_G_meta2.jsonl > logs/v3_${fam}_meta2.log 2>&1 &
done
wait
