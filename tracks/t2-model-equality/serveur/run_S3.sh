#!/bin/bash
# monde échantillonné v2 : T 0,7 + top-p 0,9, un seul processus GPU à la fois
cd ~/eac/t2
for fam in olmo qwen; do
 if [ $fam = olmo ]; then M=allenai/OLMo-2-0425-1B-Instruct; else M=Qwen/Qwen2.5-1.5B-Instruct; fi
 C="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python adem/bench/world.py --model $M --fam $fam --regime S --top_p 0.9 --probes adem/probes/probes_S3.jsonl --lora lora_$fam --bs 40 --max_new 96"
 env $C --weights base --disguises plain,plain2,sysacme,syshelp,sysspace,persona,wm,t1 --out worldS3/${fam}_S_base.jsonl
 env $C --weights lora0.7 --disguises plain,sysacme,t1 --out worldS3/${fam}_S_lora0.7.jsonl
 env $C --weights fp8,int8,lora0.3,lora1.5,nf4 --disguises plain --out worldS3/${fam}_S_var.jsonl
done
echo TERMINE
