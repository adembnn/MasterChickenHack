#!/bin/bash
# h200e : ref a 100 reponses par guichet (LoRA 0,3 visible ?) puis faux endpoints du bloc 1
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
$PY mk_pool.py > pool_prep.log 2>&1
setsid nohup $PY -u gl_reader.py --data ref_pool100.jsonl --null null_ref4.jsonl --model HuggingFaceTB/SmolLM2-360M-Instruct --eps ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09 --fp8 e4m3_tensor,w8a8_tensor --bs 32 --kdir 13 --scales 0.6,1.0,1.5 --stages fp8,base,dir --out gl_pool100.json > gl_pool100.log 2>&1 < /dev/null &
( $PY -u gen_null.py --data questions_bloc1.jsonl --model HuggingFaceTB/SmolLM2-360M-Instruct --one q --out null1_ref.jsonl > null1_ref.log 2>&1 ; $PY -u gen_null.py --data questions_bloc1.jsonl --model unsloth/Llama-3.2-1B-Instruct --one q --out null1_F3.jsonl > null1_F3.log 2>&1 ; $PY -u gen_null.py --data questions_bloc1.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --one q --out null1_F2.jsonl > null1_F2.log 2>&1 ) < /dev/null &
sleep 1; echo lance-e
