#!/bin/bash
# Les 100 reponses par ref (bloc 0 + 3 tirages frais) : la LoRA 0,3 devient-elle visible avec tout le budget ? Faux endpoints a 100 reponses.
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
RF=collecte_v4_ref_bloc0_20261121T191611Z.jsonl
( $PY -u gen_null.py --data $RF --model HuggingFaceTB/SmolLM2-360M-Instruct --one ref-01 --k 104 --seed 5 --out null_ref_k104.jsonl > null_ref4.log 2>&1 && $PY - <<'PY' && \
import json
with open("null_ref4.jsonl", "w", encoding="utf-8") as f:
    for l in open("null_ref_k104.jsonl", encoding="utf-8"):
        r = json.loads(l); a, T, k = r["endpoint"].split("-"); r["endpoint"] = f"fk-{T}-{int(k) // 4:03d}"; f.write(json.dumps(r, ensure_ascii=False) + chr(10))
with open("ref_pool100.jsonl", "w", encoding="utf-8") as f:
    for p in ("collecte_v4_ref_bloc0_20261121T191611Z.jsonl", "ref_frais_1.jsonl", "ref_frais_2.jsonl", "ref_frais_3.jsonl"):
        for l in open(p, encoding="utf-8"):
            if l.strip() and json.loads(l).get("type", "answer") == "answer": f.write(l if l.endswith(chr(10)) else l + chr(10))
print("ok")
PY
$PY -u gl_reader.py --data ref_pool100.jsonl --null null_ref4.jsonl --model HuggingFaceTB/SmolLM2-360M-Instruct --eps ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09 --fp8 e4m3_tensor,w8a8_tensor --bs 16 --kdir 13 --scales 0.6,1.0,1.5 --stages fp8,base,dir --out gl_pool100.json > gl_pool100.log 2>&1 ) < /dev/null &
sleep 1; echo lance
