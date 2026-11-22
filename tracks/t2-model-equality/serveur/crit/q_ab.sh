#!/bin/bash
# Controle croise du lecteur fp8 : memes schemas, lus separement sur les questions du bloc 0 et du bloc 1 (guichets + 200 faux endpoints).
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
python3 - <<'PY'
import json
B1 = {json.loads(l)["prompt"] for l in open("questions_bloc1.jsonl", encoding="utf-8") if l.strip()}
for F in ("F1", "F3"):
    for src in ("data50", "null50"):
        out = [open(f"{src}_{F}_b0.jsonl", "w", encoding="utf-8"), open(f"{src}_{F}_b1.jsonl", "w", encoding="utf-8")]
        n = [0, 0]
        for l in open(f"{src}_{F}.jsonl", encoding="utf-8"):
            if not l.strip(): continue
            d = json.loads(l); p = d.get("prompt")
            h = 1 if p in B1 else 0
            out[h].write(l if l.endswith("\n") else l + "\n"); n[h] += 1
        print(F, src, n)
PY
run() { setsid nohup $PY -u gl_reader.py --data data50_$1_$4.jsonl --null null50_$1_$4.jsonl --model $2 --eps $3 --fp8 e4m3_tensor,w8a8_tensor --bs 16 --stages fp8,base --out qab_$1_$4.json > qab_$1_$4.log 2>&1 < /dev/null & }
for b in b0 b1; do
run F1 Qwen/Qwen2.5-3B-Instruct ep-01,ep-04,ep-12,ep-17,ep-18,ep-24,ep-25 $b
run F3 unsloth/Llama-3.2-1B-Instruct ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-19,ep-27 $b
done
sleep 1; echo lance-qab
