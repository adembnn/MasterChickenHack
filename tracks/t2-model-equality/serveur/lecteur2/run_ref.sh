#!/bin/bash
# Controle sur les 9 ref (corrige connu) : loi nulle simple + deguisee, directions, lecteur ajuste.
cd /home/adem/eac/t2/lecteur2; export HF_HOME=/home/adem/eac/t2/lecteur2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
M=HuggingFaceTB/SmolLM2-360M-Instruct; EPS=ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09
$PY dl_zoo.py $M
cat collecte_v4_ref_bloc0_20261121T191611Z.jsonl ref_bloc1.jsonl > data50_ref.jsonl
[ -s null50_ref.jsonl ] || $PY -u gen_null.py --data data50_ref.jsonl --model $M --one ref-01 --k 25 --out null50_ref.jsonl
[ -s dg_ref.jsonl ] || $PY -u gen_disg2.py --data data50_ref.jsonl --model $M --one ref-01 --k 4 --out dg_ref.jsonl
$PY mkdgn.py dg_ref.jsonl dgn_ref.jsonl 2
# directions : loi nulle simple (comme le lecteur existant) puis loi nulle deguisee
$PY -u gl_reader.py --data data50_ref.jsonl --null null50_ref.jsonl --model $M --eps $EPS --fp8 e4m3_tensor --bs 32 --stages fp8,base,dir --out gl50_ref.json
$PY -u gl_reader.py --data data50_ref.jsonl --null dgn_ref.jsonl --model $M --eps $EPS --fp8 e4m3_tensor --bs 32 --kdir 40 --stages fp8,base,dir --out gld_ref.json
$PY -u fitdisg.py --data data50_ref.jsonl --null null50_ref.jsonl,dg_ref.jsonl --kplain 8 --kdisg 4 --model $M --eps $EPS --out fd_ref.json
echo FINIREF
