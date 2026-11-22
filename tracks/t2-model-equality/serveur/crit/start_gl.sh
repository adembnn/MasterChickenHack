#!/bin/bash
cd /home/adem/eac/t2/crit
EP=collecte_v4_ep_bloc0_20261121T193826Z.jsonl; RF=collecte_v4_ref_bloc0_20261121T191611Z.jsonl
nohup bash run_gl.sh ref HuggingFaceTB/SmolLM2-360M-Instruct $RF ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09 > /dev/null 2>&1 < /dev/null &
nohup bash run_gl.sh F1 Qwen/Qwen2.5-3B-Instruct $EP ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 > /dev/null 2>&1 < /dev/null &
nohup bash run_gl.sh F2 Qwen/Qwen2.5-1.5B-Instruct $EP ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 > /dev/null 2>&1 < /dev/null &
nohup bash run_gl.sh F3 unsloth/Llama-3.2-1B-Instruct $EP ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 > /dev/null 2>&1 < /dev/null &
sleep 1; echo lances
