#!/bin/bash
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=4; PY=/home/hermes/eac/kit/.venv/bin/python
$PY -u t1-reel/detect.py trustmark g_s1 50 '{"enc":"resnet34","arch":"unet","stem1":0,"flag":0,"hp":0,"prec":"fp16","s1":1,"st":0,"method":"trustmark"}' 2>&1 | grep -v -i -E "warn|state_dict|torch.load" | cut -c1-330; echo FINI
grep "\"step\"" t1-reel/logs/g_s1_trustmark.log | tail -n 3 | cut -c1-330; grep "\"step\"" t1-reel/logs/g_dec_pixelseal.log | tail -n 2 | cut -c1-330
