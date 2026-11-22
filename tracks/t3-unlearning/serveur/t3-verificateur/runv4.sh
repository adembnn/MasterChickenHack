#!/bin/bash
cd ~/eac/t3/t3-verificateur; export T3ROOT=$HOME/eac/t3; S=$HOME/eac/t3
free=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $2-$1}')
[ "$free" -lt 12000 ] && export LOWMEM=1
args=""; for n in "$@"; do args="$args $n=$S/t3-methodes/candidates/$n.pt"; done
echo "libre $free Mo LOWMEM=$LOWMEM" > out/vlogits4.log
../.venv/bin/python vlogits.py $args >> out/vlogits4.log 2>&1
