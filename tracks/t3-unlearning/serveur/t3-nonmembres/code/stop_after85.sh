R=$HOME/eac/t3/t3-nonmembres/sim
until [ -f $R/forget/root/meas/eval_sdxl_i2i_85.json ] && [ -f $R/nonmembers/root/meas/eval_sdxl_i2i_85.json ]; do sleep 15; done
pkill -f "[h]chain.sh"; sleep 1; pkill -f "[g]en_sdxl.py"; echo stopped $(date +%H:%M:%S)
