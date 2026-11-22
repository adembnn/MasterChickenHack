cd ~/eac/t3/t3-nonmembres
pkill -f "[a]fter_simN.sh"
# ne garder que la force 0,85 pour les deux jeux de calibration, puis arrêter
cat > code/stop_after85.sh <<'EOS'
R=$HOME/eac/t3/t3-nonmembres/sim
until [ -f $R/forget/root/meas/eval_sdxl_i2i_85.json ] && [ -f $R/nonmembers/root/meas/eval_sdxl_i2i_85.json ]; do sleep 15; done
pkill -f "[h]chain.sh"; sleep 1; pkill -f "[g]en_sdxl.py"; echo stopped $(date +%H:%M:%S)
EOS
setsid nohup bash code/stop_after85.sh > logs/stop_after85.log 2>&1 < /dev/null &
sleep 1; pgrep -af "[a]fter_simN|[s]top_after85|[h]chain" | cut -c1-80; for m in forget nonmembers; do for f in sim/$m/root/logs/sdxl_i2i_85.log; do [ -f $f ] && echo "$m $(grep -v CLIP $f | tail -n 1 | cut -c1-60)"; done; done; date +%H:%M:%S
