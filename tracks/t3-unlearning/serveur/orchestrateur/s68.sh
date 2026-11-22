for p in $(pgrep -u adem -f "name PG_C9"); do kill $p; done
for p in $(pgrep -u adem -f "finish.sh PG_C9"); do pkill -P $p; kill $p; done
sleep 2; date +%H:%M:%S; tail -2 ~/eac/t3/orchestrateur/sub_PG_T85_f70_l814c.log | cut -c1-200
nvidia-smi --query-gpu=memory.used --format=csv,noheader
cd ~/eac/t3/t3-methodes; ls -t candidates | head -6; ls -t logs/*.log | head -8
