for n in PGs3kl6 PGs6kl25 PGB6l6; do for p in $(pgrep -u adem -f "finish.sh ${n}_f70_l814c"); do pkill -P $p; kill $p; done; done; echo nettoye
date +%H:%M:%S; tail -2 ~/eac/t3/orchestrateur/sub_PG_RDt24_60_l915c.log | cut -c1-200
