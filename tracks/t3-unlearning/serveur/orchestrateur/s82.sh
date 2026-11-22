cd ~/eac/t3/orchestrateur
for p in $(pgrep -u adem -f "bash file.sh"); do kill $p; done
for p in $(pgrep -u adem -f "sub.sh .*r_70_l1117"); do pkill -P $p; kill $p; done
sleep 1
(nohup bash file.sh PG_RDt24r_60_l1016c PG_RDt24r_70_l1117c > file6.log 2>&1 &)
echo "file 6 lancee"; pgrep -u adem -fa "sub.sh" | cut -c1-100
