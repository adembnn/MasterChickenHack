cd ~/eac/t3/orchestrateur
for p in $(pgrep -u adem -f "bash file.sh"); do kill $p; done
for p in $(pgrep -u adem -f "sub.sh .*r_70_l1117"); do pkill -P $p; kill $p; done
sleep 1
(nohup bash file.sh SOUPs1 SOUP4_RDt24r_70 PG_RDt24r_65_l1016c PG_RDt24r_75_l1016c > file7.log 2>&1 &)
echo "file 7 lancee"; pgrep -u adem -fa "sub.sh" | cut -c1-100
