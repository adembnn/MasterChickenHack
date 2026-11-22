cd ~/eac/t3/orchestrateur
for p in $(pgrep -u adem -f "bash file.sh"); do kill $p; done; sleep 1
(nohup bash file.sh PG_RDt24r_72_l1016c PG_RDt24r_68_l1016c PGB6_RDt24r_70 PG_RDp_70_l1016c > file8.log 2>&1 &)
sleep 1; pgrep -u adem -fa "sub.sh" | cut -c1-100
